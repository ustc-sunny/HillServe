#include "policy/full_gqa_cpu.hpp"
#include "policy/host_policy.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define HILLSERVE_X86_F16C 1
#include <immintrin.h>
#else
#define HILLSERVE_X86_F16C 0
#endif

namespace hillserve {
namespace {
constexpr std::size_t H = 32, KH = 8, D = 128, GROUP = 4, TILE = 8;
// Each channel has the four GQA values duplicated for two independent tokens.
using Query = std::array<float, KH * D * 8>;
void scalar_tile(const Query& query, const std::uint16_t* const* keys,
                 std::size_t n, std::size_t begin, std::size_t end,
                 std::size_t kh, std::uint16_t* output) noexcept {
    for (std::size_t t = begin; t < end; ++t) {
        const auto* key = keys[t] + kh * D;
        float sums[GROUP] = {};
        for (std::size_t c = 0; c < D; ++c) {
            const float k = half_value(key[c]);
            for (std::size_t g = 0; g < GROUP; ++g)
                sums[g] += query[(kh * D + c) * 8 + g] * k;
        }
        for (std::size_t g = 0; g < GROUP; ++g)
            output[(kh * GROUP + g) * n + t] = half_bits(sums[g]);
    }
}
#if HILLSERVE_X86_F16C
// No horizontal reduction: each lane is one independent (token,Q-head) score.
// Four independent accumulators hide add latency. F16C converts four channels
// from each of two physical keys; permutations broadcast the requested channel
// to its four Q heads. Every lane still executes c=0..127 in original order.
template<std::size_t pairs>
__attribute__((always_inline,target("avx2,f16c,no-fma")))
inline void avx2_tile_pairs(const Query& query, const std::uint16_t* const* keys,
               std::size_t n, std::size_t begin, std::size_t end,
               std::size_t kh, std::uint16_t* output) noexcept {
    __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                      _mm256_setzero_ps(), _mm256_setzero_ps()};
    const std::uint16_t* first[4];
    const std::uint16_t* second[4];
    for (std::size_t p = 0; p < pairs; ++p) {
        const std::size_t t = begin + p * 2;
        first[p] = keys[t] + kh * D;
        second[p] = keys[std::min(t + 1, end - 1)] + kh * D;
    }
    const __m256i indices[4] = {
        _mm256_setr_epi32(0,0,0,0,4,4,4,4),
        _mm256_setr_epi32(1,1,1,1,5,5,5,5),
        _mm256_setr_epi32(2,2,2,2,6,6,6,6),
        _mm256_setr_epi32(3,3,3,3,7,7,7,7)};
    for (std::size_t c = 0; c < D; c += 4) {
        __m256 k[4];
        for (std::size_t p = 0; p < pairs; ++p) {
            const auto a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(first[p] + c));
            const auto b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(second[p] + c));
            k[p] = _mm256_cvtph_ps(_mm_unpacklo_epi64(a, b));
        }
        for (std::size_t j = 0; j < 4; ++j) {
            const auto q = _mm256_loadu_ps(query.data() + (kh * D + c + j) * 8);
            for (std::size_t p = 0; p < pairs; ++p) {
                const auto value = _mm256_permutevar8x32_ps(k[p], indices[j]);
                const auto product = _mm256_mul_ps(q, value);
                sums[p] = _mm256_add_ps(sums[p], product);
            }
        }
    }
    for (std::size_t p = 0; p < pairs; ++p) {
        alignas(16) std::uint16_t packed[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(packed),
            _mm256_cvtps_ph(sums[p], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        for (std::size_t lane = 0; lane < 8; ++lane) {
            const std::size_t t = begin + p * 2 + lane / GROUP;
            if (t >= end) break;
            auto value = packed[lane];
            // Match half_bits' canonical NaN if nonfinite inputs were supplied.
            if ((value & 0x7c00) == 0x7c00 && (value & 0x3ff))
                value = std::uint16_t((value & 0x8000) | 0x7e00);
            output[(kh * GROUP + lane % GROUP) * n + t] = value;
        }
    }
}
__attribute__((target("avx2,f16c,no-fma")))
void avx2_tile(const Query& query, const std::uint16_t* const* keys,
               std::size_t n, std::size_t begin, std::size_t end,
               std::size_t kh, std::uint16_t* output) noexcept {
    switch ((end - begin + 1) / 2) {
    case 4: avx2_tile_pairs<4>(query,keys,n,begin,end,kh,output); break;
    case 3: avx2_tile_pairs<3>(query,keys,n,begin,end,kh,output); break;
    case 2: avx2_tile_pairs<2>(query,keys,n,begin,end,kh,output); break;
    default: avx2_tile_pairs<1>(query,keys,n,begin,end,kh,output); break;
    }
}
#endif
}

// Original arithmetic and per-call worker loop are shared with the legacy
// replay. Keep the automatic pooled/AVX2 implementation below unchanged.
std::vector<std::uint16_t> score_full_gqa_legacy_cpu(const std::vector<std::uint16_t>& q,
        const std::uint16_t* input,std::size_t n,std::size_t token_stride,unsigned threads){
    if(!(q.size()==H*D&&input&&n>0&&(token_stride==1024||token_stride==2048)
         &&threads>=1&&threads<=32))
        throw std::invalid_argument("raw128 physical GQA CPU shape/stride/thread bound");
    if(n>std::numeric_limits<std::size_t>::max()/H ||
       n>std::numeric_limits<std::size_t>::max()/token_stride)
        throw std::invalid_argument("legacy full GQA CPU extent overflow");
    if(std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("legacy full GQA CPU requires round-to-nearest FP32 mode");
    alignas(16) std::array<std::array<std::array<float,GROUP>,D>,KH> query{};
    for(std::size_t kh=0;kh<KH;++kh)for(std::size_t c=0;c<D;++c)for(std::size_t group=0;group<GROUP;++group)
        query[kh][c][group]=half_value(q[(kh*GROUP+group)*D+c]);
    std::vector<std::uint16_t> result(H*n);
    auto worker=[&](unsigned index){
        for(std::size_t item=index;item<n*KH;item+=threads){
            const auto t=item/KH,kh=item%KH;const auto* key=input+t*token_stride+kh*D;
            alignas(16) std::array<float,GROUP> sum{};
            for(std::size_t c=0;c<D;++c){const float value=half_value(key[c]);
                for(std::size_t group=0;group<GROUP;++group)sum[group]+=query[kh][c][group]*value;
            }
            for(std::size_t group=0;group<GROUP;++group)result[(kh*GROUP+group)*n+t]=half_bits(sum[group]);
        }
    };
    std::vector<std::thread> workers;
    try{for(unsigned i=1;i<threads;++i)workers.emplace_back(worker,i);worker(0);}
    catch(...){for(auto& thread:workers)thread.join();throw;}
    for(auto& thread:workers)thread.join();
    return result;
}

struct FullGqaCpuScorer::Impl {
    unsigned count;
    bool avx;
    std::mutex invocation, mutex;
    std::condition_variable start, completed;
    bool stop = false;
    std::size_t generation = 0, pending = 0;
    std::vector<std::thread> workers;
    alignas(32) Query query{};
    std::vector<const std::uint16_t*> keys;
    std::uint16_t* output = nullptr;
    std::size_t n = 0;

    Impl(unsigned threads, FullGqaCpuImplementation implementation) : count(threads), avx(false) {
        if (std::fegetround() != FE_TONEAREST)
            throw std::invalid_argument("full GQA CPU workers require round-to-nearest FP32 mode");
        if (count == 0 || count > 128) throw std::invalid_argument("full GQA CPU threads must be 1..128");
        if (implementation != FullGqaCpuImplementation::automatic &&
            implementation != FullGqaCpuImplementation::scalar &&
            implementation != FullGqaCpuImplementation::avx2_f16c)
            throw std::invalid_argument("unknown full GQA CPU implementation");
        const bool available = FullGqaCpuScorer::avx2_f16c_available();
        if (implementation == FullGqaCpuImplementation::avx2_f16c && !available)
            throw std::runtime_error("explicit AVX2/F16C scorer is unavailable on this CPU");
        avx = implementation != FullGqaCpuImplementation::scalar && available;
        try {
            for (unsigned i = 1; i < count; ++i) workers.emplace_back([this,i] { worker(i); });
        } catch (...) {
            shutdown();
            throw;
        }
    }
    ~Impl() { shutdown(); }
    void shutdown() noexcept {
        { std::lock_guard<std::mutex> guard(mutex); stop = true; }
        start.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }
    void work(unsigned index) noexcept {
        const std::size_t tiles = (n + TILE - 1) / TILE;
        const std::size_t total = tiles * KH;
        const std::size_t first = total * index / count;
        const std::size_t last = total * (index + 1) / count;
        for (std::size_t item = first; item < last; ++item) {
            const auto t = (item / KH) * TILE;
            const auto kh = item % KH;
#if HILLSERVE_X86_F16C
            if (avx) avx2_tile(query, keys.data(), n, t, std::min(n, t + TILE), kh, output);
            else
#endif
                scalar_tile(query, keys.data(), n, t, std::min(n, t + TILE), kh, output);
        }
    }
    void worker(unsigned index) noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        std::size_t observed = 0;
        while (true) {
            start.wait(lock, [&] { return stop || generation != observed; });
            if (stop) return;
            observed = generation;
            lock.unlock();
            work(index);
            lock.lock();
            if (--pending == 0) completed.notify_one();
        }
    }
    void score(const std::vector<std::uint16_t>& q,
               const std::vector<FullGqaKeyView>& views,
               std::vector<std::uint16_t>& result) {
        std::lock_guard<std::mutex> call(invocation);
        if (std::fegetround() != FE_TONEAREST)
            throw std::invalid_argument("full GQA CPU scoring requires round-to-nearest FP32 mode");
        if (&q == &result)
            throw std::invalid_argument("full GQA CPU output must not alias the query");
        if (q.size() != H * D || views.empty())
            throw std::invalid_argument("full GQA CPU query/views shape");
        std::size_t tokens = 0;
        for (const auto& view : views) {
            if (!view.data || view.tokens == 0 ||
                (view.token_stride_words != 1024 && view.token_stride_words != 2048) ||
                view.tokens > (std::numeric_limits<std::size_t>::max() / view.token_stride_words) ||
                view.tokens > std::numeric_limits<std::size_t>::max() / H - tokens)
                throw std::invalid_argument("full GQA CPU key view shape/stride/extent");
            tokens += view.tokens;
        }
        // total work item multiplication also needs to fit even for tiny views.
        if (tokens > std::numeric_limits<std::size_t>::max() / (KH * count))
            throw std::invalid_argument("full GQA CPU work extent");
        result.resize(tokens * H);
        keys.resize(tokens);
        std::size_t cursor = 0;
        for (const auto& view : views)
            for (std::size_t t = 0; t < view.tokens; ++t)
                keys[cursor++] = view.data + t * view.token_stride_words;
        for (std::size_t kh = 0; kh < KH; ++kh)
            for (std::size_t c = 0; c < D; ++c)
                for (std::size_t g = 0; g < GROUP; ++g) {
                    const float value = half_value(q[(kh * GROUP + g) * D + c]);
                    query[(kh * D + c) * 8 + g] = value;
                    query[(kh * D + c) * 8 + g + GROUP] = value;
                }
        {
            std::lock_guard<std::mutex> guard(mutex);
            output = result.data(); n = tokens;
            pending = workers.size(); ++generation;
        }
        start.notify_all();
        work(0);
        std::unique_lock<std::mutex> lock(mutex);
        completed.wait(lock, [&] { return pending == 0; });
    }
};

FullGqaCpuScorer::FullGqaCpuScorer(unsigned threads, FullGqaCpuImplementation implementation)
    : impl_(new Impl(threads, implementation)) {}
FullGqaCpuScorer::~FullGqaCpuScorer() = default;
void FullGqaCpuScorer::score(const std::vector<std::uint16_t>& query,
    const std::vector<FullGqaKeyView>& views, std::vector<std::uint16_t>& output) {
    impl_->score(query, views, output);
}
unsigned FullGqaCpuScorer::threads() const noexcept { return impl_->count; }
const char* FullGqaCpuScorer::implementation_name() const noexcept {
    return impl_->avx ? "avx2_f16c_ordered_fp32" : "scalar_ordered_fp32";
}
std::size_t FullGqaCpuScorer::workspace_bytes() const noexcept {
    return sizeof(Impl) + impl_->keys.capacity() * sizeof(const std::uint16_t*) +
           impl_->workers.capacity() * sizeof(std::thread);
}
bool FullGqaCpuScorer::avx2_f16c_available() noexcept {
#if HILLSERVE_X86_F16C
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}
}
