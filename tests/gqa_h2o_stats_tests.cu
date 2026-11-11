#include "runtime/gqa_h2o_stats.hpp"
#include "policy/h2o_residency.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
template<class F> void rejects(F operation, const char* reason) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    require(threw, reason);
}
struct CUDAResources {
    void* gpu = nullptr;
    cudaStream_t stream = nullptr;
    ~CUDAResources() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (gpu) (void)cudaFree(gpu);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
} // namespace

int main() { try {
    // Holes between keys, padding queries/keys, and extra per-head spacing.
    // Poison padding with NaNs so accidental reduction cannot pass silently.
    const std::array<std::int64_t, 4> ne{{11, 6, 32, 1}};
    const std::array<std::size_t, 4> nb{{8, 112, 704, 22528}};
    std::vector<float> storage(nb[3]/sizeof(float), std::numeric_limits<float>::quiet_NaN());
    CUDAResources resources;
    check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "fixture stream");
    check(cudaMalloc(&resources.gpu, nb[3]), "fixture device allocation");
    hillserve::GQAH2OStatsCuda reducer(resources.gpu, 7);
    require(reducer.gpu_bytes() == 8*7*sizeof(float) && reducer.host_bytes() == 8*7*sizeof(float),
            "reducer allocated more than its linear payload bound");
    hillserve::policy::H2OResidency::Config config;
    config.layers = 1; config.prompt_tokens = 5; config.generated_capacity = 2;
    config.prompt_resident_tokens_per_head = 3; config.metadata_budget_bytes = 1 << 20;
    hillserve::policy::H2OResidency gpu_policy(config), cpu_policy(config);
    for (std::size_t first : {0u, 3u}) {
        const std::size_t queries = first == 0 ? 3 : 2, keys = first + queries;
        std::fill(storage.begin(), storage.end(), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> contiguous(32*queries*keys, 0);
        for (std::size_t head = 0; head < 32; ++head) {
            for (std::size_t q = 0; q < queries; ++q) {
                const auto available = first + q + 1;
                double denominator = 0;
                for (std::size_t k = 0; k < available; ++k) denominator += 1 + ((k + head/4)%5);
                for (std::size_t k = 0; k < keys; ++k) {
                    const float value = k < available ? float((1 + ((k + head/4)%5))/denominator) : 0;
                    const auto offset = head*nb[2] + q*nb[1] + k*nb[0];
                    std::memcpy(reinterpret_cast<unsigned char*>(storage.data()) + offset, &value, sizeof(float));
                    contiguous[(head*queries + q)*keys + k] = value;
                }
            }
        }
        check(cudaMemcpyAsync(resources.gpu, storage.data(), nb[3], cudaMemcpyHostToDevice, resources.stream),
              "fixture asynchronous probabilities upload");
        const auto reduced = reducer.reduce(resources.gpu, ne, nb, keys, queries, resources.stream);
        require(reduced.sums && reduced.elements == 8*keys && reduced.d2h_bytes == 8*keys*sizeof(float),
                "reduced shape or D2H bytes are incorrect");
        require(reduced.ready_observed_ns >= reduced.started_ns && reduced.reduction_ms >= 0 && reduced.d2h_ms >= 0,
                "completion timing invalid");
        for (std::size_t kh = 0; kh < 8; ++kh) {
            for (std::size_t k = 0; k < keys; ++k) {
                float expected = 0;
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    for (std::size_t q = 0; q < queries; ++q) {
                        expected += contiguous[((kh*4 + lane)*queries + q)*keys + k];
                    }
                }
                require(reduced.sums[kh*keys + k] == expected,
                        "strided CUDA F32 sum disagrees with independent CPU order");
            }
        }
        gpu_policy.observe_group_sums(0, first, queries, keys, reduced.sums, reduced.elements);
        cpu_policy.observe_attention(0, first, queries, keys, contiguous.data(), contiguous.size());
        require(gpu_policy.resident_ids(0) == cpu_policy.resident_ids(0),
                "GPU-reduced actual attention changed resident IDs");
        for (std::size_t kh = 0; kh < 8; ++kh) for (std::size_t k = 0; k < keys; ++k) {
            require(gpu_policy.importance(0, kh, k) == cpu_policy.importance(0, kh, k),
                    "GPU and CPU accumulated importance differs");
        }
    }
    auto bad_heads = ne; bad_heads[2] = 8;
    rejects([&] { reducer.reduce(resources.gpu, bad_heads, nb, 5, 2, resources.stream); }, "non-GQA32 head shape admitted");
    auto bad_streams = ne; bad_streams[3] = 2;
    rejects([&] { reducer.reduce(resources.gpu, bad_streams, nb, 5, 2, resources.stream); }, "multi-sequence shape admitted");
    rejects([&] { reducer.reduce(resources.gpu, ne, nb, 8, 2, resources.stream); }, "capacity overrun admitted");
    rejects([&] { reducer.reduce(resources.gpu, ne, nb, 5, 7, resources.stream); }, "padding query overrun admitted");
    rejects([&] { reducer.reduce(resources.gpu, ne, nb, 5, 2, nullptr); }, "missing actual attention stream admitted");
    std::cout << "passed: CUDA real-stream F32 GQA sum, strided/padded probabilities, bounded buffers, CPU policy agreement and shape guards\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
} }
