#include "runtime/gqa_h2o_stats.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

namespace hillserve {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
std::size_t checked_product(std::size_t a, std::size_t b) {
    require(!b || a <= std::numeric_limits<std::size_t>::max()/b,
            "H2O attention statistics size overflow");
    return a*b;
}
class DeviceScope {
    int previous_ = 0;
public:
    explicit DeviceScope(int device) {
        check(cudaGetDevice(&previous_), "H2O get CUDA device");
        if (previous_ != device) check(cudaSetDevice(device), "H2O set CUDA device");
    }
    ~DeviceScope() { (void)cudaSetDevice(previous_); }
};

// Adjacent threads read adjacent keys for normal ggml strides. Each output
// uses deterministic F32 accumulation and never allocates a quadratic tensor.
// Keep the order (group lane, query) identical to H2OResidency's CPU path.
__global__ void sum_actual_attention(const unsigned char* probabilities,
                                    std::size_t keys, std::size_t queries,
                                    std::size_t nb0, std::size_t nb1,
                                    std::size_t nb2, float* output) {
    const std::size_t key = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const std::size_t kv_head = blockIdx.y;
    if (key >= keys) return;
    float sum = 0.0f;
    for (std::size_t lane = 0; lane < 4; ++lane) {
        for (std::size_t q = 0; q < queries; ++q) {
            const auto* p = reinterpret_cast<const float*>(probabilities +
                key*nb0 + q*nb1 + (kv_head*4 + lane)*nb2);
            sum = __fadd_rn(sum, *p);
        }
    }
    output[kv_head*keys + key] = sum;
}
} // namespace

struct GQAH2OStatsCuda::Impl {
    int device = 0;
    std::size_t keys = 0, bytes = 0;
    float* gpu = nullptr;
    float* host = nullptr;
    cudaEvent_t begin = nullptr, reduced = nullptr, copied = nullptr;
    cudaStream_t pending_stream = nullptr;
    bool pending = false;
    ~Impl() {
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        (void)cudaSetDevice(device);
        // If submission failed before the completion event was recorded, the
        // original stream may still refer to our allocations. Drain it first.
        if (pending) (void)cudaStreamSynchronize(pending_stream);
        if (begin) (void)cudaEventDestroy(begin);
        if (reduced) (void)cudaEventDestroy(reduced);
        if (copied) (void)cudaEventDestroy(copied);
        if (gpu) (void)cudaFree(gpu);
        if (host) (void)cudaFreeHost(host);
        if (restore) (void)cudaSetDevice(previous);
    }
};

GQAH2OStatsCuda::GQAH2OStatsCuda(const void* reference, std::size_t capacity)
    : impl_(new Impl) {
    require(reference && capacity, "H2O statistics needs CUDA pointer and nonzero capacity");
    auto& p = *impl_;
    p.keys = capacity;
    p.bytes = checked_product(checked_product(capacity, 8), sizeof(float));
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, reference), "H2O reference pointer attributes");
    require(attributes.type == cudaMemoryTypeDevice,
            "H2O attention statistics requires CUDA device memory");
    p.device = attributes.device;
    DeviceScope device(p.device);
    check(cudaMalloc(reinterpret_cast<void**>(&p.gpu), p.bytes), "H2O reduction allocation");
    check(cudaMallocHost(reinterpret_cast<void**>(&p.host), p.bytes), "H2O pinned sums allocation");
    check(cudaEventCreate(&p.begin), "H2O begin event");
    check(cudaEventCreate(&p.reduced), "H2O reduction event");
    check(cudaEventCreate(&p.copied), "H2O D2H completion event");
}
GQAH2OStatsCuda::~GQAH2OStatsCuda() = default;
std::size_t GQAH2OStatsCuda::gpu_bytes() const noexcept { return impl_->bytes; }
std::size_t GQAH2OStatsCuda::host_bytes() const noexcept { return impl_->bytes; }

GQAH2OStatsCuda::Result GQAH2OStatsCuda::reduce(
    const void* probabilities, const std::array<std::int64_t, 4>& ne,
    const std::array<std::size_t, 4>& nb, std::size_t keys,
    std::size_t queries, void* raw_stream) {
    auto& p = *impl_;
    require(probabilities && raw_stream && keys && queries && keys <= p.keys,
            "H2O reduction invalid pointer, actual CUDA stream or capacity");
    require(ne[0] > 0 && ne[1] > 0 && ne[2] == 32 && ne[3] == 1 &&
            keys <= static_cast<std::uint64_t>(ne[0]) &&
            queries <= static_cast<std::uint64_t>(ne[1]),
            "H2O reduction requires [keys,queries,32,1] actual attention");
    std::size_t maximum_offset = 0;
    const std::size_t extents[3] = { keys, queries, 32 };
    for (unsigned axis = 0; axis < 3; ++axis) {
        require(nb[axis] >= sizeof(float) && nb[axis] % alignof(float) == 0,
                "H2O reduction invalid F32 tensor stride");
        const auto offset = checked_product(extents[axis] - 1, nb[axis]);
        require(offset <= std::numeric_limits<std::size_t>::max() - maximum_offset,
                "H2O attention statistics address overflow");
        maximum_offset += offset;
    }
    require(maximum_offset <= std::numeric_limits<std::size_t>::max() - sizeof(float),
            "H2O attention statistics final address overflow");
    require(reinterpret_cast<std::uintptr_t>(probabilities) % alignof(float) == 0,
            "H2O reduction requires aligned F32 tensor");
    const auto blocks = (keys - 1)/256 + 1;
    require(blocks <= 2147483647u, "H2O reduction exceeds CUDA grid limit");
    DeviceScope device(p.device);
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, probabilities), "H2O softmax pointer attributes");
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == p.device,
            "H2O softmax tensor is not on the expected CUDA device");
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    Result out;
    out.elements = keys*8;
    out.d2h_bytes = out.elements*sizeof(float);
    out.started_ns = now_ns();
    p.pending_stream = stream;
    p.pending = true;
    try {
        check(cudaEventRecord(p.begin, stream), "H2O reduction start record");
        sum_actual_attention<<<dim3(static_cast<unsigned>(blocks), 8), 256, 0, stream>>>(
            static_cast<const unsigned char*>(probabilities), keys, queries,
            nb[0], nb[1], nb[2], p.gpu);
        check(cudaGetLastError(), "H2O attention reduction launch");
        check(cudaEventRecord(p.reduced, stream), "H2O reduction ready record");
        check(cudaMemcpyAsync(p.host, p.gpu, out.d2h_bytes, cudaMemcpyDeviceToHost, stream),
              "H2O group sums D2H");
        check(cudaEventRecord(p.copied, stream), "H2O D2H ready record");
        check(cudaEventSynchronize(p.copied), "H2O group sums completion");
        out.ready_observed_ns = now_ns();
        p.pending = false;
        p.pending_stream = nullptr;
        float reduction_ms = 0, d2h_ms = 0;
        check(cudaEventElapsedTime(&reduction_ms, p.begin, p.reduced), "H2O reduction interval");
        check(cudaEventElapsedTime(&d2h_ms, p.reduced, p.copied), "H2O D2H interval");
        out.reduction_ms = reduction_ms;
        out.d2h_ms = d2h_ms;
        out.sums = p.host;
        return out;
    } catch (...) {
        if (cudaStreamSynchronize(stream) == cudaSuccess) {
            p.pending = false;
            p.pending_stream = nullptr;
        }
        throw;
    }
}
} // namespace hillserve
