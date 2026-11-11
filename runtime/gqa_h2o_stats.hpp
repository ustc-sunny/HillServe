#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace hillserve {

// Reduces the *actual* non-flash F32 softmax tensor consumed by attention.
// This is instrumentation for H2O residency, not a second QK scoring path.
// Its only payload allocations are 8*capacity_keys floats on CUDA and host.
class GQAH2OStatsCuda {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    struct Result {
        // Borrowed until the next reduce() or destruction. Layout [8][valid_keys].
        const float* sums = nullptr;
        std::size_t elements = 0;
        std::uint64_t started_ns = 0;
        std::uint64_t ready_observed_ns = 0;
        std::uint64_t d2h_bytes = 0;
        double reduction_ms = 0;
        double d2h_ms = 0;
    };
    GQAH2OStatsCuda(const void* actual_gpu_pointer, std::size_t capacity_keys);
    ~GQAH2OStatsCuda();
    GQAH2OStatsCuda(const GQAH2OStatsCuda&) = delete;
    GQAH2OStatsCuda& operator=(const GQAH2OStatsCuda&) = delete;
    std::size_t gpu_bytes() const noexcept;
    std::size_t host_bytes() const noexcept;

    // ne = [key, query, Q head, stream]; byte_strides are the actual ggml nb[].
    // Only B=1, 32 Q heads / 8 physical KV heads are supported. Padding keys
    // and queries are excluded. Each output sums all valid queries and the
    // four Q heads sharing its physical KV head; no max/mean approximation.
    // The caller verifies GGML_TYPE_F32 and that key slots are absolute token
    // positions. It must invoke this before probabilities storage is reused.
    // The borrowed stream must be the actual attention backend CUDA stream.
    // A completion event on that stream makes the returned host data ready.
    Result reduce(const void* probabilities,
                  const std::array<std::int64_t, 4>& ne,
                  const std::array<std::size_t, 4>& byte_strides,
                  std::size_t valid_keys, std::size_t valid_queries,
                  void* actual_attention_stream);
};

} // namespace hillserve
