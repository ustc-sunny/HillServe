#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace hillserve {
// CUDA-only implementation. No ggml state is accessed from background threads.
class GQAPrefetchCuda {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    struct Upload { std::uint64_t started_ns=0,ready_observed_ns=0,bytes=0; float device_ms=0; };
    struct Delivery { std::uint64_t started_ns=0,ready_observed_ns=0,h2d_bytes=0,d2d_bytes=0; double wait_ms=0; };
    GQAPrefetchCuda(void* actual_gpu_pointer,std::size_t max_rows);
    ~GQAPrefetchCuda();
    GQAPrefetchCuda(const GQAPrefetchCuda&)=delete;
    GQAPrefetchCuda& operator=(const GQAPrefetchCuda&)=delete;
    std::size_t gpu_bytes() const;
    std::size_t host_bytes() const;
    Upload stage(unsigned slot,const std::vector<std::uint8_t>& payload);
    // map[head*count+slot]: >=0 prefetched row, <=-2 miss row (-2-index), -1 padding.
    Delivery deliver(unsigned slot,const std::vector<std::int32_t>& map,
        const std::vector<std::uint8_t>& misses,const float* mask,std::size_t mask_elements,
        std::size_t count,void* k,void* v,void* gpu_mask,void* actual_backend_stream);
};
}
