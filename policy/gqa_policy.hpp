#pragma once
#include "smartssd/adapter.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hillserve::policy {
// Named online extension of the inspected public Llama accuracy profile.
// Scores are F16 AFTER dividing the F16 dot by sqrt(128), with head0 maximum
// broadcast, inclusive threshold, F16 count/mean rounding, and original-ID ties.
std::vector<std::uint16_t> gqa_skew_head_f16(const std::uint16_t* input,const float* basis,const std::uint8_t* query_mask=nullptr);
struct GQASelection {
    std::vector<std::vector<std::uint64_t>> ids;
    std::vector<std::uint32_t> counts;
    std::size_t k=0, boundary_tied_heads=0;
};
GQASelection select_gqa_scores(const std::vector<std::uint16_t>& scaled_scores,
    std::size_t heads,std::size_t stride,std::size_t valid_tokens,float alpha,float cap_ratio,
    bool fp32_counts=false);

class GQAPolicy {
public:
    struct Stats {
        std::uint64_t file_extent_bytes=0, append_bytes=0, score_read_bytes=0,
            logical_valid_score_key_bytes=0, key_host_staging_bytes=0,
            query_h2d_bytes=0, key_h2d_bytes=0, score_d2h_bytes=0, score_calls=0,
            append_syscall_bytes=0, append_rmw_read_bytes=0, cache_hits=0, cache_misses=0,
            cache_hit_valid_bytes=0, verification_score_count=0, verification_bit_errors=0,
            valid_candidates=0, kernel_rows=0, actual_fpga_calls=0, actual_fpga_rows=0, selected_pairs=0, drain_calls=0;
        double transform_ms=0, write_ms=0, file_read_ms=0, score_ms=0,
            select_ms=0, launch_wait_ms=0, drain_ms=0, verification_ms=0;
        bool p2p_api_path_completed=false;
    };
    GQAPolicy(const std::string& artifact,int capacity,float alpha,float cap_ratio,
              const std::string& score_path="");
    ~GQAPolicy();
    GQAPolicy(const GQAPolicy&)=delete;
    GQAPolicy& operator=(const GQAPolicy&)=delete;
    void configure_backend(const std::string& backend,const HardwareContract& contract={},bool direct_reads=false);
    void enable_direct_writes();
    void configure_cpu_cache(std::size_t cache_budget_bytes,unsigned threads);
    void verify_optimized_scores(bool enabled){verify_optimized_=enabled;}
    void append(int layer,const std::vector<std::uint64_t>& positions,const std::vector<std::uint16_t>& keys);
    std::vector<std::vector<std::uint64_t>> select(int layer,std::uint64_t position,const std::vector<float>& rehearsal);
    void drain();
    std::size_t managed_bytes() const;
    const Stats& stats() const {return stats_;}
    void write_stats(const std::string& path) const;
private:
    static constexpr std::size_t layers_=32,heads_=32,kv_heads_=8,dim_=128;
    std::size_t capacity_,layer_bytes_;float alpha_,ratio_;
    std::string path_,backend_="host";
    int fd_=-1,direct_fd_=-1,direct_write_fd_=-1,cached_basis_layer_=-1;
    unsigned cpu_threads_=1;bool optimized_cpu_=false,verify_optimized_=false;
    std::size_t cache_budget_=0,cache_allocated_=0;
    std::array<std::vector<std::uint16_t>,32> compressed_cache_;
    std::array<std::array<std::vector<std::size_t>,32>,32> channels_;
    std::array<std::array<std::size_t,32>,32> cache_head_offsets_{};
    std::array<std::uint64_t,32> completed_{};
    std::vector<std::uint16_t> basis_;
    std::vector<std::uint8_t> mask_;
    std::vector<float> active_basis_;
    std::unique_ptr<FpgaScoreBackend> fpga_;
    Stats stats_;
    struct Event {int layer=0;std::uint64_t position=0,started_ns=0,ended_ns=0,read_bytes=0,key_host_bytes=0,key_h2d=0,q_h2d=0,d2h=0;std::size_t k=0;bool hot=false;std::vector<std::uint32_t> counts;};
    std::vector<Event> events_;
    void basis_for(int layer);
    std::vector<std::uint16_t> transform(int layer,std::size_t qhead,const std::uint16_t* input,bool mask_query) const;
    void read_layer(int layer,std::vector<std::uint16_t>& keys);
    void write_region(const void*,std::size_t,std::uint64_t);
    std::vector<std::uint16_t> cpu_scores(int layer,const std::vector<std::uint16_t>& q,
        const std::vector<std::uint16_t>* keys,std::size_t valid) const;
};
}
