#pragma once
#include "smartssd/adapter.hpp"
#include "runtime/gqa_prefetch.hpp"
#include "storage/kv_read_plan.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hillserve::policy {
// Fixed full128 contract. This class never changes the rehearsal producer,
// head0 threshold, per-query-head selected IDs, or the original-position mask.
class FullGqaRuntimePolicy {
public:
    struct Config {
        std::size_t layers=32, capacity=0, prompt_tokens=0, generated_capacity=64;
        bool allow_partial_prompt=false;
        bool fp32_selection_counts=false;
        double beta=1.0;
        // Explicit sensitivity variant only. The historical v6 default stays5;
        // prefetch and final selection share this immutable normalized-score alpha.
        float selector_alpha=5.f;
        std::size_t hot_kv_budget_bytes=0, total_host_workspace_budget_bytes=0;
        std::size_t chunk_tokens=4096;
        unsigned hot_threads=3, cold_threads=1;
        std::string mode="HOST_MATCHED", score_file;
        // auto preserves the pooled optimized scorer; legacy selects the shared
        // original software-half/per-call-thread implementation explicitly.
        std::string cpu_implementation="auto";
        bool verify_scores=false;
        bool hpm=false;
        std::size_t hpm_swaps=1;
        bool payload_prefetch=false;
        storage::KVReadLimits payload_read_limits;
        unsigned payload_read_queue_depth=4; // complete-KV host IO only; FPGA raw-K QD unchanged
        std::string full_kv_backing_path;
        std::shared_ptr<runtime::GQAPrefetchIOFence> full_kv_io_fence;
        HardwareContract hardware;
    };
    struct LayerStats {
        std::uint64_t completed_tokens=0, append_calls=0, append_key_bytes=0;
        std::uint64_t append_syscall_bytes=0, append_rmw_read_bytes=0;
        std::uint64_t score_calls=0, fpga_calls=0, scored_candidates=0;
        std::uint64_t cold_read_bytes=0, query_h2d_bytes=0, score_d2h_bytes=0;
        std::uint64_t hot_fetch_bytes=0, cold_fetch_bytes=0, fetch_calls=0;
        std::uint64_t verification_scores=0, verification_read_bytes=0;
        std::uint64_t history_prefetch_calls=0,current_token_score_calls=0;
        std::uint64_t hot_cpu_packed_key_bytes=0,hot_cpu_calls=0,cold_cpu_calls=0;
        double current_token_score_ms=0;
        std::uint64_t hpm_updates=0,hpm_promoted_blocks=0,hpm_migration_bytes=0;
        double hpm_ms=0;
        std::uint64_t payload_prefetch_calls=0,payload_prefetch_bytes=0;
        std::uint64_t payload_prefetch_hot_bytes=0,payload_prefetch_cold_bytes=0;
        std::uint64_t payload_prefetch_syscall_bytes=0,payload_prefetch_read_calls=0;
        std::uint64_t payload_prefetch_useful_bytes=0,payload_prefetch_wasted_bytes=0;
        std::uint64_t payload_prefetch_miss_bytes=0,payload_prefetch_current_bytes=0;
        std::uint64_t dense_fetch_read_calls=0,dense_fetch_syscall_bytes=0;
        std::uint64_t dense_fetch_pread_ns=0,dense_fetch_checksum_ns=0;
    };
    struct Stats {
        std::string cpu_implementation_requested, cpu_implementation_actual;
        std::size_t hot_prompt_tokens=0, cold_prompt_tokens=0;
        std::size_t hot_kv_cache_bytes=0, workspace_reserved_bytes=0;
        std::size_t managed_reserved_bytes=0, score_file_extent_bytes=0;
        std::size_t payload_prefetch_reserved_bytes=0,legacy_cpu_workspace_bytes=0;
        std::uint64_t payload_prefetch_setup_read_bytes=0,payload_prefetch_setup_read_calls=0;
        double payload_prefetch_setup_read_ms=0;
        std::uint64_t drain_calls=0;
        double drain_ms=0;
        std::vector<LayerStats> layers;
    };
    using ColdFetch=std::function<std::vector<std::uint8_t>(
        int,const std::vector<std::uint64_t>&)>;
    explicit FullGqaRuntimePolicy(const Config& config);
    ~FullGqaRuntimePolicy();
    FullGqaRuntimePolicy(const FullGqaRuntimePolicy&)=delete;
    FullGqaRuntimePolicy& operator=(const FullGqaRuntimePolicy&)=delete;
    // Full KV is token-major: K[8,128] followed by V[8,128], all F16.
    // Caller has already appended these bytes to the complete backing store.
    void append(int layer,const std::vector<std::uint64_t>& positions,
                const std::vector<std::uint16_t>& keys,
                const std::vector<std::uint8_t>& full_kv);
    // Optional legal prefetch: Q is ready, history [0,current_position) already
    // exists; current KV does not. Optional callback runs after immutable KV
    // payload is ready, without policy/IO locks. It may enqueue device copies;
    // it must not reenter this policy or retain >2 artifacts after consumption.
    // select() joins scores only; fetch() joins payload IO and this callback.
    void prepare_history(int layer,std::uint64_t current_position,
                         const std::vector<float>& rehearsal,
                         runtime::GQAPrefetchReady payload_ready={});
    std::vector<std::vector<std::uint64_t>> select(
        int layer,std::uint64_t position,const std::vector<float>& rehearsal);
    // Returns bytes in exactly physical_ids order; hot members never call the
    // cold fetcher. With payload prefetch, dense layers share its bounded
    // O_DIRECT reader. Other cold callbacks remain bounded at 64 tokens.
    std::vector<std::uint8_t> fetch(int layer,
        const std::vector<std::uint64_t>& physical_ids,const ColdFetch& cold_fetch);
    void drain();
    // Teardown-only wait; preserves pending jobs/errors for select/fetch and
    // never calls callbacks or consumes artifacts on the caller thread.
    void wait_pending() noexcept;
    std::size_t managed_bytes() const;
    std::size_t hot_prompt_tokens() const;
    std::size_t cold_prompt_tokens() const;
    bool is_hot(std::uint64_t position) const; // fixed placement only
    bool is_hot(int layer,std::uint64_t position) const;
    const Stats& stats() const;
    void write_stats(const std::string& path) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
