#pragma once
#include "runtime/gqa_prefetch.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace hillserve::policy {
// Request-local, no-SSD model policy. Every absent KV row is supplied by a
// caller-owned actual model recomputation callback, never an offline trace.
class NoSSDPolicy {
public:
    struct Config {
        std::string mode="PREFETCH"; // PREFETCH, H2O, or FULL_RECOMPUTE
        std::size_t layers=32,capacity=0,prompt_tokens=0,generated_capacity=64;
        std::size_t total_host_budget_bytes=0,staging_budget_bytes=0;
        // FULL_RECOMPUTE only: the same offline APP partition and exact
        // all-layer KV payload allowance used by FullGqaRuntimePolicy.
        // Metadata/workspace do not enlarge the resident KV allowance.
        double offline_beta=std::numeric_limits<double>::quiet_NaN();
        bool allow_partial_prompt=false;
        bool extended_full_context=false; // capacity only; FULL_RECOMPUTE never scores
        std::size_t resident_kv_payload_budget_bytes=0;
        float selector_alpha=10.f;
        unsigned cpu_threads=4;
        std::string cpu_implementation="auto";
    };
    // Each returned row is exactly K[kv_head,128] then V[kv_head,128], F16,
    // in requested-position order. Only evicted original prompt rows are
    // requested; all generated rows remain pinned to preserve model history.
    using Recompute=std::function<std::vector<std::uint8_t>(
        int layer,int kv_head,const std::vector<std::uint64_t>& positions)>;
    // One actual prompt replay supplies all eight physical heads together.
    // Outer input/output sizes are exactly eight; each head preserves request
    // order and returns 512 F16 K/V bytes per position. Empty heads stay empty.
    using BatchRecompute=std::function<std::vector<std::vector<std::uint8_t>>(
        int layer,const std::vector<std::vector<std::uint64_t>>& positions_by_head)>;
    struct Stats {
        std::size_t managed_reserved_bytes=0,score_key_reserved_bytes=0,
            resident_reserved_bytes=0,staging_reserved_bytes=0,
            metadata_reserved_bytes=0,resident_rows_per_head=0;
        std::size_t resident_kv_payload_reserved_bytes=0,
            hot_prompt_tokens=0,cold_prompt_tokens=0,generated_reserved_tokens=0;
        bool all_KV_fits=false;
        std::string cpu_implementation_actual;
        std::uint64_t append_records=0,score_calls=0,current_token_scores=0,
            selections=0,fetch_calls=0,fetch_bytes=0,resident_fetch_bytes=0,
            recompute_calls=0,recomputed_rows=0,recomputed_bytes=0,
            prefetch_calls=0,prefetch_bytes=0,prefetch_useful_bytes=0,
            evicted_rows=0,restored_rows=0,cold_append_discarded_bytes=0,
            fixed_resident_records=0,peak_fixed_resident_records=0;
        double score_ms=0,selection_ms=0,fetch_ms=0,recompute_ms=0;
    };
    explicit NoSSDPolicy(const Config&,Recompute,BatchRecompute={});
    ~NoSSDPolicy();
    NoSSDPolicy(const NoSSDPolicy&)=delete;
    NoSSDPolicy& operator=(const NoSSDPolicy&)=delete;
    void append_full(int layer,const std::vector<std::uint64_t>& positions,
        const std::vector<std::uint16_t>& keys,const std::vector<std::uint8_t>& full_kv);
    // current_position is the newly appended token's position, NOT a count.
    std::vector<std::vector<std::uint64_t>> select(int layer,
        std::uint64_t current_position,const std::vector<float>& rehearsal);
    std::vector<std::uint8_t> fetch(int layer,const std::vector<std::uint64_t>& physical_ids);
    // At most two asynchronous history score/fetch/upload tasks. History
    // excludes current_position; final select joins and adds that token.
    void prepare_history(int layer,std::uint64_t current_position,
        const std::vector<float>& rehearsal,runtime::GQAPrefetchReady ready={});
    // External H2O importance policy supplies eight physical-head resident
    // sets. When the complete KV fits, this cannot force artificial eviction.
    void set_h2o_resident(int layer,const std::vector<std::vector<std::uint64_t>>& positions);
    void wait_pending();
    void drain();
    std::size_t managed_bytes() const;
    std::size_t resident_rows_per_head() const;
    bool all_kv_fits() const;
    const Stats& stats() const;
    void write_stats(const std::string& path) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
