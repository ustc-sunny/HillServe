#pragma once
#include "llama.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hillserve::runtime {

// Correctness-first prompt-KV recovery, not a mature H2O recomputation engine.
// By default each call replays the ORIGINAL prompt for one target layer.
// The opt-in layer-group mode captures several layers in one real replay;
// its bounded CPU recovery cache exists only within the current decode step.
// No SSD backing or recovery KV survives end_decode(). Generated KV stays hot.
class PromptRecompute {
public:
    struct Config {
        std::size_t host_budget_bytes = 0;
        // Incremental worker allocation; the caller separately budgets its main
        // context and the shared model weights. Actual free GPU memory is also
        // checked immediately before this private context is created.
        std::size_t gpu_budget_bytes = 0;
        std::size_t gpu_graph_budget_bytes = 256ULL << 20;
        std::size_t host_context_reserve_bytes = 32ULL << 20;
        std::size_t gpu_free_reserve_bytes = 64ULL << 20;
        // Actual device-wide used memory (total - free), including the shared
        // model, main context, worker, and any concurrent processes. Zero
        // disables this additional cap; the incremental guards remain active.
        std::size_t total_gpu_budget_bytes = 0;
        // Optional new JSONL file: existing paths are rejected. Every call,
        // including a failed admission/decode, is written before it returns.
        std::string event_log_path;
        unsigned cpu_threads = 4;
        std::size_t chunk_tokens = 256;
        std::size_t max_calls = 4096;
        std::size_t max_requested_rows = 0; // zero uses original prompt size
        // Bound simultaneously returned heads. Keep single-head callers' former
        // reservation by default; the all-head baseline explicitly sets eight.
        std::size_t max_batch_heads = 1;
        // Upper bound, reduced to fit host_budget_bytes. Grouping requires
        // eight identical fixed missing-head requests, layers0..31 in order,
        // and explicit begin_decode/end_decode around each generation step.
        std::size_t max_target_layers = 1;
        float rope_freq_base = 3580165449.0f;
        float rope_freq_scale = 1.0f;
    };
    struct Call {
        int layer = -1, kv_head = -1;
        bool passed = false;
        std::uint64_t target_layers = 1, decode_position = 0;
        std::uint64_t requested_heads = 0;
        std::uint64_t requested_rows = 0, returned_bytes = 0;
        std::uint64_t replayed_tokens = 0, decode_calls = 0;
        std::uint64_t layer_steps = 0, attention_steps = 0;
        std::uint64_t captured_rows = 0, capture_D2H_bytes = 0;
        std::size_t native_GPU_peak_bytes = 0, graph_GPU_peak_bytes = 0;
        std::size_t graph_host_peak_bytes = 0, observed_GPU_increment_peak_bytes = 0;
        std::size_t GPU_free_before_bytes = 0;
        std::size_t global_GPU_used_peak_bytes = 0;
        std::uint64_t GPU_memory_samples = 0;
        double context_init_ms = 0, replay_ms = 0, context_destroy_ms = 0, wall_ms = 0;
    };
    struct Stats {
        std::uint64_t calls = 0, successful_calls = 0, failed_calls = 0;
        std::uint64_t replayed_tokens = 0, layer_steps = 0, attention_steps = 0;
        std::uint64_t returned_rows = 0, returned_bytes = 0, capture_D2H_bytes = 0;
        std::size_t host_managed_reserved_bytes = 0, native_GPU_planned_bytes = 0;
        std::size_t native_GPU_peak_bytes = 0, graph_GPU_peak_bytes = 0;
        std::size_t graph_host_peak_bytes = 0, observed_GPU_increment_peak_bytes = 0;
        std::size_t global_GPU_used_peak_bytes = 0;
        std::size_t target_layers_per_replay = 1, recovery_cache_reserved_bytes = 0;
        std::size_t recovery_cache_peak_bytes = 0, recovery_cache_live_bytes = 0;
        std::uint64_t layer_requests = 0, reused_layer_requests = 0, completed_decode_steps = 0;
        double wall_ms = 0;
    };

    // model is shared read-only and must outlive this object. Calls are
    // serialized and block until the worker and its private context are gone.
    PromptRecompute(llama_model* model, const std::vector<llama_token>& original_prompt,
                    const Config&);
    ~PromptRecompute();
    PromptRecompute(const PromptRecompute&) = delete;
    PromptRecompute& operator=(const PromptRecompute&) = delete;
    // Row layout: K[kv_head,128] then V[kv_head,128], all F16; 512 bytes/row.
    std::vector<std::uint8_t> call(int layer, int kv_head,
                                 const std::vector<std::uint64_t>& positions);
    // Exactly eight input/output vectors, indexed by KV head. An empty head
    // needs no rows. Per-head order and duplicate positions are preserved.
    // All nonempty heads share one prompt replay and target-layer transfer;
    // requested_rows in the receipt counts head-rows across the whole batch.
    std::vector<std::vector<std::uint8_t>> batch_call(int layer,
        const std::vector<std::vector<std::uint64_t>>& positions_by_head);
    void begin_decode(std::uint64_t position);
    void end_decode();
    std::vector<std::uint8_t> operator()(int layer, int kv_head,
        const std::vector<std::uint64_t>& positions) { return call(layer, kv_head, positions); }
    // Read after calls have returned; these references do not create unbudgeted
    // copies of the bounded event array. The caller may serialize them to logs.
    const Stats& stats() const;
    const std::vector<Call>& events() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hillserve::runtime
