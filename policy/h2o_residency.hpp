#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hillserve::policy {

// Independently implemented H2O-style residency policy, not sparse attention.
// Reference: FMInference/H2O ac75c2a8a9e76832b2a4139b9363373b56336bfb,
// h2o_hf/utils_real_drop/modify_llama.py, H2OKVCache_LayerWise.
// Explicit research extensions: actual F32 model softmax probabilities;
// sum the Q heads sharing each physical KV head; keep scores for every past
// position; pin actual generated KV separately. Required nonresident KV must
// be recomputed by the caller before full attention. This class owns no KV.
class H2OResidency {
public:
    struct Config {
        std::size_t layers = 32;
        std::size_t query_heads = 32;
        std::size_t kv_heads = 8;
        std::size_t prompt_tokens = 0;
        std::size_t generated_capacity = 0;
        // Prompt KV slots per physical KV head. Generated KV has a separate
        // reservation and never displaces an already budgeted prompt slot.
        std::size_t prompt_resident_tokens_per_head = 0;
        std::uint64_t metadata_budget_bytes = 0;
    };

    explicit H2OResidency(Config);
    // Rows are [Q head][query][absolute key]. Each call is a new contiguous
    // causal query block; replaying an already observed block is rejected.
    // This path is useful for independent software checks. A model adapter
    // should use the GPU reduction path below rather than copying a quadratic
    // prefill attention matrix to the CPU.
    void observe_attention(std::size_t layer, std::size_t first_query,
                           std::size_t query_count, std::size_t key_count,
                           const float* probabilities, std::size_t elements);
    // Input [physical KV head][absolute key] is SUM, never MAX, over this
    // query block and all Q heads assigned to that KV head. Padding is absent.
    // Values must originate in the same real model softmax used by attention.
    void observe_group_sums(std::size_t layer, std::size_t first_query,
                            std::size_t query_count, std::size_t key_count,
                            const float* sums, std::size_t elements);

    // Increasing absolute IDs per physical KV head: heavy prompt positions,
    // recent prompt positions, then all actual generated positions. Equal
    // importance is broken by smaller absolute position, deterministically.
    std::vector<std::vector<std::uint64_t>> resident_ids(std::size_t layer);
    float importance(std::size_t layer, std::size_t kv_head,
                     std::size_t position) const;
    std::size_t observed_tokens(std::size_t layer) const;
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t recent_prompt_slots() const noexcept { return recent_; }
    std::size_t heavy_prompt_slots() const noexcept { return heavy_; }
    // Includes persistent score arrays, reusable reduction/ranking scratch,
    // vector objects, and an upper bound for one returned ID collection.
    // KV payload and model reduction buffers are the caller's responsibility.
    std::uint64_t reserved_metadata_bytes() const noexcept { return reserved_; }
    const Config& config() const noexcept { return config_; }
    static constexpr const char* contract_name =
        "h2o_style_F32_group_sum_full_history_prompt_half_recent_generated_pinned_v1";
    static constexpr const char* reference_commit =
        "ac75c2a8a9e76832b2a4139b9363373b56336bfb";
    static constexpr const char* reference_file_sha256 =
        "c20cfc046c89db86e21430c1efeb4a9c4a99328037d3cd66887399dba83c246d";

private:
    Config config_;
    std::size_t capacity_, recent_, heavy_;
    std::uint64_t reserved_;
    std::vector<float> scores_, grouped_;
    std::vector<std::size_t> observed_;
    std::vector<std::uint64_t> ranked_;
    void validate_block(std::size_t layer, std::size_t first_query,
                        std::size_t query_count, std::size_t key_count) const;
};
}
