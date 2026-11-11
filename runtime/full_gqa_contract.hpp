#pragma once

// Application-owned capacity accounting for one Gradient 8B request. This is
// not a hardware measurement or a directory-wide authorization check. The
// caller accounts for pre-existing files before creating either backing file.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hillserve::runtime {

inline constexpr std::uint64_t full_gqa_layers = 32;
inline constexpr std::uint64_t full_gqa_key_bytes = 2048;
inline constexpr std::uint64_t full_gqa_kv_bytes = 4096;
inline constexpr std::uint64_t full_gqa_block_tokens = 64;
inline constexpr std::uint64_t full_gqa_max_bo_bytes = 256ULL << 20;

inline std::uint64_t full_gqa_checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("full GQA byte addition overflow");
    return a + b;
}
inline std::uint64_t full_gqa_checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("full GQA byte multiplication overflow");
    return a * b;
}
inline std::uint64_t full_gqa_align_page(std::uint64_t bytes) {
    return full_gqa_checked_add(bytes, (4096 - bytes % 4096) % 4096);
}
inline void full_gqa_require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

// The caller chooses beta OFFLINE and uses this one partition for all layers.
// Decode positions do not change the prompt partition. Positive infinity is
// an explicit all-hot capacity control; NaN and negative beta are errors.
inline std::uint64_t full_gqa_hot_blocks(std::uint64_t prompt_tokens, double beta, bool allow_partial_prompt = false) {
    full_gqa_require(prompt_tokens >= 128 && (allow_partial_prompt || prompt_tokens % 64 == 0) &&
                     prompt_tokens <= 65536,
                     "full GQA prompt must be >=128; partial tail requires explicit hot-tail mode");
    full_gqa_require(!std::isnan(beta) && beta >= 0,
                     "full GQA offline beta must be nonnegative");
    const auto blocks = prompt_tokens / 64;
    if (std::isinf(beta)) return blocks;
    const long double b = beta;
    const auto rounded = static_cast<std::uint64_t>(
        std::floor(static_cast<long double>(blocks) * (b / (1 + b))));
    // Finite beta has a nonempty cold set mathematically, even if floating
    // division rounds rho to one for an extremely large supplied beta.
    return std::min(rounded, blocks - 1);
}

struct FullGqaRuntimeSpec {
    std::uint64_t layers = full_gqa_layers;
    std::uint64_t capacity = 0; // actual common store capacity, 64-token aligned
    std::uint64_t prompt_tokens = 0;
    std::uint64_t generated_capacity = 0;
    bool allow_partial_prompt = false; // incomplete final block is always hot
    bool fp32_selection_counts = false; // explicit, user-approved64K count extension
    double beta = 1;
    std::uint64_t hot_kv_budget_bytes = 0;
    std::uint64_t total_host_workspace_budget_bytes = 0;
    // Caller enumerates bounded scorer, selected IDs, fetch, append, validation
    // and index workspaces. Do not pass a single-layer hot allowance here.
    std::uint64_t additional_workspace_bytes = 0;
    std::uint64_t chunk_tokens = 4096;
    std::uint64_t bo_budget_bytes = full_gqa_max_bo_bytes;
    // Optional local cap for the two NEW files. Existing files and directory
    // authorization must be checked separately by the experiment runner.
    std::uint64_t storage_quota_bytes = std::numeric_limits<std::uint64_t>::max();
};

struct FullGqaRuntimePlan {
    std::uint64_t capacity = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t generated_capacity = 0;
    std::uint64_t hot_prompt_blocks = 0;
    std::uint64_t prompt_hot_tokens = 0;
    std::uint64_t prompt_cold_tokens = 0;
    std::uint64_t prompt_tail_hot_tokens = 0;
    std::uint64_t recent_pin_tokens = 0;
    std::uint64_t reserved_all_layers_hot_bytes = 0;
    std::uint64_t additional_workspace_bytes = 0;
    std::uint64_t managed_total = 0;
    std::uint64_t raw_K_layer_stride_bytes = 0;
    std::uint64_t raw_K_extent_bytes = 0;
    std::uint64_t segmented_KV_extent_bytes = 0;
    std::uint64_t storage_total_bytes = 0;
    std::uint64_t bo_reserved_bytes = 0;
    std::vector<std::uint64_t> hot_positions; // prompt only, sorted ascending
    // Generated tail slots are reserved, but not readable until appended.
    // Unused padded capacity and cold prompt positions have slot -1.
    std::vector<std::int64_t> hot_slot_by_position;

    void require_visible(std::uint64_t layer, std::uint64_t position,
                         std::uint64_t layer_completed) const {
        full_gqa_require(layer < full_gqa_layers,
                         "full GQA layer outside fixed 32-layer model");
        full_gqa_require(layer_completed <= prompt_tokens + generated_capacity &&
                         position < layer_completed,
                         "full GQA token not appended in this layer");
    }
};

inline FullGqaRuntimePlan plan_full_gqa_runtime(const FullGqaRuntimeSpec& s) {
    full_gqa_require(s.layers == full_gqa_layers,
                     "full GQA must reserve the complete 32-layer model");
    full_gqa_require(s.generated_capacity >= 1 && s.generated_capacity <= 512,
                     "full GQA generated capacity must be 1..512");
    const auto hot_blocks = full_gqa_hot_blocks(s.prompt_tokens, s.beta, s.allow_partial_prompt);
    full_gqa_require(s.capacity >= full_gqa_checked_add(s.prompt_tokens, s.generated_capacity) &&
                     s.capacity <= (s.fp32_selection_counts ? 66048 : 65536) && s.capacity % 64 == 0,
                     "full GQA common store capacity must cover prompt+decode and align64");
    full_gqa_require(s.fp32_selection_counts || s.prompt_tokens + s.generated_capacity <= 65504,
                     "full GQA visible count exceeds finite F16 selection contract");
    full_gqa_require(s.chunk_tokens >= 64 && s.chunk_tokens <= 32768 &&
                     s.chunk_tokens % 64 == 0,
                     "full GQA scoring chunks must be bounded and align64");
    full_gqa_require(s.bo_budget_bytes <= full_gqa_max_bo_bytes,
                     "full GQA BO budget exceeds 256MiB limit");
    FullGqaRuntimePlan p;
    p.capacity = s.capacity;
    p.prompt_tokens = s.prompt_tokens;
    p.generated_capacity = s.generated_capacity;
    p.hot_prompt_blocks = hot_blocks;
    p.prompt_tail_hot_tokens = s.prompt_tokens % 64;
    p.prompt_hot_tokens = hot_blocks * 64 + p.prompt_tail_hot_tokens;
    p.prompt_cold_tokens = s.prompt_tokens - p.prompt_hot_tokens;
    p.recent_pin_tokens = std::min<std::uint64_t>(16, hot_blocks) * 64 + p.prompt_tail_hot_tokens;
    const auto hot_capacity = full_gqa_checked_add(p.prompt_hot_tokens, s.generated_capacity);
    p.reserved_all_layers_hot_bytes = full_gqa_checked_mul(
        full_gqa_checked_mul(full_gqa_layers, hot_capacity), full_gqa_kv_bytes);
    p.additional_workspace_bytes = s.additional_workspace_bytes;
    p.managed_total = full_gqa_checked_add(p.reserved_all_layers_hot_bytes,
                                          s.additional_workspace_bytes);
    if (p.reserved_all_layers_hot_bytes > s.hot_kv_budget_bytes)
        throw std::length_error("full GQA total 32-layer hot KV exceeds budget");
    if (p.managed_total > s.total_host_workspace_budget_bytes)
        throw std::length_error("full GQA hot KV plus workspace exceeds host budget");

    p.raw_K_layer_stride_bytes = full_gqa_checked_mul(s.capacity, full_gqa_key_bytes);
    p.raw_K_extent_bytes = full_gqa_checked_mul(full_gqa_layers, p.raw_K_layer_stride_bytes);
    // Existing SegmentedKVStore has a 4096-byte file header and a 32-byte
    // per-record header. Preserve that format instead of inventing another.
    p.segmented_KV_extent_bytes = full_gqa_align_page(full_gqa_checked_add(4096,
        full_gqa_checked_mul(full_gqa_checked_mul(full_gqa_layers, s.capacity),
                              full_gqa_kv_bytes + 32)));
    if (p.segmented_KV_extent_bytes > ((s.fp32_selection_counts ? 9ULL : 8ULL) << 30))
        throw std::length_error("full GQA exceeds existing SegmentedKVStore 8GiB file limit");
    p.storage_total_bytes = full_gqa_checked_add(p.raw_K_extent_bytes,
                                               p.segmented_KV_extent_bytes);
    if (p.storage_total_bytes > s.storage_quota_bytes)
        throw std::length_error("full GQA new-file extents exceed local storage cap");
    if (p.prompt_cold_tokens) {
        const auto tokens = std::min(s.chunk_tokens, p.prompt_cold_tokens);
        const auto slots = std::min<std::uint64_t>(2,
            1 + (p.prompt_cold_tokens - 1) / s.chunk_tokens);
        const auto one_slot = full_gqa_checked_add(32 * 128 * 2,
            full_gqa_checked_mul(tokens, full_gqa_key_bytes + 32 * 2));
        p.bo_reserved_bytes = full_gqa_checked_mul(slots, one_slot);
    }
    if (p.bo_reserved_bytes > s.bo_budget_bytes)
        throw std::length_error("full GQA double-buffer reservation exceeds BO budget");

    p.hot_slot_by_position.assign(static_cast<std::size_t>(s.capacity), -1);
    p.hot_positions.reserve(static_cast<std::size_t>(p.prompt_hot_tokens));
    // Mandatory recent block window plus early blocks is deterministic. No
    // historical hit frequencies or future selected IDs are fabricated.
    const auto early = p.prompt_hot_tokens - p.recent_pin_tokens;
    const auto recent_begin = s.prompt_tokens - p.recent_pin_tokens;
    for (std::uint64_t t = 0; t < s.prompt_tokens; ++t) {
        if (t < early || t >= recent_begin) {
            p.hot_slot_by_position[t] = static_cast<std::int64_t>(p.hot_positions.size());
            p.hot_positions.push_back(t);
        }
    }
    for (std::uint64_t t = 0; t < s.generated_capacity; ++t)
        p.hot_slot_by_position[s.prompt_tokens + t] = static_cast<std::int64_t>(p.prompt_hot_tokens + t);
    return p;
}
} // namespace hillserve::runtime
