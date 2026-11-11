#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hillserve {

// Approximate HPM placement only: neither scores nor attention-selected IDs are
// changed. IDs can represent tokens or equal-sized placement units (for example,
// 64-token KV blocks). Every unit [0, token_count) must already have complete KV
// backing. The caller fixes and reports this granularity, maps selected physical
// tokens to units, and owns all payload bytes. Hot capacity and recent_pin_count
// are unit counts, not bytes. A partial trailing block must be managed separately.
struct HkmConfig {
    std::size_t token_count = 0;
    std::size_t hot_capacity = 0;
    std::size_t recent_pin_count = 0;
    std::size_t max_swaps_per_update = 0;
    // Zero disables decay; otherwise halve counters before counting every kth
    // update. update() counts a selected unit once; update_weighted() can count
    // its actual unique selected physical tokens (at most 64).
    std::uint64_t decay_interval_updates = 0;
};

struct HkmUpdate {
    std::uint64_t epoch = 0;
    std::size_t unique_selected = 0;
    std::uint64_t selected_weight = 0;
    bool decayed = false;
    // Equal-sized lists, one old-hot eviction per newly hot token. These are
    // metadata decisions, not evidence that a payload transfer has occurred.
    std::vector<std::uint64_t> promote;
    std::vector<std::uint64_t> demote;
};

// Bounded, deterministic approximation to recent+frequency HPM:
//   * Recent positions ending at current_position are always hot.
//   * uint16 hit counters saturate and periodically decay.
//   * 17 logarithmic frequency buckets avoid a global token sort. Equal buckets
//     retain existing hot members; candidate/victim ties use ascending ID.
//   * Pin replacements consume the same per-update swap limit as other moves.
// This is deliberately not claimed to be the paper's exact frequency ranking.
// The universe/capacity stay fixed for this object; no hidden growth or cache.
class ApproxHkm {
public:
    ApproxHkm(HkmConfig config, std::uint64_t initial_current_position);

    // selected must derive from an actual selector. Duplicate unit IDs (e.g.
    // across GQA heads or token-to-block mapping) count once. current_position is
    // also a unit ID and cannot regress. A 64-token invocation is approximate
    // block-level HPM, not a claim of exact token-frequency placement.
    // Insufficient swap budget for new recent pins rejects without state change.
    // On success, metadata advances. Before the next score pass, the caller must
    // enact the returned KV moves; a transfer failure must abort that request.
    // This module never performs I/O, manufactures scores, or removes backing.
    HkmUpdate update(const std::vector<std::uint64_t>& selected,
                     std::uint64_t current_position);

    // Optional block-density metric: each (unit, weight) is the number of actual
    // unique physical tokens selected within a <=64-token block, after unioning
    // Q heads. Weights are 0..64; duplicate units (including zero-weight entries)
    // are rejected. The caller must not use score magnitudes or head duplicates.
    // This avoids treating a block with 1 selected token like one with 64, but
    // equal dense-block hits can still legitimately produce no migrations.
    HkmUpdate update_weighted(
        const std::vector<std::pair<std::uint64_t, std::uint16_t>>& selected,
        std::uint64_t current_position);

    const HkmConfig& config() const noexcept { return config_; }
    std::uint64_t epoch() const noexcept { return epoch_; }
    std::uint64_t current_position() const noexcept { return current_; }
    bool is_hot(std::uint64_t id) const;
    bool is_pinned(std::uint64_t id) const;
    std::uint16_t hit_count(std::uint64_t id) const;
    // Both lists are sorted, disjoint and jointly cover the whole universe.
    std::vector<std::uint64_t> hot_ids() const;
    std::vector<std::uint64_t> cold_ids() const;
    std::size_t resident_metadata_bytes() const noexcept;

private:
    HkmConfig config_;
    std::uint64_t current_ = 0;
    std::uint64_t epoch_ = 0;
    std::vector<std::uint8_t> hot_;
    std::vector<std::uint16_t> hits_;
    void require_id(std::uint64_t id) const;
    std::size_t pin_first(std::uint64_t current) const noexcept;
    HkmUpdate apply_weights(const std::vector<std::uint16_t>& weights,
                            std::uint64_t current_position,
                            std::size_t unique_selected,
                            std::uint64_t selected_weight);
};

} // namespace hillserve
