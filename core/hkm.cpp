#include "core/hkm.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace hillserve {
namespace {
unsigned frequency_bucket(std::uint16_t hits) noexcept {
    unsigned bucket = 0;
    while (hits) { ++bucket; hits >>= 1; }
    return bucket; // zero, then [1], [2,3], ... [32768,65535]
}
}

ApproxHkm::ApproxHkm(HkmConfig config, std::uint64_t initial_current_position)
    : config_(config), current_(initial_current_position) {
    if (!config_.token_count || config_.hot_capacity > config_.token_count ||
        config_.recent_pin_count > config_.hot_capacity ||
        config_.max_swaps_per_update > config_.hot_capacity ||
        current_ >= config_.token_count)
        throw std::invalid_argument("invalid fixed-universe HPM capacity/pin/position");
    hot_.resize(config_.token_count, 0);
    hits_.resize(config_.token_count, 0);
    std::size_t used = 0;
    if (config_.recent_pin_count) {
        for (auto id = pin_first(current_); id <= current_; ++id) {
            hot_[id] = 1; ++used;
        }
    }
    // Deterministic initial placement has no fabricated historical frequency.
    for (std::size_t id = 0; used < config_.hot_capacity; ++id) {
        if (!hot_[id]) { hot_[id] = 1; ++used; }
    }
}

void ApproxHkm::require_id(std::uint64_t id) const {
    if (id >= config_.token_count) throw std::out_of_range("HPM token outside fixed universe");
}

std::size_t ApproxHkm::pin_first(std::uint64_t current) const noexcept {
    const auto end = std::size_t(current) + 1;
    return end > config_.recent_pin_count ? end - config_.recent_pin_count : 0;
}

bool ApproxHkm::is_hot(std::uint64_t id) const { require_id(id); return hot_[id] != 0; }
bool ApproxHkm::is_pinned(std::uint64_t id) const {
    require_id(id);
    return config_.recent_pin_count && id >= pin_first(current_) && id <= current_;
}
std::uint16_t ApproxHkm::hit_count(std::uint64_t id) const { require_id(id); return hits_[id]; }

std::vector<std::uint64_t> ApproxHkm::hot_ids() const {
    std::vector<std::uint64_t> ids; ids.reserve(config_.hot_capacity);
    for (std::size_t id = 0; id < hot_.size(); ++id) if (hot_[id]) ids.push_back(id);
    return ids;
}
std::vector<std::uint64_t> ApproxHkm::cold_ids() const {
    std::vector<std::uint64_t> ids; ids.reserve(config_.token_count - config_.hot_capacity);
    for (std::size_t id = 0; id < hot_.size(); ++id) if (!hot_[id]) ids.push_back(id);
    return ids;
}
std::size_t ApproxHkm::resident_metadata_bytes() const noexcept {
    return sizeof(*this) + hot_.capacity() * sizeof(hot_[0]) + hits_.capacity() * sizeof(hits_[0]);
}

HkmUpdate ApproxHkm::update(const std::vector<std::uint64_t>& selected,
                          std::uint64_t current_position) {
    std::vector<std::uint16_t> weights(config_.token_count, 0);
    std::size_t unique = 0;
    for (auto id : selected) {
        require_id(id);
        if (!weights[id]) { weights[id] = 1; ++unique; }
    }
    return apply_weights(weights, current_position, unique, unique);
}

HkmUpdate ApproxHkm::update_weighted(
        const std::vector<std::pair<std::uint64_t, std::uint16_t>>& selected,
        std::uint64_t current_position) {
    std::vector<std::uint16_t> weights(config_.token_count, 0);
    std::vector<std::uint8_t> seen(config_.token_count, 0);
    std::size_t unique = 0;
    std::uint64_t total = 0;
    for (const auto& entry : selected) {
        const auto id = entry.first;
        const auto weight = entry.second;
        require_id(id);
        if (seen[id]) throw std::invalid_argument("duplicate weighted HPM unit");
        if (weight > 64) throw std::invalid_argument("HPM physical-token weight exceeds 64");
        if (weight > std::numeric_limits<std::uint64_t>::max() - total)
            throw std::overflow_error("HPM selected weight overflow");
        seen[id] = 1; weights[id] = weight; total += weight;
        if (weight) ++unique;
    }
    return apply_weights(weights, current_position, unique, total);
}

HkmUpdate ApproxHkm::apply_weights(const std::vector<std::uint16_t>& weights,
                                  std::uint64_t current_position,
                                  std::size_t unique_selected,
                                  std::uint64_t selected_weight) {
    require_id(current_position);
    if (current_position < current_) throw std::invalid_argument("HPM current position regressed");
    if (epoch_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("HPM update epoch overflow");
    HkmUpdate result; result.epoch = epoch_ + 1;
    result.unique_selected = unique_selected; result.selected_weight = selected_weight;
    auto pinned = [&](std::size_t id) {
        return config_.recent_pin_count && id >= pin_first(current_position) && id <= current_position;
    };
    std::size_t pins_to_promote = 0;
    for (std::size_t id = 0; id < config_.token_count; ++id)
        if (pinned(id) && !hot_[id]) ++pins_to_promote;
    if (pins_to_promote > config_.max_swaps_per_update)
        throw std::length_error("HPM recent pins exceed update migration budget");

    // Work on copies so invalid input, budget errors and allocation failure do
    // not publish half an update. O(N) scratch and O(N) work, no global sorting.
    auto next_hits = hits_;
    auto next_hot = hot_;
    result.decayed = config_.decay_interval_updates &&
                     result.epoch % config_.decay_interval_updates == 0;
    for (std::size_t id = 0; id < config_.token_count; ++id) {
        if (result.decayed) next_hits[id] >>= 1;
        const auto sum = std::uint32_t(next_hits[id]) + weights[id];
        next_hits[id] = std::uint16_t(std::min(sum, std::uint32_t(std::numeric_limits<std::uint16_t>::max())));
    }

    // Ascending scan fixes within-bucket ties. Mandatory recent pins are omitted
    // from both lists; evicted old-hot records cannot bounce back in this epoch.
    std::array<std::vector<std::uint64_t>, 17> victims, candidates;
    for (std::size_t id = 0; id < config_.token_count; ++id) {
        if (pinned(id)) continue;
        const auto bucket = frequency_bucket(next_hits[id]);
        (hot_[id] ? victims[bucket] : candidates[bucket]).push_back(id);
    }
    unsigned victim_bucket = 0;
    int candidate_bucket = 16;
    std::array<std::size_t, 17> victim_index{}, candidate_index{};
    auto has_victim = [&] {
        while (victim_bucket < victims.size() &&
               victim_index[victim_bucket] == victims[victim_bucket].size()) ++victim_bucket;
        return victim_bucket < victims.size();
    };
    auto has_candidate = [&] {
        while (candidate_bucket >= 0 &&
               candidate_index[std::size_t(candidate_bucket)] == candidates[std::size_t(candidate_bucket)].size()) --candidate_bucket;
        return candidate_bucket >= 0;
    };
    result.promote.reserve(config_.max_swaps_per_update);
    result.demote.reserve(config_.max_swaps_per_update);
    auto swap_in = [&](std::uint64_t promote) {
        if (!has_victim()) throw std::logic_error("HPM pin/capacity accounting inconsistent");
        const auto demote = victims[victim_bucket][victim_index[victim_bucket]++];
        next_hot[demote] = 0; next_hot[promote] = 1;
        result.demote.push_back(demote); result.promote.push_back(promote);
    };
    for (std::size_t id = 0; id < config_.token_count; ++id)
        if (pinned(id) && !hot_[id]) swap_in(id);
    while (result.promote.size() < config_.max_swaps_per_update && has_victim() && has_candidate()) {
        if (unsigned(candidate_bucket) <= victim_bucket) break; // equal bucket: keep hot
        swap_in(candidates[std::size_t(candidate_bucket)][candidate_index[std::size_t(candidate_bucket)]++]);
    }
    hot_.swap(next_hot); hits_.swap(next_hits);
    current_ = current_position; epoch_ = result.epoch;
    return result;
}

} // namespace hillserve
