#include "core/hkm.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

using hillserve::ApproxHkm;
using hillserve::HkmConfig;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F call, const char* message) {
    bool rejected = false;
    try { call(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
void coverage(const ApproxHkm& cache) {
    const auto hot = cache.hot_ids(), cold = cache.cold_ids();
    check(hot.size() == cache.config().hot_capacity, "hot capacity changed");
    check(std::is_sorted(hot.begin(), hot.end()) && std::is_sorted(cold.begin(), cold.end()), "membership IDs not sorted");
    std::set<std::uint64_t> all(hot.begin(), hot.end());
    for (auto id : cold) check(all.insert(id).second, "hot/cold overlap");
    check(all.size() == cache.config().token_count && *all.begin() == 0 &&
          *all.rbegin() == cache.config().token_count - 1, "lost fixed-universe coverage");
    for (std::uint64_t id = 0; id < cache.config().token_count; ++id)
        if (cache.is_pinned(id)) check(cache.is_hot(id), "recent pin evicted");
}
}

int main() { try {
    ApproxHkm cache({12, 4, 2, 2, 4}, 11);
    check(cache.hot_ids() == std::vector<std::uint64_t>({0, 1, 10, 11}), "initial recent placement or stable fill");
    coverage(cache);
    const auto moved = cache.update({6, 6, 6, 7}, 11);
    check(moved.unique_selected == 2 && cache.hit_count(6) == 1, "physical ID counted per Q head");
    check(moved.promote == std::vector<std::uint64_t>({6, 7}) &&
          moved.demote == std::vector<std::uint64_t>({0, 1}), "real selected cold tokens did not promote stably");
    coverage(cache);
    const auto tied = cache.update({8, 9}, 11);
    check(tied.promote.empty() && cache.is_hot(6) && cache.is_hot(7), "equal bucket caused churn");
    const auto later = cache.update({8, 9, 9}, 11);
    check(later.promote == std::vector<std::uint64_t>({8, 9}), "higher frequency bucket did not displace old hot entries");
    check(!later.decayed && cache.hit_count(9) == 2, "unexpected decay/count");
    const auto decay = cache.update({9}, 11);
    check(decay.decayed && cache.hit_count(8) == 1 && cache.hit_count(9) == 2,
          "periodic decay must precede the current update's unique hit");
    coverage(cache);

    // Failed updates leave both membership and counters untouched.
    const auto before = cache.hot_ids(); const auto epoch = cache.epoch();
    const auto hits = cache.hit_count(8);
    rejects([&] { cache.update({8, 12}, 11); }, "invalid selected ID accepted");
    rejects([&] { cache.update({8}, 10); }, "regressing current position accepted");
    rejects([&] { cache.update({8}, 12); }, "invalid current position accepted");
    check(cache.hot_ids() == before && cache.epoch() == epoch && cache.hit_count(8) == hits,
          "rejected update mutated metadata");

    ApproxHkm sliding({12, 4, 2, 1, 0}, 5);
    const auto slide = sliding.update({8}, 6);
    check(slide.promote == std::vector<std::uint64_t>({6}) && slide.demote.size() == 1,
          "new recent pin did not consume common migration quota");
    check(!sliding.is_hot(8), "frequency promotion exceeded quota after pin move");
    coverage(sliding);
    const auto stable = sliding.hot_ids();
    rejects([&] { sliding.update({9}, 10); }, "large recent jump bypassed migration budget");
    check(sliding.hot_ids() == stable && sliding.current_position() == 6 && sliding.epoch() == 1,
          "pin budget rejection changed state");

    // Replay the returned delta against an independent membership model. This
    // validates caller-visible moves, cap, coverage and deterministic decisions.
    ApproxHkm a({257, 41, 9, 3, 7}, 256), b({257, 41, 9, 3, 7}, 256);
    const auto initial = a.hot_ids(); std::set<std::uint64_t> enacted(initial.begin(), initial.end());
    for (std::uint64_t round = 0; round < 80; ++round) {
        std::vector<std::uint64_t> selected;
        for (std::uint64_t i = 0; i < 33; ++i) selected.push_back((round * 17 + i * 11) % 257);
        const auto left = a.update(selected, 256), right = b.update(selected, 256);
        check(left.promote == right.promote && left.demote == right.demote, "unstable placement");
        check(left.promote.size() == left.demote.size() && left.promote.size() <= 3, "migration cap broken");
        for (auto id : left.demote) check(enacted.erase(id) == 1, "demotion was not old hot");
        for (auto id : left.promote) check(enacted.insert(id).second, "promotion was already hot");
        const auto now = a.hot_ids();
        check(enacted == std::set<std::uint64_t>(now.begin(), now.end()), "delta does not reconstruct placement");
        coverage(a);
    }
    ApproxHkm saturated({2, 1, 0, 1, 0}, 1);
    for (std::uint64_t i = 0; i < 65540; ++i) saturated.update({1, 1}, 1);
    check(saturated.hit_count(1) == std::numeric_limits<std::uint16_t>::max(), "hit counter wrapped");
    ApproxHkm none({5, 0, 0, 0, 0}, 4), all({5, 5, 2, 0, 0}, 4);
    none.update({2}, 4); all.update({2}, 4); coverage(none); coverage(all);
    check(none.hot_ids().empty() && all.cold_ids().empty(), "all-hot/all-cold boundaries failed");
    // The real direct-I/O integration uses 64-token placement units. Several
    // genuinely selected tokens in one block contribute one placement hit;
    // original attention-selected token IDs are never modified by this module.
    ApproxHkm blocks({512, 64, 8, 4, 8}, 511);
    const std::vector<std::uint64_t> selected_tokens{100 * 64, 100 * 64 + 17, 100 * 64 + 63};
    std::vector<std::uint64_t> selected_blocks;
    for (auto token : selected_tokens) selected_blocks.push_back(token / 64);
    const auto block_update = blocks.update(selected_blocks, 511);
    check(block_update.unique_selected == 1 && blocks.hit_count(100) == 1 && blocks.is_hot(100),
          "64-token placement unit mapping or promotion failed");
    check(blocks.hot_ids().size() * 64 == 4096 && selected_tokens[1] == 6417,
          "block hot budget or actual selected token identity changed");
    coverage(blocks);
    ApproxHkm weighted({8, 2, 1, 1, 0}, 7);
    const auto dense = weighted.update_weighted({{0, 1}, {4, 64}, {5, 0}}, 7);
    check(dense.unique_selected == 2 && dense.selected_weight == 65 &&
          dense.promote == std::vector<std::uint64_t>({4}) &&
          dense.demote == std::vector<std::uint64_t>({0}),
          "64-versus-1 selected physical token weight did not guide promotion");
    check(weighted.hit_count(4) == 64 && weighted.hit_count(0) == 1 && weighted.hit_count(5) == 0,
          "weighted hit accounting changed physical token counts");
    const auto weighted_before = weighted.hot_ids(); const auto weighted_epoch = weighted.epoch();
    rejects([&] { weighted.update_weighted({{1, 2}, {1, 3}}, 7); }, "duplicate weighted unit accepted");
    rejects([&] { weighted.update_weighted({{1, 0}, {1, 0}}, 7); }, "duplicate zero-weight unit accepted");
    rejects([&] { weighted.update_weighted({{1, 65}}, 7); }, "more than64 physical tokens per block accepted");
    rejects([&] { weighted.update_weighted({{8, 1}}, 7); }, "weighted invalid ID accepted");
    check(weighted.hot_ids() == weighted_before && weighted.epoch() == weighted_epoch && weighted.hit_count(1) == 0,
          "invalid weighted call mutated state");
    for (std::uint64_t step = 0; step < 1024; ++step) weighted.update_weighted({{4, 64}}, 7);
    check(weighted.hit_count(4) == 65535, "weighted hit counter did not saturate");
    coverage(weighted);
    rejects([] { ApproxHkm x({0, 0, 0, 0, 0}, 0); }, "empty universe accepted");
    rejects([] { ApproxHkm x({5, 2, 3, 1, 0}, 4); }, "pins exceed hot capacity");
    rejects([] { ApproxHkm x({5, 2, 1, 3, 0}, 4); }, "migration quota exceeds capacity");
    rejects([&] { cache.is_hot(12); }, "out-of-range lookup accepted");
    std::cout << "passed: approximate HPM recent pins, unique physical hits, saturating/decayed counters, "
                 "bucket replacement, migration cap, invalid-input atomicity and fixed-universe coverage\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
