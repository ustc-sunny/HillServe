#include "core/app_partition.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hillserve {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
double cost(const AppMeasuredCost& measurement, AppCostScope scope) {
    require(measurement.scope == scope, "APP profile has an incompatible timing scope");
    require(!measurement.source.empty(), "APP profile requires measurement provenance");
    require(measurement.work_items != 0, "APP profile requires a measured candidate count");
    require(std::isfinite(measurement.total_ms) && measurement.total_ms > 0,
            "APP profile requires positive finite measured wall time");
    const double rate = measurement.total_ms / double(measurement.work_items);
    require(std::isfinite(rate) && rate > 0, "APP per-candidate cost is not representable");
    return rate;
}
double finite_double(long double value) {
    const auto result = static_cast<double>(value);
    if (!std::isfinite(result)) throw std::overflow_error("APP estimated value overflow");
    return result;
}
}

const char* app_cost_scope_name(AppCostScope scope) {
    switch (scope) {
    case AppCostScope::host_hot_score_to_cpu_ready: return "host_hot_score_to_cpu_ready";
    case AppCostScope::cold_file_read_score_return_to_cpu_ready:
        return "cold_file_read_score_return_to_cpu_ready";
    case AppCostScope::selected_kv_transfer_only: return "selected_kv_transfer_only";
    case AppCostScope::unspecified: return "unspecified";
    }
    throw std::invalid_argument("unknown APP cost scope");
}

AppPartitionPlan plan_app_partition(const AppPartitionProfile& profile,
                                    const AppPartitionConstraints& constraints) {
    require(!profile.configuration_id.empty(), "APP profile configuration identity is required");
    AppPartitionPlan result;
    result.profile = profile;
    result.constraints = constraints;
    result.hot_score_ms_per_candidate = cost(profile.hot_score,
        AppCostScope::host_hot_score_to_cpu_ready);
    result.cold_read_score_return_ms_per_candidate = cost(profile.cold_read_score_return,
        AppCostScope::cold_file_read_score_return_to_cpu_ready);
    require(profile.hot_selected_kv.has_value() == profile.cold_selected_kv.has_value(),
            "APP requires both hot and cold KV transfer measurements or neither");
    for (double retention : {profile.hot_retention, profile.cold_retention})
        require(std::isfinite(retention) && retention >= 0 && retention <= 1,
                "APP observed retention must be finite and in [0,1]");
    long double hot = result.hot_score_ms_per_candidate;
    long double cold = result.cold_read_score_return_ms_per_candidate;
    if (profile.hot_selected_kv) {
        require(!profile.retention_source.empty(), "APP skew correction requires retention provenance");
        result.includes_selected_kv_transfer = true;
        result.hot_selected_kv_ms_per_selected_candidate = cost(*profile.hot_selected_kv,
            AppCostScope::selected_kv_transfer_only);
        result.cold_selected_kv_ms_per_selected_candidate = cost(*profile.cold_selected_kv,
            AppCostScope::selected_kv_transfer_only);
        hot += static_cast<long double>(profile.hot_retention) *
            result.hot_selected_kv_ms_per_selected_candidate;
        cold += static_cast<long double>(profile.cold_retention) *
            result.cold_selected_kv_ms_per_selected_candidate;
    } else {
        require(profile.hot_retention == 0 && profile.cold_retention == 0 &&
                profile.retention_source.empty(),
                "APP score-only profile must not imply unmeasured KV transfer/skew");
    }
    result.hot_total_ms_per_candidate = finite_double(hot);
    result.cold_total_ms_per_candidate = finite_double(cold);
    result.continuous_beta = finite_double(cold / hot);
    require(result.continuous_beta > 0, "APP beta underflow");
    // Form rho without an overflowing hot+cold intermediate.
    const long double rho = cold >= hot ? 1 / (1 + hot / cold) : (cold / hot) / (1 + cold / hot);
    result.continuous_rho = finite_double(rho);

    const auto total = constraints.total_candidates;
    require(total != 0, "APP cannot partition an empty candidate set");
    require(constraints.cold_alignment != 0, "APP cold alignment must be positive");
    require(constraints.mandatory_hot_candidates <= total, "APP mandatory hot set exceeds candidates");
    const auto hot_budget = std::min(total, constraints.hot_budget_candidates);
    require(constraints.mandatory_hot_candidates <= hot_budget,
            "APP mandatory hot set exceeds the remaining hot budget");
    require(constraints.min_cold_candidates <= constraints.max_cold_candidates,
            "APP inverted kernel bounds");
    const auto low = std::max(total - hot_budget, constraints.min_cold_candidates);
    const auto high = std::min(total - constraints.mandatory_hot_candidates,
                               constraints.max_cold_candidates);
    require(low <= high, "APP budget/mandatory-hot/kernel bounds have no feasible partition");
    const auto alignment = constraints.cold_alignment;
    // Quotient bounds avoid overflow in ceil(low/alignment)*alignment.
    const auto first = low / alignment + (low % alignment != 0);
    const auto last = high / alignment;
    require(first <= last, "APP has no aligned cold partition within the budget");
    const auto aligned_low = first * alignment;
    const auto aligned_high = last * alignment;
    const long double ideal_hot = static_cast<long double>(total) * rho;
    const long double ideal_cold = static_cast<long double>(total) - ideal_hot;
    result.continuous_hot_candidates = finite_double(ideal_hot);
    result.hot_budget_clamped = ideal_hot > static_cast<long double>(hot_budget);
    result.mandatory_hot_clamped = ideal_hot < static_cast<long double>(constraints.mandatory_hot_candidates);
    result.kernel_bounds_clamped = ideal_cold < static_cast<long double>(constraints.min_cold_candidates) ||
        ideal_cold > static_cast<long double>(constraints.max_cold_candidates);
    const long double bounded_cold = std::max(static_cast<long double>(low),
                                             std::min(ideal_cold, static_cast<long double>(high)));
    std::uint64_t chosen;
    if (bounded_cold <= static_cast<long double>(aligned_low)) {
        chosen = aligned_low;
    } else if (bounded_cold >= static_cast<long double>(aligned_high)) {
        chosen = aligned_high;
    } else {
        // Cast is safe here: bounded_cold lies strictly inside integer bounds.
        const auto lower = (static_cast<std::uint64_t>(bounded_cold) / alignment) * alignment;
        const auto upper = lower + alignment; // upper <= aligned_high
        const auto imbalance = [&](std::uint64_t n) {
            return std::fabs(static_cast<long double>(total - n) * hot -
                             static_cast<long double>(n) * cold);
        };
        chosen = imbalance(lower) < imbalance(upper) ? lower : upper;
    }
    result.cold_candidates = chosen;
    result.hot_candidates = total - chosen;
    result.actual_rho = double(result.hot_candidates) / double(total);
    if (chosen) result.actual_beta = double(result.hot_candidates) / double(chosen);
    result.predicted_hot_ms = finite_double(static_cast<long double>(result.hot_candidates) * hot);
    result.predicted_cold_ms = finite_double(static_cast<long double>(chosen) * cold);
    result.predicted_imbalance_ms = finite_double(std::fabs(
        static_cast<long double>(result.hot_candidates) * hot - static_cast<long double>(chosen) * cold));
    result.alignment_adjusted = static_cast<long double>(chosen) != bounded_cold;
    result.clamped = result.hot_budget_clamped || result.mandatory_hot_clamped ||
        result.kernel_bounds_clamped || result.alignment_adjusted;
    return result;
}
} // namespace hillserve
