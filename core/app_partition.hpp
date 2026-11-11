#pragma once
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace hillserve {

enum class AppCostScope {
    unspecified,
    host_hot_score_to_cpu_ready,
    cold_file_read_score_return_to_cpu_ready,
    selected_kv_transfer_only
};

// Keep the measured numerator, denominator and provenance, not just a derived
// rate. Scoring work_items counts physical token candidates (all Q heads for
// each token). Transfer work_items counts selected physical token candidates,
// after deduplicating GQA head selections. Both paths must use the same token
// geometry, numeric contract, CPU resources and timing convention.
struct AppMeasuredCost {
    double total_ms = 0;
    std::uint64_t work_items = 0;
    std::string source;
    AppCostScope scope = AppCostScope::unspecified;
};

struct AppPartitionProfile {
    AppMeasuredCost hot_score;
    // This is one measured wall-clock interval including the actual cold K
    // file read, FPGA submission/execution and score return. Do not add its
    // nested read/kernel/DMA timers again or substitute a DRAM-only kernel.
    AppMeasuredCost cold_read_score_return;
    // Optional disjoint KV transfer stages, both supplied or both absent.
    // They must exclude scoring, selection and any interval already counted
    // above. Their wall-time contribution is weighted by observed retention.
    std::optional<AppMeasuredCost> hot_selected_kv;
    std::optional<AppMeasuredCost> cold_selected_kv;
    double hot_retention = 0;
    double cold_retention = 0;
    std::string retention_source;
    // Geometry/representation/precision/resources/profile identity; retained
    // in the result so a caller cannot report an unlabelled generic rate.
    std::string configuration_id;
};

struct AppPartitionConstraints {
    std::uint64_t total_candidates = 0;
    // Caller converts the remaining hot KV byte budget to candidate capacity
    // AFTER reserving scoring keys, staging, indexes and other working sets.
    std::uint64_t hot_budget_candidates = 0;
    // Includes current/recent tokens that the placement policy must keep hot.
    std::uint64_t mandatory_hot_candidates = 0;
    std::uint64_t cold_alignment = 64;
    // Use zero only if the runner explicitly supports bypassing an empty FPGA
    // partition. A positive kernel minimum never silently falls back to CPU.
    std::uint64_t min_cold_candidates = 64;
    std::uint64_t max_cold_candidates = std::numeric_limits<std::uint64_t>::max();
};

struct AppPartitionPlan {
    AppPartitionProfile profile;
    AppPartitionConstraints constraints;
    double hot_score_ms_per_candidate = 0;
    double cold_read_score_return_ms_per_candidate = 0;
    double hot_selected_kv_ms_per_selected_candidate = 0;
    double cold_selected_kv_ms_per_selected_candidate = 0;
    double hot_total_ms_per_candidate = 0;
    double cold_total_ms_per_candidate = 0;
    bool includes_selected_kv_transfer = false;
    // beta = M_CPU / M_SSD, rho = M_CPU / (M_CPU + M_SSD).
    double continuous_beta = 0;
    double continuous_rho = 0;
    double continuous_hot_candidates = 0;
    std::uint64_t hot_candidates = 0;
    std::uint64_t cold_candidates = 0;
    std::optional<double> actual_beta; // null for a permitted all-hot partition
    double actual_rho = 0;
    double predicted_hot_ms = 0;
    double predicted_cold_ms = 0;
    double predicted_imbalance_ms = 0;
    // These are model estimates, never an observed delivery time or speedup.
    bool hot_budget_clamped = false;
    bool mandatory_hot_clamped = false;
    bool kernel_bounds_clamped = false;
    bool alignment_adjusted = false;
    bool clamped = false;
};

// With equal retention this is HPCA Eq. 5 in cost form:
// beta = (1/fs + alpha/Bs) / (1/fc + alpha/Bc).
// Separate measured hot/cold retention supplies the explicitly recorded skew
// correction. For the real file path, 1/fs includes cold-key storage access.
// Choose the legal aligned partition nearest the continuous balance point,
// minimizing |N_hot * c_hot - N_cold * c_cold|. Ties use less host memory.
// Missing/invalid measurements or an infeasible budget throw; no fallback.
AppPartitionPlan plan_app_partition(const AppPartitionProfile& profile,
                                    const AppPartitionConstraints& constraints);
const char* app_cost_scope_name(AppCostScope scope);

} // namespace hillserve
