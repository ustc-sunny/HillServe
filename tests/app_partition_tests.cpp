#include "core/app_partition.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace hillserve;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double a, double b) {
    check(std::fabs(a - b) <= 1e-12 * std::max({1.0, std::fabs(a), std::fabs(b)}),
          "unexpected analytic result");
}
template<class F> void rejects(F&& f) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    check(caught, "invalid/infeasible profile unexpectedly accepted");
}
AppPartitionProfile sample(double hot = 1, double cold = 3) {
    AppPartitionProfile p;
    p.hot_score = {hot * 64, 64, "test-fixture-hot-not-hardware",
                  AppCostScope::host_hot_score_to_cpu_ready};
    p.cold_read_score_return = {cold * 64, 64, "test-fixture-cold-not-hardware",
                               AppCostScope::cold_file_read_score_return_to_cpu_ready};
    p.configuration_id = "unit-fixture-one-physical-token-same-geometry";
    return p;
}
AppPartitionConstraints unconstrained(std::uint64_t total) {
    AppPartitionConstraints c;
    c.total_candidates = total;
    c.hot_budget_candidates = total;
    c.min_cold_candidates = 0;
    return c;
}
}

int main() {
    unsigned passed = 0, failed = 0;
    const auto test = [&](const char* name, const std::function<void()>& fn) {
        try { fn(); ++passed; std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& e) {
            ++failed; std::cerr << "FAIL " << name << ": " << e.what() << '\n';
        }
    };
    test("analytic score-only balance and preserved measurements", [] {
        const auto p = sample(); const auto c = unconstrained(256);
        const auto r = plan_app_partition(p, c);
        near(r.continuous_beta, 3); near(r.continuous_rho, .75);
        check(r.hot_candidates == 192 && r.cold_candidates == 64 && !r.clamped,
              "exact balance partition");
        near(r.predicted_hot_ms, 192); near(r.predicted_cold_ms, 192);
        check(!r.includes_selected_kv_transfer && r.profile.configuration_id == p.configuration_id &&
              r.profile.cold_read_score_return.total_ms == p.cold_read_score_return.total_ms &&
              r.profile.cold_read_score_return.work_items == p.cold_read_score_return.work_items &&
              r.profile.cold_read_score_return.source == p.cold_read_score_return.source,
              "raw evidence must survive planning");
    });
    test("equal-retention transfer costs reproduce paper Eq5", [] {
        // Independent rates fc=20, fs=4, Bc=10, Bs=2 candidates/ms, alpha=.2.
        auto p = sample(.05, .25);
        p.hot_selected_kv = AppMeasuredCost{10, 100, "fixture-hot-KV",
                                            AppCostScope::selected_kv_transfer_only};
        p.cold_selected_kv = AppMeasuredCost{50, 100, "fixture-cold-KV",
                                             AppCostScope::selected_kv_transfer_only};
        p.hot_retention = p.cold_retention = .2;
        p.retention_source = "fixture-selected-physical-IDs";
        const auto r = plan_app_partition(p, unconstrained(384));
        near(r.continuous_beta, 5); near(r.continuous_rho, 5.0 / 6);
        check(r.hot_candidates == 320 && r.cold_candidates == 64, "Eq5 partition");
        near(r.predicted_hot_ms, 22.4); near(r.predicted_cold_ms, 22.4);
    });
    test("measured hot-cold selection skew changes allocation exactly once", [] {
        auto p = sample();
        p.hot_selected_kv = AppMeasuredCost{20, 10, "hot-KV-disjoint-stage",
                                            AppCostScope::selected_kv_transfer_only};
        p.cold_selected_kv = AppMeasuredCost{80, 10, "cold-KV-disjoint-stage",
                                             AppCostScope::selected_kv_transfer_only};
        p.hot_retention = .5; p.cold_retention = .25;
        p.retention_source = "observed-unique-selected-IDs-fixture";
        const auto r = plan_app_partition(p, unconstrained(448));
        near(r.hot_total_ms_per_candidate, 2); near(r.cold_total_ms_per_candidate, 5);
        near(r.continuous_beta, 2.5);
        check(r.hot_candidates == 320 && r.cold_candidates == 128 &&
              r.includes_selected_kv_transfer, "skew-aware allocation");
    });
    test("cold storage cost is in profile and moves the balance", [] {
        const auto faster = plan_app_partition(sample(1, 1), unconstrained(512));
        const auto slower = plan_app_partition(sample(1, 3), unconstrained(512));
        check(faster.hot_candidates == 256 && slower.hot_candidates == 384,
              "slower storage+FPGA path must allocate fewer cold candidates");
        auto p = sample(); p.cold_read_score_return.scope = AppCostScope::host_hot_score_to_cpu_ready;
        rejects([&] { plan_app_partition(p, unconstrained(512)); });
    });
    test("hot budget and mandatory recent pins are both enforced", [] {
        auto c = unconstrained(1024); c.hot_budget_candidates = 500;
        auto r = plan_app_partition(sample(), c);
        check(r.hot_budget_clamped && r.alignment_adjusted && r.clamped &&
              r.hot_candidates == 448 && r.cold_candidates == 576, "budget-aligned clamp");
        c = unconstrained(1024); c.mandatory_hot_candidates = 900;
        r = plan_app_partition(sample(), c);
        check(r.mandatory_hot_clamped && r.hot_candidates == 960 && r.cold_candidates == 64,
              "pin-aligned clamp");
        c.hot_budget_candidates = 899;
        rejects([&] { plan_app_partition(sample(), c); });
    });
    test("kernel min max alignment and infeasible interval", [] {
        auto c = unconstrained(1024); c.min_cold_candidates = 400;
        auto r = plan_app_partition(sample(), c);
        check(r.kernel_bounds_clamped && r.cold_candidates == 448, "kernel minimum respected");
        c.min_cold_candidates = 0; c.max_cold_candidates = 100;
        r = plan_app_partition(sample(), c);
        check(r.kernel_bounds_clamped && r.cold_candidates == 64, "kernel maximum respected");
        c.min_cold_candidates = 65;
        rejects([&] { plan_app_partition(sample(), c); });
        c.min_cold_candidates = 101;
        rejects([&] { plan_app_partition(sample(), c); });
        c = unconstrained(1024); c.hot_budget_candidates = 1; c.mandatory_hot_candidates = 1;
        rejects([&] { plan_app_partition(sample(), c); });
    });
    test("nearest aligned tie uses less host memory", [] {
        auto c = unconstrained(192);
        const auto r = plan_app_partition(sample(1, 1), c);
        check(r.cold_candidates == 128 && r.hot_candidates == 64 && r.alignment_adjusted,
              "deterministic tie rule");
    });
    test("real first-decode shape preserves current token and all candidates", [] {
        auto c = unconstrained(32769); c.mandatory_hot_candidates = 1;
        c.min_cold_candidates = 64; c.max_cold_candidates = 32768;
        const auto r = plan_app_partition(sample(1, 1), c);
        check(r.hot_candidates == 16385 && r.cold_candidates == 16384 &&
              r.hot_candidates + r.cold_candidates == 32769 && r.cold_candidates % 64 == 0,
              "candidate must not disappear for kernel alignment");
    });
    test("all-hot bypass is explicit and zero-hot boundary remains possible", [] {
        auto c = unconstrained(128); c.mandatory_hot_candidates = 128;
        auto r = plan_app_partition(sample(), c);
        check(r.cold_candidates == 0 && !r.actual_beta && r.actual_rho == 1,
              "all-hot beta must not be a fake finite value");
        c.min_cold_candidates = 64;
        rejects([&] { plan_app_partition(sample(), c); });
        c = unconstrained(128); c.hot_budget_candidates = 0;
        r = plan_app_partition(sample(), c);
        check(r.hot_candidates == 0 && r.cold_candidates == 128 && *r.actual_beta == 0,
              "all-cold partition");
    });
    test("exhaustive small legal partitions minimize measured balance error", [] {
        // Enumerate the independent feasible set; do not mirror planner rounding.
        for (std::uint64_t n = 64; n < 600; n += 17) {
            for (std::uint64_t budget = 0; budget <= n; budget += 29) {
                for (std::uint64_t pins : {0ULL, 1ULL, 31ULL}) {
                    auto c = unconstrained(n); c.hot_budget_candidates = budget;
                    c.mandatory_hot_candidates = pins;
                    long double best = std::numeric_limits<long double>::infinity();
                    std::uint64_t expected = 0; bool feasible = false;
                    for (std::uint64_t cold = 0; cold <= n; cold += 64) {
                        const auto hot = n - cold;
                        if (hot > budget || hot < pins) continue;
                        const auto error = std::fabs(static_cast<long double>(hot) * 2 -
                                                     static_cast<long double>(cold) * 5);
                        if (error <= best) { best = error; expected = cold; feasible = true; }
                    }
                    if (!feasible) { rejects([&] { plan_app_partition(sample(2, 5), c); }); continue; }
                    const auto r = plan_app_partition(sample(2, 5), c);
                    check(r.cold_candidates == expected, "planner differs from exhaustive optimum");
                    near(r.predicted_imbalance_ms, static_cast<double>(best));
                }
            }
        }
    });
    test("missing provenance measurement counts and numeric errors rejected", [] {
        const auto c = unconstrained(256);
        for (int change = 0; change < 10; ++change) {
            auto p = sample();
            switch (change) {
            case 0: p.configuration_id.clear(); break;
            case 1: p.hot_score.source.clear(); break;
            case 2: p.cold_read_score_return.work_items = 0; break;
            case 3: p.hot_score.total_ms = 0; break;
            case 4: p.hot_score.total_ms = -1; break;
            case 5: p.hot_score.total_ms = std::numeric_limits<double>::quiet_NaN(); break;
            case 6: p.cold_read_score_return.total_ms = std::numeric_limits<double>::infinity(); break;
            case 7: p.cold_read_score_return.scope = AppCostScope::unspecified; break;
            case 8: p.hot_retention = -.1; break;
            case 9: p.cold_retention = 1.1; break;
            }
            rejects([&] { plan_app_partition(p, c); });
        }
        auto empty = c; empty.total_candidates = 0;
        rejects([&] { plan_app_partition(sample(), empty); });
        empty = c; empty.cold_alignment = 0;
        rejects([&] { plan_app_partition(sample(), empty); });
    });
    test("partial double-counted and unmeasured transfer inputs rejected", [] {
        auto p = sample(); const auto c = unconstrained(256);
        p.hot_selected_kv = AppMeasuredCost{1, 1, "transfer", AppCostScope::selected_kv_transfer_only};
        rejects([&] { plan_app_partition(p, c); });
        p.cold_selected_kv = p.hot_selected_kv;
        rejects([&] { plan_app_partition(p, c); }); // missing observed retention provenance
        p.retention_source = "fixture"; p.hot_retention = .2; p.cold_retention = .1;
        p.cold_selected_kv->scope = AppCostScope::cold_file_read_score_return_to_cpu_ready;
        rejects([&] { plan_app_partition(p, c); }); // score interval cannot be charged again
        p = sample(); p.hot_retention = .2;
        rejects([&] { plan_app_partition(p, c); }); // no silent transfer-cost omission
    });
    test("large integer limits do not overflow alignment arithmetic", [] {
        auto c = unconstrained(std::numeric_limits<std::uint64_t>::max());
        c.hot_budget_candidates = 0;
        rejects([&] { plan_app_partition(sample(), c); }); // max is not a multiple of 64
        c.hot_budget_candidates = 63;
        const auto r = plan_app_partition(sample(), c);
        check(r.hot_candidates == 63 && r.cold_candidates == c.total_candidates - 63,
              "maximum integer boundary");
    });
    std::cout << "APP partition unit tests: " << passed << " passed, " << failed
              << " failed; fixtures_only=true hardware_measurement=false\n";
    return failed ? 1 : 0;
}
