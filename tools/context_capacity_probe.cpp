// Read-only capacity evidence from the production planner/converter. This is
// NOT a model run, hardware replay, simulated latency, or performance sample.
#include "runtime/full_gqa_contract.hpp"
#include "policy/host_policy.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto model_bytes = std::stoull(argv[1]);
    using namespace hillserve::runtime;
    std::cout << "{\"evidence_mode\":\"STATIC_CAPACITY_AND_ACTUAL_CPU_CONTRACT_CHECK\","
                 "\"performance_samples_generated\":false,\"model_file_bytes\":" << model_bytes
              << ",\"cases\":[";
    bool comma = false;
    for (std::uint64_t prompt : {4096, 8192, 16384, 32768, 65536}) {
        if (comma) std::cout << ',';
        comma = true;
        const auto capacity = (prompt + 32 + 63) / 64 * 64;
        const auto native = (capacity + 255) / 256 * 256 * 32 * 8 * 128 * 4;
        const auto required = model_bytes + native + (4ULL << 30);
        const auto full_kv = full_gqa_align_page(4096 + 32 * capacity * 4128);
        const auto keys = 32 * capacity * 2048;
        FullGqaRuntimeSpec s;
        s.prompt_tokens = prompt; s.generated_capacity = 32; s.capacity = capacity;
        s.beta = 0.49707602339181306; s.hot_kv_budget_bytes = 4ULL << 30;
        s.total_host_workspace_budget_bytes = 12ULL << 30;
        std::string error;
        try { (void) plan_full_gqa_runtime(s); }
        catch (const std::exception& e) { error = e.what(); }
        const auto hot = full_gqa_hot_blocks(prompt, s.beta) * 64;
        std::cout << "{\"P\":" << prompt << ",\"G\":32,\"capacity\":" << capacity
                  << ",\"hot\":" << hot << ",\"cold\":" << prompt-hot
                  << ",\"resident_KV_bytes\":" << (hot+32)*32*4096
                  << ",\"native_GPU_KV_bytes\":" << native
                  << ",\"existing_GPU_preflight_required_bytes\":" << required
                  << ",\"GPU_total_budget_bytes\":" << (24ULL << 30)
                  << ",\"GPU_preflight_fits\":" << (required <= (24ULL << 30) ? "true" : "false")
                  << ",\"complete_KV_file_bytes\":" << full_kv
                  << ",\"raw_K_file_bytes\":" << keys
                  << ",\"new_storage_total_bytes\":" << full_kv+keys
                  << ",\"planner_passed\":" << (error.empty() ? "true" : "false")
                  << ",\"planner_error\":\"" << error << "\"}";
    }
    std::cout << "],\"actual_binary16_count_conversion\":[";
    comma = false;
    for (auto count : {65504, 65519, 65520, 65536, 65568}) {
        if (comma) std::cout << ',';
        comma = true;
        const auto bits = hillserve::half_bits(float(count));
        const auto result = hillserve::half_value(bits);
        std::cout << "{\"count\":" << count << ",\"F16_bits\":" << bits
                  << ",\"finite\":" << (std::isfinite(result) ? "true" : "false") << '}';
    }
    std::cout << "]}\n";
}
