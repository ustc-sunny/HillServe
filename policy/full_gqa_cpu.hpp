#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace hillserve {
// Fixed V3 scoring contract: 32 Q heads, 8 physical KV heads, 128 channels.
// F16 inputs; increasing-channel FP32 multiply then add (no FMA/reassociation);
// one final round-to-nearest-even F16 score; head-major output. Views are
// concatenated in caller order and may point directly into interleaved KV.
enum class FullGqaCpuImplementation { automatic, scalar, avx2_f16c };
// Shared original research baseline, extracted from hkm_app_replay::score_cpu.
// Software half conversion and striped token/KV-head work; creates/joins its
// worker threads on each call. K input may have stride 1024 or interleaved 2048.
inline constexpr char full_gqa_legacy_cpu_name[] = "legacy_software_half_per_call_threads";
std::vector<std::uint16_t> score_full_gqa_legacy_cpu(
    const std::vector<std::uint16_t>& query,const std::uint16_t* input,
    std::size_t tokens,std::size_t token_stride_words,unsigned threads);

struct FullGqaKeyView {
    const std::uint16_t* data = nullptr;
    std::size_t tokens = 0;
    std::size_t token_stride_words = 1024;
};

class FullGqaCpuScorer {
public:
    explicit FullGqaCpuScorer(unsigned threads,
        FullGqaCpuImplementation implementation = FullGqaCpuImplementation::automatic);
    ~FullGqaCpuScorer();
    FullGqaCpuScorer(const FullGqaCpuScorer&) = delete;
    FullGqaCpuScorer& operator=(const FullGqaCpuScorer&) = delete;
    // Keeps worker threads and the pointer index across calls. Output capacity
    // is caller owned/reusable. Calls on one instance are serialized; use two
    // instances for concurrent hot/cold scoring. All referenced storage must
    // remain valid until this synchronous operation returns.
    void score(const std::vector<std::uint16_t>& query,
               const std::vector<FullGqaKeyView>& views,
               std::vector<std::uint16_t>& output);
    unsigned threads() const noexcept;
    const char* implementation_name() const noexcept;
    // Heap/index/query storage only; OS thread stacks are included in measured
    // process RSS, not this estimate. Output must not overlap input storage.
    std::size_t workspace_bytes() const noexcept;
    static bool avx2_f16c_available() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
