#pragma once
#include <cstdint>
#include <vector>
#include <string>
namespace hillserve {
// Binary16 inputs/output; FP32 products/accumulation. No quantization or head reduction.
float half_value(std::uint16_t);
std::uint16_t half_bits(float);
struct PolicySelection {
    std::size_t k=0, boundary_tied_heads=0;
    std::vector<std::uint64_t> ids; // [q_head][k], original token IDs
    std::vector<std::uint32_t> above_threshold;
};
std::vector<std::uint16_t> host_partial_scores(const std::vector<std::uint16_t>& q,
 const std::vector<std::uint16_t>& keys, std::size_t heads, std::size_t tokens, std::size_t channels);
// keys are [head][token][channel]. Caller must supply the approved rehearsal Q/partial K.
PolicySelection select_partial_scores(const std::vector<std::uint16_t>& scores,
 std::size_t heads, std::size_t tokens, float alpha, std::size_t cap, bool inclusive=false);
}
