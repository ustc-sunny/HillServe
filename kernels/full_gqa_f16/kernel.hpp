#pragma once
#include <cstdint>

// HS_FULL_GQA_F16_V2. Raw, already-RoPE Q/K; this kernel does not change the
// producer's query, skew/mask channels, normalize scores, or select token IDs.
// Q: F16[32][128]; K: F16[tokens][8][128]; S: F16[32][tokens].
// Valid tokens: 1..32768. Invalid counts perform no memory accesses; callers
// must reject them, not interpret an unchanged output as a successful score.
extern "C" void hillserve_full_gqa_f16(const std::uint16_t* query,
                                      const std::uint16_t* keys,
                                      std::uint16_t* scores,
                                      std::uint32_t tokens);
