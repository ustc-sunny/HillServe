#pragma once
#include <cstdint>
#if defined(HILLSERVE_HLS) || defined(__SYNTHESIS__)
#include <ap_int.h>
using hs_pipeline_word = ap_uint<512>;
inline std::uint16_t hs_pipeline_get(const hs_pipeline_word& word,unsigned lane) {
#pragma HLS INLINE
    return std::uint16_t(word.range(lane*16+15,lane*16));
}
inline void hs_pipeline_set(hs_pipeline_word& word,unsigned lane,std::uint16_t value) {
#pragma HLS INLINE
    word.range(lane*16+15,lane*16)=value;
}
#else
// Native C-simulation only. Same64-byte payload without vendor headers,
// unaligned casts, type punning, or reliance on vector over-alignment.
struct hs_pipeline_word { std::uint16_t lane[32]{}; };
static_assert(sizeof(hs_pipeline_word)==64,"512-bit native payload");
inline std::uint16_t hs_pipeline_get(const hs_pipeline_word& word,unsigned lane){return word.lane[lane];}
inline void hs_pipeline_set(hs_pipeline_word& word,unsigned lane,std::uint16_t value){word.lane[lane]=value;}
#endif
extern "C" void hillserve_full_gqa_pipeline(const hs_pipeline_word* query,
                                           const hs_pipeline_word* keys,
                                           hs_pipeline_word* scores,
                                           std::uint32_t tokens);
