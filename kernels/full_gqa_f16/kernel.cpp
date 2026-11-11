// Independent full-GQA kernel:32 scores in parallel, with ordered arithmetic.
// Four query heads share each physical KV head. Only independent scores are
// parallelized: every individual score still accumulates c=0..127 in order.
#include "kernel.hpp"

namespace {
constexpr unsigned Q_HEADS=32, KV_HEADS=8, DIM=128, GROUP=4, TILE=16;

float decode_half(std::uint16_t x) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
    const unsigned sign=unsigned(x&0x8000u)<<16;
    const unsigned exponent=(x>>10)&31u, fraction=x&1023u;
    unsigned bits;
    if(exponent==0) {
        if(fraction==0) bits=sign;
        else {
            // Fixed priority encoder, not a data-dependent normalization loop.
            const unsigned leading=fraction>=512?0:fraction>=256?1:
                fraction>=128?2:fraction>=64?3:fraction>=32?4:
                fraction>=16?5:fraction>=8?6:fraction>=4?7:
                fraction>=2?8:9;
            bits=sign|((112u-leading)<<23)|(((fraction<<(leading+1))&1023u)<<13);
        }
    } else if(exponent==31) bits=sign|0x7f800000u|(fraction<<13);
    else bits=sign|((exponent+112u)<<23)|(fraction<<13);
    union {std::uint32_t u;float f;} value;value.u=bits;return value.f;
}

std::uint16_t encode_half(float x) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
    union {std::uint32_t u;float f;} value;value.f=x;
    const unsigned sign=(value.u>>16)&32768u;
    unsigned mantissa=value.u&0x7fffffu;
    const unsigned raw_exponent=(value.u>>23)&255u;
    const int exponent=int(raw_exponent)-112;
    if(raw_exponent==255)return std::uint16_t(sign|31744u|(mantissa?512u:0u));
    if(exponent>=31)return std::uint16_t(sign|31744u);
    unsigned output,remainder,midpoint;
    if(exponent<=0) {
        if(exponent < -10)return std::uint16_t(sign);
        mantissa|=0x800000u;
        const unsigned shift=unsigned(14-exponent);
        output=mantissa>>shift;remainder=mantissa&((1u<<shift)-1);
        midpoint=1u<<(shift-1);
    } else {
        output=(unsigned(exponent)<<10)|(mantissa>>13);
        remainder=mantissa&8191u;midpoint=4096u;
    }
    return std::uint16_t(sign|(output+(remainder>midpoint ||
                                   (remainder==midpoint && (output&1u)))));
}
}

extern "C" void hillserve_full_gqa_f16(const std::uint16_t* query,
                                      const std::uint16_t* keys,
                                      std::uint16_t* scores,
                                      std::uint32_t tokens) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE m_axi port=query offset=slave bundle=gmem0 depth=4096 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=keys offset=slave bundle=gmem1 depth=33554432 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=scores offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=query bundle=control
#pragma HLS INTERFACE s_axilite port=keys bundle=control
#pragma HLS INTERFACE s_axilite port=scores bundle=control
#pragma HLS INTERFACE s_axilite port=tokens bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif
    if(tokens==0 || tokens>32768)return;
    constexpr unsigned PACK=32, WORDS_PER_TOKEN=KV_HEADS*DIM/PACK;
    float q_cache[KV_HEADS][GROUP][DIM];
    std::uint16_t k_tile[TILE][KV_HEADS][DIM];
    float accumulator[KV_HEADS][TILE][GROUP];
#ifdef __SYNTHESIS__
#pragma HLS ARRAY_PARTITION variable=q_cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_cache complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_tile complete dim=2
#pragma HLS ARRAY_RESHAPE variable=k_tile cyclic factor=32 dim=3
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=3
#endif
    load_query: for(unsigned index=0;index<Q_HEADS*DIM;++index) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
        const unsigned head=index/DIM, channel=index%DIM;
        q_cache[head/GROUP][head%GROUP][channel]=decode_half(query[index]);
    }
    token_tiles: for(unsigned base=0;base<tokens;base+=TILE) {
        const unsigned valid=tokens-base<TILE?tokens-base:TILE;
        if(valid==TILE) {
            // No per-element condition: every full tile reads exactly32KiB in
            // address order.32 contiguous u16 lanes match the local512-bit word.
            full_tile_words: for(unsigned word=0;word<TILE*WORDS_PER_TOKEN;++word) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                const unsigned lane=word/WORDS_PER_TOKEN;
                const unsigned kv=(word/(DIM/PACK))%KV_HEADS;
                const unsigned channel=(word%(DIM/PACK))*PACK;
                full_word_lanes: for(unsigned p=0;p<PACK;++p) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    k_tile[lane][kv][channel+p]=keys[base*KV_HEADS*DIM+word*PACK+p];
                }
            }
        } else {
            // Tail has the same contiguous physical layout. Never read a
            // padded token: zero-fill unused on-chip lanes in a separate loop.
            tail_read_words: for(unsigned word=0;word<valid*WORDS_PER_TOKEN;++word) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                const unsigned lane=word/WORDS_PER_TOKEN;
                const unsigned kv=(word/(DIM/PACK))%KV_HEADS;
                const unsigned channel=(word%(DIM/PACK))*PACK;
                tail_word_lanes: for(unsigned p=0;p<PACK;++p) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    k_tile[lane][kv][channel+p]=keys[base*KV_HEADS*DIM+word*PACK+p];
                }
            }
            tail_zero_words: for(unsigned word=valid*WORDS_PER_TOKEN;word<TILE*WORDS_PER_TOKEN;++word) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                const unsigned lane=word/WORDS_PER_TOKEN;
                const unsigned kv=(word/(DIM/PACK))%KV_HEADS;
                const unsigned channel=(word%(DIM/PACK))*PACK;
                zero_word_lanes: for(unsigned p=0;p<PACK;++p) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    k_tile[lane][kv][channel+p]=0;
                }
            }
        }
        clear_accumulators: for(unsigned lane=0;lane<TILE;++lane) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            clear_physical_heads: for(unsigned kv=0;kv<KV_HEADS;++kv) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                clear_query_group: for(unsigned g=0;g<GROUP;++g) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    accumulator[kv][lane][g]=0.0f;
                }
            }
        }
        // One score receives channels0..127 in the original order, with16
        // different token lanes between updates. Let HLS infer the true RAW
        // dependence; there is no DEPENDENCE false or arithmetic reassociation.
        ordered_channels: for(unsigned step=0;step<DIM*TILE;++step) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            const unsigned channel=step/TILE,lane=step%TILE;
            parallel_physical_heads: for(unsigned kv=0;kv<KV_HEADS;++kv) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                const float key=decode_half(k_tile[lane][kv][channel]);
                parallel_query_group: for(unsigned g=0;g<GROUP;++g) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    const float product=q_cache[kv][g][channel]*key;
                    accumulator[kv][lane][g]=accumulator[kv][lane][g]+product;
                }
            }
        }
        // Output ABI stays head-major, without exposing tile padding. Keep
        // writes bounded even when N is not a multiple of16 or32.
        store_query_heads: for(unsigned h=0;h<Q_HEADS;++h) {
            store_valid_scores: for(unsigned lane=0;lane<valid;++lane) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                scores[h*tokens+base+lane]=encode_half(accumulator[h/GROUP][lane][h%GROUP]);
            }
        }
    }
}
