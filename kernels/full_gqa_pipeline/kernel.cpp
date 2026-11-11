// Independent raw128 GQA pipeline. V3 byte layout matches V2 only for the
// explicitly supported N64..32768, N%64==0. No partial-word memory access.
#include "kernel.hpp"
#if defined(HILLSERVE_HLS) || defined(__SYNTHESIS__)
#include <hls_stream.h>
template<class T> using PipelineStream=hls::stream<T>;
#else
#include <deque>
#include <stdexcept>
template<class T> class PipelineStream {
    std::deque<T> data;
public:
    explicit PipelineStream(const char*){}
    void write(const T& value){data.push_back(value);}
    T read(){if(data.empty())throw std::runtime_error("native pipeline FIFO underflow");T value=data.front();data.pop_front();return value;}
};
#endif
#ifndef HILLSERVE_PIPELINE_TOKEN_PAR
#define HILLSERVE_PIPELINE_TOKEN_PAR 4
#endif
namespace {
constexpr unsigned Q_HEADS=32,KV_HEADS=8,DIM=128,GROUP=4,TILE=64,PACK=32;
constexpr unsigned TOKEN_PAR=HILLSERVE_PIPELINE_TOKEN_PAR;
constexpr unsigned LANES=TILE/TOKEN_PAR;
static_assert(TOKEN_PAR==2 || TOKEN_PAR==4,"only separately tested2/4-token variants");
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

void read_keys(const hs_pipeline_word* keys,PipelineStream<hs_pipeline_word>& output,unsigned tokens) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE off
#endif
    const unsigned words=tokens*KV_HEADS*DIM/PACK;
    read_contiguous_keys: for(unsigned word=0;word<words;++word) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
        output.write(keys[word]);
    }
}

void compute_scores(const hs_pipeline_word* query,PipelineStream<hs_pipeline_word>& input,
                    PipelineStream<hs_pipeline_word>& output,unsigned tokens) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE off
#endif
    float q_cache[KV_HEADS][GROUP][DIM];
    hs_pipeline_word k_tile[TOKEN_PAR][KV_HEADS][LANES][DIM/PACK];
    float accumulator[TOKEN_PAR][KV_HEADS][GROUP][LANES];
#ifdef __SYNTHESIS__
#pragma HLS ARRAY_PARTITION variable=q_cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_cache complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=2
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=3
#pragma HLS BIND_STORAGE variable=accumulator type=ram_2p impl=lutram
#endif
    // Query load is once per invocation. Sequential element consumption keeps
    // each query-bank write port single-use without requiring32 bank writes.
    load_query_words: for(unsigned wi=0;wi<Q_HEADS*DIM/PACK;++wi) {
        const hs_pipeline_word word=query[wi];
        load_query_elements: for(unsigned p=0;p<PACK;++p) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            const unsigned index=wi*PACK+p,head=index/DIM;
            q_cache[head/GROUP][head%GROUP][index%DIM]=decode_half(hs_pipeline_get(word,p));
        }
    }
    compute_tiles: for(unsigned base=0;base<tokens;base+=TILE) {
        fill_local_tile: for(unsigned wi=0;wi<TILE*KV_HEADS*DIM/PACK;++wi) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            const hs_pipeline_word word=input.read();
            const unsigned token=wi/(KV_HEADS*DIM/PACK),kv=(wi/(DIM/PACK))%KV_HEADS;
            // A single full512-bit bank write. The former32 scalar writes
            // through ARRAY_RESHAPE inferred read/modify/write and actual II16.
            k_tile[token%TOKEN_PAR][kv][token/TOKEN_PAR][wi%(DIM/PACK)]=word;
        }
        clear_accumulators: for(unsigned lane=0;lane<LANES;++lane) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            clear_tokens: for(unsigned p=0;p<TOKEN_PAR;++p) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                clear_heads: for(unsigned kv=0;kv<KV_HEADS;++kv) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    clear_groups: for(unsigned g=0;g<GROUP;++g) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                        accumulator[p][kv][g][lane]=0.0f;
                    }
                }
            }
        }
        // Per score: channel0..127, independent rounded FP32 multiply and add.
        // With TOKEN_PAR4 the same accumulator is revisited every16 iterations.
        // No false-dependence directive or channel-tree reassociation is used.
        ordered_channels: for(unsigned step=0;step<DIM*LANES;++step) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
            const unsigned channel=step/LANES,lane=step%LANES;
            parallel_tokens: for(unsigned p=0;p<TOKEN_PAR;++p) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                parallel_KV_heads: for(unsigned kv=0;kv<KV_HEADS;++kv) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    const hs_pipeline_word packed_key=k_tile[p][kv][lane][channel/PACK];
                    const float key=decode_half(hs_pipeline_get(packed_key,channel%PACK));
                    parallel_Q_group: for(unsigned g=0;g<GROUP;++g) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                        const float product=q_cache[kv][g][channel]*key;
                        accumulator[p][kv][g][lane]=accumulator[p][kv][g][lane]+product;
                    }
                }
            }
        }
        // Preserve actual RAM banks: one read per token-parallel bank per
        // cycle. A32-score word is assembled in PACK/TOKEN_PAR steps instead
        // of requesting eight reads per small RAM bank or making all lanes
        // registers (which introduced a large feedback mux and actual II7).
        pack_score_words: for(unsigned wi=0;wi<Q_HEADS*TILE/PACK;++wi) {
            const unsigned h=wi/(TILE/PACK),base_token=(wi%(TILE/PACK))*PACK;
            hs_pipeline_word word{};
            pack_score_groups: for(unsigned lane=0;lane<PACK/TOKEN_PAR;++lane) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                pack_parallel_tokens: for(unsigned p=0;p<TOKEN_PAR;++p) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
                    const unsigned score_lane=base_token/TOKEN_PAR+lane;
                    hs_pipeline_set(word,lane*TOKEN_PAR+p,
                        encode_half(accumulator[p][h/GROUP][h%GROUP][score_lane]));
                }
            }
            output.write(word);
        }
    }
}

void write_scores(PipelineStream<hs_pipeline_word>& input,hs_pipeline_word* scores,unsigned tokens) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE off
#endif
    const unsigned words_per_head=tokens/PACK;
    store_tiles: for(unsigned base=0;base<tokens;base+=TILE) {
        store_heads: for(unsigned head=0;head<Q_HEADS;++head) {
            store_wide_words: for(unsigned word=0;word<TILE/PACK;++word) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
                scores[head*words_per_head+base/PACK+word]=input.read();
            }
        }
    }
}

void run_pipeline(const hs_pipeline_word* query,const hs_pipeline_word* keys,
                  hs_pipeline_word* scores,unsigned tokens) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE off
#endif
    PipelineStream<hs_pipeline_word> key_words("key_words");
    PipelineStream<hs_pipeline_word> score_words("score_words");
#ifdef __SYNTHESIS__
#pragma HLS STREAM variable=key_words depth=2048
#pragma HLS STREAM variable=score_words depth=64
#pragma HLS DATAFLOW
#endif
    read_keys(keys,key_words,tokens);
    compute_scores(query,key_words,score_words,tokens);
    write_scores(score_words,scores,tokens);
}
}

extern "C" void hillserve_full_gqa_pipeline(const hs_pipeline_word* query,
                                           const hs_pipeline_word* keys,
                                           hs_pipeline_word* scores,
                                           std::uint32_t tokens) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE m_axi port=query offset=slave bundle=gmem0 depth=128 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=keys offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=scores offset=slave bundle=gmem2 depth=32768 max_write_burst_length=64 num_write_outstanding=8
#pragma HLS INTERFACE s_axilite port=query bundle=control
#pragma HLS INTERFACE s_axilite port=keys bundle=control
#pragma HLS INTERFACE s_axilite port=scores bundle=control
#pragma HLS INTERFACE s_axilite port=tokens bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif
    if(tokens<64 || tokens>32768 || tokens%64!=0)return;
    run_pipeline(query,keys,scores,tokens);
}
