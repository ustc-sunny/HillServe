// Independent scalar specification for per-head partial F16 dot products.
// ABI HS_PARTIAL_F16_V1: Q[H,D], K[H,N,D], output[H,N].
// GQA, RoPE, skewing and channel selection happen in the validated producer.
#include <cstdint>
static float decode_half(std::uint16_t x) {
    unsigned sign=unsigned(x&0x8000)<<16, exp=(x>>10)&31, frac=x&1023, bits;
    if(!exp) {
        if(!frac)bits=sign;
        else {int e=-14;while(!(frac&1024)){frac<<=1;--e;}bits=sign|(unsigned(e+127)<<23)|((frac&1023)<<13);}
    } else if(exp==31) bits=sign|0x7f800000|(frac<<13);
    else bits=sign|((exp+112)<<23)|(frac<<13);
    union {unsigned u;float f;} value;value.u=bits;return value.f;
}
static std::uint16_t encode_half(float x) {
    union {unsigned u;float f;} value;value.f=x;
    unsigned sign=(value.u>>16)&32768,mant=value.u&0x7fffff,rawexp=(value.u>>23)&255;
    int exp=int(rawexp)-112;
    if(rawexp==255)return std::uint16_t(sign|31744|(mant?512:0));
    if(exp>=31)return std::uint16_t(sign|31744);
    unsigned output,remainder,midpoint;
    if(exp<=0){if(exp < -10)return std::uint16_t(sign);mant|=0x800000;unsigned shift=unsigned(14-exp);output=mant>>shift;remainder=mant&((1u<<shift)-1);midpoint=1u<<(shift-1);}
    else {output=(unsigned(exp)<<10)|(mant>>13);remainder=mant&8191;midpoint=4096;}
    return std::uint16_t(sign|(output+(remainder>midpoint||(remainder==midpoint&&(output&1)))));
}
extern "C" void hillserve_partial_f16(const std::uint16_t* query,const std::uint16_t* keys,std::uint16_t* scores,
                                      std::uint32_t heads,std::uint32_t tokens,std::uint32_t channels) {
#ifdef __SYNTHESIS__
#pragma HLS INTERFACE m_axi port=query offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=keys offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=scores offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=query bundle=control
#pragma HLS INTERFACE s_axilite port=keys bundle=control
#pragma HLS INTERFACE s_axilite port=scores bundle=control
#pragma HLS INTERFACE s_axilite port=heads bundle=control
#pragma HLS INTERFACE s_axilite port=tokens bundle=control
#pragma HLS INTERFACE s_axilite port=channels bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif
    for(unsigned h=0;h<heads;++h)for(unsigned t=0;t<tokens;++t){
        float sum=0;
        for(unsigned c=0;c<channels;++c){
            float product=decode_half(query[h*channels+c])*decode_half(keys[(h*tokens+t)*channels+c]);
            sum=sum+product;
        }
        scores[h*tokens+t]=encode_half(sum);
    }
}
