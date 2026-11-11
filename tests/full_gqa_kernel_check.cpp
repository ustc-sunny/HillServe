#include "kernels/full_gqa_f16/kernel.hpp"
#include "policy/host_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
// The oracle deliberately does not use kernel conversion routines, GQA tiles,
// interleaved accumulators, or shared-key caches.
std::vector<std::uint16_t> oracle(const std::vector<std::uint16_t>& q,
                                 const std::vector<std::uint16_t>& k,unsigned n) {
    std::vector<std::uint16_t> out(32*n);
    for(unsigned h=0;h<32;++h)for(unsigned t=0;t<n;++t){
        float sum=0;
        for(unsigned c=0;c<128;++c){
            const float product=hillserve::half_value(q[h*128+c])*
                hillserve::half_value(k[(t*8+h/4)*128+c]);
            sum=sum+product;
        }
        out[h*n+t]=hillserve::half_bits(sum);
    }
    return out;
}
std::uint32_t rng=0x62c83e17;
std::uint32_t random_word(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
std::uint16_t sample(){
    // Finite mixed signs/exponents, includes F16 subnormals. Kept bounded so
    // ordinary random-score tests do not merely agree on overflow to infinity.
    return std::uint16_t((random_word()&0x8000u)|
                        ((random_word()%17u)<<10)|(random_word()&1023u));
}
std::uint64_t comparisons=0,cases=0;
void compare(const std::vector<std::uint16_t>& q,
             const std::vector<std::uint16_t>& k,unsigned n,bool rtl_depth=false) {
    const std::uint16_t canary=0x55aau;
    // RTL co-simulation can marshal the interface's declared maximum depth.
    // Back those pointer extents even though smoke only computes16/17 tokens.
    std::vector<std::uint16_t> got((rtl_depth?32u*32768u:32u*n)+2,canary);
    hillserve_full_gqa_f16(q.data(),k.data(),got.data()+1,n);
    need(got.front()==canary&&got.back()==canary,"kernel wrote output guard");
    const auto expected=oracle(q,k,n);
    for(std::size_t i=0;i<expected.size();++i) {
        need(std::isfinite(hillserve::half_value(expected[i])),"oracle score nonfinite/overflow");
        if(got[i+1]!=expected[i])throw std::runtime_error("raw F16 mismatch at score "+std::to_string(i));
    }
    comparisons+=expected.size();++cases;
}
template<class T>std::vector<T> read(const std::string& path,std::size_t count) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    need(bool(f)&&f.tellg()==std::streamoff(count*sizeof(T)),"input file extent mismatch");
    f.seekg(0);std::vector<T> out(count);
    need(bool(f.read(reinterpret_cast<char*>(out.data()),count*sizeof(T))),"short input file");
    return out;
}
}
int main(int argc,char**argv){try {
    const bool f32_rehearsal=argc==5&&std::string(argv[1])=="--files-f32-rehearsal";
    const bool rtl_smoke=argc==2&&std::string(argv[1])=="--rtl-smoke";
    if(rtl_smoke) {
        // Two transactions cover one whole16-lane tile and a one-token tail.
        // Only the ABI-declared pointer allocation is large; N and oracle work
        // remain16/17. Zero-filled unused K avoids uninitialized driver reads.
        for(unsigned n:{16u,17u}) {
            std::vector<std::uint16_t> q(4096),k(32768u*1024u,0);
            for(auto& x:q)x=sample();
            for(std::size_t i=0;i<std::size_t(n)*1024;++i)k[i]=sample();
            std::fill(q.begin(),q.begin()+128,0);q[0]=0x3c00;
            std::fill(q.begin()+28*128,q.begin()+29*128,0);q[28*128+3]=0x3c00;
            std::fill(q.begin()+31*128,q.end(),0);
            q[31*128]=0x6c00;q[31*128+1]=0x3c00;q[31*128+2]=0xec00;
            for(unsigned t=0;t<n;++t) {
                k[t*1024]=std::uint16_t(t+1); // head0: exact positive subnormal
                const auto last=(t*8+7)*128;
                k[last]=0x6c00;k[last+1]=0x3c00;k[last+2]=0x6c00;
                k[last+3]=std::uint16_t(0x8000u|unsigned(t+1)); // head28: negative marker
            }
            const auto check=oracle(q,k,n);
            for(unsigned t=0;t<n;++t) {
                need(check[t]==t+1,"positive subnormal marker oracle");
                need(check[28*n+t]==std::uint16_t(0x8000u|unsigned(t+1)),"negative subnormal marker oracle");
                need(check[31*n+t]==0,"ordered-cancellation oracle");
            }
            compare(q,k,n,true);
        }
    } else if(argc==5&&(std::string(argv[1])=="--files"||f32_rehearsal)) {
        std::size_t consumed=0;const auto parsed=std::stoul(argv[4],&consumed);
        need(consumed==std::strlen(argv[4])&&parsed>=1&&parsed<=32768,"file token bound");
        const unsigned n=unsigned(parsed);
        std::vector<std::uint16_t> q(4096);
        if(f32_rehearsal){const auto original=read<float>(argv[2],4096);
            for(std::size_t i=0;i<q.size();++i){need(std::isfinite(original[i]),"nonfinite rehearsal F32 query");q[i]=hillserve::half_bits(original[i]);}}
        else q=read<std::uint16_t>(argv[2],4096);
        auto k=read<std::uint16_t>(argv[3],std::size_t(n)*1024);
        for(auto x:q)need((x&0x7c00)!=0x7c00,"nonfinite raw query input");
        for(auto x:k)need((x&0x7c00)!=0x7c00,"nonfinite raw key input");
        compare(q,k,n);
    } else {
        need(argc==1||(argc==2&&std::string(argv[1])=="--max-shape"),
             "usage: full-gqa-kernel-check [--rtl-smoke | --max-shape | --files QUERY_F16 KEY_F16 TOKENS | --files-f32-rehearsal QUERY_F32 RAW_K_F16 TOKENS]");
        // Invalid counts must not dereference even null Q/K; outputs stay intact.
        std::uint16_t sentinel=0x1234;
        hillserve_full_gqa_f16(nullptr,nullptr,&sentinel,0);
        hillserve_full_gqa_f16(nullptr,nullptr,&sentinel,32769);
        hillserve_full_gqa_f16(nullptr,nullptr,&sentinel,0xffffffffu);
        need(sentinel==0x1234,"invalid-shape call wrote output");
        for(unsigned n:{1u,15u,16u,17u,31u,32u,33u,129u,1024u}){
            std::vector<std::uint16_t> q(4096),k(std::size_t(n)*1024);
            for(auto& x:q)x=sample();for(auto& x:k)x=sample();compare(q,k,n);
        }
        // Independent physical head and channel markers expose wrong head/4,
        // token-major addressing, score strides, and cross-lane contamination.
        {const unsigned n=17;std::vector<std::uint16_t> q(4096),k(n*1024);
         for(unsigned h=0;h<32;++h)q[h*128+(h*7)%128]=hillserve::half_bits(float(h+1)/32);
         for(unsigned t=0;t<n;++t)for(unsigned h=0;h<8;++h)for(unsigned c=0;c<128;++c)
             k[(t*8+h)*128+c]=hillserve::half_bits(float(t*1024+h*128+c)/16384);
         compare(q,k,n);}
        // Cancellation/order witness: (4096*4096 +1 -4096*4096) rounds to0;
        // reassociation can produce1. Product and sum order are never relaxed.
        // Include signed zero, subnormal, maximal finite and halfway-RNE terms.
        {const unsigned n=17;std::vector<std::uint16_t> q(4096),k(n*1024);
         const std::uint16_t edge[]={0,0x8000,1,0x8001,0x3ff,0x400,0x3c00,0xbc00,0x7bff,0xfbff};
         for(unsigned h=0;h<32;++h){q[h*128]=0x6c00;q[h*128+1]=0x3c00;q[h*128+2]=0xec00;
             for(unsigned c=3;c<128;++c)q[h*128+c]=edge[(h+c)%10];}
         for(unsigned t=0;t<n;++t)for(unsigned h=0;h<8;++h){k[(t*8+h)*128]=0x6c00;k[(t*8+h)*128+1]=0x3c00;k[(t*8+h)*128+2]=0x6c00;
             for(unsigned c=3;c<128;++c)k[(t*8+h)*128+c]=std::uint16_t((t+h+c)%1024);}
         compare(q,k,n);}
        if(argc==2){const unsigned n=32768;std::vector<std::uint16_t> q(4096),k(std::size_t(n)*1024);
            for(auto& x:q)x=sample();for(auto& x:k)x=sample();compare(q,k,n);}
    }
    std::cout<<"{\"status\":\"passed\",\"evidence\":\""<<(rtl_smoke?"SCALAR_ORACLE_TESTBENCH_NOT_BOARD_TIMING":"CPU_C_SIMULATION_NOT_FPGA")<<"\",\"abi\":\"HS_FULL_GQA_F16_V2\",\"cases\":"<<cases<<",\"score_comparisons\":"<<comparisons<<",\"raw_F16_bit_mismatches\":0,\"rtl_smoke\":"<<(rtl_smoke?"true":"false")<<",\"RTL_acceptance_requires_tool_cosim_report\":"<<(rtl_smoke?"true":"false")<<",\"F32_rehearsal_query_converted_to_F16_once\":"<<(f32_rehearsal?"true":"false")<<",\"hardware_executed\":false}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
