#include "policy/full_gqa_cpu.hpp"
#include "policy/host_policy.hpp"
#include <cfenv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace hillserve;
namespace {
constexpr std::size_t H=32, KH=8, D=128;
void require(bool ok, const std::string& msg) { if(!ok) throw std::runtime_error(msg); }
template<class F> void rejects(F action) {
    bool caught=false;
    try { action(); } catch(const std::exception&) { caught=true; }
    require(caught,"invalid argument accepted");
}
std::uint32_t random_state=0x7f123451;
std::uint32_t random_word() { random_state^=random_state<<13; random_state^=random_state>>17; random_state^=random_state<<5; return random_state; }
std::uint16_t finite_half() {
    static constexpr std::uint16_t edge[]={0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
        0x3c00,0xbc00,0x7bff,0xfbff,0x3555,0xb555};
    auto word=random_word();
    return word%5==0?edge[(word>>8)%(sizeof(edge)/sizeof(edge[0]))]:
        std::uint16_t((word&0x83ff)|(((word>>10)%31)<<10));
}
std::vector<std::uint16_t> independent_oracle(const std::vector<std::uint16_t>& q,
                                            const std::vector<FullGqaKeyView>& views) {
    std::vector<const std::uint16_t*> keys;
    for(const auto& view:views) for(std::size_t t=0;t<view.tokens;++t)
        keys.push_back(view.data+t*view.token_stride_words);
    const auto n=keys.size();
    std::vector<std::uint16_t> output(H*n);
    for(std::size_t h=0;h<H;++h) for(std::size_t t=0;t<n;++t) {
        volatile float sum=0;
        for(std::size_t c=0;c<D;++c) {
            volatile float mul=half_value(q[h*D+c])*half_value(keys[t][(h/4)*D+c]);
            sum=sum+mul;
        }
        output[h*n+t]=half_bits(sum);
    }
    return output;
}
void check(FullGqaCpuScorer& scorer, const std::vector<std::uint16_t>& q,
           const std::vector<FullGqaKeyView>& views, std::size_t& count) {
    const auto oracle=independent_oracle(q,views);
    std::vector<std::uint16_t> out;
    scorer.score(q,views,out);
    if(out!=oracle) {
        for(std::size_t i=0;i<out.size();++i) if(out[i]!=oracle[i])
            throw std::runtime_error(std::string(scorer.implementation_name())+" bit mismatch at "+
                std::to_string(i)+": "+std::to_string(out[i])+" != "+std::to_string(oracle[i]));
    }
    count+=out.size();
    // Reuse both the worker pool and output allocation with a different query.
    auto next=q;
    for(auto& value:next) value^=0x8000;
    scorer.score(next,views,out);
    require(out==independent_oracle(next,views),"second pooled call mismatch");
    count+=out.size();
}
}
int main() {
    try {
        const bool avx=FullGqaCpuScorer::avx2_f16c_available();
        if(std::getenv("HILLSERVE_REQUIRE_AVX2")) require(avx,"AVX2/F16C required for this validation");
        std::vector<FullGqaCpuImplementation> backends={FullGqaCpuImplementation::scalar,
            FullGqaCpuImplementation::automatic};
        if(avx) backends.push_back(FullGqaCpuImplementation::avx2_f16c);
        else rejects([]{FullGqaCpuScorer scorer(1,FullGqaCpuImplementation::avx2_f16c);});
        std::size_t compared=0;
        for(const auto impl:backends) for(unsigned threads:{1u,2u,4u,7u}) {
            FullGqaCpuScorer scorer(threads,impl);
            require(scorer.threads()==threads,"thread count");
            for(std::size_t n:{1u,2u,7u,8u,9u,17u,65u,129u}) for(std::size_t stride:{1024u,2048u}) {
                std::vector<std::uint16_t> q(H*D), data(n*stride,0x7e00);
                for(auto& x:q) x=finite_half();
                for(std::size_t t=0;t<n;++t) for(std::size_t c=0;c<KH*D;++c)
                    data[t*stride+c]=finite_half();
                check(scorer,q,{{data.data(),n,stride}},compared);
                // Mixed noncontiguous views, including an odd tail and a
                // duplicate token. No contiguity/uniqueness requirement.
                if(n>=9) check(scorer,q,{{data.data()+7*stride,2,stride},
                    {data.data(),3,stride},{data.data()+stride,1,stride}},compared);
            }
            // Every product is representable in FP32; exercise minimum F16
            // subnormals, exact cancellation, negative zero and overflow.
            std::vector<std::uint16_t> q(H*D,1), data(9*1024,1);
            check(scorer,q,{{data.data(),9,1024}},compared);
            for(std::size_t i=0;i<q.size();++i) q[i]=i%2?0xfbff:0x7bff;
            std::fill(data.begin(),data.end(),0x7bff);
            check(scorer,q,{{data.data(),9,1024}},compared);
            std::fill(q.begin(),q.end(),0x7bff);
            check(scorer,q,{{data.data(),9,1024}},compared);
            std::fill(q.begin(),q.end(),0x8000);
            check(scorer,q,{{data.data(),9,1024}},compared);
            require(scorer.workspace_bytes()>0,"workspace accounting");
        }
        rejects([]{FullGqaCpuScorer scorer(0);});
        rejects([]{FullGqaCpuScorer scorer(129);});
        rejects([]{FullGqaCpuScorer scorer(1,static_cast<FullGqaCpuImplementation>(99));});
        FullGqaCpuScorer scorer(2);
        std::vector<std::uint16_t> q(H*D), key(1024), output;
        rejects([&]{scorer.score({},{{key.data(),1,1024}},output);});
        rejects([&]{scorer.score(q,{},output);});
        rejects([&]{scorer.score(q,{{nullptr,1,1024}},output);});
        rejects([&]{scorer.score(q,{{key.data(),0,1024}},output);});
        rejects([&]{scorer.score(q,{{key.data(),1,1}},output);});
        rejects([&]{scorer.score(q,{{key.data(),std::numeric_limits<std::size_t>::max(),1024}},output);});
        rejects([&]{scorer.score(q,{{key.data(),1,1024}},q);});
        std::fesetround(FE_DOWNWARD);
        rejects([]{FullGqaCpuScorer invalid_rounding_workers(2);});
        rejects([&]{scorer.score(q,{{key.data(),1,1024}},output);});
        std::fesetround(FE_TONEAREST);
        check(scorer,q,{{key.data(),1,1024}},compared);
        std::cout<<"{\"status\":\"passed\",\"avx2_f16c_tested\":"<<(avx?"true":"false")
                 <<",\"bitwise_scores_compared\":"<<compared
                 <<",\"ordered_fp32_no_fma\":true,\"persistent_workers\":true}"<<std::endl;
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
