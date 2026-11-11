#include "policy/h2o_residency.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using hillserve::policy::H2OResidency;
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f,const char* why){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,why);}
H2OResidency::Config configuration(){
    H2OResidency::Config c;c.layers=2;c.query_heads=8;c.kv_heads=2;
    c.prompt_tokens=6;c.generated_capacity=3;c.prompt_resident_tokens_per_head=3;
    c.metadata_budget_bytes=1<<20;return c;
}
// A valid causal attention tensor, whose groups disagree on the important
// older token. No QK/rehearsal approximation is involved in this fixture.
std::vector<float> probabilities(std::size_t first,std::size_t count){
    const auto n=first+count;std::vector<float> p(8*count*n,0);
    for(std::size_t h=0;h<8;++h)for(std::size_t q=0;q<count;++q){
        const auto position=first+q;
        const auto preferred=h<4?0:std::min<std::size_t>(2,position);
        p[(h*count+q)*n+preferred]=1;
    }
    return p;
}
std::vector<float> independent_group_sum(const std::vector<float>& p,std::size_t queries,std::size_t keys){
    std::vector<float> grouped(2*keys,0);
    for(std::size_t kh=0;kh<2;++kh)for(std::size_t k=0;k<keys;++k){
        double sum=0;
        for(std::size_t q=0;q<queries;++q)for(std::size_t lane=0;lane<4;++lane)
            sum+=p[((kh*4+lane)*queries+q)*keys+k];
        grouped[kh*keys+k]=float(sum);
    }
    return grouped;
}
}
int main(){try{
    auto c=configuration();H2OResidency raw(c),reduced(c);
    check(raw.heavy_prompt_slots()==1&&raw.recent_prompt_slots()==2,"odd budget assigns extra recent slot");
    for(std::size_t first:{0u,3u}){
        const auto p=probabilities(first,3),g=independent_group_sum(p,3,first+3);
        raw.observe_attention(0,first,3,first+3,p.data(),p.size());
        reduced.observe_group_sums(0,first,3,first+3,g.data(),g.size());
    }
    const auto selected=raw.resident_ids(0);
    check(selected==std::vector<std::vector<std::uint64_t>>{{0,4,5},{2,4,5}},"physical GQA groups must have independent heavy tokens");
    check(selected==reduced.resident_ids(0),"GPU-reduced and direct-attention interfaces disagree");
    check(raw.importance(0,0,0)==24&&raw.importance(0,1,2)==16,"importance sums queries and all four Q heads");
    check(raw.observed_tokens(1)==0&&raw.resident_ids(1)[0].empty(),"layer state leaked");
    // A formerly nonresident prompt position becomes important again. Full
    // history scores must remain addressable, unlike permanent-drop H2O.
    std::vector<float> g(2*7,0);g[1]=4;g[7+1]=4;
    raw.observe_group_sums(0,6,1,7,g.data(),g.size());
    check(raw.importance(0,0,1)==4,"discarded importance was not kept for future updates");
    auto after=raw.resident_ids(0);
    check(after[0]==std::vector<std::uint64_t>({0,4,5,6})&&after[1]==std::vector<std::uint64_t>({2,4,5,6}),"actual generated token must be pinned outside prompt capacity");
    check(std::find(after[0].begin(),after[0].end(),7)==after[0].end(),"uncomputed generated reservation became resident");
    const auto before=raw.importance(0,0,0);
    rejects([&]{raw.observe_group_sums(0,6,1,7,g.data(),g.size());},"replayed attention silently counted twice");
    check(raw.importance(0,0,0)==before&&raw.observed_tokens(0)==7,"rejected observation mutated persistent state");
    std::vector<float> bad(2*8,0);bad[0]=4;bad[8]=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{raw.observe_group_sums(0,7,1,8,bad.data(),bad.size());},"nonfinite mass accepted");
    check(raw.importance(0,0,0)==before&&raw.observed_tokens(0)==7,"partial validation mutated prior heads");
    bad[8]=3;
    rejects([&]{raw.observe_group_sums(0,7,1,8,bad.data(),bad.size());},"incorrect summed mass accepted");
    auto small=configuration();small.prompt_resident_tokens_per_head=6;H2OResidency fits(small);
    auto p=probabilities(0,6);fits.observe_attention(0,0,6,6,p.data(),p.size());
    check(fits.resident_ids(0)[0]==std::vector<std::uint64_t>({0,1,2,3,4,5}),"all-fitting KV was artificially evicted");
    // Equal older scores select the smaller absolute ID, not container order.
    H2OResidency ties(c);std::vector<float> equal(12,4.f);
    ties.observe_group_sums(0,0,6,6,equal.data(),equal.size());
    check(ties.resident_ids(0)[1]==std::vector<std::uint64_t>({0,4,5}),"tie selection is not deterministic");
    auto bounded=c;bounded.metadata_budget_bytes=raw.reserved_metadata_bytes()-1;
    rejects([&]{H2OResidency no(bounded);},"metadata budget overrun admitted");
    auto invalid=c;invalid.query_heads=7;rejects([&]{H2OResidency no(invalid);},"incomplete GQA groups accepted");
    auto future=probabilities(0,3);future[0]=0;future[1]=1;H2OResidency causal(c);
    rejects([&]{causal.observe_attention(0,0,3,3,future.data(),future.size());},"future attention accepted");
    check(causal.observed_tokens(0)==0,"causal rejection advanced policy");
    std::cout<<"passed: actual-attention group sums, cumulative history, recent/heavy, generated pins, budgets, causality and replay rejection\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
