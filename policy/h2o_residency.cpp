#include "policy/h2o_residency.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace hillserve::policy {
namespace {
void need(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    need(!b || a <= std::numeric_limits<std::uint64_t>::max()/b,
         "H2O metadata size overflow");
    return a*b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    need(a <= std::numeric_limits<std::uint64_t>::max()-b,
         "H2O metadata size overflow");
    return a+b;
}
}

H2OResidency::H2OResidency(Config c) : config_(c), capacity_(0), recent_(0),
        heavy_(0), reserved_(0) {
    need(c.layers && c.query_heads && c.kv_heads && c.prompt_tokens &&
         c.query_heads%c.kv_heads==0, "H2O geometry must define complete GQA groups");
    const auto cap=add(c.prompt_tokens,c.generated_capacity);
    need(cap<=std::numeric_limits<std::size_t>::max(),"H2O token capacity overflow");
    capacity_=std::size_t(cap);
    need(c.prompt_resident_tokens_per_head<=c.prompt_tokens,
         "H2O prompt residency cannot exceed actual prompt");
    heavy_=c.prompt_resident_tokens_per_head/2;
    recent_=c.prompt_resident_tokens_per_head-heavy_;
    const auto score_count=mul(mul(c.layers,c.kv_heads),cap);
    const auto group_count=mul(c.kv_heads,cap);
    const auto result_ids=mul(c.kv_heads,add(c.prompt_resident_tokens_per_head,c.generated_capacity));
    reserved_=sizeof(*this);
    for(auto bytes:{mul(score_count,sizeof(float)),mul(group_count,sizeof(float)),
                   mul(c.layers,sizeof(std::size_t)),mul(c.prompt_tokens,sizeof(std::uint64_t)),
                   mul(result_ids,sizeof(std::uint64_t)),
                   mul(c.kv_heads,sizeof(std::vector<std::uint64_t>))})
        reserved_=add(reserved_,bytes);
    need(reserved_<=c.metadata_budget_bytes,"H2O metadata exceeds explicit budget");
    need(score_count<=scores_.max_size() && group_count<=grouped_.max_size(),
         "H2O metadata vector exceeds addressable size");
    scores_.assign(std::size_t(score_count),0.f);
    grouped_.assign(std::size_t(group_count),0.f);
    observed_.assign(c.layers,0);
    ranked_.reserve(c.prompt_tokens);
}

void H2OResidency::validate_block(std::size_t layer,std::size_t first,
        std::size_t queries,std::size_t keys) const {
    need(layer<config_.layers && queries && keys && keys<=capacity_,
         "H2O observation geometry out of bounds");
    need(first==observed_[layer] && first<=capacity_ && queries<=capacity_-first &&
         keys==first+queries,"H2O observation must be a new contiguous causal block");
}

void H2OResidency::observe_attention(std::size_t layer,std::size_t first,
        std::size_t queries,std::size_t keys,const float* probabilities,std::size_t elements) {
    validate_block(layer,first,queries,keys);
    need(probabilities && elements==mul(mul(config_.query_heads,queries),keys),
         "H2O actual attention shape mismatch");
    std::fill_n(grouped_.data(),config_.kv_heads*keys,0.f);
    const auto group_size=config_.query_heads/config_.kv_heads;
    for(std::size_t h=0;h<config_.query_heads;++h)
        for(std::size_t q=0;q<queries;++q) {
            double mass=0;
            for(std::size_t k=0;k<keys;++k) {
                const float p=probabilities[(h*queries+q)*keys+k];
                need(std::isfinite(p)&&p>=0.f&&p<=1.f,"H2O requires finite softmax probabilities");
                need(k<=first+q || p==0.f,"H2O future attention must be causally masked");
                mass+=p;
                grouped_[(h/group_size)*keys+k]+=p;
            }
            need(std::abs(mass-1.)<=2e-3,"H2O attention row is not a probability distribution");
        }
    observe_group_sums(layer,first,queries,keys,grouped_.data(),config_.kv_heads*keys);
}

void H2OResidency::observe_group_sums(std::size_t layer,std::size_t first,
        std::size_t queries,std::size_t keys,const float* sums,std::size_t elements) {
    validate_block(layer,first,queries,keys);
    need(sums && elements==mul(config_.kv_heads,keys),"H2O grouped attention shape mismatch");
    const double expected=double(queries)*double(config_.query_heads/config_.kv_heads);
    // Validate the whole block before mutating persistent state.
    for(std::size_t h=0;h<config_.kv_heads;++h) {
        double mass=0;
        for(std::size_t k=0;k<keys;++k) {
            const auto p=sums[h*keys+k];
            need(std::isfinite(p)&&p>=0.f,"H2O grouped attention contains invalid probability mass");
            mass+=p;
            need(std::isfinite(scores_[(layer*config_.kv_heads+h)*capacity_+k]+p),
                 "H2O accumulated importance overflow");
        }
        need(std::abs(mass-expected)<=2e-3*std::max(1.,expected),
             "H2O grouped mass is inconsistent with actual attention");
    }
    for(std::size_t h=0;h<config_.kv_heads;++h)
        for(std::size_t k=0;k<keys;++k)
            scores_[(layer*config_.kv_heads+h)*capacity_+k]+=sums[h*keys+k];
    observed_[layer]=keys;
}

std::vector<std::vector<std::uint64_t>> H2OResidency::resident_ids(std::size_t layer) {
    need(layer<config_.layers,"H2O resident layer out of bounds");
    const auto seen=observed_[layer],prompt=std::min(seen,config_.prompt_tokens);
    const auto recent=std::min(recent_,prompt),old=prompt-recent;
    const auto heavy=std::min(heavy_,old);
    std::vector<std::vector<std::uint64_t>> result(config_.kv_heads);
    for(std::size_t h=0;h<config_.kv_heads;++h) {
        auto& ids=result[h];
        ids.reserve(std::min(prompt,config_.prompt_resident_tokens_per_head)+
                    (seen>config_.prompt_tokens?seen-config_.prompt_tokens:0));
        if(prompt<=config_.prompt_resident_tokens_per_head) {
            ids.resize(prompt);std::iota(ids.begin(),ids.end(),0);
        } else {
            ranked_.resize(old);std::iota(ranked_.begin(),ranked_.end(),0);
            const auto base=(layer*config_.kv_heads+h)*capacity_;
            const auto better=[&](auto a,auto b) {
                const auto x=scores_[base+a],y=scores_[base+b];
                return x>y || (x==y && a<b);
            };
            if(heavy<ranked_.size())std::nth_element(ranked_.begin(),ranked_.begin()+heavy,ranked_.end(),better);
            ids.insert(ids.end(),ranked_.begin(),ranked_.begin()+heavy);
            std::sort(ids.begin(),ids.end());
            for(std::size_t k=old;k<prompt;++k)ids.push_back(k);
        }
        for(std::size_t k=config_.prompt_tokens;k<seen;++k)ids.push_back(k);
    }
    return result;
}

float H2OResidency::importance(std::size_t layer,std::size_t head,std::size_t position) const {
    need(layer<config_.layers&&head<config_.kv_heads&&position<observed_[layer],
         "H2O unobserved importance requested");
    return scores_[(layer*config_.kv_heads+head)*capacity_+position];
}

std::size_t H2OResidency::observed_tokens(std::size_t layer) const {
    need(layer<config_.layers,"H2O observation layer out of bounds");return observed_[layer];
}
}
