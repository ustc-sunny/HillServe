#include "core/core.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace hillserve {
bool Scope::operator==(const Scope & b) const {
    return std::tie(request,layer,step,head,epoch)==std::tie(b.request,b.layer,b.step,b.head,b.epoch);
}
ScoreMerge::ScoreMerge(Scope s, std::string r, const std::map<std::string,std::vector<Token>>& p,const std::vector<Token>& universe)
    : scope_(s), representation_(std::move(r)) {
    if (representation_.empty()) throw std::invalid_argument("missing representation");
    for (const auto& part:p) for (Token id:part.second)
        if (!owners_.emplace(id,part.first).second) throw std::invalid_argument("overlapping partitions");
    std::set<Token> expected(universe.begin(),universe.end());
    if(expected.size()!=universe.size()||owners_.size()!=expected.size())throw std::invalid_argument("partition does not cover universe");
    for(auto id:expected)if(!owners_.count(id))throw std::invalid_argument("partition identity outside universe");
}
void ScoreMerge::accept(const ScoreBlock& b) {
    if (!(b.scope==scope_) || b.representation!=representation_) throw std::invalid_argument("scope/scale mismatch");
    std::set<Token> pending;
    for (const auto& s:b.scores) {
        auto owner=owners_.find(s.token);
        if (!std::isfinite(s.value) || owner==owners_.end() || owner->second!=b.partition ||
            received_.count(s.token) || !pending.insert(s.token).second)
            throw std::invalid_argument("invalid/duplicate score or partition");
    }
    for (const auto& s:b.scores) received_.emplace(s.token,s.value);
}
bool ScoreMerge::complete() const { return received_.size()==owners_.size(); }
std::vector<Score> ScoreMerge::finish() const {
    if (!complete()) throw std::logic_error("global selection barrier incomplete");
    std::vector<Score> r;
    for (auto s:received_) r.push_back({s.first,s.second});
    return r;
}
std::vector<Token> diagnostic_top_k(std::vector<Score> s,std::size_t k) {
    if(k>s.size()) throw std::invalid_argument("selection budget exceeds candidates");
    std::set<Token> unique;
    for(auto v:s) if(!std::isfinite(v.value)||!unique.insert(v.token).second)
        throw std::invalid_argument("invalid diagnostic scores");
    std::sort(s.begin(),s.end(),[](auto a,auto b){return a.value!=b.value?a.value>b.value:a.token<b.token;});
    std::vector<Token> r;
    for(std::size_t i=0;i<k;++i) r.push_back(s[i].token);
    return r;
}
std::vector<double> diagnostic_full_dot_product(const std::vector<float>& q,const std::vector<float>& k) {
    if(q.empty()||k.size()%q.size()) throw std::invalid_argument("invalid dot shape");
    std::vector<double> r(k.size()/q.size(),0.0);
    for(std::size_t t=0;t<r.size();++t) for(std::size_t d=0;d<q.size();++d) {
        if(!std::isfinite(q[d])||!std::isfinite(k[t*q.size()+d])) throw std::invalid_argument("non-finite input");
        r[t]+=double(q[d])*double(k[t*q.size()+d]);
    }
    return r;
}
ScoreRing::ScoreRing(std::size_t s,std::size_t b):slots_(s),bytes_limit_(b) {
    if(!s||!b) throw std::invalid_argument("empty ring budget");
}
void ScoreRing::push(ScoreBlock b) {
    const auto n=b.scores.size()*sizeof(Score);
    if(n>bytes_limit_) throw std::length_error("score block exceeds ring budget");
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock,[&]{return closed_||!failure_.empty()||(blocks_.size()<slots_&&bytes_<=bytes_limit_-n);});
    if(!failure_.empty()) throw std::runtime_error(failure_);
    if(closed_) throw std::logic_error("push after close");
    bytes_+=n; blocks_.push_back(std::move(b)); changed_.notify_all();
}
bool ScoreRing::pop(ScoreBlock& b) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock,[&]{return closed_||!failure_.empty()||!blocks_.empty();});
    if(!failure_.empty()) throw std::runtime_error(failure_);
    if(blocks_.empty()) return false;
    b=std::move(blocks_.front()); blocks_.pop_front(); bytes_-=b.scores.size()*sizeof(Score);
    changed_.notify_all(); return true;
}
void ScoreRing::close(){std::lock_guard<std::mutex> l(mutex_);closed_=true;changed_.notify_all();}
void ScoreRing::fail(std::string s){std::lock_guard<std::mutex> l(mutex_);failure_=std::move(s);changed_.notify_all();}
void HotPool::apply(std::uint64_t e,const std::set<Token>& backing,const std::set<Token>& evict,
                    std::map<Token,std::vector<std::uint8_t>> promote) {
    if(e!=epoch_) throw std::invalid_argument("stale migration epoch");
    for(auto id:evict) {
        if(!backing.count(id)) throw std::logic_error("cannot demote without durable backing");
    }
    for(const auto& p:promote) {
        if(!backing.count(p.first)) throw std::logic_error("promotion lacks backing");
    }
    std::size_t total=0;
    for(const auto& p:payloads_) {
        if(evict.count(p.first)||promote.count(p.first))continue;
        if(p.second.size()>limit_-total) throw std::length_error("hot pool budget exceeded");
        total+=p.second.size();
    }
    for(const auto& p:promote){
        if(p.second.size()>limit_-total)throw std::length_error("hot pool budget exceeded");
        total+=p.second.size();
    }
    // All validation precedes mutation. Move allocated map nodes; do not clone the existing hot pool.
    for(auto id:evict)payloads_.erase(id);
    while(!promote.empty()){
        auto node=promote.extract(promote.begin());payloads_.erase(node.key());payloads_.insert(std::move(node));
    }
    used_=total; ++epoch_;
}
std::set<Token> HotPool::members() const {std::set<Token> r;for(auto& p:payloads_)r.insert(p.first);return r;}
void Budget::validate() const {
    std::uint64_t n=0;
    for(auto x:{hot,pending,score,staging,metadata}) {
        if(x>limit-n) throw std::length_error("host budget exceeded");
        n+=x;
    }
}
double paper_beta_estimate(Profile p,double r) {
    for(auto x:{p.cpu_score,p.fpga_score,p.cpu_transfer,p.fpga_transfer})
        if(!std::isfinite(x)||x<=0)throw std::invalid_argument("invalid profile");
    if(!std::isfinite(r)||r<0||r>1)throw std::invalid_argument("invalid retention");
    // Algebraic form of Eq. 5 avoids multiplication overflow; all rates must use common work units.
    double b=(1.0/p.fpga_score+r/p.fpga_transfer)/(1.0/p.cpu_score+r/p.cpu_transfer);
    if(!std::isfinite(b)||b<0)throw std::overflow_error("profile overflow");
    return b;
}
double hot_fraction(double b){if(!std::isfinite(b)||b<0)throw std::invalid_argument("invalid beta");return b/(1+b);}
void validate_prefetch(bool app,bool ready,bool rehearsal){
    if(app&&(!ready||!rehearsal))throw std::logic_error("APP requires confirmed causally available rehearsal");
}
}
