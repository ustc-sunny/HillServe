// Independent implementation of the inspected performance-reference operations.
#include "policy/host_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
namespace hillserve {
float half_value(std::uint16_t h) {
    std::uint32_t e=(h>>10)&31, m=h&1023, bits=std::uint32_t(h&0x8000)<<16;
    if(e==0){if(!m){float zero;std::memcpy(&zero,&bits,4);return zero;}float v=std::ldexp(float(m),-24);return h&32768?-v:v;}
    bits|=e==31?0x7f800000|(m<<13):((e+112)<<23)|(m<<13);
    float result;std::memcpy(&result,&bits,4);return result;
}
std::uint16_t half_bits(float f) {
    std::uint32_t v;std::memcpy(&v,&f,4);std::uint32_t sign=(v>>16)&0x8000, mant=v&0x7fffff;
    int exp=int((v>>23)&255)-127+15;
    if(((v>>23)&255)==255)return std::uint16_t(sign|0x7c00|(mant?0x200:0));
    if(exp>=31)return std::uint16_t(sign|0x7c00);
    if(exp<=0){
        if(exp < -10)return std::uint16_t(sign);
        mant|=0x800000;unsigned shift=unsigned(14-exp), base=mant>>shift;
        unsigned rest=mant&((1u<<shift)-1), mid=1u<<(shift-1);
        return std::uint16_t(sign| (base+(rest>mid || (rest==mid && (base&1)))));
    }
    unsigned base=(unsigned(exp)<<10)|(mant>>13),rest=mant&8191;
    return std::uint16_t(sign|(base+(rest>4096||(rest==4096&&(base&1)))));
}
std::vector<std::uint16_t> host_partial_scores(const std::vector<std::uint16_t>& q,
 const std::vector<std::uint16_t>& keys,std::size_t h,std::size_t n,std::size_t d){
    if(!h||h>128||!n||n>65504||!d||d>256||q.size()!=h*d||keys.size()!=h*n*d)throw std::invalid_argument("partial score shape");
    std::vector<float> qf(q.size());for(std::size_t i=0;i<q.size();++i)qf[i]=half_value(q[i]);
    std::vector<std::uint16_t> scores(h*n);
    for(std::size_t head=0;head<h;++head)for(std::size_t t=0;t<n;++t){
        float sum=0;for(std::size_t c=0;c<d;++c)sum+=qf[head*d+c]*half_value(keys[(head*n+t)*d+c]);
        if(!std::isfinite(sum))throw std::runtime_error("nonfinite score");
        auto result=half_bits(sum);if(!std::isfinite(half_value(result)))throw std::runtime_error("half score overflow");scores[head*n+t]=result;
    }return scores;
}
PolicySelection select_partial_scores(const std::vector<std::uint16_t>& scores,
 std::size_t h,std::size_t n,float alpha,std::size_t cap,bool inclusive){
    if(!h||h>128||!n||n>65504||scores.size()!=h*n||!std::isfinite(alpha)||alpha<0)throw std::invalid_argument("selection contract");
    PolicySelection r;std::vector<float> values(scores.size());
    for(std::size_t i=0;i<scores.size();++i){values[i]=half_value(scores[i]);if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite score");}
    float total=0;
    for(std::size_t head=0;head<h;++head){
        auto begin=values.begin()+head*n;float mx=*std::max_element(begin,begin+n);
        float threshold=half_value(half_bits(mx-alpha));std::uint32_t count=0;
        for(std::size_t t=0;t<n;++t)count+=inclusive?begin[t]>=threshold:begin[t]>threshold;
        r.above_threshold.push_back(count);
        // torch.sum(count) outputs F16; torch.mean accumulates then returns F16.
        total+=half_value(half_bits(float(count)));
    }
    r.k=std::min({std::size_t(half_value(half_bits(total/float(h)))),cap,n});
    for(std::size_t head=0;head<h;++head){
        std::vector<std::uint64_t> order(n);std::iota(order.begin(),order.end(),0);
        // PyTorch topk leaves equal-score ordering unspecified. This implementation
        // has an explicit ID order; the validator reports exact-set failures/ties.
        // Only K plus its boundary neighbor are needed; deterministic ordering
        // and the selected set are identical to the previous full sort.
        auto end=order.begin()+std::min(n,r.k+1);
        std::partial_sort(order.begin(),end,order.end(),[&](auto a,auto b){return values[head*n+a]!=values[head*n+b]?values[head*n+a]>values[head*n+b]:a<b;});
        if(r.k&&r.k<n&&values[head*n+order[r.k-1]]==values[head*n+order[r.k]])++r.boundary_tied_heads;
        r.ids.insert(r.ids.end(),order.begin(),order.begin()+r.k);
    }return r;
}
}
