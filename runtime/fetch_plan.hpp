#pragma once
#include <cstdint>
#include <vector>
#include <set>
#include <stdexcept>
namespace hillserve {
struct FetchPlan {
    std::uint64_t request,layer,step,epoch;
    std::vector<std::uint64_t> ids;
    void validate(std::uint64_t last,std::size_t slots,const std::vector<std::uint64_t>& queries)const{
        if(ids.empty()||ids.size()>slots)throw std::length_error("selection exceeds GPU slots or is empty");
        std::set<std::uint64_t> seen;
        for(auto id:ids){if(id>last||!seen.insert(id).second)throw std::invalid_argument("future/invalid/duplicate selected ID");}
        for(auto q:queries){bool any=false;for(auto id:ids)if(id<=q)any=true;if(!any)throw std::invalid_argument("all-masked query row");}
    }
};
inline void require_shared_heads(const std::vector<std::vector<std::uint64_t>>& heads){
    for(const auto& ids:heads)if(ids!=heads.front())throw std::invalid_argument("per-query-head differing SelectedIds unsupported; no union fallback");
}
}
