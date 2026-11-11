#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
namespace hillserve {
// A bounded direct index replaces one tree insertion per head/token. The
// membership bits still reject duplicates within a head; cross-head overlap
// only deduplicates physical IO, never changes logical attention ordering.
inline std::vector<std::uint64_t> gqa_physical_union(
 const std::vector<std::vector<std::uint64_t>>& ids,std::size_t visible){
 if(ids.size()!=32||!visible||visible>66048)throw std::invalid_argument("GQA union bounds");
 std::vector<std::uint32_t> seen(visible,0);
 for(std::size_t h=0;h<ids.size();++h){const auto bit=std::uint32_t(1)<<h;
  for(auto id:ids[h]){if(id>=visible||(seen[id]&bit))throw std::invalid_argument("GQA duplicate/future selected ID");seen[id]|=bit;}
 }
 std::vector<std::uint64_t> physical;physical.reserve(visible);
 for(std::size_t i=0;i<visible;++i)if(seen[i])physical.push_back(i);
 return physical;
}

// Shared by the actual external model executor and single-layer delivery replay.
// K is [slot,Qhead,D]; V is [Qhead,D,slot]; mask is [Qhead,paddedQuery,slot].
inline void pack_gqa_workspace(const std::vector<std::vector<std::uint64_t>>& ids,
 const std::vector<std::uint64_t>& positions,const std::vector<std::uint64_t>& physical,
 const std::vector<std::uint8_t>& payload,std::size_t count,std::uint16_t* k,std::uint16_t* v,float* mask){
 if(ids.size()!=32||positions.empty()||positions.size()>256||positions.back()>=66048||
    !count||count>66048||payload.size()!=physical.size()*4096)throw std::invalid_argument("GQA packing shape");
 std::vector<std::int32_t> offsets(positions.back()+1,-1);
 for(std::size_t i=0;i<physical.size();++i){const auto id=physical[i];
  if(id>=offsets.size()||offsets[id]!=-1)throw std::invalid_argument("duplicate/future physical position");
  offsets[id]=std::int32_t(i);
 }
 const auto qpad=(positions.size()+63)/64*64;
 std::fill_n(k,count*4096,0);std::fill_n(v,count*4096,0);
 std::fill_n(mask,count*qpad*32,-std::numeric_limits<float>::infinity());
 std::vector<const std::uint8_t*> rows;rows.reserve(count);
 for(std::size_t h=0;h<32;++h){
  if(ids[h].empty()||ids[h].size()>count)throw std::invalid_argument("per-head selection exceeds slots");
  rows.clear();const auto kh=h/4;
  for(std::size_t s=0;s<ids[h].size();++s){const auto id=ids[h][s];
   if(id>=offsets.size()||offsets[id]<0)throw std::invalid_argument("future or missing selected KV");
   const auto* src=payload.data()+std::size_t(offsets[id])*4096;rows.push_back(src);
   std::memcpy(k+s*4096+h*128,src+kh*256,256);
   for(std::size_t q=0;q<positions.size();++q)if(id<=positions[q])mask[(h*qpad+q)*count+s]=0;
  }
  // Tile the transpose: each destination cache line receives consecutive
  // token values. F16 bits, slot ordering, head expansion and padding match
  // the previous scalar packer exactly; no model/graph change is required.
  for(std::size_t first=0;first<rows.size();first+=32)
   for(std::size_t channel=0;channel<128;channel+=32)
    for(std::size_t d=channel;d<channel+32;++d)
     for(std::size_t s=first;s<std::min(rows.size(),first+32);++s)
      std::memcpy(v+(h*128+d)*count+s,rows[s]+2048+(kh*128+d)*2,2);
 }

}
}
