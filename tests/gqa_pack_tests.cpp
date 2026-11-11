#include "runtime/gqa_pack.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
int main(){try{
 std::vector<std::vector<std::uint64_t>> ids(32,{8,12});ids[4]={12,8};std::vector<std::uint64_t> physical={8,12};std::vector<std::uint16_t> raw(4096);
 for(std::size_t t=0;t<2;++t)for(std::size_t h=0;h<8;++h)for(std::size_t d=0;d<128;++d){raw[t*2048+h*128+d]=1000*t+100*h+d;raw[t*2048+1024+h*128+d]=5000+1000*t+100*h+d;}
 std::vector<std::uint8_t> payload(raw.size()*2);std::memcpy(payload.data(),raw.data(),payload.size());std::vector<std::uint16_t> k(32*4096),v(k.size());std::vector<float> mask(32*64*32);
 hillserve::pack_gqa_workspace(ids,{10,12},physical,payload,32,k.data(),v.data(),mask.data());
 auto check=[](bool x){if(!x)throw std::runtime_error("independent GQA ABI value/causal mask check");};
 check(k[0]==0&&k[4096]==1000&&k[4*128]==1100&&k[4096+4*128]==100);check(v[(4*128)*32]==6100&&v[(4*128)*32+1]==5100);check(k[2*4096+127]==0&&v[127*32+2]==0);
 check(mask[0]==0&&std::isinf(mask[1])&&mask[32+1]==0);check(std::isinf(mask[(4*64)*32])&&mask[(4*64)*32+1]==0);check(std::isinf(mask[2*32]));
 bool rejected=false;ids[0]={13};try{hillserve::pack_gqa_workspace(ids,{12},physical,payload,32,k.data(),v.data(),mask.data());}catch(const std::exception&){rejected=true;}check(rejected);
 std::mt19937 rng(781);
 for(int trial=0;trial<48;++trial){
  const std::size_t n=65+trial*4,slots=(n+31)/32*32;
  std::vector<std::uint64_t> rows(n);std::iota(rows.begin(),rows.end(),0);
  std::shuffle(rows.begin(),rows.end(),rng);
  std::vector<std::uint8_t> bytes(n*4096);for(auto& x:bytes)x=std::uint8_t(rng());
  std::vector<std::vector<std::uint64_t>> selected(32);
  for(auto& head:selected){head=rows;if(trial%2)head.resize(1+rng()%n);std::shuffle(head.begin(),head.end(),rng);}
  std::vector<std::uint64_t> queries{n/2,n-1};
  std::vector<std::uint16_t> got_k(slots*4096),got_v(got_k.size()),ref_k(got_k.size(),0),ref_v(got_k.size(),0);
  std::vector<float> got_mask(slots*64*32),ref_mask(got_mask.size(),-std::numeric_limits<float>::infinity());
  // Independent scalar layout oracle, including arbitrary physical order.
  for(std::size_t h=0;h<32;++h)for(std::size_t slot=0;slot<selected[h].size();++slot){
   const auto id=selected[h][slot];const auto row=std::size_t(std::find(rows.begin(),rows.end(),id)-rows.begin());
   const auto* src=bytes.data()+row*4096;
   for(std::size_t d=0;d<128;++d){std::memcpy(&ref_k[slot*4096+h*128+d],src+(h/4*128+d)*2,2);std::memcpy(&ref_v[(h*128+d)*slots+slot],src+2048+(h/4*128+d)*2,2);}
   for(std::size_t q=0;q<queries.size();++q)if(id<=queries[q])ref_mask[(h*64+q)*slots+slot]=0;
  }
  hillserve::pack_gqa_workspace(selected,queries,rows,bytes,slots,got_k.data(),got_v.data(),got_mask.data());
  check(got_k==ref_k&&got_v==ref_v&&got_mask==ref_mask);
  const auto physical_union=hillserve::gqa_physical_union(selected,n);
  std::vector<std::uint64_t> expected;for(const auto& head:selected)expected.insert(expected.end(),head.begin(),head.end());
  std::sort(expected.begin(),expected.end());expected.erase(std::unique(expected.begin(),expected.end()),expected.end());check(physical_union==expected);
  selected[31].push_back(selected[31].front());bool duplicate=false;try{hillserve::gqa_physical_union(selected,n);}catch(const std::exception&){duplicate=true;}check(duplicate);
 }
 std::cout<<"GQA token-major K, transposed V, physical-head map, zero padding and per-query causal mask passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
