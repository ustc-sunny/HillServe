#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "policy/half_order.hpp"
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>
using namespace hillserve;
using namespace hillserve::policy;
void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<class F>void rejects(F fn,const char* why){bool threw=false;try{fn();}catch(const std::exception&){threw=true;}check(threw,why);}
void u32(std::ofstream& f,unsigned x){for(unsigned i=0;i<4;++i)f.put(char(x>>(8*i)));}
int main(int argc,char** argv){
 try{
 if(argc==3&&std::string(argv[1])=="--selector-probe"){
  std::ifstream f(argv[2],std::ios::binary);uint32_t h=0,n=0;float alpha=0,ratio=0;f.read(reinterpret_cast<char*>(&h),4);f.read(reinterpret_cast<char*>(&n),4);f.read(reinterpret_cast<char*>(&alpha),4);f.read(reinterpret_cast<char*>(&ratio),4);check(h>0&&h<=128&&n>0&&n<=131072,"probe shape");std::vector<uint16_t> scores(size_t(h)*n);f.read(reinterpret_cast<char*>(scores.data()),scores.size()*2);check(bool(f),"probe file");auto selected=select_gqa_scores(scores,h,n,n,alpha,ratio);std::cout<<selected.k<<"\n";for(const auto& ids:selected.ids){for(auto id:ids)std::cout<<id<<" ";std::cout<<"\n";}return 0;
 }
 // Distinguish head0 broadcast from perhead max; distinguish inclusive from >.
 std::vector<uint16_t> score;for(float x:{10.f,9.f,8.f,7.f,1.f,1.f,1.f,1.f})score.push_back(half_bits(x));
 auto x=select_gqa_scores(score,2,4,4,2.f,1.f);check(x.counts==std::vector<uint32_t>({3,0})&&x.k==1,"head0/inclusive/count floor semantics");check(x.ids[0][0]==0&&x.ids[1][0]==0,"deterministic original-ID ties");
 // Partial final capacity is not a real candidate even if its score is largest.
 std::vector<uint16_t> padded(2*8,half_bits(1000));for(size_t h=0;h<2;++h)for(size_t t=0;t<5;++t)padded[h*8+t]=half_bits(float(t));
 auto p=select_gqa_scores(padded,2,8,5,10,0.4f);check(p.k==2&&p.ids[0]==std::vector<uint64_t>({4,3}),"padding/current candidate selection");
 // F16 count rounding is observable at2049. No wide-integer semantic swap.
 std::vector<uint16_t> many(2049,half_bits(0));auto rounded=select_gqa_scores(many,1,2049,2049,0,1);check(rounded.counts[0]==2049&&rounded.k==2048,"F16 count2049 rounds2048");
 // Long addressing is allowed when representable threshold count stays finite.
 std::vector<uint16_t> long_scores(131072,half_bits(-10));long_scores.back()=half_bits(1);auto last=select_gqa_scores(long_scores,1,long_scores.size(),long_scores.size(),0,1);check(last.k==1&&last.ids[0][0]==131071,"global long-token ID not truncated");
 std::vector<uint16_t> overflow(65520,half_bits(0));rejects([&]{select_gqa_scores(overflow,1,overflow.size(),overflow.size(),0,1);},"F16 count overflow must fail");
 // The user-approved64K extension changes counts/mean ONLY, never scores,
 // head0 threshold, inclusive comparison, ratio cap or deterministic ordering.
 auto wide=select_gqa_scores(overflow,1,overflow.size(),overflow.size(),0,.2f,true);
 check(wide.counts[0]==65520&&wide.k==13104&&wide.ids[0].front()==0&&wide.ids[0].back()==13103,"explicit FP32 count extension keeps ordered cap");
 auto exact=select_gqa_scores(many,1,2049,2049,0,1,true);
 check(exact.k==2049,"explicit FP32 count has no hidden F16 rounding");
 // Exhaust every finite F16 encoding, including subnormals and signed zero.
 std::vector<std::uint16_t> finite;
 for(unsigned bits=0;bits<65536;++bits)if((bits&0x7c00)!=0x7c00)finite.push_back(std::uint16_t(bits));
 std::sort(finite.begin(),finite.end(),[](auto a,auto b){return half_value(a)<half_value(b);});
 for(std::size_t i=1;i<finite.size();++i){const auto a=finite[i-1],b=finite[i];
  check((finite_half_order_key(a)<finite_half_order_key(b))==(half_value(a)<half_value(b)),"finite half order must match FP32 comparator");
  check((finite_half_order_key(a)==finite_half_order_key(b))==(half_value(a)==half_value(b)),"finite half ties include signed zero");
 }
 std::mt19937 rng(1741);
 for(int trial=0;trial<24;++trial){const std::size_t heads=32,n=257,stride=288;
  std::vector<std::uint16_t> input(heads*stride,half_bits(1000));
  for(std::size_t head=0;head<heads;++head)for(std::size_t t=0;t<n;++t)
   input[head*stride+t]=(t%17==0)?std::uint16_t(t%2?0:0x8000):half_bits(float(int(rng()%97)-48)/2);
  const auto selected=select_gqa_scores(input,heads,stride,n,10,.2f);
  for(std::size_t head=0;head<heads;++head){std::vector<std::uint64_t> expected(n);std::iota(expected.begin(),expected.end(),0);
   std::sort(expected.begin(),expected.end(),[&](auto a,auto b){const auto av=half_value(input[head*stride+a]),bv=half_value(input[head*stride+b]);return av!=bv?av>bv:a<b;});
   expected.resize(selected.k);check(expected==selected.ids[head],"optimized finite ordering changes original ordered selected IDs");
  }
 }
 // File/layout and GQA mapping test with explicitly diagnostic identity basis.
 auto dir=std::filesystem::temp_directory_path()/("hillserve-gqa-policy-test-"+std::to_string(getpid()));check(std::filesystem::create_directory(dir),"new test directory");auto artifact=dir/"identity-test.bin";
 {std::ofstream f(artifact,std::ios::binary);f.write("HSGLPF01",8);for(unsigned n:{32,32,8,128})u32(f,n);std::vector<uint16_t> basis(32*32*128*128,0);for(size_t lh=0;lh<32*32;++lh)for(size_t d=0;d<128;++d)basis[(lh*128+d)*128+d]=half_bits(1);f.write(reinterpret_cast<char*>(basis.data()),basis.size()*2);std::vector<char> mask(32*32*128,0);for(size_t c=0;c<mask.size();++c)mask[c]=(c%128)%7==0;f.write(mask.data(),mask.size());}
 check(setenv("HILLSERVE_SCRATCH_ROOT",dir.c_str(),1)==0,"configure isolated test scratch root");
 auto keyfile=dir/"keys.bin";GQAPolicy policy(artifact.string(),32,100,0.25f,keyfile.string());
 std::vector<uint64_t> pos(8);std::iota(pos.begin(),pos.end(),0);std::vector<uint16_t> keys(8*8*128,0);
 for(size_t t=0;t<8;++t)for(size_t kh=0;kh<8;++kh)keys[(t*8+kh)*128]=half_bits(float(kh%2?t:7-t));
 policy.append(2,pos,keys);std::vector<float> query(32*128,0);for(size_t h=0;h<32;++h)query[h*128]=1;
 auto ids=policy.select(2,7,query);for(size_t h=0;h<32;++h)check(ids[h]==(h/4%2?std::vector<uint64_t>{7,6}:std::vector<uint64_t>{0,1}),"physical KV qhead/4 mapping");
 rejects([&]{policy.append(2,pos,keys);},"duplicate append must fail");rejects([&]{policy.select(2,8,query);},"unwritten current token cannot be selected");

 GQAPolicy cached(artifact.string(),32,100,0.25f,(dir/"cache-keys.bin").string());cached.configure_cpu_cache(1024*1024,4);cached.verify_optimized_scores(true);cached.append(2,pos,keys);check(cached.select(2,7,query)==ids,"compressed cache/parallel scoring must preserve exact scores and IDs");check(cached.stats().cache_hits==1,"fixed compressed cache must hit");check(cached.stats().verification_score_count==32*8&&cached.stats().verification_bit_errors==0,"same-input exact score validation accounting");
 GQAPolicy cold(artifact.string(),32,100,0.25f,(dir/"cold-keys.bin").string());cold.configure_cpu_cache(0,4);cold.verify_optimized_scores(true);cold.append(2,pos,keys);check(cold.select(2,7,query)==ids,"cold masked parallel score exactness");check(cold.stats().cache_misses==1,"zero hot budget requires actual file read");
#ifdef __linux__
 GQAPolicy dio(artifact.string(),32,100,0.25f,(dir/"direct-keys.bin").string());dio.enable_direct_writes();dio.configure_backend("host",{},true);dio.configure_cpu_cache(0,4);dio.verify_optimized_scores(true);dio.append(2,pos,keys);check(dio.select(2,7,query)==ids,"O_DIRECT RMW must preserve existing key bytes");
 std::vector<uint16_t> current(8*128,0);for(size_t kh=0;kh<8;++kh)current[kh*128]=half_bits(float(kh%2?8:-1));dio.append(2,{8},current);auto extended=dio.select(2,8,query);check(extended[4]==std::vector<uint64_t>({8,7})&&extended[0]==std::vector<uint64_t>({0,1}),"partial-page direct append preserves old rows and new token");dio.drain();check(dio.stats().append_rmw_read_bytes>0&&dio.stats().append_syscall_bytes>dio.stats().append_bytes,"direct RMW physical traffic reported");
#endif
 policy.drain();policy.write_stats((dir/"stats.json").string());check(policy.stats().score_calls==1&&policy.stats().selected_pairs==64,"observed policy stats");
 std::filesystem::remove_all(dir);
 std::cout<<"GQA policy: reference semantics,131072-ID indexing,F16 overflow,head mapping,file visibility passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
