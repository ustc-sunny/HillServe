#include "storage/segmented_store.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
using namespace hillserve;
void check(bool ok){if(!ok)throw std::runtime_error("segmented contract check failed");}
template<class F>void rejects(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw);}
int main(){auto dir=std::filesystem::temp_directory_path()/("hs-segmented-"+std::to_string(getpid()));
 try{check(std::filesystem::create_directory(dir));
  SegmentedKVStore store(dir/"kv",9,28,16384,2048,1ULL<<30);
  check(store.index_bytes()==28*8);check(store.extent_bytes()==4096+28ULL*16384*2080);
  std::vector<std::uint8_t> batch(3*2048);for(std::size_t i=0;i<batch.size();++i)batch[i]=(i/2048)*17+i%251;
  store.append(0,0,batch);store.append(27,0,batch);check(store.completed(27)==3);
  auto data=store.fetch(27,{2,0,2},6144);check(std::equal(data.begin(),data.begin()+2048,batch.begin()+4096));
  check(std::equal(data.begin()+2048,data.begin()+4096,batch.begin()));
  rejects([&]{store.append(0,1,batch);});rejects([&]{store.fetch(0,{3},2048);});rejects([&]{store.fetch(0,{0,1},2048);});
  std::vector<std::uint8_t> tail(2048,43);store.append(0,3,tail);check(store.fetch(0,{3},2048)==tail);
  store.drain();check(store.metrics.drain_calls==1);
  std::fstream corrupt(dir/"kv",std::ios::in|std::ios::out|std::ios::binary);corrupt.seekp(4096+32);corrupt.put('\xff');corrupt.flush();
  rejects([&]{store.fetch(0,{0},2048);});
  rejects([&]{SegmentedKVStore fail(dir/"too-big",9,28,16384,2048,1<<20);});check(!std::filesystem::exists(dir/"too-big"));
  rejects([&]{SegmentedKVStore fail(dir/"kv",9,1,1,8,4096+40);});
#ifdef O_DIRECT
  // Direct appends share unaligned pages across records and layer boundaries.
  // RMW must preserve older records in the same page and publish only complete batches.
  SegmentedKVStore direct(dir/"direct",11,2,41,4096,1ULL<<20);
  std::ofstream(dir/"wrong").put('x');rejects([&]{direct.enable_direct_writes(dir/"wrong");});
  direct.enable_direct_writes(dir/"direct");direct.enable_direct_reads(dir/"direct");check(direct.direct_writes_enabled());
  rejects([&]{direct.enable_direct_writes(dir/"direct");});
  std::vector<std::uint8_t> a(41*4096),b(23*4096);
  for(std::size_t i=0;i<a.size();++i)a[i]=std::uint8_t(i/4096*13+i%251);
  for(std::size_t i=0;i<b.size();++i)b[i]=std::uint8_t(97+i/4096*7+i%241);
  direct.append(0,0,{a.begin(),a.begin()+17*4096});
  direct.append(1,0,b); // neighbor layer before layer0 tail exercises shared page preservation
  direct.append(0,17,{a.begin()+17*4096,a.end()});
  check(direct.completed(0)==41&&direct.completed(1)==23&&direct.metrics.drain_calls==0);
  auto all_a=direct.fetch(0,{0,16,17,40},4*4096);auto all_b=direct.fetch(1,{0,22,0},3*4096);
  std::size_t j=0;for(auto id:{0,16,17,40}){check(std::equal(all_a.begin()+j*4096,all_a.begin()+(j+1)*4096,a.begin()+id*4096));++j;}
  j=0;for(auto id:{0,22,0}){check(std::equal(all_b.begin()+j*4096,all_b.begin()+(j+1)*4096,b.begin()+id*4096));++j;}
  check(direct.metrics.direct_write_bytes>=a.size()+b.size());check(direct.metrics.direct_write_bytes%4096==0);
  check(direct.metrics.direct_rmw_read_bytes==direct.metrics.direct_write_bytes);
  check(direct.metrics.max_append_staging_bytes==65536);check(direct.metrics.syscall_write_bytes==4096+direct.metrics.direct_write_bytes);
  rejects([&]{direct.append(0,41,std::vector<std::uint8_t>(4096));});rejects([&]{direct.fetch(1,{23},4096);});
  // Sorted contiguous retrieval merges overlapping/adjacent pages instead of
  // issuing one read per token. Unordered >tile-capacity callers preserve order.
  std::vector<std::uint64_t> sorted_ids(41);for(std::size_t i=0;i<sorted_ids.size();++i)sorted_ids[i]=i;
  auto calls_before=direct.metrics.direct_fetch_read_calls;check(direct.fetch(0,sorted_ids,a.size())==a);
  auto merged_calls=direct.metrics.direct_fetch_read_calls-calls_before;check(merged_calls<=5&&merged_calls<sorted_ids.size());
  std::vector<std::uint64_t> mixed_ids(1005);for(std::size_t i=0;i<mixed_ids.size();++i)mixed_ids[i]=(i*17+9)%41;
  auto mixed=direct.fetch(0,mixed_ids,mixed_ids.size()*4096);
  for(std::size_t i=0;i<mixed_ids.size();++i)check(std::equal(mixed.begin()+i*4096,mixed.begin()+(i+1)*4096,a.begin()+mixed_ids[i]*4096));
  check(direct.metrics.max_fetch_staging_bytes<=65536);
  // Largest supported payload spans multiple bounded windows, including duplicates.
  SegmentedKVStore large(dir/"large",13,1,3,65536,1ULL<<20);large.enable_direct_writes(dir/"large");large.enable_direct_reads(dir/"large");
  std::vector<std::uint8_t> big(3*65536);for(std::size_t i=0;i<big.size();++i)big[i]=std::uint8_t(i/65536*31+i%239);large.append(0,0,big);
  auto large_values=large.fetch(0,{2,0,2,1},4*65536);j=0;for(auto id:{2,0,2,1}){check(std::equal(large_values.begin()+j*65536,large_values.begin()+(j+1)*65536,big.begin()+id*65536));++j;}
  check(large.metrics.max_fetch_staging_bytes<=65536);
  direct.drain();check(direct.metrics.drain_calls==1);
  int corrupt_fd=open((dir/"direct").c_str(),O_WRONLY|O_CLOEXEC);check(corrupt_fd>=0);const unsigned char damage=255;
  check(pwrite(corrupt_fd,&damage,1,4096+20*4128+32)==1);check(fdatasync(corrupt_fd)==0);close(corrupt_fd);
  rejects([&]{direct.fetch(0,{0,20,40},3*4096);});check(direct.fetch(1,{0},4096)==std::vector<std::uint8_t>(b.begin(),b.begin()+4096));
  std::cout<<"{\"coalesced_fetch_status\":\"passed\",\"sorted_records\":41,\"sorted_read_calls\":"<<merged_calls<<",\"unordered_duplicate_records\":1005,\"max_fetch_staging_bytes\":"<<direct.metrics.max_fetch_staging_bytes<<",\"large_record_staging_bytes\":"<<large.metrics.max_fetch_staging_bytes<<",\"corruption_rejected\":true,\"evidence\":\"CPU_tmp_regular_file_correctness_only\"}\n";

  SegmentedKVStore late(dir/"late",12,1,2,4096,32768);late.append(0,0,std::vector<std::uint8_t>(4096,3));
  rejects([&]{late.enable_direct_writes(dir/"late");});
  std::cout<<"{\"direct_append_status\":\"passed\",\"direct_write_bytes\":"<<direct.metrics.direct_write_bytes<<",\"RMW_read_bytes\":"<<direct.metrics.direct_rmw_read_bytes<<",\"max_staging_bytes\":"<<direct.metrics.max_append_staging_bytes<<",\"drain_calls\":"<<direct.metrics.drain_calls<<",\"evidence\":\"CPU_tmp_regular_file_correctness_only\"}\n";
#endif
  std::cout<<"{\"status\":\"passed\",\"geometry_positions\":16384,\"layers\":28,\"index_bytes\":"<<store.index_bytes()<<",\"logical_append_bytes\":"<<store.metrics.logical_append_bytes<<",\"extent_bytes\":"<<store.extent_bytes()<<",\"evidence\":\"software_contract_not_long_context_performance\"}\n";
  std::filesystem::remove_all(dir);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';std::filesystem::remove_all(dir);return 1;}}
