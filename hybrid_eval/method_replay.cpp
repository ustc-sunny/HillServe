#include "policy/host_policy.hpp"
#include "storage/file_store.hpp"
#ifdef HILLSERVE_METHOD_GPU
#include "hybrid_eval/method_attention.hpp"
#endif
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sys/resource.h>
using namespace hillserve;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
template<class T>static std::vector<T> read(std::ifstream& f,std::size_t n){std::vector<T> v(n);if(!f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T)))throw std::runtime_error("short trace");return v;}
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("usage: hillserve-method-replay TRACE NEW_DIR alpha cap chunk_tokens repeats strict|inclusive");
 std::ifstream in(argv[1],std::ios::binary);auto hdr=read<std::uint32_t>(in,10);
 if(hdr[0]!=0x48535033||(hdr[1]!=1&&hdr[1]!=2))throw std::runtime_error("trace version");
 const std::size_t h=hdr[2],n=hdr[3],d=hdr[4],full=hdr[5],rk=hdr[6];
 if(!h||h>128||!n||n>8192||!d||d>256||!full||full>256||rk>n)throw std::runtime_error("trace dimensions");
 float alpha=std::stof(argv[3]);std::size_t cap=std::stoul(argv[4]),chunk=std::stoul(argv[5]),repeats=std::stoul(argv[6]);std::string boundary=argv[7];
 if(boundary!="strict"&&boundary!="inclusive")throw std::runtime_error("threshold mode");
 if(!chunk||chunk>n||!repeats||repeats>10||cap>n)throw std::runtime_error("pilot bounds");
 const std::size_t host_budget=8*1024*1024, file_budget=64*1024*1024;
 if(h*n*4*full+h*n*d*2>file_budget)throw std::runtime_error("64MiB total owned file limit");
 auto dir=std::filesystem::path(argv[2]);if(!std::filesystem::create_directory(dir))throw std::runtime_error("new dir required");
 auto q=read<std::uint16_t>(in,h*d),keys=read<std::uint16_t>(in,h*n*d);
 {std::ofstream out(dir/"partial-keys.bin",std::ios::binary);out.write(reinterpret_cast<const char*>(keys.data()),keys.size()*2);if(!out)throw std::runtime_error("key write");}
 keys.clear();keys.shrink_to_fit();
 IndexedKVStore store(dir/"kv.bin",file_budget,h*n,host_budget,KVDurability::volatile_query_kv);
 std::vector<std::uint64_t> expected(h*n);auto prep=Clock::now();store.begin_append_batch();
 for(std::size_t i=0;i<h*n;++i){auto payload=read<std::uint8_t>(in,4*full);expected[i]=checksum(payload.data(),payload.size());store.append_kv(1,hdr[7],i,payload);}
 store.end_append_batch();store.drain();double prep_ms=ms(prep);
 auto oracle_scores=read<std::uint16_t>(in,h*n);auto oracle_ids=read<std::uint64_t>(in,h*rk);std::vector<float> actual_q,attention_reference;std::vector<std::uint16_t> current_kv;
 if(hdr[1]==2){actual_q=read<float>(in,h*full);current_kv=read<std::uint16_t>(in,h*full*2);attention_reference=read<float>(in,h*full);}
 bool use_gpu=std::getenv("HILLSERVE_METHOD_GPU")&&std::string(std::getenv("HILLSERVE_METHOD_GPU"))=="1";
 if(use_gpu&&hdr[1]!=2)throw std::runtime_error("GPU requires actual full query/current KV trace v2");
#ifdef HILLSERVE_METHOD_GPU
 std::unique_ptr<MethodAttention> attention;if(use_gpu)attention=std::make_unique<MethodAttention>(h,full,cap);
#else
 if(use_gpu)throw std::runtime_error("GPU method target was not built");
#endif
 char tail;if(in.read(&tail,1))throw std::runtime_error("trailing trace");
 in.close();std::ifstream keyfile(dir/"partial-keys.bin",std::ios::binary);
 const auto working_bound=store.metadata().peak+expected.capacity()*8+oracle_scores.capacity()*2+oracle_ids.capacity()*8+q.capacity()*2
     +h*n*6+chunk*(d*2+2)+d*6+h*std::min(cap,n)*(16+4*full+sizeof(KVPayload)+128)+n*8+h*8+512*1024+(use_gpu?3*1024*1024:0);
 if(working_bound>host_budget)throw std::runtime_error("host working set reservation exceeds 8MiB");
 std::ofstream out(dir/"runs.jsonl");out<<std::setprecision(12);bool all_match=true;
 for(std::size_t rep=0;rep<=repeats;++rep){
  auto begin=Clock::now();double read_ms=0,score_ms=0;std::uint64_t read_before=store.selected_read_bytes;
  std::vector<std::uint16_t> scores(h*n); // global score barrier; no early selection
  for(std::size_t head=0;head<h;++head)for(std::size_t t=0;t<n;t+=chunk){auto count=std::min(chunk,n-t);auto start=Clock::now();
   keyfile.clear();keyfile.seekg((head*n+t)*d*2);auto block=read<std::uint16_t>(keyfile,count*d);read_ms+=ms(start);
   std::vector<std::uint16_t> oneq(q.begin()+head*d,q.begin()+(head+1)*d);start=Clock::now();auto partial=host_partial_scores(oneq,block,1,count,d);score_ms+=ms(start);
   std::copy(partial.begin(),partial.end(),scores.begin()+head*n+t);
  }
  auto start=Clock::now();auto chosen=select_partial_scores(scores,h,n,alpha,cap,boundary=="inclusive");double select_ms=ms(start);
  std::vector<std::uint64_t> physical;physical.reserve(chosen.ids.size());for(std::size_t head=0;head<h;++head)for(std::size_t j=0;j<chosen.k;++j)physical.push_back(head*n+chosen.ids[head*chosen.k+j]);
  start=Clock::now();auto payload=store.fetch_selected(1,hdr[7],physical,host_budget-working_bound+physical.size()*4*full);double gather_ms=ms(start);
  double gpu_total=0,gpu_h2d=0,gpu_compute=0,gpu_d2h=0,gpu_pack=0,attention_max=0,attention_rel_l2=0;std::size_t gpu_bytes=0,gpu_transfer=0;
#ifdef HILLSERVE_METHOD_GPU
  AttentionOutput gpu_out;if(use_gpu){gpu_out=attention->run(actual_q,current_kv,payload,chosen.k);gpu_total=gpu_out.total_ms;gpu_h2d=gpu_out.h2d_ms;gpu_compute=gpu_out.compute_wall_ms;gpu_d2h=gpu_out.d2h_ms;gpu_pack=gpu_out.pack_ms;gpu_bytes=gpu_out.gpu_bytes;gpu_transfer=gpu_out.h2d_bytes;}
#endif
  auto chain_ms=ms(begin); // verification and reporting explicitly outside measured chain
#ifdef HILLSERVE_METHOD_GPU
  if(use_gpu){double err2=0,ref2=0;for(std::size_t i=0;i<gpu_out.values.size();++i){double delta=double(gpu_out.values[i])-attention_reference[i];if(!std::isfinite(delta))throw std::runtime_error("nonfinite attention output");attention_max=std::max(attention_max,std::abs(delta));err2+=delta*delta;ref2+=double(attention_reference[i])*attention_reference[i];}attention_rel_l2=std::sqrt(err2/std::max(ref2,1e-30));}
#endif
  std::size_t byte_errors=0;for(auto&p:payload)byte_errors+=checksum(p.bytes.data(),p.bytes.size())!=expected[p.token];
  float max_error=0;unsigned max_ulp=0;std::size_t bit_errors=0,set_errors=0;for(std::size_t i=0;i<scores.size();++i){bit_errors+=scores[i]!=oracle_scores[i];auto ordered=[](std::uint16_t x){return x&0x8000?0x8000-int(x&0x7fff):0x8000+int(x);};max_ulp=std::max(max_ulp,unsigned(std::abs(ordered(scores[i])-ordered(oracle_scores[i]))));max_error=std::max(max_error,std::abs(half_value(scores[i])-half_value(oracle_scores[i])));}
  if(chosen.k!=rk)set_errors=h;else for(std::size_t head=0;head<h;++head){std::set<std::uint64_t> a(chosen.ids.begin()+head*rk,chosen.ids.begin()+(head+1)*rk),b(oracle_ids.begin()+head*rk,oracle_ids.begin()+(head+1)*rk);set_errors+=a!=b;}
  bool matched=!set_errors&&!byte_errors;all_match&=matched;
  struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
  out<<"{\"repeat\":"<<rep<<",\"warmup\":"<<(rep==0?"true":"false")<<",\"status\":\""<<(matched?"passed":"failed")<<"\",\"evidence\":\"real_model_method_replay_host\",\"target_device_verified\":false,\"actual_fpga\":false,\"actual_model_feedback\":false,\"N\":"<<n<<",\"heads\":"<<h<<",\"channels\":"<<d<<",\"layer\":"<<hdr[7]<<",\"step\":"<<hdr[8]<<",\"alpha\":"<<alpha<<",\"cap\":"<<cap<<",\"chunk_tokens\":"<<chunk<<",\"selected_k\":"<<chosen.k<<",\"selected_pairs\":"<<chosen.ids.size()<<",\"threshold_mode\":\""<<boundary<<"\",\"reference_set_mismatching_heads\":"<<set_errors<<",\"boundary_tied_heads\":"<<chosen.boundary_tied_heads<<",\"score_bit_differences\":"<<bit_errors<<",\"score_max_abs_error\":"<<max_error<<",\"score_max_ulp_error\":"<<max_ulp<<",\"kv_byte_errors\":"<<byte_errors<<",\"key_file_read_ms\":"<<read_ms<<",\"host_score_ms\":"<<score_ms<<",\"select_ms\":"<<select_ms<<",\"selected_gather_ms\":"<<gather_ms<<",\"retrieval_chain_wall_ms\":"<<chain_ms<<",\"actual_gpu_attention\":"<<(use_gpu?"true":"false")<<",\"gpu_attention_wall_ms\":"<<gpu_total<<",\"gpu_pack_ms\":"<<gpu_pack<<",\"gpu_h2d_ms\":"<<gpu_h2d<<",\"gpu_compute_host_wall_ms\":"<<gpu_compute<<",\"gpu_d2h_ms\":"<<gpu_d2h<<",\"gpu_h2d_bytes\":"<<gpu_transfer<<",\"gpu_tensor_allocation_bytes\":"<<gpu_bytes<<",\"attention_max_abs_error_vs_fp32_reference\":"<<attention_max<<",\"attention_relative_l2_error_vs_fp32_reference\":"<<attention_rel_l2<<",\"attention_numerical_gate_asserted\":false,\"status_scope\":\"selected_set_and_storage\",\"setup_kv_write_and_drain_ms\":"<<prep_ms<<",\"setup_kv_write_bytes\":"<<store.append_write_bytes<<",\"score_read_bytes\":"<<h*n*d*2<<",\"selected_read_bytes\":"<<store.selected_read_bytes-read_before<<",\"host_budget_bytes\":"<<host_budget<<",\"host_working_reservation_bytes\":"<<working_bound<<",\"index_allocator_peak_bytes\":"<<store.metadata().peak<<",\"file_cache_reservation_bytes\":"<<file_budget<<",\"process_peak_rss_kib\":"<<ru.ru_maxrss<<",\"physical_nand_bytes\":null,\"formal_speedup_eligible\":false}\n";out.flush();
  if(rep==1){std::ofstream ids(dir/"selected-ids.bin",std::ios::binary);ids.write(reinterpret_cast<const char*>(chosen.ids.data()),chosen.ids.size()*8);std::ofstream sf(dir/"host-scores.bin",std::ios::binary);sf.write(reinterpret_cast<const char*>(scores.data()),scores.size()*2);}
 }
 return all_match?0:3;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
