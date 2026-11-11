// Real same-input host/XRT chain. Direct regular-file reads on the granted SSD.
#include "policy/host_policy.hpp"
#include "smartssd/adapter.hpp"
#include "storage/segmented_store.hpp"
#include "storage/file_store.hpp"
#include "hybrid_eval/method_attention.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <future>
#include <set>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
using namespace hillserve;using json=nlohmann::ordered_json;using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static void need(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
template<class T>std::vector<T> read(std::ifstream& f,std::size_t n){std::vector<T> v(n);need(bool(f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T))),"short real method trace");return v;}
struct DirectFile{
 int fd=-1;std::uint64_t bytes=0;
 explicit DirectFile(const std::filesystem::path& p){fd=open(p.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);need(fd>=0,"score O_DIRECT open failed; no fallback");}
 ~DirectFile(){if(fd>=0)close(fd);}
 std::vector<std::uint16_t> get(std::size_t offset,std::size_t count){
  std::size_t begin=offset/4096*4096,aligned=(offset-begin+count*2+4095)/4096*4096;void* buffer=nullptr;need(posix_memalign(&buffer,4096,aligned)==0,"aligned allocation");
  ssize_t n;do{n=pread(fd,buffer,aligned,begin);}while(n<0&&errno==EINTR);if(n!=ssize_t(aligned)){free(buffer);throw std::runtime_error("score direct read failed; no fallback");}
  std::vector<std::uint16_t> result(count);std::memcpy(result.data(),static_cast<char*>(buffer)+offset-begin,count*2);free(buffer);bytes+=aligned;return result;
 }
};
int main(int argc,char**argv){json report={{"schema","hillserve-real-device-chain-v1"},{"status","failed"},{"simulation",false},{"actual_model_feedback",false},{"physical_nand_bytes",nullptr},{"direct_storage_to_fpga",false},{"formal_paper_speedup_eligible",false}};std::filesystem::path out;
 try{
  need((argc-1)%2==0,"key/value arguments required");std::map<std::string,std::string>a;for(int i=1;i<argc;i+=2)need(a.emplace(argv[i],argv[i+1]).second,"duplicate option");
  auto get=[&](const char* key){auto it=a.find(key);if(it==a.end())throw std::runtime_error(std::string("missing ")+key);return it->second;};
  out=get("--output");need(std::filesystem::create_directory(out),"new output directory required");
  auto root=std::filesystem::canonical(get("--scratch-root"));auto scratch=std::filesystem::path(get("--scratch"));need(std::filesystem::canonical(scratch.parent_path())==root,"only granted dedicated scratch parent");
  struct stat target{},partition{};need(stat(root.c_str(),&target)==0&&stat(get("--ssd-device").c_str(),&partition)==0&&S_ISBLK(partition.st_mode)&&target.st_dev==partition.st_rdev,"scratch is not the verified SmartSSD block device");
  std::uint64_t existing=0;for(auto& e:std::filesystem::recursive_directory_iterator(root)){need(!e.is_symlink(),"symlink in dedicated scratch");if(e.is_regular_file())existing+=e.file_size();}need(existing<8ULL*1024*1024*1024,"8GiB aggregate scratch bound");
  need(std::filesystem::create_directory(scratch),"new scratch directory required");
  std::ifstream f(get("--trace"),std::ios::binary);auto hdr=read<std::uint32_t>(f,10);need(hdr[0]==0x48535033&&hdr[1]==2,"actual method trace v2 required");
  std::size_t h=hdr[2],n=hdr[3],d=hdr[4],full=hdr[5],rk=hdr[6],cap=std::stoul(get("--cap")),chunk=std::stoul(get("--chunk"));float alpha=std::stof(get("--alpha"));
  need(h>0&&h<=32&&n>0&&n<=16384&&d>0&&d<=128&&full==128&&rk<=n&&chunk&&chunk<=n&&cap<=n,"bounded trace shape/cap");
  const std::uint64_t host_budget=128ULL*1024*1024,file_budget=256ULL*1024*1024;
  need(existing+file_budget<8ULL*1024*1024*1024,"aggregate scratch admission including full file reservation");
  auto q=read<std::uint16_t>(f,h*d),keys=read<std::uint16_t>(f,h*n*d);std::size_t padded=(keys.size()*2+4095)/4096*4096;
  auto prep=Clock::now();
  {int fd=open((scratch/"score-keys.bin").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);need(fd>=0,"new score file");std::size_t done=0;while(done<keys.size()*2){ssize_t k=write(fd,reinterpret_cast<char*>(keys.data())+done,keys.size()*2-done);if(k<0&&errno==EINTR)continue;need(k>0,"score write");done+=k;}need(ftruncate(fd,padded)==0&&fdatasync(fd)==0,"initial score materialization");close(fd);}
  keys.clear();keys.shrink_to_fit();
  SegmentedKVStore store(scratch/"kv.bin",1,h,n+1,4*full,file_budget-padded);
  std::vector<std::uint64_t> expected(h*n);
  for(std::size_t head=0;head<h;++head){auto payload=read<std::uint8_t>(f,n*4*full);for(std::size_t t=0;t<n;++t)expected[head*n+t]=checksum(payload.data()+t*4*full,4*full);store.append(head,0,payload);}
  store.drain();double setup=ms(prep);store.enable_direct_reads(scratch/"kv.bin");
  auto oracle_scores=read<std::uint16_t>(f,h*n);auto oracle_ids=read<std::uint64_t>(f,h*rk);auto actual_q=read<float>(f,h*full);auto current=read<std::uint16_t>(f,h*2*full);auto attention_ref=read<float>(f,h*full);char tail;need(!f.read(&tail,1),"trailing trace data");f.close();
  auto backend=get("--backend");need(backend=="host"||backend=="fpga"||backend=="direct","backend must be host, fpga or direct");bool direct=backend=="direct";bool verify=get("--verify")=="1";bool stream=a.count("--stream")&&get("--stream")=="1";
  std::unique_ptr<FpgaScoreBackend> device;double initialization=0;
  if(backend!="host"){HardwareContract c;c.abi="HS_PARTIAL_F16_V1";c.image_path=get("--image");c.image_sha256=get("--image-sha256");c.image_execute_authorized=get("--image-grant")=="1";auto t=Clock::now();device=make_fpga_backend(direct?HardwareMode::fpga_storage_streaming:HardwareMode::fpga_host_staged,c);initialization=ms(t);report["image_sha256"]=c.image_sha256;}
  need(!direct||(chunk==n&&!stream),"direct v1 pilot requires whole-block serial; no silent staging");
  report["requested_method"]=direct?"FPGA_DIRECT":device?"FPGA_STAGED":"HOST_MATCHED";report["score_file"]=(scratch/"score-keys.bin").string();report["score_file_st_dev"]=std::uint64_t(target.st_dev);report["score_file_bytes"]=padded;report["key_kernel_argument"]=1;report["p2p_requested"]=direct;
  MethodAttention attention(h,full,cap);DirectFile keyfile(scratch/"score-keys.bin");
  const std::size_t reserved=expected.capacity()*8+oracle_scores.capacity()*2+oracle_ids.capacity()*8+store.index_bytes()+h*n*8+4*h*chunk*d*2+h*cap*(4*full+256)+8*1024*1024;
  need(reserved+store.extent_bytes()+padded<=host_budget,"host staging/index/validation plus worst-case file cache reservation exceeded");
  // Initial materials are independently constructed above. Only this episode is timed.
  FileDeviceScores direct_result;
  const auto episode=Clock::now();double key_ms=0,score_ms=0,verify_ms=0;std::uint64_t score_h2d=0,score_d2h=0;std::size_t calls=0,bit_errors=0;double h2d_ms=0,d2h_ms=0,launch_ms=0;
  std::vector<std::uint16_t> scores(h*n);
  struct ChunkResult{std::vector<std::uint16_t> values;ScoreDeviceTimings times;double wall=0,validation=0,start=0,end=0;std::size_t errors=0,first=0,count=0;};
  json timeline=json::array();
  auto compute=[&](std::vector<std::uint16_t> block,std::size_t first,std::size_t count){ChunkResult result;result.first=first;result.count=count;result.start=ms(episode);auto t=Clock::now();
   if(device){auto r=device->score(q,block,h,count,d);result.values=std::move(r.values);result.times=r.timings;}
   else result.values=host_partial_scores(q,block,h,count,d);
   result.wall=ms(t);result.end=ms(episode);
   if(verify&&device){t=Clock::now();auto ref=host_partial_scores(q,block,h,count,d);for(std::size_t i=0;i<ref.size();++i)result.errors+=result.values[i]!=ref[i];result.validation=ms(t);}
   return result;
  };
  auto merge=[&](ChunkResult part){score_ms+=part.wall;score_h2d+=part.times.h2d_bytes;score_d2h+=part.times.d2h_bytes;h2d_ms+=part.times.h2d_ms;d2h_ms+=part.times.d2h_ms;launch_ms+=part.times.launch_wait_ms;bit_errors+=part.errors;verify_ms+=part.validation;++calls;
   for(std::size_t head=0;head<h;++head)std::copy(part.values.begin()+head*part.count,part.values.begin()+(head+1)*part.count,scores.begin()+head*n+part.first);
   timeline.push_back({{"stage","score"},{"first",part.first},{"count",part.count},{"start_ms",part.start},{"end_ms",part.end}});
  };
  std::future<ChunkResult> pending;
  if(direct){auto direct_call=Clock::now();auto part=device->score_file(q,{keyfile.fd,0,h*n*d*2,verify},h,n,d);scores=std::move(part.values);calls=1;key_ms=part.file_read_ms;keyfile.bytes=part.file_read_bytes;score_ms=ms(direct_call)-part.file_read_ms;score_h2d=part.timings.h2d_bytes;score_d2h=part.timings.d2h_bytes;h2d_ms=part.timings.h2d_ms;d2h_ms=part.timings.d2h_ms;launch_ms=part.timings.launch_wait_ms;
   direct_result=std::move(part);
  } else
  for(std::size_t first=0;first<n;first+=chunk){auto count=std::min(chunk,n-first);auto t=Clock::now();double read_start=ms(episode);std::vector<std::uint16_t> block;
   if(first==0&&count==n)block=keyfile.get(0,h*n*d);
   else{block.resize(h*count*d);for(std::size_t head=0;head<h;++head){auto part=keyfile.get((head*n+first)*d*2,count*d);std::copy(part.begin(),part.end(),block.begin()+head*count*d);}}
   key_ms+=ms(t);timeline.push_back({{"stage","read"},{"first",first},{"count",count},{"start_ms",read_start},{"end_ms",ms(episode)}});
   if(stream){if(pending.valid())merge(pending.get());pending=std::async(std::launch::async,compute,std::move(block),first,count);}
   else merge(compute(std::move(block),first,count));
  }
  if(pending.valid())merge(pending.get()); // required global barrier before selection
  auto t=Clock::now();auto chosen=select_partial_scores(scores,h,n,alpha,cap);double selection_ms=ms(t);std::vector<KVPayload> payload;payload.reserve(chosen.ids.size());auto read_before=store.metrics.syscall_read_bytes;t=Clock::now();
  for(std::size_t head=0;head<h;++head){std::vector<std::uint64_t> ids(chosen.ids.begin()+head*chosen.k,chosen.ids.begin()+(head+1)*chosen.k);auto raw=store.fetch(head,ids,host_budget-reserved);for(std::size_t j=0;j<ids.size();++j){KVPayload p;p.token=head*n+ids[j];p.bytes.assign(raw.begin()+j*4*full,raw.begin()+(j+1)*4*full);payload.push_back(std::move(p));}}
  double gather_ms=ms(t);auto gpu=attention.run(actual_q,current,payload,chosen.k);
  // Current KV visibility and final drain belong to the measured episode.
  auto append_before=store.metrics.syscall_write_bytes;t=Clock::now();for(std::size_t head=0;head<h;++head){std::vector<std::uint8_t> raw(4*full);std::memcpy(raw.data(),current.data()+head*2*full,4*full);store.append(head,n,raw);}double append_ms=ms(t);
  t=Clock::now();store.drain();double final_drain=ms(t);double wall=ms(episode);
  if(direct){const auto& part=direct_result;
   report["p2p_evidence"]={{"allocation_verified",part.p2p_allocation_verified},{"bo_flags",part.bo_flags},{"bo_memory_group",part.bo_memory_group},{"kernel_key_bank",part.kernel_key_bank},{"device_address",part.bo_device_address},{"mapped_destination_address",part.bo_mapped_address},{"mapped_region",part.mapping_evidence},{"pread_completed_bytes",part.file_read_bytes},{"kernel_completed",part.kernel_completed},{"key_h2d_sync_called",false},{"key_host_staging_bytes",part.key_host_staging_bytes}};
   report["direct_api_path_completed"]=part.p2p_allocation_verified&&part.kernel_completed;
   report["direct_storage_to_fpga"]=nullptr;
   report["direct_path_verification_scope"]="regular-file read into P2P BO and kernel completion; physical route requires independent validation";
   report["physical_route_status"]="physical_DMA_route_not_independently_measured";
   report["driver_path_proof"]=nullptr;
  }
  double outside_score_validation_ms=0;
  if(direct&&verify){auto vstart=Clock::now();std::ifstream source(get("--trace"),std::ios::binary);source.seekg(40+h*d*2);auto original=read<std::uint16_t>(source,h*n*d);auto ref=host_partial_scores(q,original,h,n,d);for(std::size_t i=0;i<ref.size();++i)bit_errors+=ref[i]!=scores[i];outside_score_validation_ms=ms(vstart);}
  report["score_validation_ms_outside_episode"]=outside_score_validation_ms;
  double observed_read_score_overlap=0;
  for(const auto& read:timeline)if(read["stage"]=="read")for(const auto& score:timeline)if(score["stage"]=="score"){
   double overlap=std::min(read["end_ms"].get<double>(),score["end_ms"].get<double>())-std::max(read["start_ms"].get<double>(),score["start_ms"].get<double>());observed_read_score_overlap+=std::max(0.0,overlap);
  }
  std::ofstream(out/"timeline.json")<<timeline.dump(2)<<'\n';
  // Full reference checks outside measurement, except explicitly labeled device validation.
  std::size_t set_errors=0,byte_errors=0,ref_score_diff=0;double max_abs=0,attention_err=0;
  for(std::size_t i=0;i<scores.size();++i){ref_score_diff+=scores[i]!=oracle_scores[i];max_abs=std::max(max_abs,double(std::abs(half_value(scores[i])-half_value(oracle_scores[i]))));}
  if(chosen.k!=rk)set_errors=h;else for(std::size_t head=0;head<h;++head){std::set<std::uint64_t>x(chosen.ids.begin()+head*rk,chosen.ids.begin()+(head+1)*rk),y(oracle_ids.begin()+head*rk,oracle_ids.begin()+(head+1)*rk);set_errors+=x!=y;}
  for(auto&p:payload)byte_errors+=checksum(p.bytes.data(),p.bytes.size())!=expected[p.token];
  std::size_t exact_byte_errors=0;double exact_byte_validation_ms=0;
  if(verify){auto validate_start=Clock::now();std::ifstream source(get("--trace"),std::ios::binary);const std::uint64_t base=40+h*d*2+h*n*d*2;
   for(const auto& record:payload){source.seekg(base+record.token*4*full);auto original=read<std::uint8_t>(source,4*full);for(std::size_t i=0;i<original.size();++i)exact_byte_errors+=original[i]!=record.bytes[i];}
   exact_byte_validation_ms=ms(validate_start);
  }
  if(verify){std::ofstream(out/"attention-output.bin",std::ios::binary).write(reinterpret_cast<const char*>(gpu.values.data()),gpu.values.size()*sizeof(float));std::ofstream(out/"selected-ids.bin",std::ios::binary).write(reinterpret_cast<const char*>(chosen.ids.data()),chosen.ids.size()*sizeof(std::uint64_t));}
  for(std::size_t i=0;i<gpu.values.size();++i)attention_err=std::max(attention_err,std::abs(double(gpu.values[i])-attention_ref[i]));
  struct rusage ru{};getrusage(RUSAGE_SELF,&ru);bool ok=!bit_errors&&!set_errors&&!byte_errors&&!exact_byte_errors;
  report.update({{"status",ok?"passed":"failed"},{"evidence_mode","REAL_MODEL_OPERATOR_REPLAY"},{"policy_id","infinigen_public_port_3754f52_OPT_speed_v1"},{"backend",backend},{"actual_fpga",bool(device)},{"actual_target_storage",true},{"actual_gpu_attention",true},{"payload_read_mode","ext4_O_DIRECT_aligned_regular_file"},{"score_path",direct?"SSD_O_DIRECT_to_P2P_BO_same_v1":device?"SSD_to_host_to_FPGA":"SSD_to_host_CPU"},{"score_key_host_staging_bytes",direct?0:h*n*d*2},{"host_staged_score_keys_bytes",direct?0:h*n*d*2},{"query_h2d_bytes",device?h*d*2:0},{"host_cache_state","no_persistent_hot_KV; O_DIRECT_reads; no_global_cache_drop"},{"setup_materialization_drain_ms",setup},{"device_initialization_ms",initialization},{"episode_wall_ms",wall},{"same_input_verification_inside_episode",verify&&!direct},{"same_input_validation_ms",verify_ms},{"score_key_read_ms",key_ms},{"score_path_wall_ms",score_ms},{"fpga_launch_wait_ms",device?json(launch_ms):json(nullptr)},{"fpga_h2d_ms",h2d_ms},{"fpga_d2h_ms",d2h_ms},{"select_ms",selection_ms},{"selected_gather_ms",gather_ms},{"gpu_attention_wall_ms",gpu.total_ms},{"gpu_compute_host_wall_ms",gpu.compute_wall_ms},{"append_ms",append_ms},{"final_drain_ms",final_drain},{"durability","volatile_query_kv+end_of_request_sync"},{"score_input_logical_bytes",h*n*d*2},{"score_read_syscall_bytes",keyfile.bytes},{"selected_payload_logical_bytes",chosen.ids.size()*full*4},{"selected_read_syscall_bytes",store.metrics.syscall_read_bytes-read_before},{"episode_append_syscall_bytes",store.metrics.syscall_write_bytes-append_before},{"score_h2d_bytes",score_h2d},{"score_d2h_bytes",score_d2h},{"gpu_h2d_bytes",gpu.h2d_bytes},{"host_managed_budget_bytes",host_budget},{"host_budget_scope","application buffers/index plus worst-case owned file cache; CUDA/XRT runtime RSS separately observed"},{"total_process_host_peak_bound_validated",false},{"host_working_reservation_bytes",reserved},{"file_cache_reservation_bytes",store.extent_bytes()+padded},{"host_total_reservation_bytes",reserved+store.extent_bytes()+padded},{"index_bytes",store.index_bytes()},{"file_extent_bytes",store.extent_bytes()+padded},{"process_peak_rss_kib",ru.ru_maxrss},{"gpu_tensor_bytes",gpu.gpu_bytes},{"heads",h},{"tokens",n},{"channels",d},{"layer",hdr[7]},{"step",hdr[8]},{"chunk",chunk},{"score_calls",calls},{"score_read_streaming_enabled",stream},{"observed_read_score_overlap_ms",observed_read_score_overlap},{"global_selection_barrier",true},{"app_enabled",false},{"hpm_enabled",false},{"selected_k",chosen.k},{"alpha",alpha},{"cap",cap},{"host_fpga_bit_errors",verify&&device?json(bit_errors):json(nullptr)},{"reference_set_mismatching_heads",set_errors},{"kv_byte_errors",byte_errors},{"kv_error_field_scope","digest mismatch count; exact comparison separately reported"},{"exact_selected_byte_mismatches",verify?json(exact_byte_errors):json(nullptr)},{"exact_selected_bytes_checked",verify?json(chosen.ids.size()*4*full):json(nullptr)},{"exact_byte_validation_ms_outside_episode",exact_byte_validation_ms},{"reference_score_bit_differences",ref_score_diff},{"reference_score_max_abs",max_abs},{"attention_max_abs_error_vs_fp32",attention_err},{"attention_quality_gate_asserted",false}});
  std::ofstream(out/"result.json")<<report.dump(2)<<'\n';std::cout<<report.dump()<<'\n';return ok?0:3;
 }catch(const std::exception&e){report["error"]=e.what();if(!out.empty()&&std::filesystem::is_directory(out))std::ofstream(out/"failure.json")<<report.dump(2)<<'\n';std::cerr<<report.dump()<<'\n';return 1;}
}
