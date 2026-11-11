#include "hybrid_eval/gqa_frozen.hpp"
#include "smartssd/adapter.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <cerrno>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <set>
#include <iostream>
using namespace hillserve;using namespace hillserve::replay;
static std::uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
struct DirectReader {
 int fd=-1;void* aligned=nullptr;std::size_t capacity;std::uint64_t bytes=0,calls=0,extent=0;
 DirectReader(const std::string& path,std::size_t cap):capacity((cap+8191)/4096*4096){fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);need(fd>=0,"O_DIRECT existing regular file required");struct stat st{};need(!fstat(fd,&st)&&S_ISREG(st.st_mode),"regular file required");extent=st.st_size;need(posix_memalign(&aligned,4096,capacity)==0,"aligned reader allocation");}
 ~DirectReader(){if(fd>=0)close(fd);free(aligned);}
 const std::uint8_t* get(std::uint64_t off,std::size_t size){auto begin=off/4096*4096,end=(off+size+4095)/4096*4096;need(end-begin<=capacity&&end<=extent,"bounded direct read extent");ssize_t got;do{got=pread(fd,aligned,end-begin,begin);}while(got<0&&errno==EINTR);need(got==ssize_t(end-begin),"short/error direct read; no fallback");bytes+=got;++calls;return static_cast<const std::uint8_t*>(aligned)+(off-begin);}
};
static std::vector<std::uint16_t> cpu_score(const std::vector<std::uint16_t>&q,const std::uint16_t*k,std::size_t h,std::size_t n,std::size_t d){
 std::vector<std::uint16_t> out(h*n);auto work=[&](unsigned worker){for(std::size_t head=worker;head<h;head+=4){std::vector<float> qf(d);for(std::size_t c=0;c<d;++c)qf[c]=half_value(q[head*d+c]);for(std::size_t t=0;t<n;++t){float sum=0;for(std::size_t c=0;c<d;++c)sum+=qf[c]*half_value(k[(head*n+t)*d+c]);out[head*n+t]=half_bits(sum);}}};std::vector<std::thread> threads;for(unsigned i=1;i<4;++i)threads.emplace_back(work,i);work(0);for(auto& t:threads)t.join();return out;
}
struct KVResult{std::vector<std::uint64_t> positions;std::vector<std::uint8_t> payload;};
static KVResult gather(DirectReader&r,const policy::GQASelection&chosen,const json&m){
 std::set<std::uint64_t> unique;for(const auto&h:chosen.ids)unique.insert(h.begin(),h.end());KVResult out;out.positions.assign(unique.begin(),unique.end());out.payload.resize(unique.size()*4096);
 const auto stride=m.value("kv_record_stride",std::uint64_t(4096)),base=m.value("kv_data_offset",std::uint64_t(0)),header=m.value("kv_payload_offset",std::uint64_t(0));
 for(std::size_t i=0;i<out.positions.size();){auto off=base+out.positions[i]*stride;std::uint64_t aligned=off/4096*4096;std::size_t j=i+1;
  while(j<out.positions.size()&&((base+out.positions[j]*stride+header+4096+4095)/4096*4096-aligned)<=65536)++j;
  auto end=base+out.positions[j-1]*stride+header+4096;const auto* data=r.get(off,end-off);
  for(std::size_t u=i;u<j;++u){auto relative=(out.positions[u]-out.positions[i])*stride;std::memcpy(out.payload.data()+u*4096,data+relative+header,4096);if(header){std::array<std::uint64_t,3> hdr{};std::memcpy(hdr.data(),data+relative,24);need(hdr[1]==out.positions[u]&&hdr[0]==checksum(out.payload.data()+u*4096,4096),"existing segmented KV identity/integrity failed");}}
  i=j;
 }return out;
}
int main(int argc,char**argv){
 if(argc!=6){std::cerr<<"TRACE_DIR SCORE_FILE KV_FILE METHOD NEW_OUTPUT_DIR\n";return 2;}const std::filesystem::path trace=argv[1],out=argv[5];if(!std::filesystem::create_directory(out))return 2;
 try{json m;std::ifstream(trace/"manifest.json")>>m;need(m.at("status")=="passed","validated real trace required");std::string mode=argv[4];
  const bool host=mode=="HOST_COLD"||mode=="HOST_RESIDENT",resident=mode=="HOST_RESIDENT"||mode=="FPGA_RESIDENT",staged=mode=="FPGA_STAGED_SERIAL"||mode=="FPGA_STAGED_STREAM",stream=mode=="FPGA_STAGED_STREAM";
  need(host||resident||staged||mode=="FPGA_DIRECT_FRESH"||mode=="FPGA_DIRECT_REUSE","unknown explicit mode");
  std::size_t h=m.at("heads"),n=m.at("capacity"),d=m.at("compact_channels_padded"),valid=m.at("valid_candidates");need(h==32&&d<=128&&n<=65504&&valid<=n,"trace ABI");
  auto q=load<std::uint16_t>(trace/"query-f16.bin",h*d);auto oracle=load<std::uint16_t>(trace/"raw-score-oracle.bin",h*n);auto oracle_ids=load<std::uint64_t>(trace/"selected-ids.bin",h*m.at("selected_per_head").get<std::size_t>());
  const std::uint64_t reservation=4ULL*h*n*d*2+2ULL*valid*4096+32ULL*1024*1024;
  need(reservation<=268435456,"replay host reservation exceeds budget before execution");
  const std::uint64_t score_offset=m.value("score_file_offset",std::uint64_t(0));need(score_offset%4096==0,"page-aligned score region required");DirectReader score(argv[2],h*n*d*2),kv(argv[3],65536);
  need(score_offset+h*n*d*2<=score.extent,"score extent");std::mutex mutex;std::condition_variable condition;bool submitted=false;
  std::unique_ptr<FpgaScoreBackend> device;double init_ms=0;HardwareContract c;c.abi="HS_PARTIAL_F16_V1";c.image_execute_authorized=true;c.image_path="build/partial-f16-hw/hillserve_partial_f16_250.xclbin";c.image_sha256="c48f86d51667edf8f9142e98e6c99c8c108aaa40acb05cba0ab4da03c6f91c35";c.reuse_file_buffers=mode=="FPGA_DIRECT_REUSE"||mode=="FPGA_RESIDENT";
  if(stream)c.on_kernel_submitted=[&]{std::lock_guard<std::mutex>lock(mutex);submitted=true;condition.notify_all();};
  if(!host){auto t=Clock::now();device=make_fpga_backend(staged?HardwareMode::fpga_host_staged:HardwareMode::fpga_storage_streaming,c);init_ms=milliseconds(t);}
  json priming=nullptr;if(resident){auto t=Clock::now();if(host)score.get(score_offset,h*n*d*2);else{auto r=device->score_file(q,{score.fd,score_offset,h*n*d*2,false},h,n,d);need(r.values==oracle,"resident prime score bit mismatch");}priming={{"wall_ms",milliseconds(t)},{"score_file_read_bytes",h*n*d*2},{"excluded_from_query_service",true}};}
  std::ofstream records(out/"runs.jsonl"),events_file(out/"events.jsonl");json all=json::array();
  for(int repeat=0;repeat<2;++repeat){const std::string phase=repeat?"measured":"warmup";std::uint64_t read_before=score.bytes,kv_before=kv.bytes,kv_calls=kv.calls;json events=json::array();ScoreDeviceTimings times;std::uint64_t direct_read=0,key_staging=0,q_h2d=0;bool reused=false,p2p_verified=false;std::uint64_t bo_address=0;double read_ms=0,merge_ms=0;
   auto begin=Clock::now();const auto ready=ns();std::vector<std::uint16_t> raw(h*n),normalized(h*n);
   if(host){auto t=Clock::now();const auto* k=resident?static_cast<const std::uint16_t*>(score.aligned):reinterpret_cast<const std::uint16_t*>(score.get(score_offset,h*n*d*2));read_ms=milliseconds(t);t=Clock::now();raw=cpu_score(q,k,h,n,d);times.launch_wait_ms=milliseconds(t);auto mt=Clock::now();normalized=scaled(raw);merge_ms=milliseconds(mt);key_staging=resident?0:h*n*d*2;}
   else if(!staged){auto r=resident?device->score_resident_file(q,h,n,d):device->score_file(q,{score.fd,score_offset,h*n*d*2,false},h,n,d);raw=std::move(r.values);times=r.timings;direct_read=r.file_read_bytes;q_h2d=r.query_h2d_bytes;read_ms=r.file_read_ms;reused=r.bo_reused;p2p_verified=r.p2p_allocation_verified;bo_address=r.bo_device_address;auto mt=Clock::now();normalized=scaled(raw);merge_ms=milliseconds(mt);}
   else{
    struct Part{std::size_t first,count;DeviceScores r;std::uint64_t started,ended;};std::future<Part> pending;
    auto compute=[&](std::vector<std::uint16_t> keys,std::size_t first,std::size_t count){Part p{first,count,{},ns(),0};p.r=device->score(q,keys,h,count,d);p.ended=ns();return p;};
    auto merge=[&](Part p){const auto a=ns();auto mt=Clock::now();for(std::size_t head=0;head<h;++head)for(std::size_t j=0;j<p.count;++j){auto bits=p.r.values[head*p.count+j];raw[head*n+p.first+j]=bits;normalized[head*n+p.first+j]=half_bits(half_value(bits)/std::sqrt(128.f));}merge_ms+=milliseconds(mt);const auto b=ns();const auto& t=p.r.timings;times.allocation_ms+=t.allocation_ms;times.pack_ms+=t.pack_ms;times.h2d_ms+=t.h2d_ms;times.launch_wait_ms+=t.launch_wait_ms;times.d2h_ms+=t.d2h_ms;times.h2d_bytes+=t.h2d_bytes;times.d2h_bytes+=t.d2h_bytes;q_h2d+=h*d*2;
      events.push_back({{"stage","device_service"},{"chunk_first",p.first},{"count",p.count},{"start_ns",p.started},{"end_ns",p.ended}});events.push_back({{"stage","launch_wait"},{"chunk_first",p.first},{"start_ns",t.launch_started_ns},{"end_ns",t.launch_finished_ns}});events.push_back({{"stage","CPU_score_return_merge"},{"chunk_first",p.first},{"start_ns",a},{"end_ns",b}});
    };
    for(std::size_t first=0;first<n;first+=1024){auto count=std::min<std::size_t>(1024,n-first);auto t=Clock::now();auto a=ns();std::vector<std::uint16_t> keys(h*count*d);for(std::size_t head=0;head<h;++head){const auto* src=score.get(score_offset+(head*n+first)*d*2,count*d*2);std::memcpy(keys.data()+head*count*d,src,count*d*2);}read_ms+=milliseconds(t);events.push_back({{"stage","SSD_read_and_pack"},{"chunk_first",first},{"start_ns",a},{"end_ns",ns()}});key_staging+=h*count*d*2;
     if(!stream)merge(compute(std::move(keys),first,count));else if(!pending.valid()){std::lock_guard<std::mutex>lock(mutex);submitted=false;pending=std::async(std::launch::async,compute,std::move(keys),first,count);}else{auto previous=pending.get();{std::lock_guard<std::mutex>lock(mutex);submitted=false;}pending=std::async(std::launch::async,compute,std::move(keys),first,count);{std::unique_lock<std::mutex>lock(mutex);need(condition.wait_for(lock,std::chrono::seconds(60),[&]{return submitted;}),"next chunk launch did not reach pipeline barrier");}merge(std::move(previous));}
    }if(pending.valid())merge(pending.get());
   }
   const auto scores_ready=ns();auto t=Clock::now();auto selected=policy::select_gqa_scores(normalized,h,n,valid,m.at("alpha"),float(m.at("cap_ratio").get<double>()));need(selected.k>0,"zero selection, no fallback");double select_ms=milliseconds(t);auto ids_ready=ns();double filter_ms=milliseconds(begin);t=Clock::now();auto fetched=gather(kv,selected,m);double gather_ms=milliseconds(t),chain_ms=milliseconds(begin);auto kv_ready=ns();
   // Exact checks are outside both filter and retrieval-chain clocks.
   need(raw==oracle,"actual raw score differs from expanded128 reference");need(flatten(selected)==oracle_ids,"actual selected IDs differ from frozen policy");
   auto expected=load<std::uint16_t>(trace/"kv-physical.bin",valid*2048);std::size_t byte_errors=0;for(std::size_t i=0;i<fetched.positions.size();++i){const auto* left=fetched.payload.data()+i*4096;const auto* right=reinterpret_cast<const std::uint8_t*>(expected.data()+fetched.positions[i]*2048);for(std::size_t b=0;b<4096;++b)byte_errors+=left[b]!=right[b];}need(!byte_errors,"selected KV exact byte comparison failed");
   double overlap_read=0,overlap_merge=0;for(const auto&e:events)if(e["stage"]=="SSD_read_and_pack"||e["stage"]=="CPU_score_return_merge")for(const auto&k:events)if(k["stage"]=="launch_wait"){auto lo=std::max(e["start_ns"].get<std::uint64_t>(),k["start_ns"].get<std::uint64_t>()),hi=std::min(e["end_ns"].get<std::uint64_t>(),k["end_ns"].get<std::uint64_t>());if(hi>lo){if(e["stage"]=="SSD_read_and_pack")overlap_read+=(hi-lo)/1e6;else overlap_merge+=(hi-lo)/1e6;}}
   struct rusage usage{};getrusage(RUSAGE_SELF,&usage);json row={{"status","passed"},{"phase",phase},{"method",mode},{"evidence_class","real_single_layer_frozen_retrieval_replay"},{"model_feedback",false},{"P",m.at("P")},{"layer",m.at("layer")},{"valid_candidates",valid},{"kernel_capacity",n},{"channels",d},{"normalization_dimension",128},{"query_ready_ns",ready},{"scores_ready_ns",scores_ready},{"global_ids_ready_ns",ids_ready},{"selected_KV_host_ready_ns",kv_ready},{"filter_service_ms",filter_ms},{"retrieval_chain_ms",chain_ms},{"BO_allocation_ms",times.allocation_ms},{"BO_release_ms",times.release_ms},{"query_or_keys_pack_ms",times.pack_ms},{"score_file_read_and_pack_ms",read_ms},{"H2D_ms",times.h2d_ms},{"launch_wait_ms",host?json(nullptr):json(times.launch_wait_ms)},{"CPU_score_ms",host?json(times.launch_wait_ms):json(nullptr)},{"score_D2H_ms",times.d2h_ms},{"CPU_merge_scale_ms",merge_ms},{"select_ms",select_ms},{"KV_gather_ms",gather_ms},{"score_read_syscall_bytes",score.bytes-read_before+direct_read},{"cold_K_application_host_staging_bytes",key_staging},{"Q_H2D_bytes",q_h2d},{"total_H2D_bytes",times.h2d_bytes},{"score_D2H_bytes",times.d2h_bytes},{"KV_read_syscall_bytes",kv.bytes-kv_before},{"KV_read_calls",kv.calls-kv_calls},{"KV_selected_payload_bytes",fetched.payload.size()},{"selected_per_head",selected.k},{"selection_checksum",std::to_string(checksum(oracle_ids.data(),oracle_ids.size()*8))},{"payload_checksum",std::to_string(checksum(fetched.payload.data(),fetched.payload.size()))},{"exact_selected_bytes_checked_outside_timing",fetched.payload.size()},{"raw_score_bit_errors",0},{"selected_ID_errors",0},{"KV_byte_errors",byte_errors},{"BO_reused",reused},{"P2P_allocation_verified",p2p_verified},{"key_BO_device_address",bo_address},{"score_file_offset",score_offset},{"logical_score_file_bytes",h*n*d*2},{"resident_keys",resident},{"resident_scope",resident?(host?"host_RAM_partial_keys":"FPGA_P2P_BO_loaded_once"):"SSD_read_each_query"},{"read_launchwait_overlap_ms",overlap_read},{"CPU_merge_launchwait_overlap_ms",overlap_merge},{"overlap_scope","actual host launch/wait envelopes; physical FPGA start not independently instrumented"},{"chunk_tokens",staged?json(1024):json(n)},{"reader_queue_depth",1},{"score_outstanding_depth",1},{"pipeline_buffer_depth",stream?2:1},{"host_application_budget_bytes",268435456},{"host_application_conservative_reservation_bytes",reservation},{"FPGA_BO_budget_bytes",c.buffer_budget_bytes},{"process_peak_RSS_KiB",usage.ru_maxrss},{"physical_DMA_no_bounce_verified",false},{"physical_NAND_bytes",nullptr},{"whole_model_TPOT_prediction",nullptr},{"image_sha256",host?json(nullptr):json(c.image_sha256)}};
   need(row["host_application_conservative_reservation_bytes"].get<std::uint64_t>()<=268435456,"replay host reservation exceeds budget");records<<row.dump()<<'\n';for(auto e:events){e["phase"]=phase;events_file<<e.dump()<<'\n';}all.push_back(row);
  }
  auto t=Clock::now();device.reset();double teardown=milliseconds(t);std::ofstream(out/"summary.json")<<json({{"status","passed"},{"method",mode},{"initialization_ms_excluded",init_ms},{"persistent_teardown_ms_excluded",teardown},{"priming",priming},{"timed_rows",2},{"independent_process",true},{"warmup_count",1},{"measured_count",1},{"score_queries_identical_frozen_checkpoint",true}}).dump(2)<<'\n';std::cout<<all.back().dump()<<'\n';return 0;
 }catch(const std::exception&e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
