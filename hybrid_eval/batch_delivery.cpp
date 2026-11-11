#include "runtime/batch_delivery.hpp"
#include "smartssd/adapter.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <cerrno>
#include <future>
#include <thread>
#include <set>
#include <iostream>
using namespace hillserve;using namespace hillserve::replay;
static std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
struct DirectFile {
 int fd=-1;void* buffer=nullptr;std::size_t capacity,extent;std::uint64_t bytes=0,calls=0;
 DirectFile(const std::string& path,std::size_t cap):capacity((cap+4095)/4096*4096){fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);need(fd>=0,"O_DIRECT existing input required");struct stat s{};need(!fstat(fd,&s)&&S_ISREG(s.st_mode),"regular input required");extent=s.st_size;need(posix_memalign(&buffer,4096,capacity)==0,"bounded aligned allocation");}
 ~DirectFile(){if(fd>=0)close(fd);free(buffer);}
 const std::uint8_t* read(std::size_t off,std::size_t size){need(off%4096==0&&size%4096==0&&off+size<=extent&&size<=capacity,"aligned read bounds");ssize_t got;do{got=pread(fd,buffer,size,off);}while(got<0&&errno==EINTR);need(got==ssize_t(size),"short direct read; no buffered fallback");bytes+=size;++calls;return static_cast<const std::uint8_t*>(buffer);}
};
static std::vector<std::uint16_t> cpu_score(const std::vector<std::uint16_t>& q,const std::uint16_t*k,std::size_t n,std::size_t d,unsigned threads){
 std::vector<std::uint16_t> out(32*n);auto worker=[&](unsigned index){for(std::size_t h=index;h<32;h+=threads){std::vector<float> query(d);for(std::size_t c=0;c<d;++c)query[c]=half_value(q[h*d+c]);for(std::size_t t=0;t<n;++t){float sum=0;for(std::size_t c=0;c<d;++c)sum+=query[c]*half_value(k[(h*n+t)*d+c]);out[h*n+t]=half_bits(sum);}}};
 std::vector<std::thread> jobs;for(unsigned i=1;i<threads;++i)jobs.emplace_back(worker,i);worker(0);for(auto&t:jobs)t.join();return out;
}
struct Sequence {
 std::filesystem::path trace;std::size_t n=0,d=0;std::vector<std::uint16_t> q,oracle,hot_keys,raw;
 std::vector<std::uint64_t> expected_ids,physical;std::vector<std::uint8_t> hot_kv,payload;std::unique_ptr<DirectFile> score,kv;
 policy::GQASelection selection;std::uint64_t hot_start=0,hot_end=0,cold_payload=0,hot_payload=0;
};
static void gather(Sequence&s,std::size_t cold){
 std::set<std::uint64_t> ids;for(const auto&h:s.selection.ids)ids.insert(h.begin(),h.end());s.physical.assign(ids.begin(),ids.end());s.payload.resize(s.physical.size()*4096);s.cold_payload=s.hot_payload=0;
 for(std::size_t i=0;i<s.physical.size();){auto id=s.physical[i];if(id>=cold){need((id-cold+1)*4096<=s.hot_kv.size(),"hot sequence bounds");std::memcpy(s.payload.data()+i*4096,s.hot_kv.data()+(id-cold)*4096,4096);s.hot_payload+=4096;++i;continue;}
  auto j=i+1;while(j<s.physical.size()&&s.physical[j]<cold&&(s.physical[j]-id+1)*4096<=65536)++j;auto data=s.kv->read(id*4096,(s.physical[j-1]-id+1)*4096);
  for(auto k=i;k<j;++k){std::memcpy(s.payload.data()+k*4096,data+(s.physical[k]-id)*4096,4096);s.cold_payload+=4096;}i=j;
 }
}
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"BATCH_MANIFEST CPU_RETURN|HILLSERVE NEW_OUTPUT\n";return 2;}const std::filesystem::path out=argv[3];if(!std::filesystem::create_directory(out))return 2;
 try{json cfg;std::ifstream(argv[1])>>cfg;const std::string method=argv[2];const bool fpga=method=="HILLSERVE";need(fpga||method=="CPU_RETURN","method");
  constexpr std::size_t B=8,P=32768,valid=P+1,cold=P/2;need(cfg.at("B")==B&&cfg.at("P")==P&&cfg.at("cold_prompt_positions")==cold&&cfg.at("sequences").size()==B,"explicit actual B8 32K half-cold manifest");
  const std::uint64_t host_budget=cfg.at("pilot_host_budget_bytes"),gpu_budget=cfg.at("GPU_workspace_budget_bytes");need(host_budget<=4ULL*1024*1024*1024&&gpu_budget<=2ULL*1024*1024*1024,"bounded pilot budgets");
  const std::size_t slots=(std::size_t(std::floor(float(valid)*.2f))+31)/32*32;std::array<Sequence,B> seq;std::vector<float> actual;std::uint64_t reserved=slots*24576ULL+128ULL*1024*1024;
  for(std::size_t i=0;i<B;++i){auto&e=cfg["sequences"][i];seq[i].trace=e.at("trace").get<std::string>();json m;std::ifstream(seq[i].trace/"manifest.json")>>m;
   need(m.at("status")=="passed"&&m.at("P")==P&&m.at("valid_candidates")==valid&&m.at("layer")==8,"real verified 32K layer8 checkpoint required");seq[i].n=m.at("capacity");seq[i].d=m.at("compact_channels_padded");need(seq[i].d==49,"frozen same-model channels");
   reserved+=32*(seq[i].n-cold)*seq[i].d*2+(valid-cold)*4096+valid*4096+32*seq[i].n*8+32*slots*16+cold*32*seq[i].d*2;
  }need(reserved<=host_budget,"B8 host reservation before payload allocation");
  for(std::size_t i=0;i<B;++i){auto&s=seq[i];auto&e=cfg["sequences"][i];json m;std::ifstream(s.trace/"manifest.json")>>m;
   s.q=load<std::uint16_t>(s.trace/"query-f16.bin",32*s.d);s.oracle=load<std::uint16_t>(s.trace/"raw-score-oracle.bin",32*s.n);s.expected_ids=load<std::uint64_t>(s.trace/"selected-ids.bin",32*m.at("selected_per_head").get<std::size_t>());
   auto q=load<float>(s.trace/"actual-attention-query-f32.bin",4096);actual.insert(actual.end(),q.begin(),q.end());
   const auto hn=s.n-cold;s.hot_keys.resize(32*hn*s.d);std::ifstream ks(s.trace/"score-compact.bin",std::ios::binary);
   for(std::size_t h=0;h<32;++h){ks.seekg((h*s.n+cold)*s.d*2);ks.read(reinterpret_cast<char*>(s.hot_keys.data()+h*hn*s.d),hn*s.d*2);need(bool(ks),"hot score extent");}
   s.hot_kv.resize((valid-cold)*4096);std::ifstream kv(e.at("full_KV_file").get<std::string>(),std::ios::binary);kv.seekg(cold*4096);kv.read(reinterpret_cast<char*>(s.hot_kv.data()),s.hot_kv.size());need(bool(kv),"host hot cache from complete SSD backing");
   s.score=std::make_unique<DirectFile>(e.at("cold_score_file"),32*cold*s.d*2);s.kv=std::make_unique<DirectFile>(e.at("full_KV_file"),65536);
   need(s.score->extent==32*cold*s.d*2&&s.kv->extent==valid*4096,"same input file extents");
  }
  GQABatchDeliveryWorkspace gpu(B,slots,actual,gpu_budget);HardwareContract hc;hc.abi="HS_PARTIAL_F16_V1";hc.image_execute_authorized=true;hc.image_path="build/partial-f16-hw/hillserve_partial_f16_250.xclbin";hc.image_sha256="c48f86d51667edf8f9142e98e6c99c8c108aaa40acb05cba0ab4da03c6f91c35";hc.reuse_file_buffers=true;
  std::unique_ptr<FpgaScoreBackend> device;if(fpga)device=make_fpga_backend(HardwareMode::fpga_storage_streaming,hc);
  std::ofstream records(out/"runs.jsonl");for(int round=0;round<2;++round){gpu.begin_round();std::array<std::promise<std::vector<std::uint16_t>>,B> promises;std::array<std::future<std::vector<std::uint16_t>>,B> futures;
   for(std::size_t i=0;i<B;++i)futures[i]=promises[i].get_future();const auto ready=now_ns();
   auto hot_job=std::async(std::launch::async,[&]{for(std::size_t i=0;i<B;++i){try{auto&s=seq[i];s.hot_start=now_ns();auto scores=cpu_score(s.q,s.hot_keys.data(),s.n-cold,s.d,1);s.hot_end=now_ns();promises[i].set_value(std::move(scores));}catch(...){promises[i].set_exception(std::current_exception());}}});
   json rows=json::array();for(std::size_t i=0;i<B;++i){auto&s=seq[i];auto kv_before=s.kv->bytes;std::vector<std::uint16_t> scores;FileDeviceScores hw;double read_ms=0,cpu_ms=0;const auto cold_start=now_ns();
    if(fpga){hw=device->score_file(s.q,{s.score->fd,0,32*cold*s.d*2,false},32,cold,s.d);scores=std::move(hw.values);read_ms=hw.file_read_ms;}else{auto t=Clock::now();auto k=reinterpret_cast<const std::uint16_t*>(s.score->read(0,32*cold*s.d*2));read_ms=milliseconds(t);t=Clock::now();scores=cpu_score(s.q,k,cold,s.d,3);cpu_ms=milliseconds(t);}
    const auto cold_end=now_ns();auto hs=futures[i].get();s.raw.resize(32*s.n);auto t=Clock::now();for(std::size_t h=0;h<32;++h){std::copy_n(scores.data()+h*cold,cold,s.raw.data()+h*s.n);std::copy_n(hs.data()+h*(s.n-cold),s.n-cold,s.raw.data()+h*s.n+cold);}auto normalized=scaled(s.raw);double merge_ms=milliseconds(t);t=Clock::now();s.selection=policy::select_gqa_scores(normalized,32,s.n,valid,5,.2f);need(s.selection.k>0,"zero selected; no fallback");double select_ms=milliseconds(t);auto id_ready=now_ns();t=Clock::now();gather(s,cold);double gather_ms=milliseconds(t);auto host_ready=now_ns();auto delivered=gpu.deliver(i,s.selection.ids,P,s.physical,s.payload);
    rows.push_back({{"sequence",i},{"query_ready_ns",ready},{"cold_started_ns",cold_start},{"cold_finished_ns",cold_end},{"hot_started_ns",s.hot_start},{"hot_finished_ns",s.hot_end},{"global_ID_ready_ns",id_ready},{"KV_host_ready_ns",host_ready},{"KV_mask_GPU_ready_ns",delivered.gpu_ready_ns},{"sequence_T_delivery_ms",(delivered.gpu_ready_ns-ready)/1e6},{"cold_file_read_ms",read_ms},{"CPU_cold_score_ms",cpu_ms},{"FPGA_launch_wait_ms",hw.timings.launch_wait_ms},{"FPGA_score_return_ms",hw.timings.d2h_ms},{"merge_ms",merge_ms},{"selection_ms",select_ms},{"KV_gather_ms",gather_ms},{"GPU_pack_ms",delivered.pack_ms},{"GPU_upload_ms",delivered.upload_wall_ms},{"GPU_copy_stream_interval_ms",delivered.cuda_copy_stream_interval_ms},{"cold_key_read_bytes",32*cold*s.d*2},{"cold_key_SSD_to_host_bytes",fpga?0:32*cold*s.d*2},{"cold_key_SSD_to_P2P_bytes",fpga?32*cold*s.d*2:0},{"scores_FPGA_to_host_bytes",hw.timings.d2h_bytes},{"query_host_to_FPGA_bytes",hw.query_h2d_bytes},{"selected_cold_KV_bytes",s.cold_payload},{"selected_hot_KV_bytes",s.hot_payload},{"selected_KV_SSD_read_bytes",s.kv->bytes-kv_before},{"GPU_upload_bytes",delivered.upload_bytes},{"selected_per_head",s.selection.k},{"P2P_BO_verified",hw.p2p_allocation_verified}});
   }const auto end=gpu.batch_ready_ns();hot_job.get();
   // All correctness checks are after the common all-eight CUDA-completed endpoint.
   json verification=json::array();for(std::size_t i=0;i<B;++i){auto&s=seq[i];need(s.raw==s.oracle&&flatten(s.selection)==s.expected_ids,"actual batch scores/IDs differ from real oracle");std::ifstream kv(s.trace/"kv-physical.bin",std::ios::binary);std::vector<std::uint8_t> block(4096);for(std::size_t j=0;j<s.physical.size();++j){kv.seekg(s.physical[j]*4096);kv.read(reinterpret_cast<char*>(block.data()),4096);need(bool(kv)&&std::memcmp(block.data(),s.payload.data()+j*4096,4096)==0,"batch physical KV mismatch");}auto v=gpu.verify_sequence(i,s.selection.ids,P,s.physical,s.payload);v["score_bit_errors"]=0;v["ordered_ID_errors"]=0;v["KV_byte_errors"]=0;verification.push_back(v);}
   auto batch_attention=gpu.verify_and_consume();json independent=json::array();
   if(round==0){for(std::size_t i=0;i<B;++i){std::vector<float> q(actual.begin()+i*4096,actual.begin()+(i+1)*4096);GQADeliveryWorkspace single(slots,q);auto&s=seq[i];single.deliver(s.selection.ids,P,s.physical,s.payload);auto check=single.verify_and_consume();need(check["actual_attention_output_checksum"]==batch_attention["per_sequence_attention_output_checksums"][i],"B8 slice differs from independent B1 actual attention");independent.push_back({{"sequence",i},{"status","passed_bit_identical_to_B1_graph"},{"attention_output_checksum",check["actual_attention_output_checksum"]}});}}
   struct rusage usage{};getrusage(RUSAGE_SELF,&usage);json result={{"status","passed"},{"phase",round?"measured":"warmup"},{"method",method},{"classification","actual_B8_single_layer_retrieval_and_batched_GPU_attention"},{"B",B},{"P",P},{"layer",8},{"valid_candidates_per_sequence",valid},{"cold_positions_per_sequence",cold},{"hot_positions_per_sequence",valid-cold},{"cold_fraction_prompt",.5},{"cold_fraction_valid",double(cold)/valid},{"query_ready_ns",ready},{"all_sequences_GPU_ready_ns",end},{"T_batch_delivery_ms",(end-ready)/1e6},{"sequence_events",rows},{"verification",verification},{"GPU_attention",batch_attention},{"independent_B1_graph_gate",independent},{"host_reservation_bytes",reserved},{"GPU_workspace_bytes",gpu.gpu_bytes()},{"host_peak_RSS_KiB",usage.ru_maxrss},{"whole_model_feedback",false},{"TPOT_ms",nullptr},{"GPU_actual_wait_ms",nullptr}};
   records<<result.dump()<<'\n';records.flush();std::cout<<result.dump()<<'\n';
  }return 0;
 }catch(const std::exception&e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
