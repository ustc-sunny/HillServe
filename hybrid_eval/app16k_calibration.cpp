// Minimal B1/16K offline APP calibration using the existing HKM/APP implementation.
// Real capture, real O_DIRECT + V3 score stream. No GPU or delivery-time claim.
#include "core/app_partition.hpp"
#include "core/hkm.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "policy/full_gqa_cpu.hpp"
#include "storage/file_store.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace hillserve;
namespace {
using json=nlohmann::ordered_json;using Clock=std::chrono::steady_clock;
constexpr size_t P=16384,N=P+1,H=32,KH=8,D=128,GROUP=4,UNIT=64;
constexpr size_t KEY_WORDS=1024,KEY_BYTES=2048,KV_BYTES=4096;
constexpr uint64_t HOST_LIMIT=4ULL<<30,BO_LIMIT=256ULL<<20;
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
json read_json(const std::string& p){std::ifstream f(p);need(bool(f),"missing JSON "+p);json j;f>>j;return j;}
void write_json(const std::filesystem::path& p,const json& j){std::ofstream f(p);need(bool(f),"output JSON open");f<<j.dump(2)<<'\n';need(bool(f),"output JSON write");}
template<class T> std::vector<T> load(const std::filesystem::path& p,size_t n){need(std::filesystem::file_size(p)==n*sizeof(T),"input extent "+p.string());std::vector<T> v(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));need(bool(f),"short input");return v;}
struct Aligned{void* p=nullptr;size_t size;explicit Aligned(size_t n):size(n){need(n%4096==0&&!posix_memalign(&p,4096,n),"aligned allocation");}~Aligned(){free(p);}Aligned(const Aligned&)=delete;};
struct Direct{
 int fd=-1;uint64_t bytes=0,calls=0,extent;Aligned buffer;
 Direct(const std::string& path,size_t expected,size_t capacity):extent(expected),buffer(capacity){
#ifdef O_DIRECT
 fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_DIRECT);
#else
 throw std::runtime_error("O_DIRECT required; no fallback");
#endif
 need(fd>=0,"open direct "+path);struct stat st{};need(!fstat(fd,&st)&&S_ISREG(st.st_mode)&&uint64_t(st.st_size)==extent,"direct extent");}
 ~Direct(){if(fd>=0)close(fd);}Direct(const Direct&)=delete;
 void read_to(void* dst,size_t off,size_t n){need(n&&off%4096==0&&n%4096==0&&off<=extent&&n<=extent-off&&uintptr_t(dst)%4096==0,"direct read alignment/bounds");ssize_t got;do{got=pread(fd,dst,n,off);}while(got<0&&errno==EINTR);need(got==ssize_t(n),"short O_DIRECT read");bytes+=n;++calls;}
 const uint8_t* read(size_t off,size_t n){need(n<=buffer.size,"direct buffer bound");read_to(buffer.p,off,n);return static_cast<uint8_t*>(buffer.p);}
};

struct Options {
 std::string cpu_impl="legacy";
 unsigned hot_threads=3,cold_threads=1;
 size_t episodes=1,hot_blocks=0;
 std::vector<size_t> sequences{0};
};
std::string affinity(){
 std::ifstream f("/proc/self/status");std::string line;
 while(std::getline(f,line))if(line.rfind("Cpus_allowed_list:",0)==0)return line.substr(line.find(':')+1);
 return "unavailable";
}
struct CpuSession {
 std::unique_ptr<FullGqaCpuScorer> scorer;
 unsigned threads;
 std::vector<uint16_t> values;
 CpuSession(const Options& o,unsigned count):threads(count){if(o.cpu_impl!="legacy"){
  auto impl=o.cpu_impl=="scalar"?FullGqaCpuImplementation::scalar:o.cpu_impl=="avx2"?FullGqaCpuImplementation::avx2_f16c:FullGqaCpuImplementation::automatic;
  scorer=std::make_unique<FullGqaCpuScorer>(threads,impl);
 }}
 const char* name()const{return scorer?scorer->implementation_name():full_gqa_legacy_cpu_name;}
 void score(const std::vector<uint16_t>& q,const std::vector<FullGqaKeyView>& views){
  if(scorer){scorer->score(q,views,values);return;}
  need(views.size()==1,"legacy scorer requires explicitly packed single view");
  values=score_full_gqa_legacy_cpu(q,views[0].data,views[0].tokens,views[0].token_stride_words,threads);
 }
};
struct CpuTimings {double prepare_ms=0,score_ms=0,merge_ms=0,total_ms=0;uint64_t packed_key_bytes=0;};
json cpu_timing_json(const CpuTimings& t){return {{"prepare_ms",t.prepare_ms},{"score_ms",t.score_ms},{"merge_ms",t.merge_ms},{"total_ms",t.total_ms},{"packed_key_bytes",t.packed_key_bytes}};}

std::vector<uint16_t> scaled(const std::vector<uint16_t>& s){auto r=s;for(auto& x:r)x=half_bits(half_value(x)/std::sqrt(128.f));return r;}
std::vector<uint64_t> flatten(const policy::GQASelection& s){std::vector<uint64_t> out;for(const auto& h:s.ids)out.insert(out.end(),h.begin(),h.end());return out;}
std::vector<uint64_t> physical(const policy::GQASelection& s){auto out=flatten(s);std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;}
json stream_json(const FileScoreStreamReport& r){json chunks=json::array();for(const auto& c:r.chunks)chunks.push_back({{"offset",c.token_offset},{"tokens",c.tokens},{"read_started_ns",c.read_started_ns},{"read_finished_ns",c.read_finished_ns},{"kernel_started_ns",c.kernel_started_ns},{"kernel_finished_ns",c.kernel_finished_ns},{"scores_CPU_ready_ns",c.scores_CPU_ready_ns},{"callback_started_ns",c.callback_started_ns},{"callback_finished_ns",c.callback_finished_ns},{"read_ms",c.file_read_ms},{"launch_wait_ms",c.launch_wait_ms},{"d2h_ms",c.d2h_ms}});return {{"total_ms",r.total_ms},{"file_read_bytes",r.file_read_bytes},{"query_h2d_bytes",r.query_h2d_bytes},{"score_d2h_bytes",r.score_d2h_bytes},{"device_buffer_bytes",r.device_buffer_bytes},{"host_score_buffer_bytes",r.host_score_buffer_bytes},{"direct_read_queue_depth",r.direct_read_queue_depth},{"aio_peak_inflight",r.aio_peak_inflight},{"file_read_operations",r.file_read_operations},{"aio_submit_calls",r.aio_submit_calls},{"aio_completions",r.aio_completions},{"aio_context_reused",r.aio_context_reused},{"aio_setup_ms",r.aio_setup_ms},{"buffers_reused",r.buffers_reused},{"p2p_verified",r.p2p_allocation_verified},{"chunks",chunks}};}
std::vector<ScoreFileRegion> regions(Direct& file,const std::vector<uint64_t>& units){std::vector<ScoreFileRegion> r;for(auto u:units){uint64_t off=u*UNIT*KEY_BYTES;if(!r.empty()&&r.back().offset+r.back().bytes==off)r.back().bytes+=UNIT*KEY_BYTES;else r.push_back({file.fd,off,UNIT*KEY_BYTES,false});}return r;}
std::vector<uint64_t> candidates(const std::vector<uint64_t>& units,bool last){std::vector<uint64_t> ids;for(auto u:units)for(size_t j=0;j<UNIT;++j)ids.push_back(u*UNIT+j);if(last)ids.push_back(P);return ids;}
void merge_scores(std::vector<uint16_t>& raw,const std::vector<uint16_t>& v,const std::vector<uint64_t>& ids,size_t off,size_t count){need(v.size()==H*count&&off+count<=ids.size(),"merge score shape");for(size_t h=0;h<H;++h)for(size_t j=0;j<count;++j)raw[h*N+ids[off+j]]=v[h*count+j];}
struct State{
 std::vector<uint16_t> q,oracle;std::vector<uint64_t> expected_ids,selected;
 uint64_t payload_hash=0;json identity;
 Direct keys,kv;ApproxHkm placement;
 std::map<uint64_t,std::vector<uint8_t>> hot;
 alignas(16) std::array<uint8_t,KV_BYTES> last{};
 State(const json& e,size_t hot_blocks,size_t swaps=8):identity(e),keys(e.at("full_K_file"),P*KEY_BYTES,P*KEY_BYTES),kv(e.at("full_KV_file"),N*KV_BYTES,UNIT*KV_BYTES),placement(HkmConfig{P/UNIT,hot_blocks,std::min(size_t(16),hot_blocks),std::min(swaps,hot_blocks),8},P/UNIT-1){
  auto f=load<float>(std::filesystem::path(e.at("trace").get<std::string>())/"rehearsal-query-f32.bin",H*D);for(float x:f){need(std::isfinite(x),"query nonfinite");q.push_back(half_bits(x));}
  // Independent original physical capture supplies expected scores/selected bytes.
  auto original=load<uint16_t>(std::filesystem::path(e.at("trace").get<std::string>())/"kv-physical.bin",N*KEY_WORDS*2);
  oracle.resize(H*N);
  for(size_t t=0;t<N;++t)for(size_t h=0;h<H;++h){volatile float sum=0;for(size_t d=0;d<D;++d){volatile float mul=half_value(q[h*D+d])*half_value(original[t*KEY_WORDS*2+(h/GROUP)*D+d]);sum=sum+mul;}oracle[h*N+t]=half_bits(sum);}
  auto sel=policy::select_gqa_scores(scaled(oracle),H,N,N,10,.2f);expected_ids=flatten(sel);selected=physical(sel);
  std::vector<uint8_t> payload(selected.size()*KV_BYTES);for(size_t i=0;i<selected.size();++i)std::memcpy(payload.data()+i*KV_BYTES,reinterpret_cast<uint8_t*>(original.data())+selected[i]*KV_BYTES,KV_BYTES);
  payload_hash=checksum(payload.data(),payload.size());
  for(auto u:placement.hot_ids())promote(u);
  std::memcpy(last.data(),kv.read(P*KV_BYTES,KV_BYTES),KV_BYTES);
 }
 void promote(uint64_t u){auto* p=kv.read(u*UNIT*KV_BYTES,UNIT*KV_BYTES);hot[u]=std::vector<uint8_t>(p,p+UNIT*KV_BYTES);}
 std::vector<uint16_t> hot_keys(const std::vector<uint64_t>& ids)const{std::vector<uint16_t> keys_(ids.size()*KEY_WORDS);for(size_t i=0;i<ids.size();++i){auto t=ids[i];const auto* src=t==P?last.data():hot.at(t/UNIT).data()+(t%UNIT)*KV_BYTES;std::memcpy(keys_.data()+i*KEY_WORDS,src,KEY_BYTES);}return keys_;}
};
std::vector<FullGqaKeyView> hot_views(const State& s,const std::vector<uint64_t>& ids){
 std::vector<FullGqaKeyView> views;views.reserve(ids.size()/UNIT+2);
 for(size_t j=0;j<ids.size();){const auto t=ids[j];
  if(t==P){views.push_back({reinterpret_cast<const uint16_t*>(s.last.data()),1,KEY_WORDS*2});++j;continue;}
  size_t end=j+1;while(end<ids.size()&&ids[end]==ids[end-1]+1&&ids[end]/UNIT==t/UNIT)++end;
  views.push_back({reinterpret_cast<const uint16_t*>(s.hot.at(t/UNIT).data()+(t%UNIT)*KV_BYTES),end-j,KEY_WORDS*2});j=end;
 }
 return views;
}
CpuTimings score_hot(State& s,const std::vector<uint64_t>& ids,CpuSession& cpu,std::vector<uint16_t>& raw){
 CpuTimings r;auto begin=Clock::now(),t=begin;std::vector<uint16_t> packed;std::vector<FullGqaKeyView> views;
 if(cpu.scorer)views=hot_views(s,ids);
 else{packed=s.hot_keys(ids);r.packed_key_bytes=packed.size()*2;views.push_back({packed.data(),ids.size(),KEY_WORDS});}
 r.prepare_ms=ms(t);t=Clock::now();cpu.score(s.q,views);r.score_ms=ms(t);
 t=Clock::now();merge_scores(raw,cpu.values,ids,0,ids.size());r.merge_ms=ms(t);r.total_ms=ms(begin);return r;
}
void verify_subset(const State& s,const std::vector<uint16_t>& raw,const std::vector<uint64_t>& ids){
 for(size_t h=0;h<H;++h)for(auto id:ids)need(raw[h*N+id]==s.oracle[h*N+id],"CPU score bits differ from independent original capture");
}
struct Gather {std::vector<uint8_t> payload;uint64_t cold_bytes=0,hot_bytes=0,cold_selected=0,hot_selected=0;double hot_ms=0,cold_ms=0;};
Gather gather(State& s,const std::vector<uint64_t>& ids){Gather r;r.payload.resize(ids.size()*KV_BYTES);for(size_t j=0;j<ids.size();){auto t=ids[j];bool hot=t==P||s.placement.is_hot(t/UNIT);auto start=Clock::now();if(hot){auto* p=t==P?s.last.data():s.hot.at(t/UNIT).data()+(t%UNIT)*KV_BYTES;std::memcpy(r.payload.data()+j*KV_BYTES,p,KV_BYTES);r.hot_ms+=ms(start);r.hot_bytes+=KV_BYTES;++r.hot_selected;++j;}else{size_t end=j+1;while(end<ids.size()&&end-j<UNIT&&ids[end]==ids[end-1]+1&&ids[end]<P&&!s.placement.is_hot(ids[end]/UNIT))++end;size_t len=(end-j)*KV_BYTES;std::memcpy(r.payload.data()+j*KV_BYTES,s.kv.read(t*KV_BYTES,len),len);r.cold_ms+=ms(start);r.cold_bytes+=len;r.cold_selected+=end-j;j=end;}}return r;}
json plan_json(const AppPartitionPlan& p){return {{"continuous_beta",p.continuous_beta},{"continuous_rho",p.continuous_rho},{"hot_prompt_tokens",p.hot_candidates},{"cold_prompt_tokens",p.cold_candidates},{"actual_beta",p.actual_beta?json(*p.actual_beta):json(nullptr)},{"actual_rho",p.actual_rho},{"hot_ms_per_candidate",p.hot_total_ms_per_candidate},{"cold_ms_per_candidate",p.cold_total_ms_per_candidate},{"predicted_hot_ms",p.predicted_hot_ms},{"predicted_cold_ms",p.predicted_cold_ms},{"prediction_scope","partition_cost_estimate_not_delivery_or_speedup"},{"hot_budget_clamped",p.hot_budget_clamped},{"alignment_adjusted",p.alignment_adjusted},{"new_decode_token_extra_always_hot",1},{"hot_bytes_including_extra",(p.hot_candidates+1)*KV_BYTES},{"host_total_limit",HOST_LIMIT},{"measurement_endpoint","profile_cost_scopes_explicit_in_parent_no_GPU"}};}
HardwareContract contract(const json& cfg){HardwareContract c;c.abi=cfg.at("fpga_abi");c.image_path=cfg["fpga_image"].at("path");c.image_sha256=cfg["fpga_image"].at("sha256");c.image_execute_authorized=cfg["fpga_image"].at("execute_authorized");c.buffer_budget_bytes=BO_LIMIT;c.reuse_file_buffers=true;c.direct_read_queue_depth=cfg.value("hkm_io_queue_depth",1u);need(c.abi=="HS_FULL_GQA_PIPELINE_F16_V3"&&c.image_sha256=="679e85f964bf644cb1fe3bd5752b5a2452d288106e8139b46d702aafbdb2dcd6","existing validated V3 only");return c;}
void verify_payload(const State& s,const std::vector<uint64_t>& ids,const Gather& g){need(checksum(g.payload.data(),g.payload.size())==s.payload_hash,"selected KV checksum");std::ifstream f(std::filesystem::path(s.identity.at("trace").get<std::string>())/"kv-physical.bin",std::ios::binary);std::array<uint8_t,KV_BYTES> b{};for(size_t i=0;i<ids.size();++i){f.seekg(ids[i]*KV_BYTES);f.read(reinterpret_cast<char*>(b.data()),b.size());need(bool(f)&&!std::memcmp(b.data(),g.payload.data()+i*KV_BYTES,KV_BYTES),"selected KV original-capture byte mismatch");}}

json profile16k(const json& cfg,FpgaScoreBackend& fpga,const Options& o){
 constexpr size_t chunk=4096;
 CpuSession cpu(o,o.hot_threads);State s(cfg.at("sequences").at(0),P/(2*UNIT));
 const auto hot_ids=candidates(s.placement.hot_ids(),false),cold_ids=candidates(s.placement.cold_ids(),false);
 const auto rr=regions(s.keys,s.placement.cold_ids());
 json samples=json::array();double hot_ms=0,cold_ms=0,hot_kv_ms=0,cold_kv_ms=0;
 uint64_t hot_selected=0,cold_selected=0;
 // One warmup and one actual measurement: no search over chunk, precision or CPU implementation.
 for(size_t round=0;round<2;++round){
  std::vector<uint16_t> raw(H*N);auto hot=score_hot(s,hot_ids,cpu,raw);verify_subset(s,raw,hot_ids);
  const auto begin=Clock::now();
  auto stream=fpga.score_file_regions_raw_gqa_stream(s.q,rr,H,KH,D,chunk,[&](const FileScoreChunk& part){
   merge_scores(raw,part.values,cold_ids,part.info.token_offset,part.info.tokens);
  });
  const double cold=ms(begin);verify_subset(s,raw,cold_ids);
  need(stream.p2p_allocation_verified&&stream.file_read_bytes==cold_ids.size()*KEY_BYTES,
       "real P2P BO file scoring extent required");
  auto payload=gather(s,s.selected);verify_payload(s,s.selected,payload);
  samples.push_back({{"phase",round?"measured":"warmup"},{"hot_candidates",hot_ids.size()},
   {"cold_candidates",cold_ids.size()},{"hot_timing",cpu_timing_json(hot)},
   {"cold_read_score_return_merge_ms",cold},{"stream",stream_json(stream)},
   {"hot_selected",payload.hot_selected},{"cold_selected",payload.cold_selected},
   {"hot_KV_ms",payload.hot_ms},{"cold_KV_ms",payload.cold_ms},
   {"hot_KV_bytes",payload.hot_bytes},{"cold_KV_bytes",payload.cold_bytes},
   {"score_bit_errors",0},{"selected_KV_byte_errors",0}});
  if(round){hot_ms=hot.total_ms;cold_ms=cold;hot_kv_ms=payload.hot_ms;cold_kv_ms=payload.cold_ms;
   hot_selected=payload.hot_selected;cold_selected=payload.cold_selected;}
 }
 AppPartitionProfile measured;
 measured.configuration_id="V3/raw128/F16-F32-F16/B1/P16384/layer8/legacy_hot3_cold1/affinity"+affinity()+"/chunk4096/QD1";
 measured.hot_score={hot_ms,hot_ids.size(),"held-out real capture: KV pack, legacy hot3 scoring, global score merge",AppCostScope::host_hot_score_to_cpu_ready};
 measured.cold_read_score_return={cold_ms,cold_ids.size(),"held-out real capture: O_DIRECT into P2P BO, V3 execution, score return and global score merge",AppCostScope::cold_file_read_score_return_to_cpu_ready};
 const AppPartitionConstraints limits{P,P,16*UNIT,UNIT,UNIT,P};
 const auto score_plan=plan_app_partition(measured,limits);
 json kv_plan=nullptr;
 if(hot_selected&&cold_selected){
  measured.hot_selected_kv=AppMeasuredCost{hot_kv_ms,hot_selected,"same selected physical IDs: actual RAM KV gather",AppCostScope::selected_kv_transfer_only};
  measured.cold_selected_kv=AppMeasuredCost{cold_kv_ms,cold_selected,"same selected physical IDs: actual O_DIRECT KV gather",AppCostScope::selected_kv_transfer_only};
  measured.hot_retention=double(hot_selected)/(hot_ids.size()+1);
  measured.cold_retention=double(cold_selected)/cold_ids.size();
  measured.retention_source="actual held-out selected physical token counts, no fabricated missing-side cost";
  kv_plan=plan_json(plan_app_partition(measured,limits));
 }
 struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
 need(uint64_t(usage.ru_maxrss)*1024<=HOST_LIMIT,"bounded component calibration RSS");
 return {{"status","passed"},{"classification","real_component_profile_partition_estimate"},
  {"calibration_input",cfg.at("sequences").at(0)},{"P",P},{"B",1},{"layer",8},
  {"configuration_id",measured.configuration_id},{"CPU_scorer",cpu.name()},
  {"selector_alpha",10},{"selector_cap_ratio",0.2},
  {"CPU_hot_threads",3},{"CPU_cold_threads_for_HOST_comparator",1},{"CPU_affinity",affinity()},
  {"FPGA_image_sha256",cfg.at("fpga_image").at("sha256")},{"chunk_tokens",chunk},
  {"score_service_balance_plan",plan_json(score_plan)},
  {"optional_selected_KV_weighted_APP_plan",kv_plan},
  {"selected_KV_both_sides_observed",hot_selected>0&&cold_selected>0},
  {"frozen_for_pilot_only",true},{"end_to_end_optimality_claim",false},
  {"GPU_transfer_in_profile",false},{"GPU_used",false},{"profile_peak_RSS_KiB",usage.ru_maxrss},
  {"profile_host_limit_bytes",HOST_LIMIT},{"formal_model_KV_resident_budget","set separately from aligned frozen hot tokens across32layers"},
  {"scoring_endpoint","all corresponding scores merged in CPU global score array"},
  {"timing_note","hot and cold services measured separately; costs balance branches, never summed into delivery latency"},
  {"measured_count",1},{"warmup_count",1},{"samples",samples}};
}
} // namespace

int main(int argc,char** argv){
 if(argc!=3){std::cerr<<"MANIFEST NEW_OUTPUT_DIRECTORY\n";return 2;}
 try{
  const auto cfg=read_json(argv[1]);need(cfg.at("P")==P&&cfg.at("B")==1&&cfg.at("calibration_only")==true,"held-out16K calibration manifest required");
  need(cfg.at("sequences").size()==1,"exactly one independent calibration capture required");
  auto allowed=affinity();allowed.erase(std::remove_if(allowed.begin(),allowed.end(),[](unsigned char c){return std::isspace(c);}),allowed.end());
  need(allowed=="0,2,4,6","fixed four physical P-core affinity 0,2,4,6 required");
  need(std::filesystem::create_directory(argv[2]),"new output directory required");
  const Options options;auto backend=make_fpga_backend(HardwareMode::fpga_storage_streaming,contract(cfg));
  auto result=profile16k(cfg,*backend,options);write_json(std::filesystem::path(argv[2])/"profile.json",result);
  std::cout<<result.at("score_service_balance_plan").dump()<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<'\n';return 1;}
}
