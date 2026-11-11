// Bounded real-file HKM/APP replay. No GPU, frozen captured query, no TPOT claim.
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
constexpr size_t P=32768,N=P+1,H=32,KH=8,D=128,GROUP=4,UNIT=64;
constexpr size_t KEY_WORDS=1024,KEY_BYTES=2048,KV_BYTES=4096;
constexpr unsigned CPU_THREADS=32;
constexpr uint64_t HOST_LIMIT=4ULL<<30,BO_LIMIT=256ULL<<20;
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
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
 size_t episodes=3,hot_blocks=0;
 std::vector<size_t> sequences{0,1,2,3,4,5,6,7};
};
std::string affinity(){
 std::ifstream f("/proc/self/status");std::string line;
 while(std::getline(f,line))if(line.rfind("Cpus_allowed_list:",0)==0)return line.substr(line.find(':')+1);
 return "unavailable";
}
Options options(int argc,char** argv,int first){
 Options o;
 for(int i=first;i<argc;i+=2){need(i+1<argc,"option requires value");std::string k=argv[i],v=argv[i+1];
  if(k=="--cpu-impl")o.cpu_impl=v;
  else if(k=="--hot-threads")o.hot_threads=std::stoul(v);
  else if(k=="--cold-threads")o.cold_threads=std::stoul(v);
  else if(k=="--episodes")o.episodes=std::stoul(v);
  else if(k=="--hot-blocks")o.hot_blocks=std::stoul(v);
  else if(k=="--sequences"){
   o.sequences.clear();std::stringstream ss(v);std::string part;std::set<size_t> seen;
   while(std::getline(ss,part,',')){size_t id=std::stoul(part);need(id<8&&seen.insert(id).second,"unique sequence IDs0..7 required");o.sequences.push_back(id);}
  }else throw std::runtime_error("unknown option "+k);
 }
 need(o.cpu_impl=="legacy"||o.cpu_impl=="scalar"||o.cpu_impl=="avx2"||o.cpu_impl=="auto","explicit known CPU implementation required");
 need(o.hot_threads>=1&&o.hot_threads<=CPU_THREADS&&o.cold_threads>=1&&o.cold_threads<=CPU_THREADS,"CPU thread budget1..32");
 need(o.episodes>=1&&o.episodes<=20&&!o.sequences.empty(),"bounded episodes and nonempty sequences required");
 return o;
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
  auto sel=policy::select_gqa_scores(scaled(oracle),H,N,N,5,.2f);expected_ids=flatten(sel);selected=physical(sel);
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
void cpu_bench(const json& cfg,const Options& o,const std::filesystem::path& out){
 const size_t blocks=o.hot_blocks?o.hot_blocks:256;need(blocks>=16&&blocks<=512,"CPU benchmark hot blocks16..512");
 CpuSession cpu(o,o.hot_threads);std::ofstream f(out/"cpu_bench.jsonl");json rows=json::array();
 for(auto sequence:o.sequences){State s(cfg["sequences"][sequence],blocks);auto ids=candidates(s.placement.hot_ids(),false);std::vector<uint16_t> raw(H*N);
  for(size_t round=0;round<=o.episodes;++round){auto t=score_hot(s,ids,cpu,raw);verify_subset(s,raw,ids);
   json row={{"status","passed"},{"classification","real_capture_resident_CPU_score_component"},{"sequence",sequence},{"round",round},{"warmup",round==0},{"phase",round==0?"warmup":"measured"},{"CPU_scorer",cpu.name()},{"CPU_threads",o.hot_threads},{"CPU_affinity",affinity()},{"tokens",ids.size()},{"K_payload_bytes",ids.size()*KEY_BYTES},{"query_bytes",s.q.size()*2},{"score_output_bytes",ids.size()*H*2},{"prepare_ms",t.prepare_ms},{"score_ms",t.score_ms},{"merge_ms",t.merge_ms},{"total_ms",t.total_ms},{"packed_key_bytes",t.packed_key_bytes},{"score_bit_errors",0},{"scorer_workspace_bytes",cpu.scorer?cpu.scorer->workspace_bytes():0},{"reusable_score_capacity_bytes",cpu.values.capacity()*2},{"GPU_used",false},{"SSD_read_in_timer",false}};
   f<<row.dump()<<'\n';f.flush();rows.push_back(row);
  }
 }
 struct rusage ru{};getrusage(RUSAGE_SELF,&ru);need(uint64_t(ru.ru_maxrss)*1024<=HOST_LIMIT,"CPU benchmark RSS budget");
 write_json(out/"summary.json",{{"status","passed"},{"mode","cpu-bench"},{"CPU_scorer",cpu.name()},{"CPU_hot_threads",o.hot_threads},{"CPU_affinity_expected",affinity()},{"sequences",o.sequences},{"measured_episodes",o.episodes},{"peak_RSS_KiB_Linux",ru.ru_maxrss},{"host_limit_bytes",HOST_LIMIT},{"pool_creation_excluded",bool(cpu.scorer)},{"legacy_per_call_thread_creation_included",!cpu.scorer},{"rows",rows}});
}
struct Gather {std::vector<uint8_t> payload;uint64_t cold_bytes=0,hot_bytes=0,cold_selected=0,hot_selected=0;double hot_ms=0,cold_ms=0;};
Gather gather(State& s,const std::vector<uint64_t>& ids){Gather r;r.payload.resize(ids.size()*KV_BYTES);for(size_t j=0;j<ids.size();){auto t=ids[j];bool hot=t==P||s.placement.is_hot(t/UNIT);auto start=Clock::now();if(hot){auto* p=t==P?s.last.data():s.hot.at(t/UNIT).data()+(t%UNIT)*KV_BYTES;std::memcpy(r.payload.data()+j*KV_BYTES,p,KV_BYTES);r.hot_ms+=ms(start);r.hot_bytes+=KV_BYTES;++r.hot_selected;++j;}else{size_t end=j+1;while(end<ids.size()&&end-j<UNIT&&ids[end]==ids[end-1]+1&&ids[end]<P&&!s.placement.is_hot(ids[end]/UNIT))++end;size_t len=(end-j)*KV_BYTES;std::memcpy(r.payload.data()+j*KV_BYTES,s.kv.read(t*KV_BYTES,len),len);r.cold_ms+=ms(start);r.cold_bytes+=len;r.cold_selected+=end-j;j=end;}}return r;}
json plan_json(const AppPartitionPlan& p){return {{"continuous_beta",p.continuous_beta},{"continuous_rho",p.continuous_rho},{"hot_prompt_tokens",p.hot_candidates},{"cold_prompt_tokens",p.cold_candidates},{"actual_beta",p.actual_beta?json(*p.actual_beta):json(nullptr)},{"actual_rho",p.actual_rho},{"hot_ms_per_candidate",p.hot_total_ms_per_candidate},{"cold_ms_per_candidate",p.cold_total_ms_per_candidate},{"predicted_hot_ms",p.predicted_hot_ms},{"predicted_cold_ms",p.predicted_cold_ms},{"prediction_scope","partition_cost_estimate_not_delivery_or_speedup"},{"hot_budget_clamped",p.hot_budget_clamped},{"alignment_adjusted",p.alignment_adjusted},{"new_decode_token_extra_always_hot",1},{"hot_bytes_including_extra",(p.hot_candidates+1)*KV_BYTES},{"host_total_limit",HOST_LIMIT},{"measurement_endpoint","selected_KV_host_ready_no_GPU"}};}
HardwareContract contract(const json& cfg){HardwareContract c;c.abi=cfg.at("fpga_abi");c.image_path=cfg["fpga_image"].at("path");c.image_sha256=cfg["fpga_image"].at("sha256");c.image_execute_authorized=cfg["fpga_image"].at("execute_authorized");c.buffer_budget_bytes=BO_LIMIT;c.reuse_file_buffers=true;c.direct_read_queue_depth=cfg.value("hkm_io_queue_depth",1u);need(c.abi=="HS_FULL_GQA_PIPELINE_F16_V3"&&c.image_sha256=="679e85f964bf644cb1fe3bd5752b5a2452d288106e8139b46d702aafbdb2dcd6","existing validated V3 only");return c;}
void verify_payload(const State& s,const std::vector<uint64_t>& ids,const Gather& g){need(checksum(g.payload.data(),g.payload.size())==s.payload_hash,"selected KV checksum");std::ifstream f(std::filesystem::path(s.identity.at("trace").get<std::string>())/"kv-physical.bin",std::ios::binary);std::array<uint8_t,KV_BYTES> b{};for(size_t i=0;i<ids.size();++i){f.seekg(ids[i]*KV_BYTES);f.read(reinterpret_cast<char*>(b.data()),b.size());need(bool(f)&&!std::memcmp(b.data(),g.payload.data()+i*KV_BYTES,KV_BYTES),"selected KV original-capture byte mismatch");}}
json profile(const json& cfg,FpgaScoreBackend& fpga,const Options& o){
 CpuSession cpu(o,o.hot_threads);State s(cfg["sequences"][o.sequences.front()],256);auto hot_ids=candidates(s.placement.hot_ids(),false),cold_ids=candidates(s.placement.cold_ids(),false);auto rr=regions(s.keys,s.placement.cold_ids());
 json samples=json::array();double hot_sum=0;std::map<size_t,double> cold_sum;double gh=0,gc=0;uint64_t nh=0,nc=0;
 for(size_t repeat=0;repeat<=o.episodes;++repeat){std::vector<uint16_t> hot_raw(H*N);auto timing=score_hot(s,hot_ids,cpu,hot_raw);double elapsed=timing.total_ms;verify_subset(s,hot_raw,hot_ids);if(repeat)hot_sum+=elapsed;samples.push_back({{"kind","hot_score"},{"scope","cached_KV_views_or_pack_score_merge_to_global_score_ready"},{"CPU_scorer",cpu.name()},{"CPU_threads",o.hot_threads},{"repeat",repeat},{"warmup",repeat==0},{"ms",elapsed},{"candidates",hot_ids.size()},{"CPU_timing",cpu_timing_json(timing)}});auto t=Clock::now();
  for(size_t chunk:{size_t(2048),size_t(4096),size_t(8192),size_t(16384)}){std::vector<uint16_t> raw(H*N);t=Clock::now();auto r=fpga.score_file_regions_raw_gqa_stream(s.q,rr,H,KH,D,chunk,[&](const FileScoreChunk& c){merge_scores(raw,c.values,cold_ids,c.info.token_offset,c.info.tokens);});double wall=ms(t);for(size_t h=0;h<H;++h)for(auto id:cold_ids)need(raw[h*N+id]==s.oracle[h*N+id],"FPGA profile score identity");if(repeat)cold_sum[chunk]+=wall;samples.push_back({{"kind","cold_file_score_return_merge"},{"chunk",chunk},{"repeat",repeat},{"warmup",repeat==0},{"ms",wall},{"candidates",cold_ids.size()},{"stream",stream_json(r)}});}
  auto g=gather(s,s.selected);verify_payload(s,s.selected,g);if(repeat){gh+=g.hot_ms;gc+=g.cold_ms;nh+=g.hot_selected;nc+=g.cold_selected;}samples.push_back({{"kind","selected_KV_host_ready"},{"repeat",repeat},{"warmup",repeat==0},{"hot_ms",g.hot_ms},{"cold_ms",g.cold_ms},{"hot_selected",g.hot_selected},{"cold_selected",g.cold_selected}});
 }
 auto best=*std::min_element(cold_sum.begin(),cold_sum.end(),[](const auto& a,const auto& b){return a.second<b.second;});
 AppPartitionProfile p;p.hot_score={hot_sum,hot_ids.size()*o.episodes,"profile measured hot prepare/score/merge rounds",AppCostScope::host_hot_score_to_cpu_ready};p.cold_read_score_return={best.second,cold_ids.size()*o.episodes,"profile measured real file V3 stream including merge",AppCostScope::cold_file_read_score_return_to_cpu_ready};
 need(nh&&nc,"calibration requires selected KV in both partitions");p.hot_selected_kv=AppMeasuredCost{gh,nh,"profile sequence actual hot selected KV copy",AppCostScope::selected_kv_transfer_only};p.cold_selected_kv=AppMeasuredCost{gc,nc,"profile sequence actual O_DIRECT selected KV read",AppCostScope::selected_kv_transfer_only};
 p.hot_retention=double(nh)/(o.episodes*(hot_ids.size()+1));p.cold_retention=double(nc)/(o.episodes*cold_ids.size());p.retention_source="actual distinct selected physical token counts; hot denominator includes one current token; prompt-equivalent approximation";p.configuration_id="V3/raw128/FP16-FP32-FP16/B8/32K/CPU_"+std::string(cpu.name())+"/hotthreads"+std::to_string(o.hot_threads)+"/affinity"+affinity()+"/host4GiB/noGPU/IOQD"+std::to_string(cfg.value("hkm_io_queue_depth",1u));
 AppPartitionConstraints limits{P,P,16*UNIT,UNIT,UNIT,P};auto plan=plan_app_partition(p,limits);
 return {{"status","passed"},{"classification","real_component_profile_partition_estimate"},{"configuration_id",p.configuration_id},{"calibration_sequence",o.sequences.front()},{"CPU_scorer",cpu.name()},{"CPU_hot_threads",o.hot_threads},{"CPU_cold_threads",o.cold_threads},{"CPU_affinity_expected",affinity()},{"measured_episodes",o.episodes},{"chunk_tokens",best.first},{"profile",{{"hot_score_ms",hot_sum},{"hot_score_candidates",p.hot_score.work_items},{"cold_score_ms",best.second},{"cold_score_candidates",p.cold_read_score_return.work_items},{"hot_KV_ms",gh},{"cold_KV_ms",gc},{"hot_selected",nh},{"cold_selected",nc},{"hot_retention",p.hot_retention},{"cold_retention",p.cold_retention}}},{"plan",plan_json(plan)},{"samples",samples},{"GPU_transfer_in_profile",false},{"APP_cross_layer_pipeline",false},{"background_condition","concurrent_GPU_task_non_GPU_component_test"}};
}
json episode(State& s,FpgaScoreBackend* fpga,const std::string& method,size_t chunk,bool update,CpuSession& hot_cpu,CpuSession& cold_cpu){
 auto hu=s.placement.hot_ids(),cu=s.placement.cold_ids(),hi=candidates(hu,true),ci=candidates(cu,false);std::vector<uint16_t> raw(H*N);auto rbefore=s.keys.bytes;uint64_t host_start=0,host_end=0;FileScoreStreamReport stream;CpuTimings hot_timing,cold_timing;double cold_read_ms=0;
 auto begin=Clock::now();uint64_t query_ready=ns();auto hot_job=std::async(std::launch::async,[&]{host_start=ns();hot_timing=score_hot(s,hi,hot_cpu,raw);host_end=ns();});
 uint64_t cold_start=ns();
 if(method=="HOST_ALL_HOT"){need(ci.empty(),"all-hot control must have no cold candidates");}
 else if(method=="HOST_MATCHED"){
  auto read_start=Clock::now();size_t offset=0;for(auto reg:regions(s.keys,cu)){s.keys.read_to(static_cast<uint8_t*>(s.keys.buffer.p)+offset,reg.offset,reg.bytes);offset+=reg.bytes;}
  cold_read_ms=ms(read_start);auto t=Clock::now();cold_cpu.score(s.q,{{static_cast<const uint16_t*>(s.keys.buffer.p),ci.size(),KEY_WORDS}});cold_timing.score_ms=ms(t);t=Clock::now();merge_scores(raw,cold_cpu.values,ci,0,ci.size());cold_timing.merge_ms=ms(t);cold_timing.total_ms=cold_timing.score_ms+cold_timing.merge_ms;
 }else{need(fpga,"FPGA backend missing");stream=fpga->score_file_regions_raw_gqa_stream(s.q,regions(s.keys,cu),H,KH,D,method=="FPGA_SERIAL"?ci.size():chunk,[&](const FileScoreChunk& c){merge_scores(raw,c.values,ci,c.info.token_offset,c.info.tokens);});}
 auto cold_end=ns();hot_job.get();auto score_ready=ns();auto sel=policy::select_gqa_scores(scaled(raw),H,N,N,5,.2f);auto ids=physical(sel);auto id_ready=ns();auto g=gather(s,ids);auto kv_ready=ns();double delivery=ms(begin);uint64_t migration_bytes=0;HkmUpdate moves;double maintenance_ms=0;
 if(update){auto t=Clock::now();std::array<uint16_t,P/UNIT> counts{};for(auto id:ids)if(id<P)++counts[id/UNIT];std::vector<std::pair<uint64_t,uint16_t>> weights;for(size_t u=0;u<counts.size();++u)if(counts[u])weights.push_back({u,counts[u]});moves=s.placement.update_weighted(weights,P/UNIT-1);for(auto u:moves.demote)s.hot.erase(u);for(auto u:moves.promote){s.promote(u);migration_bytes+=UNIT*KV_BYTES;}maintenance_ms=ms(t);}
 double wall=ms(begin); // Completes actual cache moves, no deferred work hidden.
 need(raw==s.oracle,"complete score vector differs from independent original capture");need(flatten(sel)==s.expected_ids,"ordered selected IDs differ");verify_payload(s,ids,g);
 return {{"status","passed"},{"method",method},{"classification","real_single_layer_retrieval_to_host_KV_ready_frozen_query"},{"query_ready_ns",query_ready},{"hot_started_ns",host_start},{"hot_finished_ns",host_end},{"cold_started_ns",cold_start},{"cold_finished_ns",cold_end},{"all_scores_ready_ns",score_ready},{"IDs_ready_ns",id_ready},{"KV_host_ready_ns",kv_ready},{"retrieval_to_host_ms",delivery},{"episode_including_HPM_ms",wall},{"hot_prompt_tokens",hu.size()*UNIT},{"cold_prompt_tokens",ci.size()},{"beta",cu.empty()?json(nullptr):json(double(hu.size())/cu.size())},{"hot_score_ms",double(host_end-host_start)/1e6},{"CPU_scorer",hot_cpu.name()},{"CPU_hot_threads",hot_cpu.threads},{"CPU_cold_threads",method=="HOST_MATCHED"?cold_cpu.threads:0},{"CPU_hot_timing",cpu_timing_json(hot_timing)},{"CPU_scorer_workspace_bytes",(hot_cpu.scorer?hot_cpu.scorer->workspace_bytes():0)+(cold_cpu.scorer?cold_cpu.scorer->workspace_bytes():0)},{"CPU_reusable_score_capacity_bytes",(hot_cpu.values.capacity()+cold_cpu.values.capacity())*2},{"CPU_cold_timing",cpu_timing_json(cold_timing)},{"CPU_cold_read_ms",cold_read_ms},{"cold_score_wall_ms",double(cold_end-cold_start)/1e6},{"host_cold_K_bytes",s.keys.bytes-rbefore},{"FPGA_stream",stream_json(stream)},{"selected_per_head",sel.k},{"selected_physical_tokens",ids.size()},{"hot_selected_KV_bytes",g.hot_bytes},{"cold_selected_KV_bytes",g.cold_bytes},{"hot_KV_copy_ms",g.hot_ms},{"cold_KV_read_ms",g.cold_ms},{"HPM_maintenance_ms",maintenance_ms},{"HPM_promoted_units",moves.promote},{"HPM_demoted_units",moves.demote},{"HPM_migration_read_bytes",migration_bytes},{"hot_set_checksum",std::to_string(checksum(hu.data(),hu.size()*8))},{"score_checksum",std::to_string(checksum(raw.data(),raw.size()*2))},{"KV_payload_checksum",std::to_string(checksum(g.payload.data(),g.payload.size()))},{"score_bit_errors",0},{"ordered_ID_errors",0},{"KV_byte_errors",0},{"GPU_attention_executed",false},{"actual_model_feedback",false},{"TPOT_ms",nullptr},{"GPU_delivery_ms",nullptr}};
}
} // namespace
int main(int argc,char** argv){try{
 need(argc>=4,"usage: binary profile|cpu-bench manifest output [options] | binary run manifest output method profile.json [options]");
 std::string mode=argv[1],manifest=argv[2],out=argv[3];
 need(mode=="profile"||mode=="cpu-bench"||mode=="run","known mode required");
 need(mode!="run"||argc>=6,"run requires method and profile");
 std::string method=mode=="run"?argv[4]:"";auto o=options(argc,argv,mode=="run"?6:4);auto cfg=read_json(manifest);
 need(!std::filesystem::exists(out),"new output directory required");std::filesystem::create_directories(out);
 need(cfg.at("B")==8&&cfg.at("P")==P&&cfg.at("layer")==8&&cfg["sequences"].size()==8,"frozen real B8/P32K/layer8 inputs required");
 if(mode=="cpu-bench"){cpu_bench(cfg,o,out);std::cout<<"passed CPU benchmark\n";return 0;}
 need(mode=="profile"||method=="HOST_MATCHED"||method=="FPGA_SERIAL"||method=="FPGA_CSS"||method=="FPGA_CSS_HPM"||method=="HOST_ALL_HOT","known explicit method required");
 std::unique_ptr<FpgaScoreBackend> fpga;if(mode=="profile"||(method!="HOST_MATCHED"&&method!="HOST_ALL_HOT"))fpga=make_fpga_backend(HardwareMode::fpga_storage_streaming,contract(cfg));
 if(mode=="profile"){auto j=profile(cfg,*fpga,o);write_json(std::filesystem::path(out)/"profile.json",j);std::cout<<j["plan"].dump()<<'\n';return 0;}
 auto cal=read_json(argv[5]);size_t hot=cal["plan"]["hot_prompt_tokens"],chunk=cal["chunk_tokens"];
 if(o.hot_blocks)hot=o.hot_blocks*UNIT;
 if(method=="HOST_ALL_HOT"){need(!o.hot_blocks||o.hot_blocks==512,"all-hot control requires512blocks");hot=P;}
 need(hot>=16*UNIT&&hot<=(method=="HOST_ALL_HOT"?P:P-UNIT)&&hot%UNIT==0,"admitted APP partition");
 // Reserve selected sequence hot pools, aligned CPU cold buffers, oracle scores,
 // scorer outputs and validation temporaries. No all-KV hot shadow survives setup.
 uint64_t reservation=o.sequences.size()*((hot+1)*KV_BYTES+P*KEY_BYTES+H*N*2+P/UNIT*512)+N*KV_BYTES+N*KEY_BYTES+H*N*2*4+BO_LIMIT;
 need(reservation<=HOST_LIMIT,"managed host admission");std::vector<std::unique_ptr<State>> states;
 for(auto id:o.sequences)states.push_back(std::make_unique<State>(cfg["sequences"][id],hot/UNIT,cfg.value("hkm_max_swaps_per_episode",size_t(8))));
 CpuSession hot_cpu(o,o.hot_threads),cold_cpu(o,o.cold_threads);
 std::ofstream f(std::filesystem::path(out)/"runs.jsonl");json summary=json::array();
 for(size_t round=0;round<=o.episodes;++round){auto start=Clock::now();json rows=json::array();double sum=0;
  for(size_t i=0;i<states.size();++i){auto sequence=o.sequences[i];auto row=episode(*states[i],fpga.get(),method,chunk,round>0&&method=="FPGA_CSS_HPM",hot_cpu,cold_cpu);row["sequence"]=sequence;row["round"]=round;row["phase"]=round==0?"warmup":"measured_cache_episode";row["calibration_input"]=sequence<2;sum+=row["episode_including_HPM_ms"].get<double>();f<<row.dump()<<'\n';f.flush();rows.push_back(row);}
  summary.push_back({{"round",round},{"phase",round==0?"warmup":"measured_cache_episode"},{"sum_of_actual_sequential_episode_wall_ms",sum},{"outer_wall_including_excluded_validation_ms",ms(start)},{"rows",rows}});
 }
 struct rusage ru{};getrusage(RUSAGE_SELF,&ru);need(uint64_t(ru.ru_maxrss)*1024<=HOST_LIMIT,"observed RSS exceeds host budget");
 write_json(std::filesystem::path(out)/"summary.json",{{"status","passed"},{"method",method},{"profile_source",argv[5]},{"actual_hot_prompt_tokens",hot},{"chunk_tokens",chunk},{"HPM_granularity_tokens",UNIT},{"recent_pin_tokens",16*UNIT},{"max_migrations_per_episode_tokens",cfg.value("hkm_max_swaps_per_episode",size_t(8))*UNIT},{"managed_host_reservation_bytes",reservation},{"host_limit_bytes",HOST_LIMIT},{"peak_RSS_KiB_Linux",ru.ru_maxrss},{"CPU_affinity_expected",affinity()},{"CPU_scorer",hot_cpu.name()},{"CPU_hot_threads",o.hot_threads},{"CPU_cold_threads",method=="HOST_MATCHED"?o.cold_threads:0},{"sequences",o.sequences},{"measured_episodes",o.episodes},{"pool_creation_excluded",bool(hot_cpu.scorer)},{"legacy_per_call_thread_creation_included",!hot_cpu.scorer},{"single_layer_budget_not_full_model_budget",true},{"all_hot_resident_control",method=="HOST_ALL_HOT"},{"query_semantics","same real captured query replayed to exercise cache state; not new decode steps"},{"APP_partition_applied",true},{"APP_cross_layer_pipeline",false},{"rounds",summary}});
 std::cout<<"passed "<<method<<" hot="<<hot<<" cold="<<P-hot<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<'\n';return 1;}}
