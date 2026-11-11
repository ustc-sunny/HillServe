#include "runtime/gqa_executor.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "policy/full_gqa_runtime.hpp"
#include "storage/file_store.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sys/resource.h>

using namespace hillserve;
namespace {
using Clock=std::chrono::steady_clock;
using json=nlohmann::ordered_json;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void need(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
std::string read(const std::filesystem::path& p){std::ifstream f(p);need(bool(f),"missing file: "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
void save(const std::filesystem::path& p,const json& j){std::ofstream f(p);f<<j.dump(2)<<'\n';need(bool(f),"cannot save "+p.string());}
std::uint64_t stamp(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
}

// One actual model request. B=1 is explicit; this is not an eight-sequence replay.
// New Q/K/V are produced by every live decode and every GPU attention consumes
// the external selected K/V. The two prologue layers retain the existing dense
// rule; the remaining thirty use the unchanged rehearsal/selector contract.
int main(int argc,char** argv){
 if(argc!=2){std::cerr<<"usage: hillserve-full-gqa-runtime CONFIG.json\n";return 2;}
 std::filesystem::path out;
 try{
  const auto cfg=json::parse(read(argv[1]));out=cfg.at("output").get<std::string>();
  need(std::filesystem::create_directory(out),"new output directory required");save(out/"config.json",cfg);
  const int np=cfg.at("prompt_tokens"),ng=cfg.at("output_tokens");
  need(cfg.value("batch",1)==1,"full model runner currently supports one real request, not replay batch");
  need(np>=128&&np<=32768&&np%64==0&&ng>=2&&ng<=64,"bounded full-model P128..32768 aligned64/G2..64");
  const int capacity=(np+ng+63)/64*64;
  const int chunk=cfg.value("prefill_chunk",np>8192?16:64);
  need(chunk>=1&&chunk<=256,"bounded prefill chunk");
  const bool verify=cfg.value("verify",false),early=cfg.value("early_rehearsal",false);
  const bool payload_prefetch=cfg.value("payload_prefetch",false);
  need(!payload_prefetch||early,"payload prefetch requires early rehearsal");
  auto prefetch_io_fence=std::make_shared<runtime::GQAPrefetchIOFence>();
  const bool low_gpu=cfg.value("functional_low_gpu",false);
  const int gpu_layers=cfg.value("gpu_layers",99);
  need(gpu_layers==99||(low_gpu&&gpu_layers==0),"explicit low-GPU functional mode or all-GPU weights required");
  const auto host_budget=cfg.value("host_budget_bytes",2ULL<<30);
  const auto gpu_budget=cfg.value("gpu_workspace_budget_bytes",1ULL<<30);
  const auto debug_bytes=verify?32ULL*capacity*4096:0;
  need(!verify||debug_bytes<=cfg.value("debug_shadow_budget_bytes",512ULL<<20),"explicit debug shadow budget exceeded");
  setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);setenv("HILLSERVE_GQA_DIRECT_IO","1",1);
  ggml_backend_load_all();llama_backend_init();
  auto t=Clock::now();auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(cfg.at("model").get<std::string>().c_str(),mp),llama_model_free);
  need(bool(model),"model loading failed");const double load_ms=ms(t);
  need(llama_model_n_layer(model.get())==32&&llama_model_n_head(model.get())==32&&llama_model_n_head_kv(model.get())==8&&llama_model_n_embd(model.get())==4096,"Gradient full128 geometry mismatch");
  const auto* vocab=llama_model_get_vocab(model.get());const auto text=read(cfg.at("prompt").get<std::string>());
  const int count=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
  need(count>=np,"actual tokenizer input too short; no padding");
  std::vector<llama_token> prompt(count);need(llama_tokenize(vocab,text.data(),text.size(),prompt.data(),count,true,true)==count,"tokenization failed");prompt.resize(np);
  {std::ofstream f(out/"prompt-token-ids.bin",std::ios::binary);f.write(reinterpret_cast<const char*>(prompt.data()),prompt.size()*sizeof(llama_token));need(bool(f),"save token identities");}
  policy::FullGqaRuntimePolicy::Config pc;
  pc.layers=32;pc.capacity=capacity;pc.prompt_tokens=np;pc.generated_capacity=ng;
  pc.beta=cfg.at("beta");pc.hot_kv_budget_bytes=cfg.at("hot_kv_budget_bytes");
  pc.total_host_workspace_budget_bytes=cfg.value("policy_workspace_budget_bytes",256ULL<<20);
  pc.chunk_tokens=cfg.value("score_chunk_tokens",4096);pc.hot_threads=cfg.value("hot_threads",3);pc.cold_threads=cfg.value("cold_threads",1);
  pc.cpu_implementation=cfg.value("cpu_implementation",std::string("auto"));
  need(pc.cpu_implementation=="auto"||pc.cpu_implementation=="legacy","cpu_implementation must be auto or legacy");
  pc.mode=cfg.at("method").get<std::string>();pc.score_file=cfg.at("score_file").get<std::string>();pc.verify_scores=verify;
  pc.hpm=cfg.value("hpm",false);pc.hpm_swaps=cfg.value("hpm_swaps",1);
  pc.payload_prefetch=payload_prefetch;pc.full_kv_backing_path=cfg.at("kv_file").get<std::string>();pc.full_kv_io_fence=prefetch_io_fence;
  if(pc.mode!="HOST_MATCHED"){
   const auto image=cfg.at("fpga");need(image.at("execute_authorized").get<bool>(),"existing image action must be authorized");
   pc.hardware.abi="HS_FULL_GQA_PIPELINE_F16_V3";pc.hardware.image_path=image.at("path");pc.hardware.image_sha256=image.at("sha256");pc.hardware.image_execute_authorized=true;
   need(pc.hardware.image_sha256==HILLSERVE_V3_IMAGE_SHA256,"image does not match the build-time validated V3 SHA256");
   pc.hardware.buffer_budget_bytes=256ULL<<20;pc.hardware.reuse_file_buffers=true;pc.hardware.direct_read_queue_depth=1;
  }
  t=Clock::now();policy::FullGqaRuntimePolicy scoring(pc);const double policy_init_ms=ms(t);
  need(scoring.managed_bytes()<host_budget,"all-layer policy exceeds total host budget");
  GQARuntimeHooks hooks;hooks.early_rehearsal=early;
  hooks.payload_prefetch=payload_prefetch;hooks.full_kv_io_fence=prefetch_io_fence;
  hooks.wait_pending=[&]{scoring.wait_pending();};
  hooks.append_full=[&](int l,const auto& positions,const auto& keys,const auto& bytes){scoring.append(l,positions,keys,bytes);};
  hooks.fetch=[&](int l,const auto& ids,const GQAColdFetch& fetch){return scoring.fetch(l,ids,fetch);};
  hooks.prepare_history=[&](int l,uint64_t position,const auto& q,runtime::GQAPrefetchReady ready){scoring.prepare_history(l,position,q,std::move(ready));};
  GQALayerExecutor runtime(out.string(),cfg.at("kv_file"),capacity,chunk,.2f,false,verify,true,gpu_budget,host_budget-scoring.managed_bytes(),{},
    [&](int l,uint64_t position,const auto& q){return scoring.select(l,position,q);},hooks);
  need(runtime.stats.managed_host_bytes+scoring.managed_bytes()<=host_budget,"combined all-layer host budget exceeded");
  std::vector<uint64_t> initial(std::min(np,chunk));std::iota(initial.begin(),initial.end(),0);runtime.begin(0,initial,false);
  auto cp=llama_context_default_params();cp.n_ctx=capacity;cp.n_batch=chunk;cp.n_ubatch=chunk;cp.n_seq_max=1;cp.n_threads=cfg.value("model_cpu_threads",4);cp.n_threads_batch=cp.n_threads;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn=false;cp.offload_kqv=true;cp.op_offload=!low_gpu;cp.rope_freq_base=3580165449.0f;cp.rope_freq_scale=1;cp.cb_eval=GQALayerExecutor::callback;cp.cb_eval_user_data=&runtime;
  t=Clock::now();std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);need(bool(ctx),"model context creation failed");const double context_ms=ms(t);
  llama_hillserve_set_profile(ctx.get(),[](ggml_backend_t backend,ggml_tensor*,int phase,int,void* opaque){
   if(phase!=0)return;auto& r=*static_cast<GQALayerExecutor*>(opaque);
   try{r.observe_backend(backend);}catch(const std::exception& error){r.failure=std::string("prefetch backend observation: ")+error.what();}
  },&runtime);
  std::ofstream steps(out/"steps.jsonl"),logits(out/"logits-checksums.jsonl"),output(out/"generated.txt");
  std::vector<llama_token> generated;std::vector<std::uint64_t> emitted;
  const auto request=Clock::now();t=Clock::now();int batches=0;
  for(int offset=0;offset<np;offset+=chunk){const int n=std::min(chunk,np-offset);std::vector<uint64_t> positions(n);std::iota(positions.begin(),positions.end(),offset);runtime.begin(batches++,positions,false);const int rc=llama_decode(ctx.get(),llama_batch_get_one(prompt.data()+offset,n));need(rc==0&&runtime.failure.empty(),"prefill failed: "+runtime.failure);llama_synchronize(ctx.get());runtime.accounting(ctx.get());}
  const double prefill_ms=ms(t);double decode_ms=0;int calls=0,eos=0;
  for(int step=0;step<ng;++step){auto* scores=llama_get_logits_ith(ctx.get(),-1);need(scores,"missing model logits");const int nv=llama_vocab_n_tokens(vocab);for(int i=0;i<nv;++i)need(std::isfinite(scores[i]),"nonfinite model logits");llama_token token=std::max_element(scores,scores+nv)-scores;generated.push_back(token);emitted.push_back(stamp());eos+=llama_vocab_is_eog(vocab,token);
   logits<<json({{"output_index",step},{"greedy_token",token},{"checksum",std::to_string(checksum(scores,nv*sizeof(float)))}}).dump()<<'\n';
   std::vector<char> piece(256);int n=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);if(n<0){piece.resize(-n);n=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,true);}if(n>0)output.write(piece.data(),n);
   if(step+1==ng)break;
   runtime.begin(batches+step,{uint64_t(np+step)},true);t=Clock::now();const auto begin=stamp();const int rc=llama_decode(ctx.get(),llama_batch_get_one(&token,1));need(rc==0&&runtime.failure.empty(),"decode failed: "+runtime.failure);llama_synchronize(ctx.get());const double elapsed=ms(t);decode_ms+=elapsed;++calls;runtime.accounting(ctx.get());steps<<json({{"decode_call",calls},{"position",np+step},{"started_ns",begin},{"ended_ns",stamp()},{"wall_ms",elapsed},{"actual_model_feedback",true}}).dump()<<'\n';steps.flush();
  }
  t=Clock::now();runtime.drain();scoring.drain();const double drain_ms=ms(t);const double request_ms=ms(request);
  scoring.write_stats((out/"policy-stats.json").string());save(out/"generated-token-ids.json",generated);save(out/"token-emission-ns.json",emitted);
  const auto& s=runtime.stats;need(s.attention_calls==32ULL*(batches+ng-1),"all32-layer attention coverage failed");need(s.rehearsal_calls==30ULL*(ng-1),"live30-layer score-query coverage failed");need(!payload_prefetch||s.prefetch_layers==30ULL*(ng-1),"live30-layer payload prefetch coverage failed");need(s.appends==32ULL*(np+ng-1),"all32-layer incremental KV coverage failed");
  struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
  json result={{"status","passed"},{"evidence_mode","ACTUAL_FULL128_GQA_MODEL_FEEDBACK"},{"batch",1},{"layers",32},{"sparse_layers",30},{"dense_prologue_layers",2},{"P",np},{"output_tokens",generated.size()},{"decode_calls",calls},{"method",pc.mode},{"cpu_implementation_requested",scoring.stats().cpu_implementation_requested},{"CPU_scorer",scoring.stats().cpu_implementation_actual},{"legacy_per_call_thread_creation_included",pc.cpu_implementation=="legacy"},{"CPU_hot_threads",pc.hot_threads},{"CPU_cold_threads",pc.mode=="HOST_MATCHED"?pc.cold_threads:0},{"beta_offline_fixed_all_layers",pc.beta},{"early_history_score_overlap_enabled",early},{"payload_prefetch_enabled",payload_prefetch},{"functional_low_gpu",low_gpu},{"model_gpu_layers",gpu_layers},{"GPU_attention_required",true},{"GPU_attention_device_memory_verified",true},{"CPU_op_auto_offload",!low_gpu},{"attention_calls",s.attention_calls},{"rehearsal_calls",s.rehearsal_calls},{"append_records",s.appends},{"GPU_H2D_bytes",s.gpu_h2d_bytes},{"GPU_prefetch_H2D_bytes",s.prefetch_h2d_bytes},{"GPU_D2D_bytes",s.gpu_d2d_bytes},{"payload_prefetch_layers",s.prefetch_layers},{"payload_prefetch_GPU_workspace_bytes",s.prefetch_gpu_bytes},{"payload_prefetch_host_workspace_bytes",s.prefetch_host_bytes},{"payload_prefetch_useful_bytes",s.prefetch_useful_bytes},{"payload_prefetch_wasted_bytes",s.prefetch_wasted_bytes},{"payload_prefetch_miss_H2D_bytes",s.prefetch_miss_h2d_bytes},{"payload_delivery_event_wait_ms",s.prefetch_consume_wait_ms},{"GPU_exact_readback_bytes",s.exact_gpu_bytes},{"full_native_GPU_history_bytes",s.native_history_bytes},{"GPU_workspace_bytes",s.workspace_bytes},{"graph_GPU_bytes",s.graph_gpu_bytes},{"managed_host_budget_bytes",host_budget},{"managed_host_reservation_bytes",s.managed_host_bytes+scoring.managed_bytes()},{"debug_shadow_bytes_separate",debug_bytes},{"process_peak_RSS_KiB",ru.ru_maxrss},{"prefill_ms",prefill_ms},{"decode_call_mean_ms",decode_ms/calls},{"interarrival_TPOT_ms",double(emitted.back()-emitted.front())/1e6/(emitted.size()-1)},{"request_wall_ms",request_ms},{"final_drain_ms",drain_ms},{"load_ms",load_ms},{"context_init_ms",context_ms},{"policy_init_ms",policy_init_ms},{"EOS_count",eos},{"fixed_output_budget_ignores_EOS",true},{"verify",verify},{"numerical_rule","F16_QK_ordered_FP32_mul_add_final_F16_then_scaled_selector"},{"selection_contract","head0_max_minus5_GE_mean_count_cap20_per_head_IDs"},{"durability","volatile_query_kv+end_of_request_sync"},{"formal_speedup_eligible",false}};
  if(const auto* io=runtime.store()){const auto& m=io->metrics;save(out/"kv-storage.json",{{"extent_bytes",io->extent_bytes()},{"logical_append_bytes",m.logical_append_bytes},{"logical_read_bytes",m.logical_read_bytes},{"syscall_write_bytes",m.syscall_write_bytes},{"syscall_read_bytes",m.syscall_read_bytes},{"direct_read_calls",m.direct_fetch_read_calls},{"drain_calls",m.drain_calls}});}
  save(out/"result.json",result);std::cout<<result.dump()<<'\n';return 0;
 }catch(const std::exception& error){llama_hillserve_configure(nullptr);if(!out.empty()&&std::filesystem::is_directory(out)){std::ofstream f(out/"error.txt");f<<error.what()<<'\n';save(out/"failure.json",{{"status","failed"},{"reason",error.what()}});}std::cerr<<error.what()<<'\n';return 1;}
}
