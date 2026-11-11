#include "runtime/gqa_executor.hpp"
#include "policy/gqa_policy.hpp"
#include "storage/file_store.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <sys/resource.h>
using namespace hillserve;
namespace {using Clock=std::chrono::steady_clock;double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}void need(bool b,const std::string& x){if(!b)throw std::runtime_error(x);}std::string read(const std::string& path){std::ifstream f(path);need(bool(f),"input file missing");return std::string(std::istreambuf_iterator<char>(f),{});}}
int main(int argc,char** argv){
 if(argc!=10){std::cerr<<"usage: hillserve-gqa-runtime MODEL ARTIFACT PROMPT NEW_DIR P G file|memory verify|none KV_FILE\n";return 2;}
 const std::filesystem::path out=argv[4];if(!std::filesystem::create_directory(out)){std::cerr<<"new output directory required\n";return 2;}
 try{
 setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);ggml_backend_load_all();llama_backend_init();
 int np=std::stoi(argv[5]),ng=std::stoi(argv[6]);bool memory=std::string(argv[7])=="memory",verify=std::string(argv[8])=="verify";
 need(memory||std::string(argv[7])=="file","invalid backing mode");need(verify||std::string(argv[8])=="none","invalid verify mode");need(!memory||verify,"memory backing is a separate validation oracle, not a budgeted timed path");need(np>=32&&np<=8192&&ng>=2&&ng<=64,"bounded GQA pilot P/G");
 const uint64_t debug_bytes=verify?32ULL*((np+ng+31)/32*32)*4096:0;need(debug_bytes<=512ULL*1024*1024,"explicit GQA debug shadow budget exceeded");
 int capacity=(np+ng+31)/32*32;const int chunk=np>2048?64:256;float alpha=5,ratio=.2f;
 const char* av=std::getenv("HILLSERVE_GQA_ALPHA");if(av)alpha=std::stof(av);
 const char* rv=std::getenv("HILLSERVE_GQA_CAP_RATIO");if(rv)ratio=std::stof(rv);
 const size_t gpu_budget=std::getenv("HILLSERVE_GQA_GPU_BUDGET")?std::stoull(std::getenv("HILLSERVE_GQA_GPU_BUDGET")):256ULL*1024*1024;
 const size_t host_budget=std::getenv("HILLSERVE_GQA_HOST_BUDGET")?std::stoull(std::getenv("HILLSERVE_GQA_HOST_BUDGET")):256ULL*1024*1024;
 const std::string method=std::getenv("HILLSERVE_GQA_METHOD")?std::getenv("HILLSERVE_GQA_METHOD"):"HOST_MATCHED";
 need(method=="HOST_MATCHED"||method=="HOST_BEST"||method=="FPGA_STAGED"||method=="FPGA_DIRECT","invalid explicit method");
 bool direct_io=std::getenv("HILLSERVE_GQA_DIRECT_IO")&&std::string(std::getenv("HILLSERVE_GQA_DIRECT_IO"))=="1";
 bool direct=direct_io||(std::getenv("HILLSERVE_GQA_DIRECT_KV")&&std::string(std::getenv("HILLSERVE_GQA_DIRECT_KV"))=="1");
 need(method=="HOST_MATCHED"||(!memory&&direct_io),"paired C methods require direct score/KV file I/O, no memory fallback");
 const size_t cache_bytes=std::getenv("HILLSERVE_GQA_CACHE_BYTES")?std::stoull(std::getenv("HILLSERVE_GQA_CACHE_BYTES")):32ULL*1024*1024;
 bool verify_scores=std::getenv("HILLSERVE_VERIFY_SCORES")&&std::string(std::getenv("HILLSERVE_VERIFY_SCORES"))=="1";
 need(!verify_scores||verify,"score reference validation must be outside formal none runs");
 auto start=Clock::now();auto mp=llama_model_default_params();mp.n_gpu_layers=99;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);need(bool(model),"model load failed");double model_ms=ms(start);
 need(llama_model_n_layer(model.get())==32&&llama_model_n_head(model.get())==32&&llama_model_n_head_kv(model.get())==8&&llama_model_n_embd(model.get())==4096,"Gradient8B GQA ABI mismatch");
 const auto* vocab=llama_model_get_vocab(model.get());std::string text=read(argv[3]);int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);need(n>=np,"actual prompt shorter than requested prefix; no padding");if(std::getenv("HILLSERVE_EXPECT_EXACT_PROMPT"))need(n==np,"native prompt count differs from exact intended input; refuse question truncation");std::vector<llama_token> prompt(n);need(llama_tokenize(vocab,text.data(),text.size(),prompt.data(),n,true,true)==n,"tokenize failed");prompt.resize(np);{std::ofstream native_ids(out/"native-prompt-token-ids.bin",std::ios::binary);native_ids.write(reinterpret_cast<const char*>(prompt.data()),prompt.size()*sizeof(llama_token));need(bool(native_ids),"native token provenance write failed");}
 const std::string score_path=std::getenv("HILLSERVE_SCORE_FILE")?std::getenv("HILLSERVE_SCORE_FILE"):(out/"score-keys.bin").string();
 start=Clock::now();policy::GQAPolicy scoring(argv[2],capacity,alpha,ratio,score_path);
 HardwareContract contract;
 if(method=="FPGA_STAGED"||method=="FPGA_DIRECT"){
  const char* image=std::getenv("HILLSERVE_FPGA_IMAGE");const char* sha=std::getenv("HILLSERVE_FPGA_SHA256");const char* grant=std::getenv("HILLSERVE_FPGA_EXECUTE");
  need(image&&sha&&grant&&std::string(grant)=="1","explicit previously authorized image identity/action required");
  need(std::string(sha)=="c48f86d51667edf8f9142e98e6c99c8c108aaa40acb05cba0ab4da03c6f91c35","C comparison pins the authorized conservative v1 image");
  contract.abi="HS_PARTIAL_F16_V1";contract.image_path=image;contract.image_sha256=sha;contract.image_execute_authorized=true;
 }
 if(direct_io)scoring.enable_direct_writes();
 scoring.configure_backend(method=="FPGA_STAGED"?"fpga_staged":method=="FPGA_DIRECT"?"fpga_direct":"host",contract,direct_io);
 if(method!="HOST_MATCHED")scoring.configure_cpu_cache(cache_bytes,4);
 scoring.verify_optimized_scores(verify_scores);double policy_init_ms=ms(start);
 need(scoring.managed_bytes()<host_budget,"policy reservation exceeds combined host budget");
 GQALayerExecutor runtime(out.string(),argv[9],capacity,chunk,ratio,memory,verify,direct,gpu_budget,host_budget-scoring.managed_bytes(),
  [&](int l,const std::vector<uint64_t>& pos,const std::vector<uint16_t>& keys){scoring.append(l,pos,keys);},
  [&](int l,uint64_t pos,const std::vector<float>& q){return scoring.select(l,pos,q);});
 need(runtime.stats.managed_host_bytes+scoring.managed_bytes()<=host_budget,"combined executor+policy managed host budget exceeded");
 std::vector<uint64_t> first(std::min(np,chunk));std::iota(first.begin(),first.end(),0);runtime.begin(0,first,false);
 auto cp=llama_context_default_params();cp.n_ctx=capacity;cp.n_batch=chunk;cp.n_ubatch=chunk;cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn=false;cp.offload_kqv=true;cp.rope_freq_base=3580165449.0f;cp.rope_freq_scale=1;cp.cb_eval=GQALayerExecutor::callback;cp.cb_eval_user_data=&runtime;
 start=Clock::now();std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);need(bool(ctx),"GQA context initialization failed");double context_ms=ms(start);
 std::ofstream steps(out/"steps.jsonl"),tokens(out/"generated-token-ids.json"),output(out/"generated.txt"),logits(out/"logits-checksums.jsonl");steps<<std::setprecision(17);const auto request_start=Clock::now();start=Clock::now();int batch_index=0;
 for(int offset=0;offset<np;offset+=chunk){int count=std::min(chunk,np-offset);std::vector<uint64_t> positions(count);std::iota(positions.begin(),positions.end(),offset);runtime.begin(batch_index++,positions,false);int rc=llama_decode(ctx.get(),llama_batch_get_one(prompt.data()+offset,count));need(rc==0&&runtime.failure.empty(),"prefill: "+runtime.failure+" rc="+std::to_string(rc));llama_synchronize(ctx.get());runtime.accounting(ctx.get());}
 double prefill_ms=ms(start),ttft_ms=0,decode_ms=0;int decode_calls=0,eos_count=0;tokens<<"[";
 for(int step=0;step<ng;++step){float* scores=llama_get_logits_ith(ctx.get(),-1);need(scores,"missing logits");int nv=llama_vocab_n_tokens(vocab);for(int i=0;i<nv;++i)need(std::isfinite(scores[i]),"nonfinite actual logits");llama_token tok=std::max_element(scores,scores+nv)-scores;if(step==0)ttft_ms=ms(request_start);if(step)tokens<<",";tokens<<tok;
  if(llama_vocab_is_eog(vocab,tok))++eos_count;std::vector<char> piece(256);int len=llama_token_to_piece(vocab,tok,piece.data(),piece.size(),0,true);if(len<0){piece.resize(-len);len=llama_token_to_piece(vocab,tok,piece.data(),piece.size(),0,true);}if(len>0)output.write(piece.data(),len);
  if(verify)logits<<"{\"output_index\":"<<step<<",\"checksum\":\""<<checksum(scores,nv*4)<<"\",\"greedy_token\":"<<tok<<"}\n";
  if(step+1==ng)break;
  runtime.begin(batch_index+step,{uint64_t(np+step)},true);auto t=Clock::now();int rc=llama_decode(ctx.get(),llama_batch_get_one(&tok,1));need(rc==0&&runtime.failure.empty(),"decode: "+runtime.failure+" rc="+std::to_string(rc));llama_synchronize(ctx.get());double wall=ms(t);decode_ms+=wall;++decode_calls;runtime.accounting(ctx.get());steps<<"{\"decode_call\":"<<decode_calls<<",\"input_token\":"<<tok<<",\"original_position\":"<<np+step<<",\"started_ns\":"<<std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count()<<",\"ended_ns\":"<<std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()<<",\"wall_ms\":"<<wall<<",\"real_feedback\":true}\n";
 }
 tokens<<"]\n";start=Clock::now();runtime.drain();scoring.drain();double drain_ms=ms(start),request_ms=ms(request_start);scoring.write_stats((out/"policy-stats.json").string());const auto& s=runtime.stats;
 if(const auto* io=runtime.store()){std::ofstream storage(out/"kv-storage.json");const auto& m=io->metrics;storage<<std::setprecision(17)<<"{\"extent_bytes\":"<<io->extent_bytes()<<",\"index_bytes\":"<<io->index_bytes()<<",\"logical_append_bytes\":"<<m.logical_append_bytes<<",\"logical_read_bytes\":"<<m.logical_read_bytes<<",\"syscall_write_bytes\":"<<m.syscall_write_bytes<<",\"syscall_read_bytes\":"<<m.syscall_read_bytes<<",\"drain_calls\":"<<m.drain_calls<<",\"drain_ms\":"<<m.drain_ms<<",\"direct_reads\":"<<(direct?"true":"false")<<",\"direct_writes\":"<<(io->direct_writes_enabled()?"true":"false")<<",\"direct_write_bytes\":"<<m.direct_write_bytes<<",\"direct_RMW_read_bytes\":"<<m.direct_rmw_read_bytes<<",\"direct_append_max_staging_bytes\":"<<m.max_append_staging_bytes<<",\"direct_fetch_max_staging_bytes\":"<<m.max_fetch_staging_bytes<<",\"direct_fetch_read_calls\":"<<m.direct_fetch_read_calls<<",\"buffered_header_reservation_bytes\":4096,\"physical_NAND_bytes\":null}\n";}
 need(s.attention_calls==32ULL*((np+chunk-1)/chunk+ng-1),"GQA layer/step coverage mismatch");need(s.rehearsal_calls>=30ULL*(ng-1),"GQA rehearsal coverage missing");need(s.appends==32ULL*(np+ng-1),"GQA append volume nonincremental");
 struct rusage ru{};getrusage(RUSAGE_SELF,&ru);std::ofstream result(out/"result.json");result<<std::setprecision(17)<<"{\"status\":\"passed\",\"evidence_mode\":\"ACTUAL_GQA_MODEL_FEEDBACK\",\"method\":\""<<method<<"\",\"fixed_cache_requested_bytes\":"<<(method=="HOST_MATCHED"?0:cache_bytes)<<",\"model\":\"Gradient8B\",\"Q_heads\":32,\"KV_heads\":8,\"rope_base_explicit\":3580165449,\"native_rope_float32\":"<<cp.rope_freq_base<<",\"native_prompt_available_tokens\":"<<n<<",\"native_prompt_token_bytes_checksum\":\""<<checksum(prompt.data(),prompt.size()*sizeof(llama_token))<<"\",\"native_prompt_token_checksum_kind\":\"existing_fnv64_integrity_checksum_not_SHA\",\"P\":"<<np<<",\"output_budget\":"<<ng<<",\"output_tokens\":"<<ng<<",\"decode_calls\":"<<decode_calls<<",\"EOS_count\":"<<eos_count<<",\"EOS_ignored_for_fixed_budget\":true,\"TTFT_ms\":"<<ttft_ms<<",\"TPOT_ms\":"<<decode_ms/decode_calls<<",\"request_wall_ms\":"<<request_ms<<",\"prefill_build_ms\":"<<prefill_ms<<",\"model_load_ms\":"<<model_ms<<",\"policy_device_file_init_ms\":"<<policy_init_ms<<",\"context_init_ms\":"<<context_ms<<",\"end_request_drain_ms\":"<<drain_ms<<",\"verify\":"<<(verify?"true":"false")<<",\"backing\":\""<<(memory?"compact_memory":"incremental_file")<<"\",\"direct_KV_read\":"<<(direct?"true":"false")<<",\"direct_KV_and_score_write\":"<<(direct_io?"true":"false")<<",\"full_native_GPU_history_bytes\":"<<s.native_history_bytes<<",\"debug_shadow_budget_bytes\":536870912,\"debug_physical_KV_backing_bytes\":"<<debug_bytes<<",\"debug_full_KV_shadow\":"<<(verify?"true":"false")<<",\"gpu_workspace_bytes\":"<<s.workspace_bytes<<",\"graph_GPU_bytes\":"<<s.graph_gpu_bytes<<",\"managed_host_budget_bytes\":"<<host_budget<<",\"managed_host_reservation_bytes\":"<<(s.managed_host_bytes+scoring.managed_bytes())<<",\"runtime_libraries_RSS_in_budget\":false,\"process_peak_RSS_KiB\":"<<ru.ru_maxrss<<",\"attention_calls\":"<<s.attention_calls<<",\"rehearsal_calls\":"<<s.rehearsal_calls<<",\"append_records\":"<<s.appends<<",\"physical_selected_KV_bytes\":"<<s.physical_selected_bytes<<",\"logical_selected_KV_bytes\":"<<s.logical_selected_bytes<<",\"exact_selected_bytes_checked\":"<<s.exact_bytes<<",\"GPU_H2D_bytes\":"<<s.gpu_h2d_bytes<<",\"append_ms\":"<<s.append_ms<<",\"fetch_ms\":"<<s.fetch_ms<<",\"h2d_ms\":"<<s.h2d_ms<<",\"durability\":\"volatile_query_kv+end_of_request_sync\",\"formal_speedup_eligible\":false}\n";
 return 0;
 }catch(const std::exception& e){llama_hillserve_configure(nullptr);std::ofstream err(out/"failure.json");err<<"{\"status\":\"failed\",\"error_file\":\"error.txt\"}\n";std::ofstream log(out/"error.txt");log<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
