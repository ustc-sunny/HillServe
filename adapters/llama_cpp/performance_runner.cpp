// Formal native runner: actual text, bounded prefill, separate model/context setup,
// explicit positions and fixed token budget. Linked to pristine pinned llama.cpp.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sys/resource.h>
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static void need(bool condition,const std::string& msg){if(!condition)throw std::runtime_error(msg);}
static std::string read_text(const std::string& path){need(std::filesystem::file_size(path)<=8*1024*1024,"text exceeds 8MiB bound");std::ifstream f(path);need(bool(f),"missing text");return std::string(std::istreambuf_iterator<char>(f),{});}
static std::uint64_t token_digest(const std::vector<llama_token>& x){std::uint64_t hash=1469598103934665603ULL;for(auto v:x)for(unsigned s=0;s<32;s+=8){hash^=(std::uint32_t(v)>>s)&255;hash*=1099511628211ULL;}return hash;}
static std::vector<llama_token> tokenize(const llama_vocab* vocab,const std::string& s,bool special){if(s.empty()&&!special)return {};int n=llama_tokenize(vocab,s.data(),s.size(),nullptr,0,special,true);need(n<0,"empty/invalid input tokenization");std::vector<llama_token> out(-n);need(llama_tokenize(vocab,s.data(),s.size(),out.data(),out.size(),special,true)==int(out.size()),"tokenization mismatch");return out;}
struct Batch {llama_batch b;explicit Batch(int size):b(llama_batch_init(size,0,1)){}~Batch(){llama_batch_free(b);}};
int main(int argc,char**argv){
 json result={{"schema","hillserve-native-performance-v1"},{"status","failed"},{"method","NATIVE_FULL"},{"evidence_mode","M_E2E_NATIVE"},{"actual_fpga",false},{"actual_target_storage",false},{"simulation",false},{"durability","NOT_APPLICABLE_NATIVE"},{"fpga_kernel_ms",nullptr},{"physical_nand_bytes",nullptr},{"formal_method_speedup_eligible",false}};
 std::filesystem::path output;
 try{
  need((argc-1)%2==0,"all arguments are --key value");std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2)need(args.emplace(argv[i],argv[i+1]).second,"duplicate argument");
  for(const auto& kv:args)need(kv.first=="--model"||kv.first=="--prompt"||kv.first=="--suffix"||kv.first=="--prompt-tokens"||kv.first=="--decode"||kv.first=="--prefill-chunk"||kv.first=="--output"||kv.first=="--threads"||kv.first=="--gpu-budget"||kv.first=="--save-logits"||kv.first=="--rope-base"||kv.first=="--model-profile"||kv.first=="--exact-prompt", "unknown argument "+kv.first);
  auto get=[&](const std::string& name){auto it=args.find(name);need(it!=args.end(),"missing "+name);return it->second;};
  output=get("--output");need(std::filesystem::create_directory(output),"new output directory required");
  int np=std::stoi(get("--prompt-tokens")),ng=std::stoi(get("--decode")),chunk=std::stoi(get("--prefill-chunk")),threads=std::stoi(get("--threads"));
  const std::size_t gpu_budget=std::stoull(get("--gpu-budget"));bool save_logits=args.count("--save-logits")&&get("--save-logits")=="1";
  need(np>=1&&ng>=1&&ng<=1024&&np+ng<=32768,"requested context/output exceeds bounded model pilot");need(chunk>=32&&chunk<=512&&chunk%32==0,"bounded prefill chunk must be 32..512 in multiples of32");need(threads>=1&&threads<=16,"CPU thread bound");need(gpu_budget<=24ULL*1024*1024*1024,"GPU budget exceeds4090");
  const std::string profile=args.count("--model-profile")?get("--model-profile"):"qwen2-7b";
  need(profile=="qwen2-7b"||profile=="gradient-8b","unsupported native model profile");
  const bool gradient=profile=="gradient-8b";
  const int layers=gradient?32:28,qheads=gradient?32:28,kvheads=gradient?8:4,embedding=gradient?4096:3584;
  const std::string architecture=gradient?"llama":"qwen2";
  const bool exact_prompt=args.count("--exact-prompt")&&get("--exact-prompt")=="1";
  float rope_base=args.count("--rope-base")?std::stof(get("--rope-base")):0;
  need(gradient?rope_base==float(3580165449.0):rope_base==0||rope_base==1000000.0f,
       gradient?"Gradient requires explicit verified source RoPE3580165449":"fixed Qwen2 source-config RoPE only");
  const auto prompt=read_text(get("--prompt")),suffix=read_text(get("--suffix"));
  need(!gradient||(exact_prompt&&suffix.empty()),"Gradient boundary run requires exact whole prompt and empty suffix");
  result["model_profile"]=profile;result["model_architecture"]=architecture;
  result["verified_source_rope_base"]=gradient?json(3580165449ULL):json(1000000);
  result["query_heads"]=qheads;result["kv_heads"]=kvheads;result["layers"]=layers;result["head_dim"]=128;
  result["quality_scope"]=gradient?"separate_pristine_full_GPU_boundary_QA_not_matched_method_speedup":"existing_Qwen_native_pilot";
  ggml_backend_load_all();llama_backend_init();need(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU),"real GPU required, no CPU fallback");
  std::size_t free_gpu=0,total_gpu=0;need(cudaMemGetInfo(&free_gpu,&total_gpu)==cudaSuccess,"GPU memory query failed");
  const auto planned_kv=std::size_t((np+ng+255)/256*256)*layers*kvheads*128*4;
  const auto minimum_gpu=std::filesystem::file_size(get("--model"))+planned_kv+1024ULL*1024*1024;
  result["gpu_free_before_model_bytes"]=free_gpu;result["gpu_required_preflight_bytes"]=minimum_gpu;need(minimum_gpu<=gpu_budget&&minimum_gpu<=free_gpu,"RESOURCE_CAP: insufficient GPU memory for full model plus bounded workspace; no offload fallback");
  auto t=Clock::now();auto mp=llama_model_default_params();mp.n_gpu_layers=99;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(get("--model").c_str(),mp),llama_model_free);need(bool(model),"model load/allocation failed");result["model_load_ms"]=ms(t);
  need(llama_model_n_layer(model.get())==layers&&llama_model_n_head(model.get())==qheads&&llama_model_n_head_kv(model.get())==kvheads&&llama_model_n_embd(model.get())==embedding,"fixed native model profile ABI mismatch");
  char arch_meta[64]{};need(llama_model_meta_val_str(model.get(),"general.architecture",arch_meta,sizeof arch_meta)>0&&std::string(arch_meta)==architecture,"native model architecture mismatch");
  char filetype[32];need(llama_model_meta_val_str(model.get(),"general.file_type",filetype,sizeof filetype)>0&&std::string(filetype)=="32","expected unchanged BF16 GGUF");
  need(np+ng<=llama_model_n_ctx_train(model.get()),"requested context exceeds native model context, no RoPE override");
  const auto* vocab=llama_model_get_vocab(model.get());t=Clock::now();auto input=tokenize(vocab,prompt,true),tail=tokenize(vocab,suffix,false);
  result["source_prompt_token_count"]=input.size();result["suffix_token_count"]=tail.size();result["exact_whole_prompt"]=exact_prompt;
  if(exact_prompt)need(tail.empty()&&input.size()==std::size_t(np),"exact native prompt mismatch; refusing truncation or suffix replacement");
  need(input.size()+tail.size()>=std::size_t(np)&&tail.size()<std::size_t(np),"actual source text too short; no repeated tokens or synthetic padding");
  input.resize(np-tail.size());input.insert(input.end(),tail.begin(),tail.end());result["tokenization_ms"]=ms(t);result["input_token_digest_fnv64"]=std::to_string(token_digest(input));
  t=Clock::now();auto cp=llama_context_default_params();cp.n_ctx=(np+ng+255)/256*256;cp.n_batch=chunk;cp.n_ubatch=chunk;cp.n_seq_max=1;cp.n_threads=threads;cp.n_threads_batch=threads;
  cp.rope_freq_base=rope_base;
  char rope_meta[64]{};int rope_len=llama_model_meta_val_str(model.get(),(architecture+".rope.freq_base").c_str(),rope_meta,sizeof rope_meta);result["gguf_rope_freq_base_metadata"]=rope_len>0?json(std::string(rope_meta)):json(nullptr);result["effective_rope_base_explicit"]=rope_base?json(rope_base):json(nullptr);
  cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.offload_kqv=true;cp.flash_attn=true;cp.no_perf=false;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);need(bool(ctx),"context allocation failed");
  result["context_init_ms"]=ms(t);result["actual_context_tokens"]=llama_n_ctx(ctx.get());result["native_kv_logical_reserved_bytes"]=std::size_t(llama_n_ctx(ctx.get()))*layers*kvheads*128*4;
  Batch batch(chunk);std::vector<llama_token> generated;generated.reserve(ng);std::vector<double> emitted;emitted.reserve(ng);
  std::vector<float> first_logits;const int nv=llama_vocab_n_tokens(vocab);if(save_logits)first_logits.resize(nv);
  double prefill_ms=0,decode_ms=0;int prefill_calls=0,decode_calls=0;const auto start=Clock::now();t=start;
  for(int offset=0;offset<np;offset+=chunk){int count=std::min(chunk,np-offset);batch.b.n_tokens=count;
   for(int i=0;i<count;++i){batch.b.token[i]=input[offset+i];batch.b.pos[i]=offset+i;batch.b.n_seq_id[i]=1;batch.b.seq_id[i][0]=0;batch.b.logits[i]=(offset+i==np-1);}
   need(llama_decode(ctx.get(),batch.b)==0,"prefill decode failed");++prefill_calls;
  }
  llama_synchronize(ctx.get());prefill_ms=ms(t);
  for(int step=0;step<ng;++step){const float* logits=llama_get_logits_ith(ctx.get(),-1);need(logits,"missing logits");
   if(save_logits&&step==0)std::copy(logits,logits+nv,first_logits.begin());
   llama_token next=0;float best=-INFINITY;for(int i=0;i<nv;++i){need(std::isfinite(logits[i]),"nonfinite logits");if(logits[i]>best){best=logits[i];next=i;}}
   generated.push_back(next);emitted.push_back(ms(start));if(step+1==ng)break;
   batch.b.n_tokens=1;batch.b.token[0]=next;batch.b.pos[0]=np+step;batch.b.n_seq_id[0]=1;batch.b.seq_id[0][0]=0;batch.b.logits[0]=1;
   t=Clock::now();need(llama_decode(ctx.get(),batch.b)==0,"generation decode failed");llama_synchronize(ctx.get());decode_ms+=ms(t);++decode_calls;
  }
  llama_synchronize(ctx.get());const double request_ms=ms(start);auto perf=llama_perf_context(ctx.get());
  struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
  result.update({{"status","passed"},{"prompt_tokens",np},{"generated_token_count",generated.size()},{"decode_call_count",decode_calls},{"requested_output_tokens",ng},{"prefill_chunk_tokens",chunk},{"prefill_calls",prefill_calls},{"prefill_ms",prefill_ms},{"decode_call_wall_ms",decode_ms},{"request_wall_ms",request_ms},{"ttft_ms",emitted.front()},{"final_drain_ms",0.0},{"final_drain_applicable",false},{"native_file_append_bytes",0},{"native_file_read_bytes",0},{"gpu_budget_bytes",gpu_budget},{"process_peak_rss_kib",ru.ru_maxrss},{"flash_attention",true},{"host_threads",threads},{"weight_dtype","BF16"},{"payload_dtype","F16"},{"rope_override",rope_base!=0},{"fixed_output_budget_ignores_eos",true},{"context_and_model_setup_in_request_wall",false},{"validation_logits_export",save_logits},{"llama_prompt_eval_ms",perf.t_p_eval_ms},{"llama_decode_eval_ms",perf.t_eval_ms},{"generated_token_digest_fnv64",std::to_string(token_digest(generated))}});
  result["generated_EOS_count"]=std::count_if(generated.begin(),generated.end(),[&](llama_token token){return llama_vocab_is_eog(vocab,token);});
  result["tpot_ms"]=emitted.size()>1?json((emitted.back()-emitted.front())/(emitted.size()-1)):json(nullptr);
  // All text/JSON/logit export occurs after request timing.
  std::string text;for(auto token:generated){char buffer[256];int n=llama_token_to_piece(vocab,token,buffer,sizeof buffer,0,true);if(n>=0)text.append(buffer,n);else{std::vector<char> large(-n);n=llama_token_to_piece(vocab,token,large.data(),large.size(),0,true);need(n>=0,"token text conversion");text.append(large.data(),n);}}
  std::ofstream native_input(output/"native-prompt-token-ids.bin",std::ios::binary);native_input.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(llama_token));need(bool(native_input),"native token provenance export failed");
  std::ofstream(output/"generated.txt")<<text;std::ofstream(output/"generated-ids.json")<<json(generated).dump()<<'\n';std::ofstream(output/"emission-times.json")<<json(emitted).dump()<<'\n';
  if(save_logits){std::ofstream f(output/"first-logits.bin",std::ios::binary);f.write(reinterpret_cast<const char*>(first_logits.data()),first_logits.size()*4);}
  std::ofstream(output/"result.json")<<result.dump(2)<<'\n';std::cout<<result.dump()<<'\n';return 0;
 }catch(const std::exception&e){result["error"]=e.what();if(!output.empty()&&std::filesystem::is_directory(output))std::ofstream(output/"failure.json")<<result.dump(2)<<'\n';std::cerr<<result.dump()<<'\n';return 1;}
}
