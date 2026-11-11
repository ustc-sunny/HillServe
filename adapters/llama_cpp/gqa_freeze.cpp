// Real native-model checkpoint producer. It does not claim sparse model feedback.
#include "hybrid_eval/gqa_frozen.hpp"
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <iomanip>
#include <iostream>
using namespace hillserve;using namespace hillserve::replay;
struct Capture {
 int layer=8;std::size_t offset=0,count=0;bool decode=false;std::string failure;
 std::vector<std::uint16_t> keys,values;std::vector<float> previous_norm,current_norm,actual_query;
 std::array<char,GGML_MAX_OP_PARAMS> rope{};std::size_t key_calls=0,value_calls=0;
 static bool callback(ggml_tensor* t,bool ask,void* ptr){auto& s=*static_cast<Capture*>(ptr);const std::string name=t->name;
  bool key=name=="Kcur-"+std::to_string(s.layer)&&t->op==GGML_OP_ROPE;
  bool value=name=="Vcur-"+std::to_string(s.layer)&&t->op==GGML_OP_MUL_MAT;
  bool prev=s.decode&&name=="attn_norm-"+std::to_string(s.layer-1);
  bool norm=s.decode&&name=="attn_norm-"+std::to_string(s.layer);
  bool query=s.decode&&name=="Qcur-"+std::to_string(s.layer)&&t->op==GGML_OP_ROPE;
  if(ask)return key||value||prev||norm||query;
  try{need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"capture requires contiguous F32 actual tensor");std::vector<float> x(ggml_nelements(t));ggml_backend_tensor_get(t,x.data(),0,ggml_nbytes(t));
   for(float v:x)need(std::isfinite(v),"nonfinite actual model tensor");
   if(key||value){need(x.size()==s.count*1024,"physical GQA capture shape");auto& dest=key?s.keys:s.values;need(dest.size()==s.offset*1024,"capture duplicate or missing positions");for(float v:x)dest.push_back(half_bits(v));if(key){++s.key_calls;std::memcpy(s.rope.data(),t->op_params,GGML_MAX_OP_PARAMS);}else ++s.value_calls;}
   if(prev){need(x.size()==4096,"previous norm shape");s.previous_norm=std::move(x);}if(norm){need(x.size()==4096,"current norm shape");s.current_norm=std::move(x);}if(query){need(x.size()==4096,"actual Q shape");s.actual_query=std::move(x);}return true;
  }catch(const std::exception&e){s.failure=name+": "+e.what();return false;}
 }
};
static std::vector<float> project(llama_model* model,int layer,const std::vector<float>& norm,int pos,const std::array<char,GGML_MAX_OP_PARAMS>& rope){
 need(norm.size()==4096,"projection source missing");auto* weight=const_cast<ggml_tensor*>(model->get_tensor(("blk."+std::to_string(layer)+".attn_q.weight").c_str()));need(weight,"actual target Wq missing");
 auto device=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(device,"real GPU projection required");auto backend=ggml_backend_dev_init(device,nullptr);auto* ctx=ggml_init({2*1024*1024,nullptr,true});need(backend&&ctx,"projection graph init");
 auto* x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,1);auto* p=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);auto* q=ggml_mul_mat(ctx,weight,x);q=ggml_reshape_3d(ctx,q,128,32,1);
 q=ggml_rope_ext(ctx,q,p,nullptr,128,0,1048576,3580165449.0f,1,0,1,32,1);std::memcpy(q->op_params,rope.data(),GGML_MAX_OP_PARAMS);
 auto* g=ggml_new_graph(ctx);ggml_build_forward_expand(g,q);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buffer,"projection GPU allocation");ggml_backend_tensor_set(x,norm.data(),0,norm.size()*4);ggml_backend_tensor_set(p,&pos,0,4);
 need(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"actual projection GPU compute");ggml_backend_synchronize(backend);std::vector<float> out(4096);ggml_backend_tensor_get(q,out.data(),0,out.size()*4);
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return out;
}
int main(int argc,char** argv){
 if(argc!=7){std::cerr<<"MODEL ARTIFACT PROMPT NEW_DIR P LAYER\n";return 2;}std::filesystem::path out=argv[4];if(!std::filesystem::create_directory(out))return 2;
 try{auto start=Clock::now();int P=std::stoi(argv[5]),layer=std::stoi(argv[6]);need(P==2048||P==4096||P==8192||P==16384||P==32768,"finite real input lengths only");
  setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);ggml_backend_load_all();llama_backend_init();llama_hillserve_configure(nullptr);
  auto mp=llama_model_default_params();mp.n_gpu_layers=99;std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);need(bool(model),"model load");
  need(llama_model_n_layer(model.get())==32&&llama_model_n_head(model.get())==32&&llama_model_n_head_kv(model.get())==8&&llama_model_n_embd(model.get())==4096,"Gradient ABI");
  std::ifstream f(argv[3]);need(bool(f),"actual input missing");std::string text(std::istreambuf_iterator<char>(f),{});auto* vocab=llama_model_get_vocab(model.get());int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);need(n==P,"exact real input length required, no truncation or repetition");std::vector<llama_token> input(n);need(llama_tokenize(vocab,text.data(),text.size(),input.data(),n,true,true)==n,"tokenization");save(out/"input-token-ids.bin",input);
  need(std::size_t(P+1)*4096ULL+text.capacity()+input.capacity()*sizeof(llama_token)+16ULL*1024*1024<=256ULL*1024*1024,"capture storage exceeds managed budget before allocation");
  Capture capture;capture.layer=layer;capture.keys.reserve(std::size_t(P+1)*1024);capture.values.reserve(std::size_t(P+1)*1024);auto cp=llama_context_default_params();cp.n_ctx=(P+1+255)/256*256;cp.n_batch=256;cp.n_ubatch=256;cp.n_threads=4;cp.n_threads_batch=4;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn=false;cp.offload_kqv=true;cp.rope_freq_base=3580165449.0f;cp.rope_freq_scale=1;cp.cb_eval=Capture::callback;cp.cb_eval_user_data=&capture;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);need(bool(ctx),"native capture context");
  const auto GPU_compute_window_started_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
  for(int off=0;off<P;off+=256){capture.offset=off;capture.count=std::min(256,P-off);int rc=llama_decode(ctx.get(),llama_batch_get_one(input.data()+off,capture.count));need(rc==0&&capture.failure.empty(),"real prefill: "+capture.failure);llama_synchronize(ctx.get());}
  auto* logits=llama_get_logits_ith(ctx.get(),-1);need(logits,"actual logits");int nv=llama_vocab_n_tokens(vocab);llama_token token=std::max_element(logits,logits+nv)-logits;
  capture.decode=true;capture.offset=P;capture.count=1;int rc=llama_decode(ctx.get(),llama_batch_get_one(&token,1));need(rc==0&&capture.failure.empty(),"actual feedback step: "+capture.failure);llama_synchronize(ctx.get());
  need(capture.keys.size()==std::size_t(P+1)*1024&&capture.values.size()==capture.keys.size(),"actual KV history coverage");
  auto projected_current=project(model.get(),layer,capture.current_norm,P,capture.rope);std::size_t query_bit_errors=0;for(std::size_t i=0;i<4096;++i)query_bit_errors+=std::memcmp(&projected_current[i],&capture.actual_query[i],4)!=0;
  need(query_bit_errors==0,"same-input GPU Wq/RoPE operator not bit-exact to actual model");auto rehearsal=project(model.get(),layer,capture.previous_norm,P,capture.rope);
  const auto GPU_compute_window_finished_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
  const auto native_kv=llama_hillserve_native_kv_bytes(ctx.get());need(native_kv>0,"native checkpoint capture must declare full GPU history");ctx.reset();model.reset();
  LayerArtifact artifact(argv[2],layer);std::size_t valid=P+1,capacity=(valid+63)/64*64,D=artifact.channels;
  // Native checkpoint capture is B=1 asset generation, never a B=8 timing result.
  // Do not duplicate complete physical K/V or the 32-head compact-key file in RAM.
  // Keep numerical operation order and serialized output layout identical to v2.
  const std::size_t captured_kv_capacity=(capture.keys.capacity()+capture.values.capacity())*2ULL;
  const std::size_t miscellaneous=16ULL*1024*1024+text.capacity()+input.capacity()*sizeof(llama_token);
  const std::size_t artifact_bytes=artifact.basis.capacity()*sizeof(float)+artifact.mask.capacity();
  const std::size_t capture_phase_reservation=captured_kv_capacity+miscellaneous+artifact_bytes;
  const std::size_t scoring_phase_reservation=capture.keys.capacity()*2ULL+capacity*(128+D)*2ULL+8ULL*32*capacity+16ULL*32*std::size_t(std::floor(float(valid)*.2f))+miscellaneous+artifact_bytes;
  const std::size_t capture_reservation=std::max(capture_phase_reservation,scoring_phase_reservation);
  need(capture_reservation<=256ULL*1024*1024,"bounded streamed capture managed host reservation");
  {
   const auto path=out/"kv-physical.bin";need(!std::filesystem::exists(path),"new streamed physical KV file required");std::ofstream physical(path,std::ios::binary);
   for(std::size_t t=0;t<valid;++t){physical.write(reinterpret_cast<const char*>(capture.keys.data()+t*1024),2048);physical.write(reinterpret_cast<const char*>(capture.values.data()+t*1024),2048);}
   need(bool(physical),"streamed physical KV write failed");physical.close();need(bool(physical)&&std::filesystem::file_size(path)==valid*4096,"streamed physical KV extent");
  }
  std::vector<std::uint16_t>().swap(capture.values);
  auto q=artifact.query(rehearsal);std::vector<std::uint16_t> cq(32*D,0),reference(32*capacity);
  const auto score_path=out/"score-compact.bin";need(!std::filesystem::exists(score_path),"new streamed compact key file required");std::ofstream score_file(score_path,std::ios::binary);std::size_t errors=0;
  for(std::size_t h=0;h<32;++h){std::vector<std::uint16_t> full(capacity*128,0),compact_head(capacity*D,0);for(std::size_t c=0;c<artifact.columns[h].size();++c)cq[h*D+c]=q[h*128+artifact.columns[h][c]];
   for(std::size_t t=0;t<valid;++t){auto v=policy::gqa_skew_head_f16(capture.keys.data()+t*1024+(h/4)*128,artifact.basis.data()+h*128*128);std::copy(v.begin(),v.end(),full.begin()+t*128);for(std::size_t c=0;c<artifact.columns[h].size();++c)compact_head[t*D+c]=v[artifact.columns[h][c]];}
   std::vector<std::uint16_t> head_q(q.begin()+h*128,q.begin()+(h+1)*128),compact_q(cq.begin()+h*D,cq.begin()+(h+1)*D);auto head_ref=host_partial_scores(head_q,full,1,capacity,128);auto head_packed=host_partial_scores(compact_q,compact_head,1,capacity,D);
   for(std::size_t t=0;t<capacity;++t)errors+=head_ref[t]!=head_packed[t];std::copy(head_ref.begin(),head_ref.end(),reference.begin()+h*capacity);
   score_file.write(reinterpret_cast<const char*>(compact_head.data()),compact_head.size()*2);need(bool(score_file),"streamed compact key write failed");}
  score_file.close();need(bool(score_file)&&std::filesystem::file_size(score_path)==32*capacity*D*2,"streamed compact key extent");need(errors==0,"compact representation changed ordered raw F16 scores");
  std::vector<std::uint16_t>().swap(capture.keys);
  auto selection=policy::select_gqa_scores(scaled(reference),32,capacity,valid,5,.2f);need(selection.k>0,"reference selected zero, no fallback");std::vector<std::uint64_t> positions(valid);std::iota(positions.begin(),positions.end(),0);
  save(out/"query-f16.bin",cq);save(out/"query-full-f16.bin",q);save(out/"raw-score-oracle.bin",reference);save(out/"selected-ids.bin",flatten(selection));save(out/"positions.bin",positions);save(out/"actual-attention-query-f32.bin",capture.actual_query);save(out/"rehearsal-query-f32.bin",rehearsal);
  json result={{"status","passed"},{"evidence_class","actual_model_checkpoint_for_operator_replay"},{"sparse_model_feedback_claim",false},{"model","Gradient8B_BF16"},{"layer",layer},{"P",P},{"valid_candidates",valid},{"capacity",capacity},{"heads",32},{"physical_KV_heads",8},{"head_dim",128},{"compact_channels_padded",D},{"columns",artifact.columns},{"query_source","actual previous-layer attention RMSNorm -> actual target Wq -> copied actual target RoPE parameters -> fixed skew/mask"},{"one_actual_autoregressive_decode",true},{"native_full_KV_bytes_during_capture",native_kv},{"same_input_projection_bit_errors",query_bit_errors},{"compact_vs_expanded_score_bit_errors",errors},{"score_comparisons",reference.size()},{"selected_per_head",selection.k},{"alpha",5},{"cap_ratio",.2},{"normalization_dimension",128},{"policy_id","infinigen_public_llama_accuracy_GQA_online_v1"},{"ties","original_position_ascending"},{"capture_seconds",milliseconds(start)/1000},{"capture_timing_is_not_serving_performance",true},{"score_file_bytes",32*capacity*D*2},{"KV_file_bytes",valid*4096},{"all_input_positions_real",true},{"capture_managed_host_reservation_bytes",capture_reservation},{"reference_expanded128_max_heads_resident",1},{"capture_host_budget_bytes",268435456},{"captured_physical_KV_vector_capacity_bytes",captured_kv_capacity},{"capture_phase_reservation_bytes",capture_phase_reservation},{"scoring_phase_reservation_bytes",scoring_phase_reservation},{"compact_max_heads_resident",1},{"physical_KV_output_streamed",true},{"capture_batch_size",1},{"batch_performance_claim",false},{"GPU_model_compute_window_started_ns",GPU_compute_window_started_ns},{"GPU_model_compute_window_finished_ns",GPU_compute_window_finished_ns}};
  std::ofstream(out/"manifest.json")<<result.dump(2)<<'\n';std::cout<<result.dump()<<'\n';return 0;
 }catch(const std::exception&e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
