// Bounded native-model capture for ranking diagnostics, never serving timing.
// No upstream edits: use the existing b6000 callbacks and real target Wq/RoPE.
#include "hybrid_eval/gqa_frozen.hpp"
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <iostream>
#include <map>
#include <memory>
#include <sys/resource.h>

using namespace hillserve;
using namespace hillserve::replay;
namespace {
constexpr std::array<int,6> TARGETS{2,7,15,19,27,31}; // one-based 3/8/16/20/28/32
constexpr std::array<int,3> STEPS{1,4,8};
struct Layer {
    std::ofstream keys;
    std::size_t positions=0,key_calls=0;
    std::vector<float> previous_norm,current_norm,actual_query;
    std::array<char,GGML_MAX_OP_PARAMS> rope{};
};
struct Capture {
    std::map<int,Layer> layers;
    std::size_t offset=0,count=0;
    bool observe=false;
    std::string failure;
    static bool callback(ggml_tensor* t,bool ask,void* ptr) {
        auto& s=*static_cast<Capture*>(ptr);
        const std::string name=t->name;
        int target=-1,kind=0;
        for(auto& [l,state]:s.layers) {
            (void)state;
            if(name=="Kcur-"+std::to_string(l)&&t->op==GGML_OP_ROPE){target=l;kind=1;break;}
            if(s.observe&&name=="attn_norm-"+std::to_string(l-1)){target=l;kind=2;break;}
            if(s.observe&&name=="attn_norm-"+std::to_string(l)){target=l;kind=3;break;}
            if(s.observe&&name=="Qcur-"+std::to_string(l)&&t->op==GGML_OP_ROPE){target=l;kind=4;break;}
        }
        if(ask)return target>=0;
        if(target<0)return true;
        try {
            need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"capture requires contiguous actual F32 tensor");
            need(ggml_nelements(t)<=128*4096,"bounded callback tensor");
            std::vector<float> x(ggml_nelements(t));
            ggml_backend_tensor_get(t,x.data(),0,ggml_nbytes(t));
            for(float v:x)need(std::isfinite(v),"nonfinite model tensor");
            auto& state=s.layers.at(target);
            if(kind==1) {
                need(x.size()==s.count*1024&&state.positions==s.offset,"duplicate/missing physical K positions");
                std::vector<std::uint16_t> half(x.size());
                for(std::size_t i=0;i<x.size();++i){half[i]=half_bits(x[i]);need(std::isfinite(half_value(half[i])),"K overflows F16");}
                state.keys.write(reinterpret_cast<const char*>(half.data()),half.size()*2);
                need(bool(state.keys),"streamed K write failed");
                state.positions+=s.count;++state.key_calls;
                std::memcpy(state.rope.data(),t->op_params,GGML_MAX_OP_PARAMS);
            } else {
                need(s.count==1&&x.size()==4096,"decode Q/norm shape");
                if(kind==2)state.previous_norm=std::move(x);
                if(kind==3)state.current_norm=std::move(x);
                if(kind==4)state.actual_query=std::move(x);
            }
            return true;
        } catch(const std::exception& e){s.failure=name+": "+e.what();return false;}
    }
};
// Identical operator reconstruction to the validated single-layer freeze tool.
std::vector<float> project(llama_model* model,int layer,const std::vector<float>& norm,
                          int position,const std::array<char,GGML_MAX_OP_PARAMS>& rope) {
    need(norm.size()==4096,"projection source missing");
    auto* w=const_cast<ggml_tensor*>(model->get_tensor(("blk."+std::to_string(layer)+".attn_q.weight").c_str()));
    need(w&&w->ne[0]==4096&&w->ne[1]==4096,"actual target Wq ABI");
    auto device=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    need(device,"GPU projection required");
    auto backend=ggml_backend_dev_init(device,nullptr);
    auto* context=ggml_init({2*1024*1024,nullptr,true});
    need(backend&&context,"projection allocation");
    auto* x=ggml_new_tensor_2d(context,GGML_TYPE_F32,4096,1);
    auto* p=ggml_new_tensor_1d(context,GGML_TYPE_I32,1);
    auto* q=ggml_reshape_3d(context,ggml_mul_mat(context,w,x),128,32,1);
    q=ggml_rope_ext(context,q,p,nullptr,128,0,1048576,3580165449.0f,1,0,1,32,1);
    std::memcpy(q->op_params,rope.data(),GGML_MAX_OP_PARAMS);
    auto* graph=ggml_new_graph(context);ggml_build_forward_expand(graph,q);
    auto buffer=ggml_backend_alloc_ctx_tensors(context,backend);need(buffer,"projection GPU buffer");
    ggml_backend_tensor_set(x,norm.data(),0,4096*4);ggml_backend_tensor_set(p,&position,0,4);
    need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"GPU Q projection failed");
    ggml_backend_synchronize(backend);
    std::vector<float> result(4096);ggml_backend_tensor_get(q,result.data(),0,4096*4);
    ggml_backend_buffer_free(buffer);ggml_free(context);ggml_backend_free(backend);
    return result;
}
llama_token greedy(llama_context* context,const llama_vocab* vocab) {
    const auto* logits=llama_get_logits_ith(context,-1);need(logits,"missing actual logits");
    const int n=llama_vocab_n_tokens(vocab);
    for(int i=0;i<n;++i)need(std::isfinite(logits[i]),"nonfinite logits");
    return std::max_element(logits,logits+n)-logits;
}
}

int main(int argc,char** argv) {
    if(argc!=5){std::cerr<<"MODEL TOKEN_IDS NEW_DIR P\n";return 2;}
    const std::filesystem::path out=argv[3];
    if(!std::filesystem::create_directory(out)){std::cerr<<"new capture directory required\n";return 2;}
    try {
        const auto start=Clock::now();const int P=std::stoi(argv[4]);
        need(P==256||P==4096||P==16384||P==32768,"bounded preregistered lengths only");
        const auto input=load<llama_token>(argv[2],P);static_assert(sizeof(llama_token)==4);
        setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);ggml_backend_load_all();llama_backend_init();
        llama_hillserve_configure(nullptr); // native full attention, same trajectory for all comparisons
        auto mp=llama_model_default_params();mp.n_gpu_layers=99;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        need(bool(model),"model load");
        need(llama_model_n_layer(model.get())==32&&llama_model_n_head(model.get())==32&&
             llama_model_n_head_kv(model.get())==8&&llama_model_n_embd(model.get())==4096,"Gradient GQA ABI");
        const auto* vocab=llama_model_get_vocab(model.get());
        for(auto token:input)need(token>=0&&token<llama_vocab_n_tokens(vocab),"invalid frozen tokenizer ID");
        Capture capture;
        for(int l:TARGETS){auto& s=capture.layers[l];s.keys.open(out/("layer-"+std::to_string(l)+"-K-f16.bin"),std::ios::binary);need(bool(s.keys),"K output open");}
        auto cp=llama_context_default_params();cp.n_ctx=(P+8+255)/256*256;
        cp.n_batch=128;cp.n_ubatch=128;cp.n_threads=4;cp.n_threads_batch=4;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn=false;cp.offload_kqv=true;
        cp.rope_freq_base=3580165449.0f;cp.rope_freq_scale=1;
        cp.cb_eval=Capture::callback;cp.cb_eval_user_data=&capture;
        std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);
        need(bool(context),"native context");
        for(int off=0;off<P;off+=128){capture.offset=off;capture.count=std::min(128,P-off);
            need(llama_decode(context.get(),llama_batch_get_one(const_cast<llama_token*>(input.data()+off),capture.count))==0&&capture.failure.empty(),"prefill: "+capture.failure);
            llama_synchronize(context.get());}
        const double prefill_ms=milliseconds(start);json points=json::array(),generated=json::array();
        for(int step=1;step<=8;++step) {
            auto token=greedy(context.get(),vocab);generated.push_back(token);
            capture.offset=P+step-1;capture.count=1;
            capture.observe=std::find(STEPS.begin(),STEPS.end(),step)!=STEPS.end();
            if(capture.observe)for(auto& [l,s]:capture.layers){(void)l;s.previous_norm.clear();s.current_norm.clear();s.actual_query.clear();}
            need(llama_decode(context.get(),llama_batch_get_one(&token,1))==0&&capture.failure.empty(),"decode: "+capture.failure);
            llama_synchronize(context.get());
            if(!capture.observe)continue;
            for(auto& [l,s]:capture.layers) {
                const auto exact=project(model.get(),l,s.current_norm,P+step-1,s.rope);
                need(s.actual_query.size()==4096,"missing actual Q");
                std::size_t errors=0;for(std::size_t i=0;i<4096;++i)errors+=std::memcmp(&exact[i],&s.actual_query[i],4)!=0;
                need(errors==0,"same-input Wq/RoPE differs from model Q: layer="+std::to_string(l));
                const auto rehearsal=project(model.get(),l,s.previous_norm,P+step-1,s.rope);
                const auto prefix="layer-"+std::to_string(l)+"-step-"+std::to_string(step);
                save(out/(prefix+"-actual-Q-f32.bin"),s.actual_query);
                save(out/(prefix+"-rehearsal-Q-f32.bin"),rehearsal);
                points.push_back({{"layer_zero_based",l},{"layer_one_based",l+1},{"step",step},
                    {"position",P+step-1},{"valid_candidates",P+step},{"prefix",prefix},
                    {"same_input_projection_bit_errors",errors}});
            }
            std::cout<<"captured P="<<P<<" decode_step="<<step<<" six_layers\n"<<std::flush;
        }
        std::uint64_t key_bytes=0;
        for(auto& [l,s]:capture.layers){need(s.positions==std::size_t(P+8),"K history coverage");s.keys.close();
            const auto bytes=std::filesystem::file_size(out/("layer-"+std::to_string(l)+"-K-f16.bin"));
            need(bytes==std::uint64_t(P+8)*2048,"K streamed extent");key_bytes+=bytes;}
        struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
        const json result={{"status","passed"},{"evidence_class","actual_native_model_multi_layer_checkpoints"},
            {"P",P},{"B",1},{"decode_steps",8},{"checkpoints",points},{"generated_token_ids",generated},
            {"query_source","current_layer_attention_RMSNorm_target_Wq_target_RoPE_before_F16_rounding"},
            {"reference_trajectory","native_full_attention_greedy_same_forward_state"},
            {"K_representation","same_F16_post_RoPE_K_as_native_cache"},{"key_file_bytes",key_bytes},
            {"native_KV_bytes",llama_hillserve_native_kv_bytes(context.get())},
            {"capture_workspace_bound_bytes",16ULL<<20},{"process_peak_RSS_KiB",usage.ru_maxrss},
            {"prefill_including_load_ms",prefill_ms},{"total_ms",milliseconds(start)},
            {"serving_performance_claim",false},{"future_keys_excluded_by_each_checkpoint_valid_prefix",true}};
        std::ofstream(out/"manifest.json")<<result.dump(2)<<'\n';return 0;
    } catch(const std::exception& e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
