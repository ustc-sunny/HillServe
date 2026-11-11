// Matched model runner with opt-in fixed-output performance measurements.
// Quality defaults retain the same binary, weights, token IDs, RoPE and
// no-flash attention operators; FULL_NATIVE differs from HILLSERVE in KV policy.
// Task quality is evaluated by the dataset scorer, never by backend checksums.
#include "runtime/gqa_executor.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "policy/full_gqa_runtime.hpp"
#include "runtime/no_ssd_policy.hpp"
#include "runtime/prompt_recompute.hpp"
#include "runtime/gqa_h2o_stats.hpp"
#include "policy/h2o_residency.hpp"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace hillserve;
namespace {
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
constexpr const char* IMAGE_SHA = HILLSERVE_V3_IMAGE_SHA256;
void need(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }
std::uint64_t stamp() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
std::string read_text(const std::filesystem::path& path, std::size_t bound=256ULL<<20) {
    need(std::filesystem::file_size(path)<=bound, "input text exceeds explicit bound");
    std::ifstream f(path, std::ios::binary); need(bool(f), "cannot read "+path.string());
    return {std::istreambuf_iterator<char>(f), {}};
}
void save(const std::filesystem::path& path, const json& value) {
    std::ofstream f(path); f<<value.dump(2)<<'\n'; need(bool(f), "cannot save "+path.string());
}
std::uint64_t digest(const void* data, std::size_t bytes) {
    auto p=static_cast<const unsigned char*>(data); std::uint64_t h=1469598103934665603ULL;
    for(std::size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ULL;} return h;
}
std::string token_digest(const std::vector<llama_token>& tokens) {
    std::uint64_t h=1469598103934665603ULL;
    for(auto token:tokens) for(unsigned s=0;s<32;s+=8){h^=(std::uint32_t(token)>>s)&255;h*=1099511628211ULL;}
    return std::to_string(h);
}
void write_tokens(const std::filesystem::path& path, const std::vector<llama_token>& tokens) {
    std::ofstream f(path,std::ios::binary);
    for(auto token:tokens) for(unsigned s=0;s<32;s+=8) f.put(char((std::uint32_t(token)>>s)&255));
    need(bool(f), "cannot write little-endian int32 token file");
}
std::vector<llama_token> read_tokens(const std::filesystem::path& path, std::size_t limit) {
    auto bytes=std::filesystem::file_size(path);
    need(bytes>0&&bytes%4==0&&bytes/4<=limit,"invalid/bounded little-endian int32 token input");
    std::ifstream f(path,std::ios::binary);need(bool(f),"cannot read token file");
    std::vector<llama_token> result(bytes/4);
    for(auto& token:result){std::uint32_t u=0;for(unsigned s=0;s<32;s+=8){int c=f.get();need(c!=EOF,"truncated token file");u|=std::uint32_t(static_cast<unsigned char>(c))<<s;}need(u<=std::uint32_t(INT32_MAX),"negative token ID");token=llama_token(u);}
    return result;
}
std::string metadata(llama_model* model,const char* name) {
    int n=llama_model_meta_val_str(model,name,nullptr,0); if(n<0)return {};
    std::vector<char> buffer(std::size_t(n)+1); llama_model_meta_val_str(model,name,buffer.data(),buffer.size()); return buffer.data();
}
std::string piece(const llama_vocab* vocab,llama_token token) {
    char buffer[256];int n=llama_token_to_piece(vocab,token,buffer,sizeof buffer,0,false);
    if(n>=0)return {buffer,std::size_t(n)};
    std::vector<char> large(-n);n=llama_token_to_piece(vocab,token,large.data(),large.size(),0,false);
    need(n>=0,"token text conversion failed");return {large.data(),std::size_t(n)};
}
int tokenize_only(int argc,char** argv) {
    need(argc==5||argc==6,"usage: --tokenize MODEL.gguf INPUT.txt OUTPUT.tokens.bin [--no-special]");
    const bool special=argc==5;need(special||std::string(argv[5])=="--no-special","unknown tokenize flag");
    const std::filesystem::path output=argv[4];need(!std::filesystem::exists(output)&&!std::filesystem::exists(output.string()+".json"),"token outputs must be new");
    llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;mp.n_gpu_layers=0;
    Model model(llama_model_load_from_file(argv[2],mp),llama_model_free);need(bool(model),"vocabulary load failed");
    const auto text=read_text(argv[3]);const auto* vocab=llama_model_get_vocab(model.get());
    int n=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,special,true);need(n<0,"empty tokenization");
    std::vector<llama_token> tokens(-n);need(llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),special,true)==int(tokens.size()),"tokenization mismatch");
    write_tokens(output,tokens);
    const char* chat=llama_model_chat_template(model.get(),nullptr);
    json info={{"schema","hillserve-quality-tokenization-v1"},{"model",argv[2]},{"model_file_bytes",std::filesystem::file_size(argv[2])},{"text",argv[3]},{"text_bytes",text.size()},{"text_fnv64",std::to_string(digest(text.data(),text.size()))},{"tokens_file",output.string()},{"token_count",tokens.size()},{"token_ids_fnv64",token_digest(tokens)},{"encoding","little_endian_int32"},{"add_special",special},{"parse_special",true},{"chat_template_applied",false},{"chat_template_metadata",chat?json(chat):json(nullptr)},{"tokenizer_model",metadata(model.get(),"tokenizer.ggml.model")},{"tokenizer_pre",metadata(model.get(),"tokenizer.ggml.pre")},{"vocab_size",llama_vocab_n_tokens(vocab)},{"model_name",metadata(model.get(),"general.name")},{"vocab_only",true},{"SHA256_provenance","dataset wrapper must record model/text/token file SHA256"}};
    save(output.string()+".json",info);std::cout<<info.dump()<<'\n';return 0;
}
struct NativeCoverage {
    std::uint64_t attention=0;std::string failure;
    bool expect_cpu=false;
    std::array<std::uint64_t,32> cpu_layers{},gpu_layers{};
    ggml_backend_t completed_backend=nullptr;ggml_tensor* completed_tensor=nullptr;
    std::uint64_t cpu_pinned_host_outputs=0;
    static void profile(ggml_backend_t backend,ggml_tensor* tensor,int phase,int,void* user){
        if(phase==2){auto& self=*static_cast<NativeCoverage*>(user);self.completed_backend=backend;self.completed_tensor=tensor;}
    }
    int first_query=0,query_count=0;void* stream=nullptr;
    std::function<void(ggml_tensor*,int,std::uint64_t,std::size_t,std::size_t,void*)> h2o_observer;
    static bool callback(ggml_tensor* tensor,bool ask,void* user) {
        auto& self=*static_cast<NativeCoverage*>(user);const std::string name=tensor->name;
        const bool attention=name.rfind("kqv_out-",0)==0,h2o=name.rfind("hs_h2o_prob_",0)==0;if(ask)return attention||h2o;
        if(!self.failure.empty())return false;
        try{
            auto* b=tensor->buffer?tensor->buffer:tensor->view_src?tensor->view_src->buffer:nullptr;
            need(b,"native attention buffer missing");auto dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b));
            // CPU execution may use CUDA_Host pinned memory. Its allocator
            // device is GPU; the scheduler backend determines execution.
            if(self.expect_cpu)need(self.completed_tensor==tensor&&self.completed_backend,"native CPU attention lacks completed scheduler-backend evidence");
            const auto execution_dev=self.expect_cpu?ggml_backend_get_device(self.completed_backend):dev;
            const bool cpu=ggml_backend_buffer_is_host(b)&&execution_dev&&ggml_backend_dev_type(execution_dev)==GGML_BACKEND_DEVICE_TYPE_CPU;
            const bool gpu=!ggml_backend_buffer_is_host(b)&&dev&&ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU;
            need(self.expect_cpu?cpu:gpu,self.expect_cpu?"native CPU-KV attention output must reside on CPU":"native attention must execute on GPU device memory");
            if(h2o){int layer=-1;need(std::sscanf(name.c_str(),"hs_h2o_prob_%d",&layer)==1&&bool(self.h2o_observer),"native H2O observer missing");self.h2o_observer(tensor,layer,self.first_query,self.query_count,self.first_query+self.query_count,self.stream);}else {
                int layer=-1;need(std::sscanf(name.c_str(),"kqv_out-%d",&layer)==1&&layer>=0&&layer<32,"native attention layer identity missing");
                ++self.attention;++(cpu?self.cpu_layers:self.gpu_layers)[layer];
                if(cpu&&dev&&ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU)++self.cpu_pinned_host_outputs;
            }return true;
        }catch(const std::exception& e){self.failure=e.what();return false;}
    }
};
struct Batch {
    llama_batch value;explicit Batch(int n):value(llama_batch_init(n,0,1)){}~Batch(){llama_batch_free(value);}
    void set(const llama_token* tokens,int n,int position) {
        value.n_tokens=n;
        for(int i=0;i<n;++i){value.token[i]=tokens[i];value.pos[i]=position+i;value.n_seq_id[i]=1;value.seq_id[i][0]=0;value.logits[i]=i==n-1;}
    }
};
}

int main(int argc,char** argv) {
    std::filesystem::path out;
    try {
        if(argc>1&&std::string(argv[1])=="--tokenize")return tokenize_only(argc,argv);
        need(argc==2,"usage: hillserve-quality-runtime CONFIG.json");
        const auto cfg=json::parse(read_text(argv[1],4ULL<<20));out=cfg.at("output").get<std::string>();
        need(std::filesystem::create_directory(out),"new output directory required");save(out/"config.json",cfg);
        const auto method=cfg.at("method").get<std::string>(),task=cfg.at("task").get<std::string>();
        const bool performance=task=="performance",h2o=method=="H2O_RECOMPUTE",full_recompute=method=="FULL_RECOMPUTE",full_attention=h2o||full_recompute,no_ssd=full_attention||method=="PREFETCH_RECOMPUTE";
        const bool extended=cfg.value("long_context_extension",false);
        const auto count_precision=cfg.value("selector_count_precision",std::string("F16"));
        need(count_precision=="F16"||count_precision=="FP32","unknown selector count precision");
        need(count_precision!="FP32"||extended,"FP32 counts require explicit authorized long-context extension");
        need(!extended||(performance&&(method=="HILLSERVE"||method=="HOST_MATCHED"||full_recompute)),"extended context supports the three performance methods only");
        const auto recompute_group_limit=cfg.value("recompute_layer_group_limit",std::size_t(1));
        need(recompute_group_limit>=1&&recompute_group_limit<=32&&(full_recompute||recompute_group_limit==1),"layer-group recovery is only supported by FULL_RECOMPUTE");
        need(method=="FULL_NATIVE"||method=="HILLSERVE"||(performance&&(method=="HOST_MATCHED"||no_ssd)),"method must be FULL_NATIVE/HILLSERVE, or HOST_MATCHED for performance");
        need(task=="qa"||task=="teacher_forced"||performance,"task must be qa, teacher_forced or performance");
        bool hill=method!="FULL_NATIVE";const bool teacher=task=="teacher_forced",verify=cfg.value("verify",false);
        const auto native_kv_device=cfg.value("native_kv_device",std::string("gpu"));
        need(native_kv_device=="gpu"||native_kv_device=="cpu","native_kv_device must be gpu or cpu");
        const bool native_cpu_kv=native_kv_device=="cpu";
        need(!native_cpu_kv||method=="FULL_NATIVE","CPU KV mode is only supported by FULL_NATIVE");
        const bool early=full_attention?false:cfg.value("early_rehearsal",true),payload_prefetch=full_attention?false:cfg.value("payload_prefetch",true);
        need(!hill||!payload_prefetch||early,"payload prefetch requires early rehearsal");
        auto prompt=read_tokens(cfg.at("input_tokens_file").get<std::string>(),extended?65536:32768);
        auto targets=teacher?read_tokens(cfg.at("target_tokens_file").get<std::string>(),512):std::vector<llama_token>{};
        const int np=int(prompt.size()),ng=teacher?int(targets.size()):cfg.value("max_new_tokens",128);
        need(cfg.value("batch",1)==1&&np>=128&&np<=(extended?65536:32768)&&ng>=1&&ng<=512,"bounded quality/explicit long-context extension geometry");
        need(!performance||(ng>=2&&ng<=64),"bounded performance G2..64");
        if(cfg.contains("prompt_tokens"))need(cfg.at("prompt_tokens").get<int>()==np,"declared prompt length differs from exact token IDs");
        const int capacity=(np+ng+63)/64*64,chunk=cfg.value("prefill_chunk",256);
        need(chunk>=1&&chunk<=256,"prefill chunk must be1..256");
        const int gpu_layers=cfg.value("gpu_layers",99);const bool low_gpu=cfg.value("functional_low_gpu",false);
        bool native_prefill=hill&&cfg.value("native_dense_prefill",false);
        need(!extended||native_prefill,"extended capacity requires native prefill with deferred external workspace");
        need(!native_prefill||gpu_layers==99,"native dense prefill requires all-GPU model weights");
        need(gpu_layers==99||(low_gpu&&gpu_layers==0),"only all-GPU or explicit CPU weights/GPU attention supported");
        need(!full_recompute||native_prefill,"FULL_RECOMPUTE requires bounded native prefill export followed by released GPU history");
        const std::uint64_t host_budget=cfg.value("host_budget_bytes",8ULL<<30),gpu_budget=cfg.value("gpu_workspace_budget_bytes",3ULL<<30);
        std::uint64_t debug_bytes=hill&&verify?32ULL*capacity*4096:0;
        need(debug_bytes<=cfg.value("debug_shadow_budget_bytes",512ULL<<20),"explicit debug shadow budget exceeded");
        setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);setenv("HILLSERVE_GQA_DIRECT_IO","1",1);
        ggml_backend_load_all();llama_backend_init();llama_hillserve_configure(nullptr);
        auto device=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(device,"real GPU required");
        std::size_t gpu_free=0,gpu_total=0;ggml_backend_dev_memory(device,&gpu_free,&gpu_total);
        const std::uint64_t native_plan=std::uint64_t((capacity+255)/256*256)*32*8*128*4;
        const auto model_path=cfg.at("model").get<std::string>();
        const std::uint64_t gpu_total_budget=cfg.value("gpu_total_budget_bytes",24ULL<<30);
        const bool native_h2o=h2o&&cfg.value("native_resident_if_fits",true)&&gpu_layers==99&&std::filesystem::file_size(model_path)+native_plan+gpu_budget<=std::min<std::uint64_t>(gpu_free,gpu_total_budget);
        if(native_h2o){hill=false;native_prefill=false;debug_bytes=0;}

        need(!native_cpu_kv||native_plan<=host_budget,"RESOURCE_CAP: complete CPU KV exceeds host budget");
        std::uint64_t required_gpu=(gpu_layers==99?std::filesystem::file_size(model_path):0)+(extended?0:(hill?gpu_budget+(native_prefill?native_plan:0):(native_cpu_kv?0:native_plan)))+(1ULL<<30);
        need(required_gpu<=gpu_free&&required_gpu<=gpu_total_budget,"RESOURCE_CAP: model/KV/workspace preflight exceeds available/budget GPU memory");
        auto begin=Clock::now();auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers;
        Model model(llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);need(bool(model),"Gradient model load failed");const double load_ms=ms(begin);
        std::uint64_t actual_device_GPU_peak_bytes=0;
        auto check_total_gpu=[&]{std::size_t f=0,t=0;ggml_backend_dev_memory(device,&f,&t);need(f<=t,"invalid GPU memory observation");actual_device_GPU_peak_bytes=std::max<std::uint64_t>(actual_device_GPU_peak_bytes,t-f);need(!extended||t-f<=gpu_total_budget,"RESOURCE_CAP: actual device GPU use exceeds total budget");};
        if(extended){
            // File size includes CPU embedding tensors. After real model load,
            // charge actual device occupancy plus the maximum phase allocation.
            // Native history and external workspace are never simultaneously live.
            std::size_t f=0,t=0;ggml_backend_dev_memory(device,&f,&t);
            const auto phase=std::max(native_plan,gpu_budget)+(1ULL<<30);
            required_gpu=t-f+phase;
            need(phase<=f&&required_gpu<=gpu_total_budget,"RESOURCE_CAP: actual model plus bounded64K phase exceeds GPU memory");
        }
        check_total_gpu();
        need(llama_model_n_layer(model.get())==32&&llama_model_n_head(model.get())==32&&llama_model_n_head_kv(model.get())==8&&llama_model_n_embd(model.get())==4096,"Gradient8B geometry mismatch");
        need(metadata(model.get(),"general.architecture")=="llama"&&metadata(model.get(),"general.file_type")=="32","unchanged Llama BF16 GGUF required");
        need(np+ng<=llama_model_n_ctx_train(model.get()),"context exceeds verified model context");
        const auto* vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
        for(auto id:prompt)need(id>=0&&id<nv,"prompt ID outside model vocabulary");for(auto id:targets)need(id>=0&&id<nv,"target ID outside model vocabulary");
        write_tokens(out/"prompt-token-ids.bin",prompt);if(teacher)write_tokens(out/"target-token-ids.bin",targets);
        // Declaration order is intentional: executor waits for policy jobs before
        // it frees staging, and the context is destroyed before either object.
        std::unique_ptr<policy::FullGqaRuntimePolicy> scoring;
        std::unique_ptr<runtime::PromptRecompute> recompute;
        std::unique_ptr<policy::NoSSDPolicy> memory_policy;
        std::unique_ptr<policy::H2OResidency> h2o_policy;
        std::unique_ptr<GQAH2OStatsCuda> h2o_reduction;
        std::unique_ptr<GQALayerExecutor> runtime; // Destroy before every callback dependency above.
        std::uint64_t h2o_d2h=0,h2o_updates=0;double h2o_reduce_ms=0;
        auto observe_h2o=[&](ggml_tensor* t,int layer,std::uint64_t first,std::size_t queries,std::size_t keys,void* stream){
            need(h2o_policy&&t->type==GGML_TYPE_F32,"actual F32 attention required for H2O");
            if(!h2o_reduction)h2o_reduction=std::make_unique<GQAH2OStatsCuda>(t->data,capacity);
            auto r=h2o_reduction->reduce(t->data,{t->ne[0],t->ne[1],t->ne[2],t->ne[3]},{t->nb[0],t->nb[1],t->nb[2],t->nb[3]},keys,queries,stream);
            h2o_policy->observe_group_sums(layer,first,queries,keys,r.sums,r.elements);
            if(memory_policy)memory_policy->set_h2o_resident(layer,h2o_policy->resident_ids(layer));
            h2o_d2h+=r.d2h_bytes;h2o_reduce_ms+=r.reduction_ms+r.d2h_ms;++h2o_updates;
        };
        policy::FullGqaRuntimePolicy::Config pc;
        double policy_init_ms=0;
        if(hill&&!no_ssd){
            need(performance||(early&&payload_prefetch),"quality HillServe uses current early-rehearsal/payload-prefetch path");
            auto fence=std::make_shared<runtime::GQAPrefetchIOFence>();
            pc.layers=32;pc.capacity=capacity;pc.prompt_tokens=np;pc.generated_capacity=ng;pc.allow_partial_prompt=true;
            pc.fp32_selection_counts=count_precision=="FP32";
            pc.selector_alpha=cfg.value("selector_alpha",5.f);
            need(std::isfinite(pc.selector_alpha)&&pc.selector_alpha>=0,"selector_alpha must be finite and nonnegative");
            pc.beta=cfg.at("beta");pc.hot_kv_budget_bytes=cfg.at("hot_kv_budget_bytes");pc.total_host_workspace_budget_bytes=cfg.value("policy_workspace_budget_bytes",4ULL<<30);
            pc.chunk_tokens=cfg.value("score_chunk_tokens",4096);pc.hot_threads=cfg.value("hot_threads",3);pc.cold_threads=cfg.value("cold_threads",1);
            pc.cpu_implementation=cfg.value("cpu_implementation",std::string("auto"));
            need(pc.cpu_implementation=="auto"||(performance&&pc.cpu_implementation=="legacy"),"quality requires auto CPU scorer; performance supports auto or legacy");
            pc.mode=cfg.value("scoring_backend",method=="HOST_MATCHED"?std::string("HOST_MATCHED"):std::string("FPGA_CSS"));need(pc.mode=="FPGA_CSS"||pc.mode=="FPGA_SERIAL"||pc.mode=="HOST_MATCHED","unknown scoring backend");
            need(method!="HOST_MATCHED"||pc.mode=="HOST_MATCHED","HOST_MATCHED method cannot execute an FPGA backend");
            pc.score_file=cfg.at("score_file");pc.verify_scores=verify;pc.hpm=cfg.value("hpm",false);pc.hpm_swaps=cfg.value("hpm_swaps",1);
            pc.payload_prefetch=payload_prefetch;pc.full_kv_backing_path=cfg.at("kv_file");pc.full_kv_io_fence=fence;
            pc.payload_read_queue_depth=cfg.value("payload_read_queue_depth",4u);
            if(pc.mode!="HOST_MATCHED"){
                const auto image=cfg.at("fpga");need(image.at("execute_authorized").get<bool>(),"approved image execution required");
                pc.hardware.abi="HS_FULL_GQA_PIPELINE_F16_V3";pc.hardware.image_path=image.at("path");pc.hardware.image_sha256=image.at("sha256");pc.hardware.image_execute_authorized=true;
                need(pc.hardware.image_sha256==IMAGE_SHA,"image does not match the build-time validated V3 SHA256");pc.hardware.buffer_budget_bytes=256ULL<<20;pc.hardware.reuse_file_buffers=true;pc.hardware.direct_read_queue_depth=1;
            }
            begin=Clock::now();scoring=std::make_unique<policy::FullGqaRuntimePolicy>(pc);policy_init_ms=ms(begin);
            if(cfg.contains("resident_kv_payload_budget_bytes"))need(scoring->stats().hot_kv_cache_bytes==cfg.at("resident_kv_payload_budget_bytes").get<std::size_t>(),"fixed resident KV payload differs from common budget");
            need(scoring->managed_bytes()<host_budget,"policy exceeds total managed host budget");
            GQARuntimeHooks hooks;hooks.early_rehearsal=early;hooks.payload_prefetch=payload_prefetch;hooks.full_kv_io_fence=fence;
            hooks.compact_dense_prefill=cfg.value("compact_dense_prefill",true);
            hooks.native_dense_prefill=native_prefill;
            hooks.defer_native_workspace=extended;
            hooks.kv_file_quota_bytes=(extended?9ULL:8ULL)<<30;
            hooks.native_prefill_kv_budget_bytes=native_prefill?native_plan:0;
            hooks.wait_pending=[&]{scoring->wait_pending();};
            hooks.append_full=[&](int l,const auto& positions,const auto& keys,const auto& bytes){scoring->append(l,positions,keys,bytes);};
            hooks.fetch=[&](int l,const auto& ids,const GQAColdFetch& fetch){return scoring->fetch(l,ids,fetch);};
            hooks.prepare_history=[&](int l,std::uint64_t position,const auto& q,runtime::GQAPrefetchReady ready){scoring->prepare_history(l,position,q,std::move(ready));};
            runtime=std::make_unique<GQALayerExecutor>(out.string(),cfg.at("kv_file").get<std::string>(),capacity,chunk,.2f,false,verify,true,gpu_budget,host_budget-scoring->managed_bytes(),GQAAppend{},[&](int l,std::uint64_t position,const auto& q){return scoring->select(l,position,q);},hooks);
            need(runtime->stats.managed_host_bytes+scoring->managed_bytes()<=host_budget,"combined all-layer host budget exceeded");
            std::vector<std::uint64_t> initial(std::min(np,chunk));std::iota(initial.begin(),initial.end(),0);runtime->begin(0,initial,false);
        }
        if(no_ssd&&hill){
            const auto policy_started=Clock::now();
            GQARuntimeHooks hooks;hooks.early_rehearsal=early;hooks.payload_prefetch=payload_prefetch;
            hooks.full_kv_io_fence=std::make_shared<runtime::GQAPrefetchIOFence>();
            hooks.compact_dense_prefill=true;hooks.native_dense_prefill=native_prefill;
            hooks.defer_native_workspace=extended;hooks.compact_full_workspace=extended&&full_recompute;
            hooks.native_prefill_kv_budget_bytes=native_prefill?native_plan:0;
            hooks.external_memory_backing=true;hooks.full_attention=full_attention;
            hooks.skip_native_export_readback=full_recompute;hooks.h2o_attention_stats=h2o;
            if(h2o)hooks.observe_attention=observe_h2o;
            hooks.wait_pending=[&]{if(memory_policy)memory_policy->wait_pending();};
            hooks.append_full=[&](int l,const auto& positions,const auto& keys,const auto& bytes){memory_policy->append_full(l,positions,keys,bytes);};
            hooks.fetch=[&](int l,const auto& ids,const GQAColdFetch&){return memory_policy->fetch(l,ids);};
            hooks.prepare_history=[&](int l,std::uint64_t position,const auto& q,runtime::GQAPrefetchReady ready){memory_policy->prepare_history(l,position,q,std::move(ready));};
            const std::uint64_t auxiliary=(full_recompute?256ULL:128ULL)<<20;
            need(host_budget>auxiliary,"noSSD metadata budget");
            runtime=std::make_unique<GQALayerExecutor>(out.string(),"",capacity,chunk,full_attention?1.f:.2f,false,verify,false,gpu_budget,host_budget-auxiliary,GQAAppend{},[&](int l,std::uint64_t position,const auto& q){return memory_policy->select(l,position,q);},hooks);
            need(host_budget>runtime->stats.managed_host_bytes+auxiliary,"noSSD host budget exhausted by executor");
            runtime::PromptRecompute::Config rc;
            rc.host_budget_bytes=(full_recompute?160ULL:64ULL)<<20;rc.max_batch_heads=full_recompute?8:1;
            rc.gpu_graph_budget_bytes=(extended?512ULL:1024ULL)<<20;
            rc.gpu_budget_bytes=native_plan+rc.gpu_graph_budget_bytes;
            if(recompute_group_limit>1){
                // Reuse the existing 256MiB auxiliary reservation; the global
                // managed-host cap and persistent hot-KV quota do not increase.
                rc.host_budget_bytes=auxiliary;rc.max_target_layers=recompute_group_limit;
                rc.max_requested_rows=(np/64-runtime::full_gqa_hot_blocks(np,cfg.at("beta"),true))*64;
                need(rc.max_requested_rows>0,"grouped recovery requires a nonempty fixed cold set");
            }
            rc.cpu_threads=cfg.value("model_cpu_threads",4);rc.chunk_tokens=chunk;
            rc.event_log_path=(out/"recompute-events.jsonl").string();rc.total_gpu_budget_bytes=gpu_total_budget;
            recompute=std::make_unique<runtime::PromptRecompute>(model.get(),prompt,rc);
            policy::NoSSDPolicy::Config mc;mc.mode=full_recompute?"FULL_RECOMPUTE":h2o?"H2O":"PREFETCH";mc.capacity=capacity;mc.prompt_tokens=np;mc.generated_capacity=ng;
            mc.extended_full_context=extended&&full_recompute;
            mc.total_host_budget_bytes=host_budget-runtime->stats.managed_host_bytes-auxiliary;
            if(cfg.contains("no_ssd_policy_budget_bytes")){need(cfg.value("functional_recompute_gate",false),"separate low-memory budget only allowed in labelled functional gate");mc.total_host_budget_bytes=std::min<std::uint64_t>(mc.total_host_budget_bytes,cfg.at("no_ssd_policy_budget_bytes"));}
            mc.staging_budget_bytes=std::uint64_t(capacity)*4096*4+(16ULL<<20);mc.selector_alpha=cfg.value("selector_alpha",5.f);
            mc.cpu_threads=cfg.value("model_cpu_threads",4);mc.cpu_implementation=cfg.value("cpu_implementation",std::string("auto"));
            if(full_recompute){mc.offline_beta=cfg.at("beta");mc.allow_partial_prompt=true;mc.resident_kv_payload_budget_bytes=cfg.at("resident_kv_payload_budget_bytes");
                memory_policy=std::make_unique<policy::NoSSDPolicy>(mc,policy::NoSSDPolicy::Recompute{},[&](int l,const auto& ids){return recompute->batch_call(l,ids);});
            }else memory_policy=std::make_unique<policy::NoSSDPolicy>(mc,[&](int l,int h,const auto& ids){return (*recompute)(l,h,ids);});
            if(h2o){policy::H2OResidency::Config hc;hc.prompt_tokens=np;hc.generated_capacity=ng;hc.prompt_resident_tokens_per_head=std::min<std::size_t>(np,memory_policy->resident_rows_per_head()-ng);hc.metadata_budget_bytes=64ULL<<20;h2o_policy=std::make_unique<policy::H2OResidency>(hc);}
            policy_init_ms=ms(policy_started);
            std::vector<std::uint64_t> initial(std::min(np,chunk));std::iota(initial.begin(),initial.end(),0);runtime->begin(0,initial,false);
        }
        if(native_h2o){
            policy::H2OResidency::Config hc;hc.prompt_tokens=np;hc.generated_capacity=ng;hc.prompt_resident_tokens_per_head=np;hc.metadata_budget_bytes=64ULL<<20;
            h2o_policy=std::make_unique<policy::H2OResidency>(hc);
            need(h2o_policy->reserved_metadata_bytes()+8ULL*capacity*4<=host_budget,"native H2O host metadata budget exceeded");
            need(8ULL*capacity*4<=gpu_budget,"native H2O reduction buffer budget exceeded");
            llama_hillserve_config hc_graph{};hc_graph.h2o_attention_stats=true;llama_hillserve_configure(&hc_graph);
        }
        NativeCoverage native;native.expect_cpu=native_cpu_kv;if(native_h2o)native.h2o_observer=observe_h2o;

        auto cp=llama_context_default_params();cp.n_ctx=capacity;cp.n_batch=chunk;cp.n_ubatch=chunk;cp.n_seq_max=1;
        cp.n_threads=cfg.value("model_cpu_threads",4);cp.n_threads_batch=cp.n_threads;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn=false;cp.offload_kqv=!native_cpu_kv;cp.op_offload=!low_gpu;cp.rope_freq_base=3580165449.0f;cp.rope_freq_scale=1;
        cp.cb_eval=hill?GQALayerExecutor::callback:NativeCoverage::callback;cp.cb_eval_user_data=hill?static_cast<void*>(runtime.get()):static_cast<void*>(&native);
        begin=Clock::now();Context ctx(llama_init_from_model(model.get(),cp),llama_free);need(bool(ctx),"model context allocation failed");const double context_ms=ms(begin);
        if(hill)llama_hillserve_set_profile(ctx.get(),[](ggml_backend_t backend,ggml_tensor*,int phase,int,void* opaque){if(phase==0)static_cast<GQALayerExecutor*>(opaque)->observe_backend(backend);},runtime.get());
        if(native_cpu_kv)llama_hillserve_set_profile(ctx.get(),NativeCoverage::profile,&native);
        if(native_h2o)llama_hillserve_set_profile(ctx.get(),[](ggml_backend_t backend,ggml_tensor*,int phase,int,void* opaque){if(phase==0&&ggml_backend_is_cuda(backend))static_cast<NativeCoverage*>(opaque)->stream=ggml_backend_cuda_hillserve_stream(backend);},&native);
        if(hill)runtime->accounting(ctx.get());check_total_gpu();
        Batch batch(chunk);std::ofstream token_log(out/"token-scores.jsonl"),steps(out/"steps.jsonl"),answer(out/"answer.txt");
        need(bool(token_log)&&bool(steps)&&bool(answer),"cannot create per-token output");
        std::vector<llama_token> generated,answer_ids;std::vector<std::uint64_t> emitted;
        const auto request=Clock::now();begin=Clock::now();int batches=0,calls=0,eos_count=0;double decode_ms=0,nll_sum=0,sparse_nll_sum=0;std::uint64_t graph_gpu=0;
        auto decode=[&](const llama_token* tokens,int n,int position,bool is_decode){
            if(is_decode&&recompute)recompute->begin_decode(position);
            if(hill){std::vector<std::uint64_t> positions(n);std::iota(positions.begin(),positions.end(),position);runtime->begin(batches+calls,positions,is_decode);}
            native.first_query=position;native.query_count=n;
            batch.set(tokens,n,position);const auto started=stamp();const auto timer=Clock::now();const int rc=llama_decode(ctx.get(),batch.value);llama_synchronize(ctx.get());
            need(rc==0&&(hill?runtime->failure.empty():native.failure.empty()),"model decode failed: "+(hill?runtime->failure:native.failure));
            if(is_decode&&recompute)recompute->end_decode();
            const double elapsed=ms(timer);if(hill)runtime->accounting(ctx.get());check_total_gpu();graph_gpu=std::max<std::uint64_t>(graph_gpu,llama_hillserve_graph_bytes(ctx.get(),true));
            steps<<json({{"phase",is_decode?"decode":"prefill"},{"position",position},{"tokens",n},{"started_ns",started},{"ended_ns",stamp()},{"wall_ms",elapsed},{"input_kind",is_decode?(teacher?"teacher_target":"generated_token"):"fixed_prompt"}}).dump()<<'\n';steps.flush();
            if(is_decode){++calls;decode_ms+=elapsed;}else ++batches;
        };
        for(int offset=0;offset<np;offset+=chunk)decode(prompt.data()+offset,std::min(chunk,np-offset),offset,false);
        if(native_prefill)runtime->finish_prefill(ctx.get(),np);
        const double prefill_ms=ms(begin);std::string stop_reason=teacher?"teacher_target_exhausted":performance?"fixed_output_budget":"max_new_tokens";
        const auto generation_begin=Clock::now();
        for(int index=0;index<ng;++index){
            const float* logits=llama_get_logits_ith(ctx.get(),-1);need(logits,"missing model logits");
            for(int i=0;i<nv;++i)need(std::isfinite(logits[i]),"nonfinite logits");
            const auto greedy=llama_token(std::max_element(logits,logits+nv)-logits);const auto token=teacher?targets[index]:greedy;
            double log_z=0,log_probability=0,nll=0;
            if(!performance){
                const double maximum=logits[greedy];double sum=0;for(int i=0;i<nv;++i)sum+=std::exp(double(logits[i])-maximum);
                log_z=maximum+std::log(sum);log_probability=double(logits[token])-log_z;nll=-log_probability;
                need(std::isfinite(nll)&&nll>=-1e-9,"invalid stable logsumexp token NLL");nll_sum+=nll;if(index)sparse_nll_sum+=nll;
            }
            const bool eog=llama_vocab_is_eog(vocab,token);eos_count+=eog;
            emitted.push_back(stamp());if(!teacher){generated.push_back(token);if(!eog){answer_ids.push_back(token);if(!performance){answer<<piece(vocab,token);answer.flush();}}}
            if(performance){
                token_log<<json({{"logits_fnv64",verify?json(std::to_string(digest(logits,std::size_t(nv)*4))):json(nullptr)},{"schema","hillserve-performance-token-v1"},{"output_index",index},{"position",np+index},{"generated_token",token},{"is_eog",eog},{"EOS_ignored_for_fixed_work",true},{"prediction_from_dense_prefill",index==0},{"actual_model_feedback",true},{"observed_ns",emitted.back()}}).dump()<<'\n';
            }else{
                token_log<<json({{"schema","hillserve-quality-token-v1"},{"task",task},{"output_index",index},{"position",np+index},{"context_tokens",np+index},{"prediction_from_dense_prefill",index==0},{"greedy_token",greedy},{"target_token",teacher?json(token):json(nullptr)},{"evaluated_token",token},{"token_log_probability",log_probability},{"token_nll",nll},{"logsumexp",log_z},{"is_eog",eog},{"logits_fnv64",std::to_string(digest(logits,std::size_t(nv)*4))},{"observed_ns",emitted.back()}}).dump()<<'\n';
            }
            token_log.flush();need(bool(token_log)&&bool(steps),"per-token receipt write failed");
            if(!teacher&&!performance&&eog){stop_reason="eog";break;}
            if(index+1<ng)decode(&token,1,np+index,true);
        }
        const double generation_ms=ms(generation_begin),request_before_drain_ms=ms(request);
        begin=Clock::now();if(hill){runtime->drain();if(scoring)scoring->drain();if(memory_policy)memory_policy->drain();}const double drain_ms=hill?ms(begin):0;
        const double request_ms=ms(request);const std::size_t evaluated=emitted.size();
        need(evaluated>0&&calls==int(evaluated)-1,"actual feedback coverage mismatch");
        if(teacher)need(evaluated==targets.size(),"incomplete teacher-forced continuation");
        if(performance)need(evaluated==std::size_t(ng)&&generated.size()==std::size_t(ng),"incomplete fixed-output generated workload");
        if(h2o){need(h2o_updates==32ULL*(batches+calls),"H2O actual probability updates incomplete");for(int l=0;l<32;++l)need(h2o_policy->observed_tokens(l)==std::size_t(np+calls),"H2O history accumulation incomplete");}
        const auto attention=hill?runtime->stats.attention_calls:native.attention;
        need(attention==32ULL*(batches+calls),"all32-layer actual attention coverage failed");
        if(method=="FULL_NATIVE"){
            need(!runtime&&!recompute&&!scoring&&!memory_policy,"FULL_NATIVE must not instantiate an external KV or recompute policy");
            const auto all_kv=llama_hillserve_native_kv_bytes(ctx.get()),gpu_kv=llama_hillserve_native_gpu_kv_bytes(ctx.get());
            need(all_kv==std::uint64_t(llama_n_ctx(ctx.get()))*32*4096,"native complete F16 KV history size mismatch");
            need(native_cpu_kv?gpu_kv==0:gpu_kv==all_kv,"native KV buffer placement differs from requested device");
            need(!native_cpu_kv||all_kv<=host_budget,"actual complete CPU KV exceeds host budget");
            for(int l=0;l<32;++l)need((native_cpu_kv?native.cpu_layers:native.gpu_layers)[l]==std::uint64_t(batches+calls),"native per-layer attention device coverage incomplete");
        }
        if(hill){need(runtime->stats.rehearsal_calls==(full_attention?0:30ULL*calls)&&runtime->stats.prefetch_layers==(payload_prefetch?30ULL*calls:0),"actual sparse query/payload coverage failed");need(runtime->stats.appends==32ULL*(np+calls),"actual incremental KV coverage failed");if(scoring)scoring->write_stats((out/"policy-stats.json").string());if(memory_policy)memory_policy->write_stats((out/"policy-stats.json").string());}
        save(out/"generated-token-ids.json",generated);save(out/"answer-token-ids.json",answer_ids);save(out/"token-emission-ns.json",emitted);
        if(performance){for(auto token:answer_ids)answer<<piece(vocab,token);answer.flush();need(bool(answer),"generated text write failed");}
        struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
        json result={{"schema","hillserve-quality-model-v1"},{"status","passed"},{"method",method},{"task",task},{"evidence_mode","ACTUAL_MODEL_QUALITY_OUTPUT"},{"quality_score_source","external_dataset_scorer"},{"simulation",false},{"batch",1},{"model",model_path},{"model_file_bytes",std::filesystem::file_size(model_path)},{"weight_dtype","BF16"},{"KV_dtype","F16"},{"flash_attention",false},{"rope_base",3580165449ULL},{"rope_scale",1},{"sampling","greedy"},{"prompt_tokens",np},{"prompt_ids_fnv64",token_digest(prompt)},{"target_ids_fnv64",teacher?json(token_digest(targets)):json(nullptr)},{"actual_context_tokens",llama_n_ctx(ctx.get())},{"requested_new_tokens",ng},{"evaluated_tokens",evaluated},{"answer_tokens",answer_ids.size()},{"generated_tokens",generated.size()},{"decode_calls",calls},{"prefill_calls",batches},{"prefill_dense_all_layers",true},{"first_prediction_from_dense_prefill",true},{"attention_calls",attention},{"rehearsal_calls",hill?runtime->stats.rehearsal_calls:0},{"sparse_layer_decode_calls",hill?30ULL*calls:0},{"EOS_count",eos_count},{"stop_reason",stop_reason},{"stops_at_EOG",!teacher},{"teacher_forcing",teacher},{"fixed_output_budget_ignores_EOS",teacher},{"total_token_nll",nll_sum},{"mean_token_nll",nll_sum/evaluated},{"perplexity",teacher?json(std::exp(nll_sum/evaluated)):json(nullptr)},{"post_prefill_mean_token_nll",evaluated>1?json(sparse_nll_sum/(evaluated-1)):json(nullptr)},{"actual_fpga",hill&&pc.mode!="HOST_MATCHED"},{"actual_target_storage",hill},{"gpu_layers",gpu_layers},{"functional_low_gpu",low_gpu},{"verify",verify},{"debug_shadow_bytes_separate",debug_bytes},{"gpu_free_before_model_bytes",gpu_free},{"gpu_required_preflight_bytes",required_gpu},{"gpu_total_budget_bytes",gpu_total_budget},{"gpu_workspace_budget_bytes",gpu_budget},{"host_budget_bytes",host_budget},{"graph_GPU_bytes",graph_gpu},{"full_native_GPU_history_bytes",llama_hillserve_native_kv_bytes(ctx.get())},{"process_peak_RSS_KiB",ru.ru_maxrss},{"prefill_ms",prefill_ms},{"decode_wall_ms",decode_ms},{"decode_call_mean_ms",calls?json(decode_ms/calls):json(nullptr)},{"interarrival_TPOT_ms",evaluated>1?json(double(emitted.back()-emitted.front())/1e6/(evaluated-1)):json(nullptr)},{"request_wall_ms",request_ms},{"final_drain_ms",drain_ms},{"model_load_ms",load_ms},{"context_init_ms",context_ms},{"policy_init_ms",policy_init_ms},{"formal_speedup_eligible",false},{"includes_per_token_quality_logsumexp_and_logging",true}};
        if(hill&&!no_ssd){const auto& s=runtime->stats;result.update({{"scoring_backend",pc.mode},{"CPU_scorer",scoring->stats().cpu_implementation_actual},{"selection_contract","head0_max_minus5_GE_mean_count_cap20_per_head_IDs"},{"query_source","previous_layer_RMSNorm_target_Wq_target_RoPE_rehearsal"},{"beta",pc.beta},{"HPM",pc.hpm},{"early_rehearsal",early},{"payload_prefetch",payload_prefetch},{"managed_host_reservation_bytes",s.managed_host_bytes+scoring->managed_bytes()},{"GPU_workspace_bytes",s.workspace_bytes},{"GPU_H2D_bytes",s.gpu_h2d_bytes},{"GPU_D2D_bytes",s.gpu_d2d_bytes},{"GPU_prefetch_H2D_bytes",s.prefetch_h2d_bytes},{"logical_selected_bytes",s.logical_selected_bytes},{"physical_selected_bytes",s.physical_selected_bytes},{"durability","volatile_query_kv+end_of_request_sync"}});
            if(const auto* io=runtime->store()){const auto& m=io->metrics;save(out/"kv-storage.json",{{"extent_bytes",io->extent_bytes()},{"logical_append_bytes",m.logical_append_bytes},{"logical_read_bytes",m.logical_read_bytes},{"syscall_write_bytes",m.syscall_write_bytes},{"syscall_read_bytes",m.syscall_read_bytes},{"direct_read_calls",m.direct_fetch_read_calls},{"drain_calls",m.drain_calls}});}
        }
        if(performance){
            // A CPU scoring run uses the same external-KV executor but is never
            // labelled as an actual FPGA/HillServe hardware result.
            const std::string measured_method=no_ssd?method:hill?(pc.mode=="HOST_MATCHED"?"HOST_MATCHED":"HILLSERVE"):"FULL_NATIVE";
            result.update({{"schema","hillserve-performance-model-v1"},{"method",measured_method},
                {"requested_method",method},{"evidence_mode","ACTUAL_MODEL_PERFORMANCE_PILOT"},
                {"quality_score_source",nullptr},{"stops_at_EOG",false},{"fixed_output_budget_ignores_EOS",true},
                {"total_token_nll",nullptr},{"mean_token_nll",nullptr},{"post_prefill_mean_token_nll",nullptr},
                {"includes_per_token_quality_logsumexp_and_logging",false},
                {"includes_per_token_progress_logging",true},{"includes_runtime_integrity_checks_and_event_instrumentation",true},
                {"timing_scope","prefill+fixed_output_generation+end_of_request_sync; excludes model/context/policy initialization and final result serialization"},
                {"decode_wall_scope","sum of synchronized llama_decode calls; excludes sampling and progress receipts"},
                {"generation_stage_wall_ms",generation_ms},{"request_before_drain_ms",request_before_drain_ms},
                {"end_of_request_sync_ms",drain_ms},{"generated_token_ids_fnv64",token_digest(generated)},
                {"request_started_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(request.time_since_epoch()).count()},
                {"CPU_scorer_requested",hill?json(pc.cpu_implementation):json(nullptr)},
                {"CPU_hot_threads",hill?json(pc.hot_threads):json(nullptr)},
                {"CPU_cold_threads",hill&&pc.mode=="HOST_MATCHED"?json(pc.cold_threads):json(nullptr)},
                {"legacy_per_call_thread_creation_included",hill&&pc.cpu_implementation=="legacy"},
                {"formal_speedup_eligible",false},
                {"performance_validation_scope","single execution; external driver must validate idle devices, matching inputs/configuration, binary/library identity and output equivalence"}});
        }
        if(method=="FULL_NATIVE"){
            const auto all_kv=llama_hillserve_native_kv_bytes(ctx.get()),gpu_kv=llama_hillserve_native_gpu_kv_bytes(ctx.get());
            result.update({{"native_kv_device_requested",native_kv_device},{"native_offload_kqv",cp.offload_kqv},
                {"full_native_KV_history_bytes",all_kv},{"full_native_GPU_history_bytes",gpu_kv},{"full_native_CPU_history_bytes",all_kv-gpu_kv},
                {"native_attention_output_CPU_layer_calls",native.cpu_layers},{"native_attention_output_GPU_layer_calls",native.gpu_layers},
                {"native_attention_CPU_pinned_host_output_calls",native.cpu_pinned_host_outputs},
                {"native_attention_device_evidence",native_cpu_kv?"completed_scheduler_backend_and_host_buffer":"GPU_device_buffer"},
                {"native_model_GPU_weight_bytes",llama_hillserve_weight_bytes(model.get(),true)},
                {"native_model_CPU_weight_bytes",llama_hillserve_weight_bytes(model.get(),false)},
                {"native_recompute_enabled",false},{"native_score_select_enabled",false}});
        }
        result["long_context_extension"]=extended;
        result["selector_count_precision"]=full_recompute?"not_used_full_attention":count_precision;
        result["deferred_external_GPU_workspace"]=extended;
        result["compact_full_GPU_workspace"]=extended&&full_recompute;
        result["recompute_graph_GPU_budget_bytes"]=full_recompute?json((extended?512ULL:1024ULL)<<20):json(nullptr);
        result["actual_device_GPU_peak_observed_bytes"]=std::max<std::uint64_t>(actual_device_GPU_peak_bytes,recompute?recompute->stats().global_GPU_used_peak_bytes:0);
        result["compact_dense_prefill"]=hill&&!native_prefill&&cfg.value("compact_dense_prefill",true);
        result["compact_dense_prefill_requested"]=hill&&cfg.value("compact_dense_prefill",true);
        if(hill&&!no_ssd){
            const auto& s=runtime->stats;
            result["native_dense_prefill"]=native_prefill;
            result["prefill_KV_location"]=native_prefill?"temporary_native_GPU_exported_to_SSD_then_released":"external_store_with_compact_or_expanded_GPU_workspace";
            result["native_prefill_KV_reserved_bytes"]=s.native_prefill_reserved_bytes;
            result["native_prefill_KV_peak_bytes"]=s.native_prefill_peak_bytes;
            result["native_prefill_context_tokens"]=s.native_prefill_context_tokens;
            result["native_prefill_layers"]=s.native_prefill_layers;
            result["native_prefill_export_bytes"]=s.native_prefill_export_bytes;
            result["native_prefill_export_verified_bytes"]=s.native_prefill_export_verified_bytes;
            result["native_prefill_released_bytes"]=s.native_prefill_released_bytes;
            result["native_prefill_release_ms"]=s.native_prefill_release_ms;
            result["phase_gpu_managed_peak_bytes"]=s.phase_gpu_managed_peak_bytes;
            result["selector_alpha"]=pc.selector_alpha;
            result["selector_cap_ratio"]=0.2;
            result["selector_cap_ratio_f32"]=double(.2f);
            result["selection_variant"]=pc.selector_alpha==5.f?"default_alpha5":"explicit_alpha_override";
            if(pc.selector_alpha!=5.f)result["selection_contract"]="head0_max_minus_alpha_GE_mean_count_cap20_per_head_IDs";
        }
        if(no_ssd){
            result.update({{"actual_fpga",false},{"actual_target_storage",false},{"KV_SSD_backing",false},{"method",method},
                {"H2O_native_full_residency",native_h2o},{"early_rehearsal",early},{"payload_prefetch",payload_prefetch},
                {"attention_policy",full_recompute?"full_attention_fixed_APP_CPU_residency_true_recompute":h2o?"full_attention_H2O_residency_only":"same_HillServe_rehearsal_selected_KV"},
                {"H2O_contract",h2o?json(policy::H2OResidency::contract_name):json(nullptr)},
                {"H2O_attention_updates",h2o_updates},{"H2O_probabilities_D2H_bytes",h2o_d2h},{"H2O_reduction_and_copy_ms",h2o_reduce_ms},
                {"sparse_layer_decode_calls",full_attention?0:30ULL*calls},{"CPU_scorer_requested",full_attention?json(nullptr):json(cfg.value("cpu_implementation",std::string("auto")))},
                {"CPU_hot_threads",nullptr},{"CPU_cold_threads",nullptr},{"legacy_per_call_thread_creation_included",!full_attention&&cfg.value("cpu_implementation",std::string("auto"))=="legacy"},
                {"selector_alpha",full_attention?json(nullptr):json(cfg.value("selector_alpha",5.f))},{"selector_cap_ratio",full_attention?1.0:0.2},
                {"memory_policy_all_KV_fits",native_h2o||(memory_policy&&memory_policy->all_kv_fits())},
                {"managed_host_reservation_bytes",native_h2o?h2o_policy->reserved_metadata_bytes()+(h2o_reduction?h2o_reduction->host_bytes():0):runtime->stats.managed_host_bytes+memory_policy->managed_bytes()+((full_recompute?256ULL:128ULL)<<20)}});
            if(hill){result["GPU_prefetch_H2D_bytes"]=runtime->stats.prefetch_h2d_bytes;result["physical_selected_bytes"]=runtime->stats.physical_selected_bytes;result["GPU_H2D_bytes"]=runtime->stats.gpu_h2d_bytes;result["GPU_D2D_bytes"]=runtime->stats.gpu_d2d_bytes;result["native_prefill_released_bytes"]=runtime->stats.native_prefill_released_bytes;result["GPU_workspace_bytes"]=runtime->stats.workspace_bytes;}
            result["recompute_calls"]=memory_policy?memory_policy->stats().recompute_calls:0;
            result["recomputed_KV_rows"]=memory_policy?memory_policy->stats().recomputed_rows:0;
            result["recompute_ms"]=memory_policy?memory_policy->stats().recompute_ms:0;
            result["recomputation_scope"]=full_recompute?"one_real_full_prompt_32_layer_replay_per_missing_target_layer; all_8_KV_heads_batched; generated_KV_pinned":"real_original_prompt_prefix_replay; unoptimized reference; generated KV pinned";
            if(recompute_group_limit>1)result["recomputation_scope"]="bounded_target_layer_groups_per_decode; real_full_prompt_replay; CPU_recovery_payload_freed_within_step; generated_KV_pinned";
            if(recompute){const auto& rs=recompute->stats();
                result["recompute_model_replays"]=rs.calls;result["recompute_target_layers_per_replay"]=rs.target_layers_per_replay;
                result["recompute_layer_group_limit"]=recompute_group_limit;
                result["recompute_layer_requests"]=rs.layer_requests;result["recompute_reused_layer_requests"]=rs.reused_layer_requests;
                result["recompute_completed_decode_steps"]=rs.completed_decode_steps;
                result["recompute_recovery_cache_reserved_bytes"]=rs.recovery_cache_reserved_bytes;
                result["recompute_recovery_cache_peak_bytes"]=rs.recovery_cache_peak_bytes;
                result["recompute_recovery_cache_live_bytes_at_exit"]=rs.recovery_cache_live_bytes;
                result["recompute_actual_prompt_tokens"]=rs.replayed_tokens;result["recompute_actual_layer_steps"]=rs.layer_steps;
                result["recompute_GPU_native_peak_bytes"]=rs.native_GPU_peak_bytes;result["recompute_GPU_graph_peak_bytes"]=rs.graph_GPU_peak_bytes;
                result["recompute_host_reserved_bytes"]=rs.host_managed_reserved_bytes;result["recompute_global_GPU_used_peak_bytes"]=rs.global_GPU_used_peak_bytes;
                json rr=json::array();for(const auto& c:recompute->events())rr.push_back({{"layer",c.layer},{"kv_head",c.kv_head},{"requested_heads",c.requested_heads},{"passed",c.passed},{"requested_rows",c.requested_rows},{"returned_bytes",c.returned_bytes},{"replayed_tokens",c.replayed_tokens},{"decode_calls",c.decode_calls},{"layer_steps",c.layer_steps},{"attention_steps",c.attention_steps},{"wall_ms",c.wall_ms},{"replay_ms",c.replay_ms},{"native_GPU_peak_bytes",c.native_GPU_peak_bytes},{"graph_GPU_peak_bytes",c.graph_GPU_peak_bytes},{"capture_D2H_bytes",c.capture_D2H_bytes}});
                save(out/"recompute-calls.json",rr);
                if(recompute_group_limit>1){for(std::size_t i=0;i<rr.size();++i){rr[i]["target_layers"]=recompute->events()[i].target_layers;rr[i]["decode_position"]=recompute->events()[i].decode_position;}save(out/"recompute-calls.json",rr);}
            }
            result["H2O_metadata_reserved_bytes"]=h2o_policy?h2o_policy->reserved_metadata_bytes():0;
            result["H2O_reduction_GPU_bytes"]=h2o_reduction?h2o_reduction->gpu_bytes():0;
            result["H2O_reduction_host_bytes"]=h2o_reduction?h2o_reduction->host_bytes():0;
            if(native_h2o)result["GPU_workspace_bytes"]=h2o_reduction?h2o_reduction->gpu_bytes():0;

        }
        if(cfg.contains("resident_kv_payload_budget_bytes")){
            need(bool(scoring)||bool(memory_policy),"explicit resident quota requires an external limited-KV method");
            result["common_resident_KV_payload_budget_bytes"]=cfg.at("resident_kv_payload_budget_bytes");
            result["actual_resident_KV_payload_reserved_bytes"]=scoring?scoring->stats().hot_kv_cache_bytes:memory_policy->stats().resident_kv_payload_reserved_bytes;
            result["offline_beta"]=cfg.at("beta");result["offline_calibration_sha256"]=cfg.value("offline_calibration_sha256",std::string());
            result["decode_retained_native_GPU_history_bytes"]=llama_hillserve_native_kv_bytes(ctx.get());
            need(result["actual_resident_KV_payload_reserved_bytes"]==result["common_resident_KV_payload_budget_bytes"],"final common resident payload mismatch");
            need(llama_hillserve_native_kv_bytes(ctx.get())==0,"limited-KV decode retained hidden native GPU history");
        }
        result["shared_GPU_pilot"]=cfg.value("shared_GPU_pilot",false);
        if(cfg.value("shared_GPU_pilot",false))result["performance_validation_scope"]="shared GPU exploratory pilot; no formal speedup claim";
        save(out/"result.json",result);std::cout<<result.dump()<<'\n';return 0;
    }catch(const std::exception& e){llama_hillserve_configure(nullptr);if(!out.empty()&&std::filesystem::is_directory(out))save(out/"failure.json",{{"schema","hillserve-quality-model-v1"},{"status","failed"},{"reason",e.what()}});std::cerr<<e.what()<<'\n';return 1;}
}
