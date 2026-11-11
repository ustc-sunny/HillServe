#include "runtime/executor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <sys/resource.h>
using namespace hillserve;
namespace {
void need(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
struct Run { std::vector<llama_token> generated;std::vector<std::vector<float>> logits;double prefill_ms=0,decode_ms=0,execute_wall_ms=0,context_init_ms=0,context_teardown_ms=0; };
std::string precise(double v){std::ostringstream f;f<<std::setprecision(17)<<v;return f.str();}
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
Run execute(llama_model* m,LayerExecutor& runtime,std::vector<llama_token> prompt,int ng,bool keep_logits){
    const auto request_start=Clock::now();runtime.configure();
    auto p=llama_context_default_params();p.n_ctx=512;p.n_batch=256;p.n_ubatch=((prompt.size()+31)/32)*32;p.n_seq_max=1;
    p.n_threads=4;p.n_threads_batch=4;p.type_k=GGML_TYPE_F16;p.type_v=GGML_TYPE_F16;p.flash_attn=false;p.offload_kqv=true;
    p.cb_eval=LayerExecutor::callback;p.cb_eval_user_data=&runtime;
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(m,p),llama_free);need(bool(ctx),"context creation failed");
    runtime.attach_profile(ctx.get());
    Run result;result.context_init_ms=ms(request_start);std::vector<std::uint64_t> positions(prompt.size());std::iota(positions.begin(),positions.end(),0);
    runtime.begin(0,positions);auto t=Clock::now();int rc=llama_decode(ctx.get(),llama_batch_get_one(prompt.data(),prompt.size()));
    need(rc==0&&runtime.failure.empty(),"prefill: "+runtime.failure+" rc="+std::to_string(rc));llama_synchronize(ctx.get());result.prefill_ms=ms(t);runtime.accounting(ctx.get(),true);
    int nv=llama_vocab_n_tokens(llama_model_get_vocab(m));
    for(int step=0;step<=ng;++step){
        float* ptr=llama_get_logits_ith(ctx.get(),-1);need(ptr,"missing logits");
        for(int i=0;i<nv;++i)need(std::isfinite(ptr[i]),"nonfinite logits");
        if(keep_logits)result.logits.emplace_back(ptr,ptr+nv);
        if(step==ng)break;
        llama_token token=std::max_element(ptr,ptr+nv)-ptr;result.generated.push_back(token);
        runtime.begin(step+1,{std::uint64_t(prompt.size()+step)});t=Clock::now();
        rc=llama_decode(ctx.get(),llama_batch_get_one(&token,1));
        need(rc==0&&runtime.failure.empty(),"decode: "+runtime.failure+" rc="+std::to_string(rc));llama_synchronize(ctx.get());result.decode_ms+=ms(t);runtime.accounting(ctx.get(),false);
    }
    need(runtime.stats.attention_checks==std::size_t(28*(ng+1)),"attention coverage incomplete");runtime.release();
    auto teardown=Clock::now();ctx.reset();result.context_teardown_ms=ms(teardown);result.execute_wall_ms=ms(request_start);
    return result;
}

void execution_pilot(llama_model* model,const std::vector<llama_token>& input,const std::filesystem::path& out,
                     int ng,int slots,std::size_t gpu_budget,std::size_t host_budget){
    need(input.size()==17&&ng==8&&slots==32,"focused performance pilot requires P17/G8/S32");
    std::ofstream records(out/"request-runs.jsonl");records<<std::setprecision(17);
    std::ofstream owned(out/"owned_files.json");owned<<"{\"files\":[";bool first=true;
    int process_sequence=0;
    for(int repeat=0;repeat<4;++repeat){
        std::vector<std::string> modes={"durable_per_layer","volatile_query_kv"};if(repeat==2)std::reverse(modes.begin(),modes.end());
        for(const auto& mode:modes){
            const std::string name=mode+"-r"+std::to_string(repeat);const auto dir=out/name;
            need(std::filesystem::create_directory(dir),"new request directory required");
            if(!first)owned<<",";first=false;owned<<"\""<<name<<"/kv-records.bin\"";
            setenv("HILLSERVE_KV_DURABILITY",mode.c_str(),1);
            Run run;RuntimeStats stats;std::size_t metadata_peak=0;
            const auto outer=Clock::now();
            {
                LayerExecutor runtime(dir.string(),true,slots,32,gpu_budget,host_budget,nullptr,nullptr,nullptr,false);
                run=execute(model,runtime,input,ng,false);stats=runtime.stats;metadata_peak=runtime.metadata_peak();
                need(runtime.backing_bytes()==28*(input.size()+ng)*2048,"pilot append bytes mismatch");
                need(stats.native_history_bytes==0&&stats.own_shadow_bytes==0,"profile contains hidden history");
            }
            const double request_wall=ms(outer);
            std::ostringstream record;record<<std::setprecision(17)<<"{\"request\":\""<<name<<"\",\"durability\":\""<<mode<<"\",\"repeat\":"<<repeat<<",\"warmup\":"<<(repeat==0?"true":"false")
                <<",\"process_sequence\":"<<process_sequence++<<",\"request_wall_ms\":"<<request_wall<<",\"execute_wall_ms\":"<<run.execute_wall_ms
                <<",\"context_init_ms\":"<<run.context_init_ms<<",\"context_teardown_ms\":"<<run.context_teardown_ms<<",\"prefill_ms\":"<<run.prefill_ms<<",\"decode_ms\":"<<run.decode_ms
                <<",\"decode_call_windows_ms\":"<<(run.prefill_ms+run.decode_ms)<<",\"decode_calls\":"<<ng<<",\"total_llama_decode_calls\":"<<(ng+1)<<",\"output_token_count\":"<<run.generated.size()<<",\"logit_group_count\":"<<(ng+1)
                <<",\"evidence_class\":\"EXECUTOR_PILOT\",\"policy_id\":null,\"precision\":\"default\",\"physical_path\":\"systemfs_buffered\",\"processes\":1,\"steady_state_scope\":\"same_process_new_context_request\",\"formal_speedup_eligible\":false}";
            records<<record.str()<<'\n';records.flush();std::ofstream timing(dir/"request-timing.json");timing<<record.str()<<'\n';
            std::ofstream mem(dir/"kv-memory-accounting.json");mem<<"{\"native_full_model_gpu_history_bytes\":"<<stats.native_history_bytes<<",\"gpu_workspace_allocator_bytes\":"<<stats.gpu_workspace_bytes
               <<",\"host_kv_reserved_upper_bound_bytes\":"<<stats.host_payload_peak<<",\"index_allocator_peak_requested_bytes\":"<<metadata_peak<<",\"debug_shadow_bytes\":0,\"debug_actual_logits_bytes\":0,\"file_page_cache_reservation_bytes\":"<<stats.file_cache_reservation_bytes
               <<",\"dirty_peak_upper_bound_bytes\":"<<stats.dirty_peak_bytes<<",\"dirty_after_final_drain_bytes\":0,\"graph_prefill_gpu_bytes\":"<<stats.graph_prefill_gpu<<",\"graph_decode_gpu_bytes\":"<<stats.graph_decode_gpu<<"}\n";
        }
    }
    owned<<"],\"per_file_cap_bytes\":67108864,\"actual_payload_bytes_per_file\":1433600,\"system_cache_not_userspace_KV_budget\":true}\n";
}

Plans read_plans(const std::string& name){
    need(std::filesystem::file_size(name)<=512*1024,"plan file exceeds diagnostic budget");
    std::size_t plan_bytes=0;
    std::ifstream f(name);need(bool(f),"plan file missing");Plans r;std::string line;
    while(std::getline(f,line)){if(line.empty())continue;std::istringstream row(line);int l,s;std::string ids;need(bool(row>>l>>s>>ids),"bad FetchPlan line");
        need(l>=0&&l<28&&s>=0&&s<=64&&!r.count({l,s}),"duplicate/invalid plan identity");
        std::vector<std::uint64_t> v;std::istringstream list(ids);std::string val;
        while(std::getline(list,val,',')){need(!val.empty()&&val.find_first_not_of("0123456789")==std::string::npos,"invalid ID encoding");need(v.size()<512,"plan ID count exceeds limit");v.push_back(std::stoull(val));}plan_bytes+=128+v.capacity()*8;need(plan_bytes<=256*1024,"plan metadata exceeds budget");r.emplace(std::make_pair(l,s),std::move(v));}
    return r;
}
void stats_json(std::ostream& f,const RuntimeStats& s,bool compared){
 f<<"\"appends\":"<<s.appends<<",\"gathers\":"<<s.gathers<<",\"gpu_attention_checks\":"<<s.attention_checks
  <<",\"kv_identity_checks\":"<<s.kv_identity_checks<<",\"attention_max_abs_error\":"<<(compared?precise(s.max_attention_error):"null")
  <<",\"append_prefill_bytes\":"<<s.append_prefill<<",\"append_decode_bytes\":"<<s.append_decode<<",\"selected_kv_bytes\":"<<s.fetch_bytes
  <<",\"append_ms\":"<<s.append_ms<<",\"fetch_ms\":"<<s.fetch_ms<<",\"h2d_ms\":"<<s.h2d_ms;
}
}
int main(int argc,char**argv){
 if(argc!=11){std::cerr<<"usage: hillserve-layer-runtime MODEL NEW_DIR P G SLOTS all|provided PLAN|- verify|verify-compact|verify-numerical|none GPU_WORKSPACE_BYTES HOST_KV_BYTES\n";return 2;}
 std::filesystem::path out=argv[2];if(!std::filesystem::create_directory(out)){std::cerr<<"require new result directory\n";return 2;}
 std::ofstream report(out/"runtime-tests.json");
 try{
    setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
    int np=std::stoi(argv[3]),ng=std::stoi(argv[4]),slots=std::stoi(argv[5]);
    bool compact_ref=std::string(argv[8])=="verify-compact";
    bool numerical=std::string(argv[8])=="verify-numerical";
    bool verify=std::string(argv[8])=="verify"||compact_ref||numerical;need(verify||std::string(argv[8])=="none","invalid oracle mode");
    need(std::filesystem::space(out).available>1024ULL*1024*1024+64*1024*1024,"insufficient file-test reserve");
    need(np>=1&&np<=256&&ng>=1&&ng<=64&&np+ng<=512,"workload exceeds round2 pilot bound");
    std::size_t gpu_budget=std::stoull(argv[9]),host_budget=std::stoull(argv[10]);
    need(host_budget<=64*1024*1024&&gpu_budget<=16*1024*1024,"budget exceeds pilot limit");
    bool provided=std::string(argv[6])=="provided";need(provided||std::string(argv[6])=="all","paper_policy/per-head mode unsupported");
    need(!compact_ref||provided,"compact reference only for provided IDs");
    Plans plans;if(provided)plans=read_plans(argv[7]);else need(std::string(argv[7])=="-","all mode takes no IDs file");
    const std::size_t debug_bound=verify ? std::size_t(28)*(np+ng)*(3584*4+2048+256)+std::size_t(ng+1)*152064*4*2+1024*1024+(numerical?std::size_t(28)*(np+ng)*(2048+256):0) : 0;
    need(debug_bound<=256*1024*1024,"debug shadow preflight budget exceeded");
    std::ofstream manifest(out/"owned_files.json");manifest<<"{\"files\":[\"kv-records.bin\"],\"file_cap_bytes\":67108864,\"format\":\"KV-F16-Qwen2-v1\",\"heads\":4,\"head_dim\":128,\"record_bytes\":2048,\"key\":\"request,layer,absolute_position\"}\n";
    ggml_backend_load_all();llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=99;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);need(bool(model),"model load failed");
    need(llama_model_n_layer(model.get())==28&&llama_model_n_head(model.get())==28&&llama_model_n_head_kv(model.get())==4,"Qwen ABI mismatch");
    const auto* vocab=llama_model_get_vocab(model.get());
    need(llama_model_n_embd(model.get())==3584&&llama_vocab_n_tokens(vocab)==152064,"fixed checkpoint budget ABI mismatch");std::string text;
    std::string prompt_id=std::getenv("HILLSERVE_PROMPT")?std::getenv("HILLSERVE_PROMPT"):"A";need(prompt_id=="A"||prompt_id=="B","invalid prompt split");
    while(text.size()<10000)text+=prompt_id=="A"?"The library stores blue books on shelf seven. Readers borrow books and return them after a week. ":"The garden has red flowers beside a small pond. Every morning the gardener waters the trees and checks the soil. ";
    int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);std::vector<llama_token> input(n);
    need(llama_tokenize(vocab,text.data(),text.size(),input.data(),n,true,true)==n,"tokenize failed");input.resize(np);
    if(std::getenv("HILLSERVE_EXECUTION_PILOT")){
        need(std::string(std::getenv("HILLSERVE_EXECUTION_PILOT"))=="1"&&!verify&&!provided,"pilot requires all/none");
        const char* precision=std::getenv("HILLSERVE_ATTN_VALUE_ACCUM");need(!precision||std::string(precision)=="default","pilot precision must be default");
        manifest.close();execution_pilot(model.get(),input,out,ng,slots,gpu_budget,host_budget);
        report<<"{\"status\":\"passed\",\"measurement_mode\":\"EXECUTOR_PILOT\",\"same_process_requests\":8,\"warmups_per_mode\":1,\"measured_requests_per_mode\":3,\"independent_processes\":1,\"formal_speedup_eligible\":false}\n";return 0;
    }
    std::map<std::string,std::vector<float>> attention;
    std::map<std::string,std::vector<ggml_fp16_t>> kv;
    Run ref; std::size_t reference_history_bytes=0;
    int max_tokens=(np+31)/32*32;
    if(verify){LayerExecutor oracle(out.string(),false,compact_ref?slots:512,max_tokens,16*1024*1024,host_budget,provided?&plans:nullptr,&attention,&kv,true,compact_ref);
        ref=execute(model.get(),oracle,input,ng,true); reference_history_bytes=oracle.stats.native_history_bytes;}
    std::size_t shadow_bytes=0;for(const auto& p:attention)shadow_bytes+=p.second.capacity()*sizeof(float);for(const auto& p:kv)shadow_bytes+=p.second.capacity()*2;
    for(const auto& v:ref.logits)shadow_bytes+=v.capacity()*sizeof(float);
    need(shadow_bytes<=256*1024*1024,"debug shadow budget exceeded");
    LayerExecutor runtime(out.string(),true,slots,max_tokens,gpu_budget,host_budget,provided?&plans:nullptr,verify?&attention:nullptr,verify?&kv:nullptr,verify,false,numerical);
    Run actual=execute(model.get(),runtime,input,ng,verify);
    double max_error=0;std::size_t logit_violations=0;bool tokens_equal=!verify||ref.generated==actual.generated;
    if(verify){if(!numerical)need(tokens_equal,"generated tokens differ");need(ref.logits.size()==actual.logits.size(),"logit coverage differs");
        std::ofstream diffs(out/"logit-differences.jsonl");
        for(std::size_t s=0;s<ref.logits.size();++s){double step_max=0,sum_error2=0,sum_reference2=0;std::size_t violations=0;
            for(std::size_t i=0;i<ref.logits[s].size();++i){double a=ref.logits[s][i],b=actual.logits[s][i],e=std::abs(a-b);max_error=std::max(max_error,e);step_max=std::max(step_max,e);sum_error2+=e*e;sum_reference2+=a*a;
                need(std::isfinite(a)&&std::isfinite(b),"nonfinite comparison logits");if(e>1e-5+1e-5*std::abs(a)){++violations;if(!numerical)throw std::runtime_error("logits fixed tolerance exceeded");}}
            auto margin=[](const std::vector<float>& x){auto top=std::max_element(x.begin(),x.end());float second=-INFINITY;for(auto it=x.begin();it!=x.end();++it)if(it!=top)second=std::max(second,*it);return double(*top)-second;};
            logit_violations+=violations;diffs<<std::setprecision(17)<<"{\"step\":"<<s<<",\"max_abs_error\":"<<step_max<<",\"violations\":"<<violations<<",\"elements\":"<<ref.logits[s].size()<<",\"rms_error\":"<<std::sqrt(sum_error2/ref.logits[s].size())<<",\"relative_l2_error\":"<<(sum_reference2?precise(std::sqrt(sum_error2/sum_reference2)):"null")<<",\"reference_logits_checksum\":\""<<checksum(ref.logits[s].data(),ref.logits[s].size()*sizeof(float))<<"\",\"actual_logits_checksum\":\""<<checksum(actual.logits[s].data(),actual.logits[s].size()*sizeof(float))<<"\",\"reference_top1_top2_margin\":"<<margin(ref.logits[s])<<",\"actual_top1_top2_margin\":"<<margin(actual.logits[s])<<"}\n";}}
    bool numerical_pass=tokens_equal&&logit_violations==0&&runtime.stats.attention_violations==0;
    if(numerical){std::ofstream nreport(out/"numerical-contract.json");nreport<<std::setprecision(17)<<"{\"contract\":\"MODEL_NUMERICAL_V1\",\"status\":\""<<(numerical_pass?"passed":"failed")<<"\",\"storage_exact_status\":\"passed\",\"storage_exact_checks\":"<<runtime.stats.storage_exact_checks<<",\"cross_chain_kv_drift_words\":"<<runtime.stats.kv_drift_words<<",\"cross_chain_kv_max_abs_error\":"<<runtime.stats.kv_drift_max<<",\"attention_violations\":"<<runtime.stats.attention_violations<<",\"logit_violations\":"<<logit_violations<<",\"tokens_equal\":"<<(tokens_equal?"true":"false")<<",\"nonfinite\":false,\"own_storage_debug_shadow_bytes_upper_bound\":"<<runtime.stats.own_shadow_bytes<<"}\n";}

    need(runtime.stats.appends==std::size_t(28*(np+ng)),"append count is not incremental");
    need(runtime.stats.append_decode==std::size_t(ng)*28*2048,"decode bytes include historical KV");
    need(runtime.backing_bytes()==std::size_t(np+ng)*28*2048,"backing payload mismatch");
    std::ofstream mem(out/"kv-memory-accounting.json");const auto& s=runtime.stats;struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
    mem<<"{\"native_full_model_gpu_history_bytes\":"<<s.native_history_bytes<<",\"native_reference_history_bytes\":"<<reference_history_bytes
       <<",\"reference_context_freed_before_external\":true,\"persistent_gpu_kv_bytes\":"<<s.kv_bytes<<",\"persistent_gpu_mask_bytes\":"<<s.mask_bytes
       <<",\"gpu_workspace_allocator_bytes\":"<<s.gpu_workspace_bytes<<",\"gpu_workspace_budget_bytes\":"<<gpu_budget
       <<",\"current_kv_graph_tensor_peak_bytes\":"<<s.current_peak<<",\"graph_allocator_prefill_gpu_bytes\":"<<s.graph_prefill_gpu
       <<",\"graph_allocator_decode_gpu_bytes\":"<<s.graph_decode_gpu<<",\"graph_allocator_cpu_bytes\":"<<s.graph_cpu
       <<",\"current_kv_prefill_bytes\":"<<s.current_prefill<<",\"current_kv_decode_bytes\":"<<s.current_decode<<",\"metadata_prefill_peak_requested_bytes\":"<<s.metadata_prefill<<",\"metadata_decode_peak_requested_bytes\":"<<s.metadata_decode<<",\"host_kv_budget_bytes\":"<<host_budget<<",\"host_kv_reserved_upper_bound_bytes\":"<<s.host_payload_peak
       <<",\"file_page_cache_reservation_bytes\":"<<s.file_cache_reservation_bytes<<",\"dirty_peak_upper_bound_bytes\":"<<s.dirty_peak_bytes<<",\"dirty_after_final_drain_bytes\":0,\"index_allocator_peak_requested_bytes\":"<<runtime.metadata_peak()<<",\"debug_shadow_bytes\":"<<(shadow_bytes+runtime.stats.own_shadow_bytes)
       <<",\"debug_budget_bytes\":268435456,\"debug_prevalidated_upper_bound_bytes\":"<<debug_bound<<",\"debug_actual_logits_bytes\":"<<(verify?std::size_t(ng+1)*llama_vocab_n_tokens(vocab)*4:0)
       <<",\"process_peak_rss_kib\":"<<ru.ru_maxrss<<",\"weight_gpu_allocator_bytes\":"<<llama_hillserve_weight_bytes(model.get(),true)<<",\"weight_host_allocator_bytes\":"<<llama_hillserve_weight_bytes(model.get(),false)<<",\"output_buffers\":\"separate native allocator log\"}\n";
    report<<std::setprecision(17)<<"{\"status\":\""<<(numerical_pass?"passed":"failed")<<"\",\"selection_mode\":\""<<(provided?"provided_ids_diagnostic":"all_tokens_diagnostic")
        <<"\",\"measurement_mode\":\"correctness\",\"policy_mode\":\"diagnostic\",\"compute_backend\":\"not_run\",\"attention_backend\":\"gpu\",\"candidate_path\":\"unknown\",\"payload_path\":\"systemfs_buffered\",\"smartssd_io\":false,\"fpga_executed\":false,\"snapshot_calls\":0,\"restore_calls\":0,\"native_full_gpu_kv\":false,\"debug_oracle\":"<<(verify?"true":"false")
        <<",\"reference_mode\":\""<<(compact_ref?"compact_memory_oracle":verify?"native_masked_oracle":"none")<<"\",\"P\":"<<np<<",\"G\":"<<ng<<",\"logits_max_abs_error\":"<<(verify?precise(max_error):"null")<<",\"execute_wall_ms\":"<<actual.execute_wall_ms<<",\"decode_calls\":"<<ng<<",\"total_llama_decode_calls\":"<<(ng+1)<<",\"output_token_count\":"<<actual.generated.size()<<",\"logit_group_count\":"<<(ng+1)<<",\"prefill_ms\":"<<actual.prefill_ms<<",\"decode_ms\":"<<actual.decode_ms<<",";
    stats_json(report,runtime.stats,verify);report<<",\"formal_speedup_eligible\":false}\n";
    std::ofstream tokens(out/"generated-token-ids.json");tokens<<"[";for(std::size_t i=0;i<actual.generated.size();++i){if(i)tokens<<",";tokens<<actual.generated[i];}tokens<<"]\n";
    if(!numerical_pass){std::cerr<<"MODEL_NUMERICAL_V1 failed; full numerical/storage reports preserved\n";return 1;}
    std::cout<<"PASS incremental layer runtime; GPU history allocation zero; snapshots zero\n";return 0;
 }catch(const std::exception& e){llama_hillserve_configure(nullptr);std::ofstream error(out/"error.txt");error<<e.what()<<'\n';
    report<<"{\"status\":\"failed\",\"formal_speedup_eligible\":false,\"error_log\":\"error.txt\"}\n";std::cerr<<e.what()<<'\n';return 1;}
}
