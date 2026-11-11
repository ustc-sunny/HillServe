#include "llama.h"
#include "ggml-backend.h"
#include "storage/file_store.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
constexpr double abs_tolerance=1e-5,rtol=1e-5; // Fixed before execution: same graph, exact KV bytes; no precision change.
struct Observation {
    bool compare=false;
    int step=0;
    std::map<std::string,std::vector<float>> oracle;
    std::set<std::string> seen;
    std::size_t oracle_bytes=0,gpu_callbacks=0,callbacks=0;
    double max_error=0;
    std::string failure;
};
static std::vector<float> tensor_f32(ggml_tensor* t){
    need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"unsupported attention tensor layout/type");
    std::vector<float> x(ggml_nelements(t));
    ggml_backend_tensor_get(t,x.data(),0,ggml_nbytes(t));
    return x;
}
static bool observe(ggml_tensor* t,bool ask,void* opaque){
    auto& o=*static_cast<Observation*>(opaque);
    bool wanted=std::string(t->name).rfind("kqv_out-",0)==0;
    if(ask)return wanted;
    if(!wanted)return true;
    try {
        auto x=tensor_f32(t);
        auto key=std::to_string(o.step)+":"+t->name;
        need(o.seen.insert(key).second,"duplicate attention callback");
        auto* buf=t->buffer?t->buffer:(t->view_src?t->view_src->buffer:nullptr);
        need(buf!=nullptr,"attention buffer unavailable");
        auto dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
        need(dev&&ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU,"attention output not GPU resident");
        ++o.gpu_callbacks;++o.callbacks;
        if(!o.compare){
            need(o.oracle_bytes+x.size()*sizeof(float)<=256ULL*1024*1024,"debug oracle exceeds 256 MiB");
            o.oracle_bytes+=x.size()*sizeof(float);o.oracle.emplace(key,std::move(x));
        }else{
            auto it=o.oracle.find(key);need(it!=o.oracle.end()&&it->second.size()==x.size(),"attention coverage mismatch");
            for(std::size_t i=0;i<x.size();++i){
                double error=std::abs(double(x[i])-it->second[i]);
                need(std::isfinite(x[i])&&std::isfinite(it->second[i]),"nonfinite attention");
                o.max_error=std::max(o.max_error,error);
                need(error<=abs_tolerance+rtol*std::abs(it->second[i]),"attention tolerance exceeded");
            }
        }
        return true;
    }catch(const std::exception& e){o.failure=e.what();return false;}
}
struct Run {
    std::vector<llama_token> tokens;
    std::vector<std::vector<float>> logits;
    double prefill_ms=0,decode_ms=0,external_ms=0,total_ms=0;
    std::size_t restores=0,snapshot_peak=0;
};
static llama_token greedy(const std::vector<float>& x){
    need(!x.empty(),"empty logits");
    for(float y:x)need(std::isfinite(y),"nonfinite logits");
    return std::max_element(x.begin(),x.end())-x.begin();
}
static std::vector<std::uint8_t> state(llama_context* c){
    auto n=llama_state_seq_get_size(c,0);
    need(n>0&&n<=64ULL*1024*1024,"snapshot outside 64 MiB pilot cap");
    std::vector<std::uint8_t> x(n);
    need(llama_state_seq_get_data(c,x.data(),x.size(),0)==n,"state export failed");return x;
}
static void roundtrip(llama_context* c,hillserve::FileStore& f,Run& r,std::ofstream& events,int step){
    auto t=Clock::now();
    llama_synchronize(c);
    auto before=state(c);
    r.snapshot_peak=std::max(r.snapshot_peak,before.size());
    auto hash=hillserve::checksum(before.data(),before.size());
    auto last=llama_memory_seq_pos_max(llama_get_memory(c),0);
    f.replace(before); // completion means pwrite and fdatasync finished
    // Release the export copy before the external read; no hidden host source is used to restore.
    before.clear();before.shrink_to_fit();
    llama_memory_clear(llama_get_memory(c),true);
    need(llama_memory_seq_pos_max(llama_get_memory(c),0)==-1,"cache clear failed");
    auto disk=f.fetch();
    need(hillserve::checksum(disk.data(),disk.size())==hash,"file identity mismatch");
    need(llama_state_seq_set_data(c,disk.data(),disk.size(),0)==disk.size(),"external restore failed");
    need(llama_memory_seq_pos_max(llama_get_memory(c),0)==last,"positions changed");
    auto after=state(c);
    need(after==disk,"restored all-layer KV/metadata bytes differ");
    ++r.restores;
    double elapsed=ms(t);r.external_ms+=elapsed;
    events<<"{\"event\":\"external_roundtrip\",\"step\":"<<step<<",\"bytes\":"<<disk.size()
          <<",\"last_position\":"<<last<<",\"wall_ms\":"<<elapsed
          <<",\"identity\":true,\"storage_path_mode\":\"system_nvme_buffered_debug\"}\n";
}
static Run run(llama_model* model,std::vector<llama_token> prompt,int outputs,Observation& obs,
               hillserve::FileStore* store,std::ofstream& events){
    auto p=llama_context_default_params();
    p.n_ctx=512;p.n_batch=512;p.n_ubatch=512;p.n_seq_max=1;p.n_threads=4;p.n_threads_batch=4;
    p.type_k=GGML_TYPE_F16;p.type_v=GGML_TYPE_F16;p.flash_attn=false;p.offload_kqv=true;
    p.cb_eval=observe;p.cb_eval_user_data=&obs;p.no_perf=false;
    std::unique_ptr<llama_context,decltype(&llama_free)> c(llama_init_from_model(model,p),llama_free);
    need(bool(c),"context initialization failed");
    Run r;auto start=Clock::now();auto t=start;obs.step=0;
    need(llama_decode(c.get(),llama_batch_get_one(prompt.data(),prompt.size()))==0,"prefill failed: "+obs.failure);
    llama_synchronize(c.get());r.prefill_ms=ms(t);
    int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto capture=[&]{auto* a=llama_get_logits_ith(c.get(),-1);need(a,"missing logits");r.logits.emplace_back(a,a+nv);};
    capture();
    if(store)roundtrip(c.get(),*store,r,events,0);
    for(int i=0;i<outputs;++i){
        auto token=greedy(r.logits.back());r.tokens.push_back(token);obs.step=i+1;
        t=Clock::now();
        need(llama_decode(c.get(),llama_batch_get_one(&token,1))==0,"decode failed: "+obs.failure);
        llama_synchronize(c.get());r.decode_ms+=ms(t);capture();
        if(store)roundtrip(c.get(),*store,r,events,i+1);
    }
    r.total_ms=ms(start);
    need(obs.failure.empty(),obs.failure);
    llama_memory_clear(llama_get_memory(c.get()),true);
    need(llama_memory_seq_pos_max(llama_get_memory(c.get()),0)==-1,"release failed");
    return r;
}
int main(int argc,char**argv){
    if(argc!=5){std::cerr<<"usage: hillserve-kv-check MODEL NEW_RESULT_DIR PROMPT_TOKENS OUTPUT_TOKENS\n";return 2;}
    std::filesystem::path dir=argv[2];
    if(!std::filesystem::create_directory(dir)){std::cerr<<"result directory must be new\n";return 2;}
    std::ofstream report(dir/"summary.json"),events(dir/"events.jsonl");
    try{
        int np=std::stoi(argv[3]),ng=std::stoi(argv[4]);
        need(np>=1&&np<=256&&ng>=1&&ng<=64&&np+ng<=512,"unsupported pilot dimensions");
        std::ofstream manifest(dir/"owned_files.json");
        manifest<<"{\"files\":[\"kv-state.bin\"],\"max_file_bytes\":67108864,\"scope\":\"synthetic_public_prompt_debug\"}\n";
        ggml_backend_load_all();llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=99;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        need(bool(model),"model load failed");
        auto* vocab=llama_model_get_vocab(model.get());
        std::string prompt;
        // Authored public test text, tokenized normally (no fabricated KV or positions).
        while(prompt.size()<10000)prompt+="The library stores blue books on shelf seven. Readers borrow books and return them after a week. ";
        int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
        std::vector<llama_token> tokens(n);
        need(llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),n,true,true)==n,"tokenization failed");
        need(n>=np,"prompt too short");tokens.resize(np);
        std::ofstream input(dir/"input_tokens.json");input<<"[";for(int i=0;i<np;++i){if(i)input<<",";input<<tokens[i];}input<<"]\n";
        Observation obs;
        auto native=run(model.get(),tokens,ng,obs,nullptr,events);
        auto count=obs.callbacks;
        need(count==std::size_t(llama_model_n_layer(model.get())*(ng+1)),"missing per-layer attention observations");
        obs.compare=true;obs.seen.clear();obs.callbacks=0;
        hillserve::FileStore store(dir/"kv-state.bin",64ULL*1024*1024);
        auto external=run(model.get(),tokens,ng,obs,&store,events);
        need(obs.callbacks==count,"attention coverage changed");
        need(native.tokens==external.tokens,"greedy generated tokens differ");
        need(native.logits.size()==external.logits.size(),"logit step count differs");
        double logits_max=0;
        for(std::size_t s=0;s<native.logits.size();++s)for(std::size_t i=0;i<native.logits[s].size();++i){
            double a=native.logits[s][i],b=external.logits[s][i],e=std::abs(a-b);
            need(std::isfinite(a)&&std::isfinite(b),"nonfinite logits");logits_max=std::max(logits_max,e);
            need(e<=abs_tolerance+rtol*std::abs(a),"logit tolerance exceeded");
        }
        struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
        report<<std::setprecision(12)<<"{\"status\":\"passed\",\"scope\":\"debug_oracle_full_retention\","
          <<"\"real_gpu\":true,\"real_fpga\":false,\"smartssd_io\":false,\"full_gpu_kv_allocation_retained\":true,"
          <<"\"storage_path_mode\":\"system_nvme_buffered_debug\",\"storage_cold_verified\":false,"
          <<"\"prompt_tokens\":"<<np<<",\"output_tokens\":"<<ng<<",\"sequences\":1,\"layers\":"<<llama_model_n_layer(model.get())
          <<",\"attention_checks\":"<<count<<",\"gpu_attention_callbacks\":"<<obs.gpu_callbacks
          <<",\"kv_roundtrips\":"<<external.restores<<",\"attention_max_abs_error\":"<<obs.max_error
          <<",\"logits_max_abs_error\":"<<logits_max<<",\"atol\":"<<abs_tolerance<<",\"rtol\":"<<rtol
          <<",\"tokens_equal\":true,\"native_prefill_ms\":"<<native.prefill_ms<<",\"native_decode_ms\":"<<native.decode_ms
          <<",\"external_prefill_ms\":"<<external.prefill_ms<<",\"external_decode_ms\":"<<external.decode_ms
          <<",\"external_roundtrip_ms\":"<<external.external_ms<<",\"native_request_ms\":"<<native.total_ms
          <<",\"external_request_ms\":"<<external.total_ms<<",\"append_snapshot_logical_write_bytes\":"<<store.write_bytes
          <<",\"selected_all_state_logical_read_bytes\":"<<store.read_bytes<<",\"device_read_bytes\":null,\"nand_bytes\":null"
          <<",\"snapshot_peak_bytes\":"<<external.snapshot_peak<<",\"attention_oracle_bytes\":"<<obs.oracle_bytes
          <<",\"process_peak_rss_kib\":"<<usage.ru_maxrss<<",\"formal_speedup_eligible\":false}\n";
        std::ofstream output(dir/"generated_tokens.json");output<<"[";for(std::size_t i=0;i<native.tokens.size();++i){if(i)output<<",";output<<native.tokens[i];}output<<"]\n";
        std::cout<<"PASS full-retention debug: "<<count<<" GPU attention comparisons, "<<external.restores<<" external restores\n";
        return 0;
    }catch(const std::exception& e){
        // Error text is logged separately so JSON remains valid for arbitrary exception strings.
        std::ofstream err(dir/"error.txt");err<<e.what()<<'\n';
        report<<"{\"status\":\"failed\",\"formal_speedup_eligible\":false,\"error_log\":\"error.txt\"}\n";
        std::cerr<<e.what()<<'\n';return 1;
    }
}
