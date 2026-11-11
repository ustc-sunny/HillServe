#include "runtime/executor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <cstdlib>
#include <iomanip>
namespace hillserve {
namespace {
constexpr int D=128,H=4,E=D*H;
constexpr std::size_t record_bytes=2*E*sizeof(ggml_fp16_t);
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
std::string key(std::uint64_t p,int l){return std::to_string(p)+":"+std::to_string(l);}
}
LayerExecutor::LayerExecutor(const std::string& dir,bool external,int slots,int max_tokens,
 std::size_t gpu_budget,std::size_t host_budget,const Plans* plans,
 std::map<std::string,std::vector<float>>* oracle,std::map<std::string,std::vector<ggml_fp16_t>>* kv_oracle,bool observe,bool compact_oracle,bool numerical)
 :plans_(plans),external_(external),observe_(observe),compact_oracle_(compact_oracle),slots_(slots),max_tokens_(max_tokens),host_budget_(host_budget),
 events_(dir+(external?"/kv-io-events.jsonl":"/oracle-events.jsonl")),oracle_(oracle),kv_oracle_(kv_oracle){
    directory_=dir; numerical_=numerical;
    const std::string durability=std::getenv("HILLSERVE_KV_DURABILITY")?std::getenv("HILLSERVE_KV_DURABILITY"):"durable_per_layer";
    need(durability=="durable_per_layer" || durability=="volatile_query_kv","invalid KV durability");
    durability_=durability=="volatile_query_kv"?KVDurability::volatile_query_kv:KVDurability::durable_per_layer;
    if(observe_)selections_.open(dir+(external?"/selection-external.jsonl":"/selection-oracle.jsonl"));
    if(numerical_ && external_) numerical_log_.open(dir+"/numerical-layers.jsonl");
    need(slots>0&&slots<=512&&slots%32==0&&max_tokens>0&&max_tokens<=256,"unsupported workspace geometry");
    const std::size_t mask_size=slots*std::size_t((max_tokens+GGML_KQ_MASK_PAD-1)/GGML_KQ_MASK_PAD*GGML_KQ_MASK_PAD)*sizeof(float);
    const std::size_t kv_size=compact()?2*slots*E*sizeof(ggml_fp16_t):0;
    need(kv_size+mask_size<=gpu_budget,"GPU KV/mask workspace budget exceeded");
    // Reserve a conservative bound for payload vectors, selection bookkeeping and metadata nodes.
    std::size_t plan_bytes=0;
    if(plans_)for(const auto& entry:*plans_)plan_bytes+=128+entry.second.capacity()*sizeof(std::uint64_t);
    const std::size_t host_bound=kv_size+mask_size+max_tokens*record_bytes+slots*record_bytes+
        host_budget/4+slots*256+max_tokens*64+512*1024+plan_bytes;
    need(host_bound<=host_budget,"host KV staging/metadata budget exceeded");
    stats.host_payload_peak=host_bound;
    auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(dev,"GPU backend missing");
    try {
    tensor_ctx_=ggml_init({16*ggml_tensor_overhead(),nullptr,true});need(tensor_ctx_,"tensor metadata allocation failed");
    if(compact()){k_=ggml_new_tensor_2d(tensor_ctx_,GGML_TYPE_F16,E,slots);v_=ggml_new_tensor_2d(tensor_ctx_,GGML_TYPE_F16,slots,E);
        ggml_set_name(k_,"hillserve_shared_k");ggml_set_name(v_,"hillserve_shared_v");}
    mask_=ggml_new_tensor_1d(tensor_ctx_,GGML_TYPE_F32,mask_size/4);ggml_set_name(mask_,"hillserve_shared_mask");
    buffer_=ggml_backend_alloc_ctx_tensors_from_buft(tensor_ctx_,ggml_backend_dev_buffer_type(dev));need(buffer_,"GPU workspace allocation failed");
    stats.gpu_workspace_bytes=ggml_backend_buffer_get_size(buffer_);stats.kv_bytes=kv_size;stats.mask_bytes=mask_size;
    need(stats.gpu_workspace_bytes<=gpu_budget,"allocator-aligned GPU workspace exceeds budget");
    ggml_backend_buffer_clear(buffer_,0);
    current_.reserve(max_tokens*2*E);host_mask_.resize(mask_size/4);
    if(compact()){host_k_.resize(slots*E);host_v_.resize(slots*E);}
    if(external){
        store_=std::make_unique<IndexedKVStore>(dir+"/kv-records.bin",64*1024*1024,28*512,host_budget/4,durability_);
        stats.file_cache_reservation_bytes=store_->file_cache_reservation();
    }
    } catch (...) { if(buffer_)ggml_backend_buffer_free(buffer_); if(tensor_ctx_)ggml_free(tensor_ctx_); throw; }
}
LayerExecutor::~LayerExecutor(){llama_hillserve_configure(nullptr);if(buffer_)ggml_backend_buffer_free(buffer_);if(tensor_ctx_)ggml_free(tensor_ctx_);}
void LayerExecutor::configure(){llama_hillserve_config c{true,compact(),std::uint32_t(slots_),std::uint32_t(max_tokens_),k_,v_,mask_,false};
    const char* precision=std::getenv("HILLSERVE_ATTN_VALUE_ACCUM");
    need(!precision || std::string(precision)=="default" || std::string(precision)=="f32-control","invalid value accumulation control");
    c.f32_value_control=precision && std::string(precision)=="f32-control";llama_hillserve_configure(&c);}
void LayerExecutor::attach_profile(llama_context* ctx){
    if(std::getenv("HILLSERVE_PROFILE")){
        need(!observe_,"profile must use none oracle to exclude debug shadows");
        profiler_=std::make_unique<GraphProfiler>(directory_+"/model-segments.jsonl");
        llama_hillserve_set_profile(ctx,GraphProfiler::callback,profiler_.get());
    }
}
void LayerExecutor::begin(int step,const std::vector<std::uint64_t>& p){
    need(step==step_+1,"step/epoch must advance exactly once");
    need(!p.empty()&&p.size()<=std::size_t(max_tokens_),"invalid current positions");
    for(std::size_t i=1;i<p.size();++i)need(p[i]==p[i-1]+1,"non-sequential append positions unsupported");
    step_=step;positions_=p;if(profiler_)profiler_->step=step;
}
void LayerExecutor::prepare(ggml_tensor* t,int layer,int count){
    need(t->type==GGML_TYPE_F16&&ggml_is_contiguous(t)&&t->ne[0]==2*D&&t->ne[1]==H&&t->ne[2]==int(positions_.size()),"current KV tensor ABI mismatch");
    need(positions_.front()==next_[layer],"request/layer append epoch mismatch");
    FetchPlan plan{request_,std::uint64_t(layer),std::uint64_t(step_),std::uint64_t(step_),{}};
    if(plans_){auto it=plans_->find({layer,step_});need(it!=plans_->end(),"missing explicit FetchPlan");plan.ids=it->second;}
    else{plan.ids.resize(positions_.back()+1);std::iota(plan.ids.begin(),plan.ids.end(),0);}
    plan.validate(positions_.back(),compact()?count:512,positions_);
    if(observe_){selections_<<"{\"request\":"<<request_<<",\"layer\":"<<layer<<",\"step\":"<<step_<<",\"selected_ids\":[";
        for(std::size_t i=0;i<plan.ids.size();++i){if(i)selections_<<",";selections_<<plan.ids[i];}selections_<<"]}\n";}
    current_.resize(ggml_nelements(t));ggml_backend_tensor_get(t,current_.data(),0,ggml_nbytes(t));
    stats.current_peak=std::max(stats.current_peak,ggml_nbytes(t));
    auto& phase_current=step_==0?stats.current_prefill:stats.current_decode;phase_current=std::max(phase_current,ggml_nbytes(t));
    auto start=Clock::now();
    if(external_)store_->begin_append_batch();
    for(std::size_t j=0;j<positions_.size();++j){
        std::vector<std::uint8_t> bytes(record_bytes);auto* values=reinterpret_cast<ggml_fp16_t*>(bytes.data());
        auto packing_start=Clock::now();
        for(int h=0;h<H;++h){auto* row=current_.data()+(j*H+h)*2*D;
            std::memcpy(values+h*D,row,D*2);std::memcpy(values+E+h*D,row+D,D*2);}
        stats.pack_ms+=ms(packing_start);
        if(numerical_ && external_){own_kv_[key(positions_[j],layer)]=std::vector<ggml_fp16_t>(values,values+2*E);
            stats.own_shadow_bytes+=record_bytes+256;need(stats.own_shadow_bytes<=32*1024*1024,"own storage debug shadow budget exceeded");}
        if(kv_oracle_){auto id=key(positions_[j],layer);
            if(!external_)(*kv_oracle_)[id]=std::vector<ggml_fp16_t>(values,values+2*E);
            else {auto it=kv_oracle_->find(id);need(it!=kv_oracle_->end(),"missing KV oracle");
                if(std::memcmp(it->second.data(),bytes.data(),record_bytes)!=0){
                    std::size_t count=0,first=0;double maxerr=0;
                    for(int e=0;e<2*E;++e)if(it->second[e]!=values[e]){if(!count)first=e;++count;maxerr=std::max(maxerr,std::abs(double(ggml_fp16_to_fp32(it->second[e]))-ggml_fp16_to_fp32(values[e])));}
                    stats.kv_drift_words+=count;stats.kv_drift_max=std::max(stats.kv_drift_max,maxerr);
                    if(numerical_)numerical_log_<<std::setprecision(17)<<"{\"kind\":\"model_kv_drift\",\"step\":"<<step_<<",\"layer\":"<<layer<<",\"position\":"<<positions_[j]<<",\"mismatched_words\":"<<count<<",\"max_abs_error\":"<<maxerr<<"}\n";
                    if(!first_kv_){first_kv_=true;std::ofstream f(directory_+"/first-kv-difference.json");
                    f<<std::setprecision(17)<<"{\"step\":"<<step_<<",\"layer\":"<<layer<<",\"first_element\":"<<first<<",\"mismatched_half_words\":"<<count<<",\"max_abs_error\":"<<maxerr<<",\"legacy_strict_assertion\":"<<(numerical_?"false":"true")<<"}\n";}
                    if(!numerical_)throw std::runtime_error("current KV differs from native reference");
                }++stats.kv_identity_checks;}}
        if(external_)store_->append_kv(request_,layer,positions_[j],bytes);
        ++stats.appends;
    }
    if(external_)store_->end_append_batch();
    double append_time=ms(start);stats.append_ms+=append_time;
    next_[layer]=positions_.back()+1;
    const std::size_t appended=positions_.size()*record_bytes;
    if(step_==0)stats.append_prefill+=appended;else stats.append_decode+=appended;
    start=Clock::now();
    if(compact()){
        std::vector<KVPayload> payload;
        if(external_)payload=store_->fetch_selected(request_,layer,plan.ids,slots_*record_bytes);
        else { // Independent in-memory backing oracle, verification mode only.
            need(kv_oracle_,"compact reference requires explicit debug backing");
            for(auto id:plan.ids){const auto& ref=kv_oracle_->at(key(id,layer));KVPayload p{id,std::vector<std::uint8_t>(record_bytes)};
                std::memcpy(p.bytes.data(),ref.data(),record_bytes);payload.push_back(std::move(p));}
        }
        auto packing_start=Clock::now();
        std::fill(host_k_.begin(),host_k_.end(),0);std::fill(host_v_.begin(),host_v_.end(),0);
        for(std::size_t slot=0;slot<payload.size();++slot){
            const auto& p=payload[slot];need(p.token==plan.ids[slot]&&p.bytes.size()==record_bytes,"gather mapping corrupt");
            if(numerical_ && external_){const auto& own=own_kv_.at(key(p.token,layer));need(std::memcmp(own.data(),p.bytes.data(),record_bytes)==0,"selected stored KV differs from own append bytes");++stats.storage_exact_checks;}
            else if(kv_oracle_){const auto& ref=kv_oracle_->at(key(p.token,layer));need(std::memcmp(ref.data(),p.bytes.data(),record_bytes)==0,"selected stored KV differs from oracle");++stats.kv_identity_checks;}
            std::memcpy(host_k_.data()+slot*E,p.bytes.data(),E*2);
            for(int e=0;e<E;++e)std::memcpy(&host_v_[e*slots_+slot],p.bytes.data()+E*2+e*2,2);
        }
        stats.pack_ms+=ms(packing_start);
        ++stats.gathers;stats.fetch_bytes+=payload.size()*record_bytes;
    }
    double fetch_time=ms(start);stats.fetch_ms+=fetch_time;
    int qpad=(positions_.size()+GGML_KQ_MASK_PAD-1)/GGML_KQ_MASK_PAD*GGML_KQ_MASK_PAD;
    std::fill(host_mask_.begin(),host_mask_.end(),-std::numeric_limits<float>::infinity());
    for(std::size_t q=0;q<positions_.size();++q)for(std::size_t s=0;s<plan.ids.size();++s)
        if(plan.ids[s]<=positions_[q])host_mask_[q*count+(compact()?s:plan.ids[s])]=0;
    start=Clock::now();
    if(compact()){ggml_backend_tensor_set(k_,host_k_.data(),0,host_k_.size()*2);ggml_backend_tensor_set(v_,host_v_.data(),0,host_v_.size()*2);}
    ggml_backend_tensor_set(mask_,host_mask_.data(),0,std::size_t(count)*qpad*4);
    double h2d_time=ms(start);stats.h2d_ms+=h2d_time;
    events_<<"{\"request\":"<<request_<<",\"layer\":"<<layer<<",\"step\":"<<step_<<",\"epoch\":"<<step_
        <<",\"append_tokens\":"<<positions_.size()<<",\"append_write_bytes\":"<<(external_?appended:0)
        <<",\"selected_tokens\":"<<plan.ids.size()<<",\"selected_kv_bytes\":"<<(external_?plan.ids.size()*record_bytes:0)
        <<",\"append_ms\":"<<append_time<<",\"fetch_ms\":"<<fetch_time<<",\"h2d_ms\":"<<h2d_time
        <<",\"candidate_read_bytes\":null,\"score_return_bytes\":null,\"promotion_bytes\":0,\"measurement_mode\":\"correctness\",\"payload_path\":\""<<(external_?"systemfs_buffered":compact_oracle_?"host_shadow_gpu_oracle":"native_gpu_oracle")<<"\",\"policy_mode\":\"diagnostic\",\"candidate_path\":\"unknown\",\"compute_backend\":\"not_run\"}\n";
}
void LayerExecutor::inspect(ggml_tensor* t){
    auto* buf=t->buffer?t->buffer:t->view_src?t->view_src->buffer:nullptr;
    need(buf,"attention buffer missing");auto dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
    need(dev&&ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU,"attention did not run on GPU");
    ++stats.attention_checks;
    if(std::getenv("HILLSERVE_FREEZE") && !external_ && step_==1 && kv_oracle_ && !frozen_q_.empty()){
        int layer=-1;std::sscanf(t->name,"kqv_out-%d",&layer);
        const std::string layers=std::getenv("HILLSERVE_FREEZE_LAYERS")?std::getenv("HILLSERVE_FREEZE_LAYERS"):"0,7,9";
        need(layers=="0,7,9" || layers=="7,10","freeze scope outside declared fixtures");
        if((layers=="0,7,9"&&(layer==0||layer==7||layer==9)) || (layers=="7,10"&&(layer==7||layer==10))){
            // Validation-only frozen input, local authored prompt. Never copied to evidence package.
            auto freeze_start=Clock::now();
            std::ofstream f(directory_+"/frozen-layer"+std::to_string(layer)+".bin",std::ios::binary);
            std::uint32_t header[4]={std::uint32_t(positions_.back()+1),D,28,H};
            f.write(reinterpret_cast<char*>(header),sizeof(header));f.write(reinterpret_cast<char*>(frozen_q_.data()),28*D*4);
            for(std::uint64_t pos=0;pos<=positions_.back();++pos){auto& v=kv_oracle_->at(key(pos,layer));f.write(reinterpret_cast<const char*>(v.data()),record_bytes);}
            std::vector<float> observed(ggml_nelements(t));ggml_backend_tensor_get(t,observed.data(),0,ggml_nbytes(t));f.write(reinterpret_cast<char*>(observed.data()),observed.size()*4);f.flush();need(bool(f),"frozen input write failed");
            std::ofstream manifest(directory_+"/frozen-manifest.jsonl",std::ios::app);
            manifest<<"{\"layer\":"<<layer<<",\"step\":1,\"evidence_mode\":\"D_DIAGNOSTIC\",\"policy_bound\":false,\"method_trace\":false,\"source\":\"actual_Qwen_native_masked_authored_prompt\",\"payload_bytes\":"<<(16+frozen_q_.size()*4+(positions_.back()+1)*record_bytes+observed.size()*4)<<",\"extra_live_export_vectors_bytes\":"<<((frozen_q_.capacity()+observed.capacity())*4)<<",\"export_ms\":"<<ms(freeze_start)<<",\"export_in_performance_measurement\":false,\"full_payload_shareable\":false}\n";
        }
    }
    if(!observe_||!oracle_)return;
    need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"attention layout unsupported");
    std::vector<float> data(ggml_nelements(t));ggml_backend_tensor_get(t,data.data(),0,ggml_nbytes(t));
    auto name=std::to_string(step_)+":"+t->name;
    if(!external_){need(!oracle_->count(name),"duplicate attention observation");oracle_->emplace(name,std::move(data));return;}
    const auto& ref=oracle_->at(name);need(ref.size()==data.size(),"attention shape differs");
    if(!first_bit_ && std::memcmp(ref.data(),data.data(),data.size()*4)!=0){
        first_bit_=true;std::size_t first=0;while(first<data.size()&&std::memcmp(&ref[first],&data[first],4)==0)++first;
        std::ofstream f(directory_+"/first-attention-bit-difference.json");f<<std::setprecision(17)<<"{\"step\":"<<step_<<",\"tensor\":\""<<t->name<<"\",\"element\":"<<first<<",\"reference\":"<<ref[first]<<",\"actual\":"<<data[first]<<",\"abs_error\":"<<std::abs(double(ref[first])-data[first])<<"}\n";
    }
    double layer_max=0;std::size_t violations=0;
    for(std::size_t i=0;i<data.size();++i){double e=std::abs(double(data[i])-ref[i]);need(std::isfinite(data[i])&&std::isfinite(ref[i]),"nonfinite attention");
        layer_max=std::max(layer_max,e);stats.max_attention_error=std::max(stats.max_attention_error,e);if(e>1e-5+1e-5*std::abs(ref[i])){++violations;if(stats.attention_violations++==0){std::ofstream f(directory_+"/first-tolerance-violation.json");f<<"{\"step\":"<<step_<<",\"tensor\":\""<<t->name<<"\",\"element\":"<<i<<"}\n";}if(!numerical_)throw std::runtime_error("attention fixed tolerance exceeded");}}
    if(numerical_)numerical_log_<<std::setprecision(17)<<"{\"kind\":\"model_attention\",\"step\":"<<step_<<",\"tensor\":\""<<t->name<<"\",\"elements\":"<<data.size()<<",\"max_abs_error\":"<<layer_max<<",\"violations\":"<<violations<<",\"nonfinite\":false}\n";
}
void LayerExecutor::accounting(llama_context* c,bool prefill){
    if(store_){auto& peak=prefill?stats.metadata_prefill:stats.metadata_decode;peak=std::max(peak,store_->metadata().peak);}
    stats.native_history_bytes=llama_hillserve_native_kv_bytes(c);
    need(!compact() || stats.native_history_bytes==0,"native GPU history unexpectedly allocated");
    auto n=llama_hillserve_graph_bytes(c,true);if(prefill)stats.graph_prefill_gpu=n;else stats.graph_decode_gpu=std::max(stats.graph_decode_gpu,n);
    stats.graph_cpu=std::max(stats.graph_cpu,llama_hillserve_graph_bytes(c,false));
}
void LayerExecutor::release(){if(store_){
    stats.dirty_peak_bytes=store_->dirty_peak_upper_bound();
    store_->drain();stats.final_drain_ms=store_->final_drain_ms;
    need(store_->dirty_upper_bound()==0,"dirty bytes remain after final drain");
    const auto& t=store_->timings;std::ofstream f(directory_+"/store-phases.json");
    f<<std::setprecision(17)<<"{\"physical_path\":\"system_nvme_buffered_regular_file\",\"cache_state\":\"uncontrolled_page_cache_no_drop\",\"durability\":\""<<(durability_==KVDurability::durable_per_layer?"durable_per_layer":"volatile_query_kv")<<"\",\"pwrite_ms\":"<<t.pwrite_ms<<",\"pread_ms\":"<<t.pread_ms<<",\"checksum_ms\":"<<t.checksum_ms<<",\"index_ms\":"<<t.index_ms<<",\"fdatasync_ms\":"<<t.fdatasync_ms<<",\"fdatasync_calls\":"<<t.sync_calls<<",\"final_drain_ms\":"<<store_->final_drain_ms<<",\"final_drain_calls\":"<<store_->final_drain_calls<<",\"dirty_peak_upper_bound_bytes\":"<<stats.dirty_peak_bytes<<",\"dirty_bytes_after_drain\":"<<store_->dirty_upper_bound()<<",\"file_page_cache_reservation_bytes\":"<<store_->file_cache_reservation()<<",\"file_page_cache_observed_resident_bytes\":null,\"packing_ms\":"<<stats.pack_ms<<",\"api_append_bytes\":"<<store_->append_write_bytes<<",\"api_selected_read_bytes\":"<<store_->selected_read_bytes<<",\"physical_nand_bytes\":null,\"thread_start_ms\":null,\"thread_wait_ms\":null,\"thread_model\":\"synchronous_caller_no_worker\"}\n";
    store_->release_request(request_);need(store_->records()==0,"request release failed");}}

bool LayerExecutor::callback(ggml_tensor* t,bool ask,void* user){
    auto& r=*static_cast<LayerExecutor*>(user);std::string name=t->name;
    bool append=name.rfind("hs_append_",0)==0,attn=name.rfind("kqv_out-",0)==0;
    bool query=std::getenv("HILLSERVE_FREEZE") && r.step_==1 && !r.external_ && t->op==GGML_OP_ROPE && name.rfind("Qcur-",0)==0;
    if(ask)return append||attn||query;
    if(query){if(t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||ggml_nelements(t)!=28*128){r.failure="freeze query ABI unsupported";return false;}r.frozen_q_.resize(ggml_nelements(t));ggml_backend_tensor_get(t,r.frozen_q_.data(),0,ggml_nbytes(t));return true;}
    try{if(append){int layer,count;need(std::sscanf(name.c_str(),"hs_append_%d_%d",&layer,&count)==2,"bad runtime barrier");r.prepare(t,layer,count);}else if(attn)r.inspect(t);return true;}
    catch(const std::exception& e){r.failure="step="+std::to_string(r.step_)+" tensor="+name+": "+e.what();return false;}
}
}
