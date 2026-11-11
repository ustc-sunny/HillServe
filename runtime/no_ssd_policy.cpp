#include "runtime/no_ssd_policy.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "policy/full_gqa_cpu.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "storage/bounded_memory_kv.hpp"
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include "nlohmann/json.hpp"

namespace hillserve::policy {
namespace {
constexpr std::size_t H=32,KH=8,D=128,KEY_WORDS=KH*D,KV_BYTES=4096,HEAD_BYTES=512;
using Clock=std::chrono::steady_clock;
void need(bool yes,const std::string& message){if(!yes)throw std::runtime_error("no-SSD policy: "+message);}
std::size_t add(std::size_t a,std::size_t b){need(b<=std::numeric_limits<std::size_t>::max()-a,"size addition overflow");return a+b;}
std::size_t mul(std::size_t a,std::size_t b){need(!a||b<=std::numeric_limits<std::size_t>::max()/a,"size multiplication overflow");return a*b;}
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
std::uint64_t stamp(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
std::vector<std::uint16_t> query_half(const std::vector<float>& input){
    need(input.size()==H*D,"rehearsal query must contain 32x128 channels");
    std::vector<std::uint16_t> q(input.size());
    for(std::size_t i=0;i<q.size();++i){need(std::isfinite(input[i]),"nonfinite rehearsal query");q[i]=half_bits(input[i]);need(std::isfinite(half_value(q[i])),"F16 rehearsal overflow");}
    return q;
}
void validate_half_bytes(const std::vector<std::uint8_t>& input){
    need(input.size()%2==0,"odd F16 payload bytes");
    for(std::size_t i=0;i<input.size();i+=2){std::uint16_t v;std::memcpy(&v,input.data()+i,2);need(std::isfinite(half_value(v)),"nonfinite model KV payload");}
}
}

struct NoSSDPolicy::Impl {
    // Main-thread append/select and worker fetch/restore share metadata. A
    // future is NEVER joined while this lock is held. History score K resides
    // in a fixed slab; current-token append writes a distinct tail position.
    mutable std::recursive_mutex state_mutex;
    std::mutex score_mutex;
    Config config;
    Recompute recompute;
    BatchRecompute batch_recompute;
    Stats statistics;
    std::unique_ptr<BoundedMemoryKVStore> store;
    std::unique_ptr<FullGqaCpuScorer> scorer;
    std::vector<std::uint16_t> keys;
    std::vector<std::uint64_t> completed,initial_victim,selection_count;
    // Preference is separate from physical residency. In H2O it is supplied
    // by actual attention importance; PREFETCH uses the last exact selection.
    std::vector<std::uint8_t> preferred;
    // FULL_RECOMPUTE owns only this fixed APP hot slab. Cold append payloads
    // are discarded; missing fetch payloads have request-local lifetime only.
    std::vector<std::int64_t> fixed_slots;
    std::vector<std::uint8_t> fixed_hot_kv;
    struct Work {
        std::vector<std::uint16_t> raw;
        std::shared_ptr<const runtime::GQAPrefetchArtifact> payload;
    };
    struct Pending {
        std::uint64_t position=0;
        std::vector<std::uint16_t> query;
        std::future<Work> result;
    };
    std::map<int,std::shared_ptr<Pending>> pending;
    std::map<int,std::shared_ptr<const runtime::GQAPrefetchArtifact>> ready_payload;

    Impl(const Config& c,Recompute r,BatchRecompute b):config(c),recompute(std::move(r)),batch_recompute(std::move(b)){
        need(config.mode=="PREFETCH"||config.mode=="H2O"||config.mode=="FULL_RECOMPUTE","mode must be PREFETCH, H2O or FULL_RECOMPUTE");
        need(!config.extended_full_context||config.mode=="FULL_RECOMPUTE","extended noSSD context requires full attention");
        need(config.layers>0&&config.layers<=32&&config.capacity>0&&config.capacity<=(config.extended_full_context?66048:65504),
             "bounded Gradient GQA geometry (layers<=32, capacity<=65504)");
        need(config.prompt_tokens>0&&config.generated_capacity>0&&
             add(config.prompt_tokens,config.generated_capacity)<=config.capacity,"prompt/generated capacity");
        need(config.cpu_threads>=1&&config.cpu_threads<=32&&
             (config.cpu_implementation=="auto"||config.cpu_implementation=="legacy"),"CPU implementation/threads");
        need(std::isfinite(config.selector_alpha)&&config.selector_alpha>=0,"selector alpha");
        need(std::fegetround()==FE_TONEAREST,"F16 policy requires round-to-nearest FP32 mode");
        const auto lanes=mul(config.layers,KH),C=config.capacity;
        if(config.mode=="FULL_RECOMPUTE"){
            need(bool(batch_recompute),"FULL_RECOMPUTE requires an actual eight-head batch model recomputation callback");
            runtime::FullGqaRuntimeSpec spec;
            spec.layers=config.layers;spec.capacity=C;spec.prompt_tokens=config.prompt_tokens;
            spec.generated_capacity=config.generated_capacity;spec.beta=config.offline_beta;
            spec.allow_partial_prompt=config.allow_partial_prompt;
            spec.fp32_selection_counts=config.extended_full_context; // no selector used by this policy
            spec.hot_kv_budget_bytes=config.resident_kv_payload_budget_bytes;
            // This shared plan determines only the identical fixed APP hot
            // set. No file, score K, FPGA BO, or SSD payload is instantiated.
            spec.total_host_workspace_budget_bytes=std::numeric_limits<std::uint64_t>::max();
            auto plan=runtime::plan_full_gqa_runtime(spec);
            need(config.resident_kv_payload_budget_bytes==plan.reserved_all_layers_hot_bytes,
                 "FULL_RECOMPUTE resident payload allowance must equal the shared APP plan including generated reserve");
            statistics.hot_prompt_tokens=plan.prompt_hot_tokens;
            statistics.cold_prompt_tokens=plan.prompt_cold_tokens;
            statistics.generated_reserved_tokens=config.generated_capacity;
            statistics.resident_rows_per_head=add(plan.prompt_hot_tokens,config.generated_capacity);
            statistics.resident_kv_payload_reserved_bytes=plan.reserved_all_layers_hot_bytes;
            statistics.resident_reserved_bytes=plan.reserved_all_layers_hot_bytes;
            statistics.metadata_reserved_bytes=add(mul(C,sizeof(std::int64_t)),mul(config.layers,sizeof(std::uint64_t)));
            // Delivery plus eight returned missing-head payloads; bounded ID
            // vectors and full-attention selection lists. Caller model replay
            // context has its own separately reported resource allowance.
            const auto minimum_staging=add(mul(C,2*KV_BYTES+KH*sizeof(std::uint64_t)+sizeof(std::size_t)+H*sizeof(std::uint64_t)),4096);
            if(config.staging_budget_bytes)need(config.staging_budget_bytes>=minimum_staging,"staging budget below FULL_RECOMPUTE temporary upper bound");
            statistics.staging_reserved_bytes=config.staging_budget_bytes?config.staging_budget_bytes:minimum_staging;
            statistics.managed_reserved_bytes=add(statistics.resident_reserved_bytes,
                add(statistics.metadata_reserved_bytes,statistics.staging_reserved_bytes));
            need(statistics.managed_reserved_bytes<=config.total_host_budget_bytes,"FULL_RECOMPUTE resident payload plus metadata/workspace exceeds host budget");
            statistics.all_KV_fits=plan.prompt_cold_tokens==0;
            statistics.cpu_implementation_actual="not_used_FULL_RECOMPUTE_full_attention";
            fixed_slots=std::move(plan.hot_slot_by_position);
            fixed_hot_kv.resize(statistics.resident_kv_payload_reserved_bytes);
            completed.resize(config.layers,0);
            return;
        }
        // All persistent score K, storage maps and bounded temporary payloads
        // count against the same budget. OS thread stacks/allocator overhead
        // additionally appear in measured process RSS, not a claimed exact RSS.
        // The first two layers use full attention and never have rehearsal
        // score K. Their physical K/V still resides in the common store.
        const auto scored_layers=config.layers>2?config.layers-2:0;
        statistics.score_key_reserved_bytes=config.mode=="PREFETCH"?mul(mul(scored_layers,C),KEY_WORDS*2):0;
        statistics.metadata_reserved_bytes=add(mul(lanes,C*(config.mode=="PREFETCH"?9:1)),add(mul(lanes+config.layers,sizeof(std::uint64_t)),16*1024*1024));
        // At most two in-flight history payloads plus one delivery payload;
        // per-head gather/recompute and scoring/selection scratch are bounded.
        const auto minimum_staging=add(mul(C,3*KV_BYTES+2*HEAD_BYTES+12*H+64),
                                       add(mul(C,sizeof(void*)),H*D*2));
        if(config.staging_budget_bytes)need(config.staging_budget_bytes>=minimum_staging,"staging budget below safe temporary upper bound");
        statistics.staging_reserved_bytes=config.staging_budget_bytes?config.staging_budget_bytes:minimum_staging;
        const auto fixed=add(statistics.score_key_reserved_bytes,
                             add(statistics.metadata_reserved_bytes,statistics.staging_reserved_bytes));
        need(config.total_host_budget_bytes>fixed,"host budget cannot hold score metadata and bounded workspace");
        const auto available=config.total_host_budget_bytes-fixed;
        BoundedMemoryKVStore::Config sc{lanes,C,HEAD_BYTES,1,available};
        std::size_t low=0,high=C;
        while(low<high){const auto mid=low+(high-low+1)/2;sc.resident_rows_per_layer=mid;
            if(BoundedMemoryKVStore::estimate_bytes(sc)<=available)low=mid;else high=mid-1;}
        // Generated decode rows are never evicted/recomputed. One additional
        // prompt slot permits consecutive native-prefill admission.
        need(low>=config.generated_capacity+1,"resident budget cannot pin generated rows plus one prompt row");
        sc.resident_rows_per_layer=low;
        store=std::make_unique<BoundedMemoryKVStore>(sc);
        statistics.resident_rows_per_head=low;
        statistics.resident_reserved_bytes=store->reserved_bytes();
        statistics.resident_kv_payload_reserved_bytes=mul(mul(lanes,low),HEAD_BYTES);
        statistics.managed_reserved_bytes=add(fixed,statistics.resident_reserved_bytes);
        statistics.all_KV_fits=low==C;
        keys.resize(statistics.score_key_reserved_bytes/2);
        completed.resize(config.layers,0);initial_victim.resize(lanes,0);preferred.resize(mul(lanes,C),0);
        if(config.mode=="PREFETCH")selection_count.resize(mul(lanes,C),0);
        if(config.mode=="PREFETCH"){
            if(config.cpu_implementation=="auto"){
                scorer=std::make_unique<FullGqaCpuScorer>(config.cpu_threads);
                statistics.cpu_implementation_actual=scorer->implementation_name();
            }else statistics.cpu_implementation_actual=full_gqa_legacy_cpu_name;
        }else statistics.cpu_implementation_actual="not_used_H2O_full_attention";
    }
    void check_layer(int layer)const{need(layer>=0&&std::size_t(layer)<config.layers,"layer range");}
    std::size_t fixed_offset(int layer,std::int64_t slot)const{
        need(slot>=0&&std::size_t(slot)<statistics.resident_rows_per_head,"invalid fixed hot slot");
        return mul(add(mul(std::size_t(layer),statistics.resident_rows_per_head),std::size_t(slot)),KV_BYTES);
    }
    std::size_t lane(int layer,std::size_t head)const{return std::size_t(layer)*KH+head;}
    std::uint16_t* key(int layer,std::size_t position){need(layer>=2,"first two layers have no rehearsal score metadata");return keys.data()+((std::size_t(layer)-2)*config.capacity+position)*KEY_WORDS;}
    bool wanted(std::size_t ln,std::uint64_t pos)const{return preferred[ln*config.capacity+pos]!=0;}
    void make_slot(std::size_t ln){
        if(store->resident_records(ln)<statistics.resident_rows_per_head)return;
        const auto n=std::min<std::uint64_t>(store->completed(ln),config.prompt_tokens);
        need(n>0,"no evictable original-prompt KV; generated rows are pinned");
        std::uint64_t victim=n;
        if(config.mode=="PREFETCH"){
            if(statistics.selections==0){
                // Before any rehearsal selection all counters are zero. This
                // monotone scan is the exact oldest-ID tie break, not a clock.
                for(auto pos=initial_victim[ln];pos<n;++pos)if(store->contains(ln,pos)){victim=pos;initial_victim[ln]=pos+1;break;}
            }else{
                std::uint64_t least=std::numeric_limits<std::uint64_t>::max();
                for(std::uint64_t pos=0;pos<n;++pos)if(store->contains(ln,pos)){
                    const auto count=selection_count[ln*config.capacity+pos];
                    if(victim==n||count<least){victim=pos;least=count;}
                }
            }
        }else{
            // H2O supplies its exact residency set after real attention. A
            // newly appended pinned row displaces an undesired prompt first.
            for(unsigned pass=0;pass<2&&victim==n;++pass)for(std::uint64_t pos=0;pos<n;++pos)
                if(store->contains(ln,pos)&&(pass||!wanted(ln,pos))){victim=pos;break;}
        }
        if(victim<n){store->evict(ln,{victim});++statistics.evicted_rows;return;}
        throw std::runtime_error("no-SSD policy: no resident prompt victim");
    }
    std::vector<std::uint16_t> score(int layer,std::size_t valid,const std::vector<std::uint16_t>& q){
        need(config.mode=="PREFETCH"&&valid>0&&valid<=config.capacity,"score candidate range");
        const auto begin=Clock::now();std::vector<std::uint16_t> compact;
        {std::lock_guard<std::mutex> score_lock(score_mutex);
            if(scorer)scorer->score(q,{{key(layer,0),valid,KEY_WORDS}},compact);
            else compact=score_full_gqa_legacy_cpu(q,key(layer,0),valid,KEY_WORDS,config.cpu_threads);}
        need(compact.size()==H*valid,"CPU score output shape");
        std::vector<std::uint16_t> raw(H*config.capacity,0);
        for(std::size_t h=0;h<H;++h)for(std::size_t t=0;t<valid;++t){
            const auto value=compact[h*valid+t];need(std::isfinite(half_value(value)),"CPU score overflow");raw[h*config.capacity+t]=value;}
        {std::lock_guard<std::recursive_mutex> lock(state_mutex);++statistics.score_calls;statistics.score_ms+=elapsed(begin);}
        return raw;
    }
    void score_current(int layer,std::uint64_t pos,const std::vector<std::uint16_t>& q,std::vector<std::uint16_t>& raw){
        std::lock_guard<std::recursive_mutex> lock(state_mutex);
        need(std::fegetround()==FE_TONEAREST,"current-token arithmetic rounding mode");
        const auto begin=Clock::now();const auto* k=key(layer,pos);
        for(std::size_t h=0;h<H;++h){volatile float sum=0;
            for(std::size_t d=0;d<D;++d){volatile float product=half_value(q[h*D+d])*half_value(k[(h/4)*D+d]);sum=sum+product;}
            const auto value=half_bits(sum);need(std::isfinite(half_value(value)),"current-token score overflow");raw[h*config.capacity+pos]=value;}
        ++statistics.current_token_scores;statistics.score_ms+=elapsed(begin);
    }
    std::vector<std::uint16_t> scale(const std::vector<std::uint16_t>& raw)const{
        std::vector<std::uint16_t> result(raw.size());
        for(std::size_t i=0;i<raw.size();++i)result[i]=half_bits(half_value(raw[i])/std::sqrt(128.f));
        return result;
    }
    void set_preference(int layer,const std::vector<std::vector<std::uint64_t>>& ids,bool physical){
        need(ids.size()==(physical?KH:H),"residency head count");
        auto* first=preferred.data()+std::size_t(layer)*KH*config.capacity;
        std::fill(first,first+KH*config.capacity,0);
        for(std::size_t h=0;h<ids.size();++h)for(auto id:ids[h]){
            need(id<completed[layer],"residency ID must already exist");first[(physical?h:h/4)*config.capacity+id]=1;}
    }
    std::vector<std::uint8_t> fetch_rows(int layer,const std::vector<std::uint64_t>& ids){
        std::lock_guard<std::recursive_mutex> lock(state_mutex);
        check_layer(layer);const auto begin=Clock::now();
        need(ids.size()<=config.capacity,"fetch row count exceeds bounded workspace");
        for(auto id:ids)need(id<completed[layer],"fetch contains future position");
        std::vector<std::uint8_t> out(mul(ids.size(),KV_BYTES));
        if(config.mode=="FULL_RECOMPUTE"){
            std::vector<std::vector<std::uint64_t>> missing(KH);
            std::vector<std::size_t> destination;
            for(auto& head:missing)head.reserve(ids.size());
            destination.reserve(ids.size());
            for(std::size_t i=0;i<ids.size();++i){
                const auto slot=fixed_slots[ids[i]];
                if(slot>=0){
                    std::memcpy(out.data()+i*KV_BYTES,fixed_hot_kv.data()+fixed_offset(layer,slot),KV_BYTES);
                    statistics.resident_fetch_bytes+=KV_BYTES;
                }else{
                    need(ids[i]<config.prompt_tokens,"generated KV unexpectedly missing from fixed hot slab");
                    for(auto& head:missing)head.push_back(ids[i]);
                    destination.push_back(i);
                }
            }
            if(!destination.empty()){
                const auto started=Clock::now();auto rows=batch_recompute(layer,missing);
                statistics.recompute_ms+=elapsed(started);++statistics.recompute_calls;
                need(rows.size()==KH,"actual batch recomputation returned wrong physical head count");
                for(std::size_t h=0;h<KH;++h){
                    need(rows[h].size()==mul(destination.size(),HEAD_BYTES),"actual batch recomputation returned wrong head row bytes");
                    validate_half_bytes(rows[h]);
                    for(std::size_t i=0;i<destination.size();++i){auto* dst=out.data()+destination[i]*KV_BYTES;
                        std::memcpy(dst+h*D*2,rows[h].data()+i*HEAD_BYTES,D*2);
                        std::memcpy(dst+KEY_WORDS*2+h*D*2,rows[h].data()+i*HEAD_BYTES+D*2,D*2);}
                    statistics.recomputed_rows+=destination.size();statistics.recomputed_bytes+=rows[h].size();
                }
                // Deliberately no restore: the same cold IDs miss next time.
            }
            ++statistics.fetch_calls;statistics.fetch_bytes+=out.size();statistics.fetch_ms+=elapsed(begin);return out;
        }
        for(std::size_t h=0;h<KH;++h){
            const auto ln=lane(layer,h);
            std::vector<std::uint64_t> resident_ids,missing_ids;
            std::vector<std::size_t> resident_dest,missing_dest;
            resident_ids.reserve(ids.size());missing_ids.reserve(ids.size());
            resident_dest.reserve(ids.size());missing_dest.reserve(ids.size());
            for(std::size_t i=0;i<ids.size();++i){
                if(store->contains(ln,ids[i])){resident_ids.push_back(ids[i]);resident_dest.push_back(i);}
                else{need(ids[i]<config.prompt_tokens,"generated KV unexpectedly missing; dense recompute would change sparse history");missing_ids.push_back(ids[i]);missing_dest.push_back(i);}
            }
            auto copy_rows=[&](const std::vector<std::uint8_t>& rows,const std::vector<std::size_t>& dest){
                need(rows.size()==mul(dest.size(),HEAD_BYTES),"KV head response byte count");
                for(std::size_t i=0;i<dest.size();++i){auto* dst=out.data()+dest[i]*KV_BYTES;
                    std::memcpy(dst+h*D*2,rows.data()+i*HEAD_BYTES,D*2);
                    std::memcpy(dst+KEY_WORDS*2+h*D*2,rows.data()+i*HEAD_BYTES+D*2,D*2);}
            };
            if(!resident_ids.empty()){
                auto rows=store->fetch(ln,resident_ids,statistics.staging_reserved_bytes);copy_rows(rows,resident_dest);
                statistics.resident_fetch_bytes+=rows.size();
            }
            if(!missing_ids.empty()){
                need(bool(recompute),"absent KV requires an actual model recomputation callback");
                const auto started=Clock::now();auto rows=recompute(layer,int(h),missing_ids);
                statistics.recompute_ms+=elapsed(started);++statistics.recompute_calls;
                need(rows.size()==mul(missing_ids.size(),HEAD_BYTES),"actual recomputation returned wrong head row bytes");
                validate_half_bytes(rows);copy_rows(rows,missing_dest);
                statistics.recomputed_rows+=missing_ids.size();statistics.recomputed_bytes+=rows.size();
                // Restore only into an available slot. Missing rows can be
                // consumed directly without allocating a second full cache.
                for(std::size_t i=0;i<missing_ids.size()&&store->resident_records(ln)<statistics.resident_rows_per_head;++i){
                    const auto id=missing_ids[i];if(store->contains(ln,id)||(config.mode=="H2O"&&!wanted(ln,id)))continue;
                    std::vector<std::uint8_t> row(rows.begin()+i*HEAD_BYTES,rows.begin()+(i+1)*HEAD_BYTES);
                    store->restore(ln,id,row);++statistics.restored_rows;
                }
            }
        }
        ++statistics.fetch_calls;statistics.fetch_bytes+=out.size();statistics.fetch_ms+=elapsed(begin);return out;
    }
};

NoSSDPolicy::NoSSDPolicy(const Config& config,Recompute callback,BatchRecompute batch):impl_(std::make_unique<Impl>(config,std::move(callback),std::move(batch))){}
NoSSDPolicy::~NoSSDPolicy()=default;

void NoSSDPolicy::append_full(int layer,const std::vector<std::uint64_t>& positions,
        const std::vector<std::uint16_t>& keys,const std::vector<std::uint8_t>& full_kv){
    auto& p=*impl_;std::lock_guard<std::recursive_mutex> lock(p.state_mutex);p.check_layer(layer);need(!positions.empty(),"empty append");
    need(keys.size()==mul(positions.size(),KEY_WORDS)&&full_kv.size()==mul(positions.size(),KV_BYTES),"append model K/KV shape");
    need(positions.front()==p.completed[layer]&&positions.size()<=p.config.capacity-p.completed[layer],"consecutive bounded append progress");
    for(std::size_t i=0;i<positions.size();++i){need(positions[i]==p.completed[layer]+i,"append positions must be consecutive");
        need(std::memcmp(keys.data()+i*KEY_WORDS,full_kv.data()+i*KV_BYTES,KEY_WORDS*2)==0,"model K and complete KV bytes differ");}
    validate_half_bytes(full_kv);
    if(p.config.mode=="FULL_RECOMPUTE"){
        need(positions.back()<p.config.prompt_tokens+p.config.generated_capacity,"append exceeds explicit generated reserve");
        for(std::size_t i=0;i<positions.size();++i){const auto slot=p.fixed_slots[positions[i]];
            if(slot>=0){std::memcpy(p.fixed_hot_kv.data()+p.fixed_offset(layer,slot),full_kv.data()+i*KV_BYTES,KV_BYTES);
                p.statistics.fixed_resident_records+=KH;
            }else p.statistics.cold_append_discarded_bytes+=KV_BYTES;
        }
        p.statistics.peak_fixed_resident_records=std::max(p.statistics.peak_fixed_resident_records,p.statistics.fixed_resident_records);
        p.completed[layer]+=positions.size();p.statistics.append_records+=positions.size();return;
    }
    if(p.config.mode=="PREFETCH"&&layer>=2)std::copy(keys.begin(),keys.end(),p.key(layer,positions.front()));
    for(std::size_t h=0;h<KH;++h){const auto ln=p.lane(layer,h);
        if(positions.size()<=p.statistics.resident_rows_per_head-p.store->resident_records(ln)){
            std::vector<std::uint8_t> rows(mul(positions.size(),HEAD_BYTES));
            for(std::size_t i=0;i<positions.size();++i){const auto* src=full_kv.data()+i*KV_BYTES;
                std::memcpy(rows.data()+i*HEAD_BYTES,src+h*D*2,D*2);
                std::memcpy(rows.data()+i*HEAD_BYTES+D*2,src+KEY_WORDS*2+h*D*2,D*2);}
            p.store->append(ln,positions.front(),rows);
        }else{
            std::vector<std::uint8_t> row(HEAD_BYTES);
            for(std::size_t i=0;i<positions.size();++i){p.make_slot(ln);const auto* src=full_kv.data()+i*KV_BYTES;
                std::memcpy(row.data(),src+h*D*2,D*2);std::memcpy(row.data()+D*2,src+KEY_WORDS*2+h*D*2,D*2);
                p.store->append(ln,positions[i],row);}
        }
    }
    p.completed[layer]+=positions.size();p.statistics.append_records+=positions.size();
}

std::vector<std::vector<std::uint64_t>> NoSSDPolicy::select(int layer,std::uint64_t position,const std::vector<float>& rehearsal){
    auto& p=*impl_;p.check_layer(layer);
    {std::lock_guard<std::recursive_mutex> lock(p.state_mutex);need(position<p.config.capacity&&p.completed[layer]==position+1,"selection requires newly appended current token");}
    const auto begin=Clock::now();
    if(p.config.mode=="H2O"||p.config.mode=="FULL_RECOMPUTE"){
        std::vector<std::vector<std::uint64_t>> ids(H,std::vector<std::uint64_t>(position+1));
        for(auto& head:ids)std::iota(head.begin(),head.end(),0);
        {std::lock_guard<std::recursive_mutex> lock(p.state_mutex);++p.statistics.selections;p.statistics.selection_ms+=elapsed(begin);}return ids;
    }
    const auto q=query_half(rehearsal);std::vector<std::uint16_t> raw;
    std::shared_ptr<Impl::Pending> prepared;
    {std::lock_guard<std::recursive_mutex> lock(p.state_mutex);const auto at=p.pending.find(layer);
        if(at!=p.pending.end()){prepared=at->second;need(prepared->position==position&&prepared->query==q,"prefetched/final rehearsal query or position mismatch");}}
    if(prepared){
        auto work=prepared->result.get(); // No state lock: worker may restore KV.
        raw=std::move(work.raw);p.score_current(layer,position,q,raw);
        std::lock_guard<std::recursive_mutex> lock(p.state_mutex);
        if(work.payload){
            p.ready_payload[layer]=std::move(work.payload);
        }
        p.pending.erase(layer);
    }else raw=p.score(layer,position+1,q);
    auto result=select_gqa_scores(p.scale(raw),H,p.config.capacity,position+1,p.config.selector_alpha,.2f);
    need(result.k>0,"selector selected no KV; no hidden full-attention fallback");
    std::lock_guard<std::recursive_mutex> lock(p.state_mutex);p.set_preference(layer,result.ids,false);
    for(std::size_t h=0;h<KH;++h)for(std::uint64_t t=0;t<=position;++t)if(p.wanted(p.lane(layer,h),t)){
        auto& count=p.selection_count[p.lane(layer,h)*p.config.capacity+t];
        need(count<std::numeric_limits<std::uint64_t>::max(),"selection-frequency counter overflow");++count;
    }
    ++p.statistics.selections;p.statistics.selection_ms+=elapsed(begin);return result.ids;
}

std::vector<std::uint8_t> NoSSDPolicy::fetch(int layer,const std::vector<std::uint64_t>& ids){
    auto& p=*impl_;std::lock_guard<std::recursive_mutex> lock(p.state_mutex);p.check_layer(layer);const auto at=p.ready_payload.find(layer);
    if(at==p.ready_payload.end())return p.fetch_rows(layer,ids);
    const auto artifact=at->second;p.ready_payload.erase(at);
    need(artifact->position+1==p.completed[layer],"stale no-SSD prepared payload");
    need(ids.size()<=p.config.capacity,"prepared fetch workspace bound");
    std::vector<std::uint8_t> result(mul(ids.size(),KV_BYTES));
    std::vector<std::uint64_t> missing;std::vector<std::size_t> destination;
    for(std::size_t i=0;i<ids.size();++i){need(ids[i]<p.completed[layer],"prepared fetch future ID");
        const auto found=std::lower_bound(artifact->physical_ids.begin(),artifact->physical_ids.end(),ids[i]);
        if(found!=artifact->physical_ids.end()&&*found==ids[i]){
            const auto index=std::size_t(found-artifact->physical_ids.begin());std::memcpy(result.data()+i*KV_BYTES,artifact->kv_bytes.data()+index*KV_BYTES,KV_BYTES);
            p.statistics.prefetch_useful_bytes+=KV_BYTES;
        }else{missing.push_back(ids[i]);destination.push_back(i);}
    }
    // Ordinary selected delivery misses at most the newly appended current
    // token. Keep even an arbitrary diagnostic fetch bounded without another
    // capacity-sized complete-KV vector on top of two history artifacts.
    constexpr std::size_t refill_rows=64;
    for(std::size_t first=0;first<missing.size();first+=refill_rows){const auto n=std::min(refill_rows,missing.size()-first);
        std::vector<std::uint64_t> chunk(missing.begin()+first,missing.begin()+first+n);
        auto extra=p.fetch_rows(layer,chunk);for(std::size_t i=0;i<n;++i)
            std::memcpy(result.data()+destination[first+i]*KV_BYTES,extra.data()+i*KV_BYTES,KV_BYTES);}
    return result;
}

void NoSSDPolicy::prepare_history(int layer,std::uint64_t position,const std::vector<float>& rehearsal,runtime::GQAPrefetchReady ready){
    auto& p=*impl_;p.check_layer(layer);need(p.config.mode=="PREFETCH","only PREFETCH performs rehearsal prefetch");
    auto task=std::make_shared<Impl::Pending>();task->position=position;task->query=query_half(rehearsal);
    {std::lock_guard<std::recursive_mutex> lock(p.state_mutex);
        need(position>0&&position<p.config.capacity&&p.completed[layer]==position,"history prepare must precede current KV append");
        need(p.pending.count(layer)==0&&p.ready_payload.count(layer)==0,"duplicate or unconsumed layer prefetch");
        need(p.pending.size()+p.ready_payload.size()<2,"two-artifact prefetch budget exceeded");
        p.pending.emplace(layer,task);
    }
    try{task->result=std::async(std::launch::async,[&p,layer,position,q=task->query,ready=std::move(ready)]() mutable {
    Impl::Work work;work.raw=p.score(layer,position,q);
    auto a=std::make_shared<runtime::GQAPrefetchArtifact>();a->layer=layer;a->position=position;a->score_ready_ns=stamp();
    auto scaled=p.scale(work.raw);
    const auto cap=std::min<std::size_t>(position,std::floor(double(position+1)*double(.2f)));
    need(cap>0,"historical prefetch cap is empty");
    float maximum=-std::numeric_limits<float>::infinity();
    for(std::size_t t=0;t<position;++t)maximum=std::max(maximum,half_value(scaled[t]));
    const auto threshold=half_value(half_bits(maximum-p.config.selector_alpha));need(std::isfinite(threshold),"historical threshold overflow");
    float total=0;
    for(std::size_t h=0;h<H;++h){std::uint64_t count=1;
        for(std::size_t t=0;t<position;++t)count+=half_value(scaled[h*p.config.capacity+t])>=threshold;
        const auto rounded=half_value(half_bits(float(count)));need(std::isfinite(rounded),"historical F16 count overflow");total+=rounded;}
    const auto mean=half_value(half_bits(total/float(H)));need(std::isfinite(mean)&&mean>=1,"historical mean upper bound");
    const auto bound=std::min(cap,std::size_t(mean));
    std::vector<std::uint64_t> ranked(position);std::vector<std::uint8_t> used(position,0);a->per_head_ids.resize(H);
    for(std::size_t h=0;h<H;++h){std::iota(ranked.begin(),ranked.end(),0);
        auto compare=[&](std::uint64_t x,std::uint64_t y){const auto vx=half_value(scaled[h*p.config.capacity+x]),vy=half_value(scaled[h*p.config.capacity+y]);return vx!=vy?vx>vy:x<y;};
        std::partial_sort(ranked.begin(),ranked.begin()+bound,ranked.end(),compare);
        a->per_head_ids[h].assign(ranked.begin(),ranked.begin()+bound);for(auto id:a->per_head_ids[h])used[id]=1;
    }
    for(std::size_t t=0;t<position;++t)if(used[t])a->physical_ids.push_back(t);
    a->read_started_ns=stamp();
    {std::lock_guard<std::recursive_mutex> lock(p.state_mutex);const auto resident_before=p.statistics.resident_fetch_bytes;
        a->kv_bytes=p.fetch_rows(layer,a->physical_ids);a->hot_bytes=p.statistics.resident_fetch_bytes-resident_before;}
    a->read_finished_ns=stamp();work.payload=a;
    {std::lock_guard<std::recursive_mutex> lock(p.state_mutex);++p.statistics.prefetch_calls;p.statistics.prefetch_bytes+=a->kv_bytes.size();}
    // This entire task is asynchronous with current-layer GPU work. The
    // caller's upload callback may synchronize its own CUDA completion event;
    // select joins the task before consuming this target layer's artifact.
    if(ready){
        ready(std::move(a));
    }
    return work;
    });}catch(...){std::lock_guard<std::recursive_mutex> lock(p.state_mutex);p.pending.erase(layer);throw;}
}

void NoSSDPolicy::set_h2o_resident(int layer,const std::vector<std::vector<std::uint64_t>>& ids){
    auto& p=*impl_;std::lock_guard<std::recursive_mutex> lock(p.state_mutex);p.check_layer(layer);need(p.config.mode=="H2O","H2O residency on wrong mode");
    need(ids.size()==KH,"H2O requires eight physical-head sets");
    const auto pinned=p.completed[layer]>p.config.prompt_tokens?p.completed[layer]-p.config.prompt_tokens:0;
    for(const auto& head:ids){need(head.size()<=p.statistics.resident_rows_per_head,"H2O residency exceeds head capacity");
        std::vector<std::uint64_t> sorted=head;std::sort(sorted.begin(),sorted.end());
        need(std::adjacent_find(sorted.begin(),sorted.end())==sorted.end(),"duplicate H2O resident ID");
        std::size_t prompt_rows=0;for(auto id:head){need(id<p.completed[layer],"future H2O resident ID");prompt_rows+=id<p.config.prompt_tokens;}
        need(prompt_rows+pinned<=p.statistics.resident_rows_per_head,"H2O prompt residency leaves no space for pinned generated rows");}
    p.set_preference(layer,ids,true);if(p.statistics.all_KV_fits)return;
    for(std::size_t h=0;h<KH;++h){const auto ln=p.lane(layer,h);std::vector<std::uint64_t> evict;
        for(std::uint64_t pos=0;pos<std::min<std::uint64_t>(p.completed[layer],p.config.prompt_tokens);++pos)
            if(!p.wanted(ln,pos)&&p.store->contains(ln,pos))evict.push_back(pos);
        p.store->evict(ln,evict);p.statistics.evicted_rows+=evict.size();}
}

void NoSSDPolicy::wait_pending(){
    std::vector<std::shared_ptr<Impl::Pending>> tasks;
    {std::lock_guard<std::recursive_mutex> lock(impl_->state_mutex);for(const auto& task:impl_->pending)tasks.push_back(task.second);}
    for(const auto& task:tasks)if(task->result.valid())task->result.wait();
}
void NoSSDPolicy::drain(){wait_pending();std::lock_guard<std::recursive_mutex> lock(impl_->state_mutex);need(impl_->pending.empty(),"request ended with unconsumed historical scoring work");impl_->ready_payload.clear();}
std::size_t NoSSDPolicy::managed_bytes()const{return impl_->statistics.managed_reserved_bytes;}
std::size_t NoSSDPolicy::resident_rows_per_head()const{return impl_->statistics.resident_rows_per_head;}
bool NoSSDPolicy::all_kv_fits()const{return impl_->statistics.all_KV_fits;}
const NoSSDPolicy::Stats& NoSSDPolicy::stats()const{return impl_->statistics;}
void NoSSDPolicy::write_stats(const std::string& path)const{
    const auto& p=*impl_;std::lock_guard<std::recursive_mutex> lock(p.state_mutex);const auto& s=p.statistics;
    const bool fixed=p.config.mode=="FULL_RECOMPUTE";
    const auto resident_records=fixed?s.fixed_resident_records:p.store->metrics().resident_records;
    const auto peak_resident_records=fixed?s.peak_fixed_resident_records:p.store->metrics().peak_resident_records;
    nlohmann::json j{{"schema","hillserve.no_ssd_policy.v1"},{"mode",p.config.mode},
        {"storage","bounded_memory_only"},{"ssd_read_bytes",0},{"ssd_write_bytes",0},
        {"actual_fpga",false},{"missing_KV_source","caller_actual_model_recompute"},
        {"generated_KV","always_pinned_never_dense_recomputed"},
        {"prefetch_preparation",p.config.mode=="PREFETCH"?"bounded_async_history_CPU_score_fetch_and_GPU_callback; target_select_joins_then_scores_current":"disabled"},
        {"max_inflight_prefetch_tasks",p.config.mode=="PREFETCH"?2:0},
        {"residency_policy",fixed?"fixed_offline_APP_same_full_gqa_plan_early_plus_recent_prompt_and_generated_pins_no_cold_restore":(p.config.mode=="H2O"?"external_actual_attention_H2O_per_KV_head; all-fit-kept":"selection_frequency_per_physical_KV_head; minimum_counter_then_oldest_ID; all-fit-kept")},
        {"selector",p.config.mode!="PREFETCH"?"all_historical_positions_full_attention":"full128_F16dot_sqrt128_F16_head0max_inclusive_F16countmean_cap0.2_originalIDties"},
        {"selector_alpha",p.config.selector_alpha},{"cpu_threads",p.config.cpu_threads},
        {"first_rehearsal_layer",2},
        {"cpu_implementation",s.cpu_implementation_actual},{"capacity",p.config.capacity},
        {"prompt_tokens",p.config.prompt_tokens},{"layers",p.config.layers},
        {"total_host_budget_bytes",p.config.total_host_budget_bytes},{"managed_reserved_bytes",s.managed_reserved_bytes},
        {"score_key_reserved_bytes",s.score_key_reserved_bytes},{"resident_reserved_bytes",s.resident_reserved_bytes},
        {"resident_kv_payload_reserved_bytes",s.resident_kv_payload_reserved_bytes},
        {"resident_kv_payload_budget_bytes",p.config.resident_kv_payload_budget_bytes},
        {"staging_reserved_bytes",s.staging_reserved_bytes},{"metadata_reserved_bytes",s.metadata_reserved_bytes},
        {"budget_scope","persistent_requested_bytes_plus_bounded_temporary_reserve; caller_recompute_context_and_GPU_buffers_separate; OS_stacks_and_allocator_in_measured_RSS"},
        {"resident_rows_per_head",s.resident_rows_per_head},{"all_KV_fits",s.all_KV_fits},
        {"append_records",s.append_records},{"score_calls",s.score_calls},{"current_token_scores",s.current_token_scores},
        {"selections",s.selections},{"fetch_calls",s.fetch_calls},{"fetch_bytes",s.fetch_bytes},
        {"resident_fetch_bytes",s.resident_fetch_bytes},{"recompute_calls",s.recompute_calls},
        {"recomputed_rows",s.recomputed_rows},{"recomputed_bytes",s.recomputed_bytes},
        {"prefetch_calls",s.prefetch_calls},{"prefetch_bytes",s.prefetch_bytes},{"prefetch_useful_bytes",s.prefetch_useful_bytes},
        {"evicted_rows",s.evicted_rows},{"restored_rows",s.restored_rows},
        {"score_ms",s.score_ms},{"selection_including_score_ms",s.selection_ms},
        {"fetch_including_recompute_ms",s.fetch_ms},{"recompute_ms",s.recompute_ms},
        {"store_resident_records",resident_records},{"store_peak_resident_records",peak_resident_records}};
    if(fixed){
        j["requested_beta"]=std::isfinite(p.config.offline_beta)?nlohmann::json(p.config.offline_beta):nlohmann::json("all_hot");
        j["actual_prompt_rho"]=double(s.hot_prompt_tokens)/p.config.prompt_tokens;
        j["hot_prompt_tokens"]=s.hot_prompt_tokens;j["cold_prompt_tokens"]=s.cold_prompt_tokens;
        j["generated_reserved_always_hot"]=s.generated_reserved_tokens;
        j["cold_append_discarded_bytes"]=s.cold_append_discarded_bytes;
        j["cold_KV_retained_after_fetch_bytes"]=0;
        j["recompute_callback_scope"]="one_batch_all_eight_physical_heads_per_missing_fetch";
        j["first_rehearsal_layer"]=nullptr;
    }
    std::ofstream out(path);need(bool(out),"cannot write statistics");out<<j.dump(2)<<'\n';need(bool(out),"statistics write failed");
}
}
