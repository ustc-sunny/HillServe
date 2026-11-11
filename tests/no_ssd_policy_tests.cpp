#include "runtime/no_ssd_policy.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "policy/full_gqa_cpu.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "storage/bounded_memory_kv.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Constructed unit-test fixtures and a MOCK missing-row supplier. This tests
// storage/policy contracts only; it is NOT actual model recomputation evidence.
using hillserve::policy::NoSSDPolicy;
namespace {
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
template<class F>void rejects(F f,const char* message){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,message);}
std::vector<std::uint8_t> records(int layer,std::uint64_t first,std::size_t count){
    std::vector<std::uint16_t> words(count*2048);
    for(std::size_t t=0;t<count;++t)for(std::size_t h=0;h<8;++h)for(std::size_t d=0;d<128;++d){
        const float k=d==0?float((first+t)%17)*.25f+float(h)*.03125f+float(layer)*.125f:0;
        words[t*2048+h*128+d]=hillserve::half_bits(k);
        words[t*2048+1024+h*128+d]=hillserve::half_bits(float((first+t+d+layer)%31)*.015625f+float(h)*.03125f);
    }
    std::vector<std::uint8_t> out(words.size()*2);std::memcpy(out.data(),words.data(),out.size());return out;
}
std::vector<std::uint16_t> score_keys(const std::vector<std::uint8_t>& rows){
    std::vector<std::uint16_t> out(rows.size()/4);
    for(std::size_t t=0;t<rows.size()/4096;++t)std::memcpy(out.data()+t*1024,rows.data()+t*4096,2048);
    return out;
}
void append(NoSSDPolicy& p,int layer,std::uint64_t first,std::size_t count){
    std::vector<std::uint64_t> ids(count);std::iota(ids.begin(),ids.end(),first);
    const auto rows=records(layer,first,count);p.append_full(layer,ids,score_keys(rows),rows);
}
std::vector<float> query(){
    std::vector<float> q(4096,0);
    for(std::size_t h=0;h<32;++h){q[h*128]=1;}
    return q;
}
std::vector<std::vector<std::uint64_t>> oracle(int layer,std::size_t count,const std::vector<float>& q){
    const auto k=score_keys(records(layer,0,count));std::vector<std::uint16_t> qh(q.size());
    for(std::size_t i=0;i<qh.size();++i)qh[i]=hillserve::half_bits(q[i]);
    hillserve::FullGqaCpuScorer scorer(1,hillserve::FullGqaCpuImplementation::scalar);
    std::vector<std::uint16_t> raw;scorer.score(qh,{{k.data(),count,1024}},raw);
    for(auto& v:raw)v=hillserve::half_bits(hillserve::half_value(v)/std::sqrt(128.f));
    return hillserve::policy::select_gqa_scores(raw,32,count,count,10,.2f).ids;
}
std::vector<std::uint64_t> physical(const std::vector<std::vector<std::uint64_t>>& per_head){
    std::vector<std::uint64_t> ids;for(const auto& h:per_head)ids.insert(ids.end(),h.begin(),h.end());
    std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());return ids;
}
void check_rows(const std::vector<std::uint8_t>& actual,int layer,const std::vector<std::uint64_t>& ids){
    check(actual.size()==ids.size()*4096,"full KV output byte count");
    for(std::size_t i=0;i<ids.size();++i){const auto wanted=records(layer,ids[i],1);
        check(std::equal(wanted.begin(),wanted.end(),actual.begin()+i*4096),"full KV bytes/order changed");}
}
NoSSDPolicy::Config config(std::string mode="PREFETCH"){
    NoSSDPolicy::Config c;c.mode=std::move(mode);c.layers=3;c.capacity=80;c.prompt_tokens=64;c.generated_capacity=8;
    c.total_host_budget_bytes=64ULL<<20;c.cpu_threads=2;return c;
}
NoSSDPolicy::Recompute mock_callback(std::uint64_t& calls){
    return [&calls](int layer,int head,const std::vector<std::uint64_t>& ids){
        ++calls;std::vector<std::uint8_t> out(ids.size()*512);
        for(std::size_t i=0;i<ids.size();++i){check(ids[i]<64,"generated KV must never be recomputed");const auto row=records(layer,ids[i],1);
            std::memcpy(out.data()+i*512,row.data()+head*256,256);
            std::memcpy(out.data()+i*512+256,row.data()+2048+head*256,256);}
        return out;
    };
}
NoSSDPolicy::Config tight_config(const std::string& mode,std::size_t rows){
    auto c=config(mode);NoSSDPolicy all(c,{});const auto s=all.stats();
    hillserve::BoundedMemoryKVStore::Config sc{c.layers*8,c.capacity,512,rows,0};
    c.total_host_budget_bytes=s.managed_reserved_bytes-s.resident_reserved_bytes+hillserve::BoundedMemoryKVStore::estimate_bytes(sc);
    return c;
}
NoSSDPolicy::Config fixed_config(double beta=1){
    NoSSDPolicy::Config c;c.mode="FULL_RECOMPUTE";c.layers=32;c.capacity=320;
    c.prompt_tokens=256;c.generated_capacity=4;c.offline_beta=beta;
    c.resident_kv_payload_budget_bytes=32*(hillserve::runtime::full_gqa_hot_blocks(c.prompt_tokens,beta)*64+c.generated_capacity)*4096;
    c.total_host_budget_bytes=64ULL<<20;return c;
}
NoSSDPolicy::BatchRecompute mock_batch_callback(std::uint64_t& calls,
        std::vector<std::vector<std::uint64_t>>& observed,std::size_t prompt_tokens){
    return [&calls,&observed,prompt_tokens](int layer,const std::vector<std::vector<std::uint64_t>>& ids){
        ++calls;observed=ids;check(ids.size()==8,"MOCK batch callback requires all eight heads together");
        std::vector<std::vector<std::uint8_t>> out(8);
        for(std::size_t h=0;h<8;++h){out[h].resize(ids[h].size()*512);
            for(std::size_t i=0;i<ids[h].size();++i){check(ids[h][i]<prompt_tokens,"MOCK must never recompute generated position");
                const auto row=records(layer,ids[h][i],1);
                std::memcpy(out[h].data()+i*512,row.data()+h*256,256);
                std::memcpy(out[h].data()+i*512+256,row.data()+2048+h*256,256);
            }
        }
        return out;
    };
}
}

int main(){try{
    std::uint64_t mock_calls=0;
    for(const auto& implementation:{"legacy","auto"}){
        auto c=config();c.cpu_implementation=implementation;NoSSDPolicy p(c,mock_callback(mock_calls));
        check(p.all_kv_fits()&&p.resident_rows_per_head()==80,"ample budget must keep all KV");
        check(p.managed_bytes()<=c.total_host_budget_bytes,"managed memory exceeds budget");
        append(p,2,0,64);
        check_rows(p.fetch(2,{63,0,7,7}),2,{63,0,7,7});
        rejects([&]{append(p,2,65,1);},"nonconsecutive append accepted");
        auto q=query();std::shared_ptr<const hillserve::runtime::GQAPrefetchArtifact> prepared;
        p.prepare_history(2,64,q,[&](auto value){prepared=std::move(value);});
        // Append a distinct current-token tail while history score/fetch runs.
        append(p,2,64,1);p.wait_pending();
        check(prepared&&prepared->layer==2&&prepared->position==64,"actual history-ready callback missing");
        for(auto id:prepared->physical_ids)check(id<64,"prefetch included current/future token");
        check_rows(prepared->kv_bytes,2,prepared->physical_ids);
        auto wrong=q;wrong[0]=2;
        rejects([&]{p.select(2,64,wrong);},"changed rehearsal query silently reused historical scores");
        const auto ids=p.select(2,64,q);check(ids==oracle(2,65,q),"prefetch/current scores differ from full CPU ordered selector");
        for(const auto& head:ids)for(auto id:head)if(id<64)
            check(std::binary_search(prepared->physical_ids.begin(),prepared->physical_ids.end(),id),"safe historical prefetch omitted a final selected ID");
        const auto unique=physical(ids);check_rows(p.fetch(2,unique),2,unique);
        prepared.reset();
        p.prepare_history(2,65,q);append(p,2,65,1);
        const auto second=p.select(2,65,q);check(second==oracle(2,66,q),"second-step ordered selector mismatch");
        check_rows(p.fetch(2,physical(second)),2,physical(second));p.wait_pending();p.drain();
        check(p.stats().evicted_rows==0&&p.stats().recompute_calls==0&&mock_calls==0,"all-fit cache fabricated a miss");
        check(p.stats().prefetch_calls==2&&p.stats().current_token_scores==2&&p.stats().prefetch_useful_bytes>0,"prefetch/current evidence absent");
    }
    {
        const auto c=tight_config("PREFETCH",12);NoSSDPolicy p(c,mock_callback(mock_calls));
        check(p.resident_rows_per_head()==12&&!p.all_kv_fits(),"budget-derived resident cap mismatch");
        append(p,2,0,64);append(p,2,64,1);
        const auto ids=p.select(2,64,query());check(ids==oracle(2,65,query()),"evicted KV score representation not preserved");
        check_rows(p.fetch(2,{0,63,64,0}),2,{0,63,64,0});
        check(p.stats().recomputed_bytes>0&&p.stats().evicted_rows>0,"tight memory did not exercise explicit MOCK missing callback");
        // Only one low-frequency prompt row per physical head is removed as
        // new generated KV is admitted. Missing rows are not silently zeroed.
        const auto before=p.stats().evicted_rows;append(p,2,65,1);
        check(p.stats().evicted_rows==before+8,"generated append did not obey per-head residency budget");
        check_rows(p.fetch(2,{64,65}),2,{64,65});
        auto bad=c;--bad.total_host_budget_bytes;NoSSDPolicy slightly_smaller(bad,{});
        check(slightly_smaller.resident_rows_per_head()==11,"one-byte budget decrease did not reduce slab capacity");
        bad.total_host_budget_bytes=1;rejects([&]{NoSSDPolicy tiny(bad,{});},"impossible host budget accepted");
        auto stage=c;stage.staging_budget_bytes=1;rejects([&]{NoSSDPolicy tiny(stage,{});},"undersized staging reserve accepted");
    }
    {
        auto c=config();c.layers=5;NoSSDPolicy p(c,{});
        for(int l=2;l<5;++l)append(p,l,0,64);
        std::promise<void> release,entered2,entered3;auto released=release.get_future().share();
        p.prepare_history(2,64,query(),[&](auto){entered2.set_value();released.wait();});
        p.prepare_history(3,64,query(),[&](auto){entered3.set_value();released.wait();});
        rejects([&]{p.prepare_history(4,64,query());},"third outstanding history task exceeded the two-artifact budget");
        const bool both_ready=entered2.get_future().wait_for(std::chrono::seconds(5))==std::future_status::ready&&
                              entered3.get_future().wait_for(std::chrono::seconds(5))==std::future_status::ready;
        // These appends must not wait for the blocked upload callbacks. The
        // subsequent select joins only after release, without holding state.
        append(p,2,64,1);append(p,3,64,1);release.set_value();
        check(both_ready,"asynchronous history workers did not reach callback independently of target append");
        for(int l=2;l<4;++l){const auto ids=p.select(l,64,query());check(ids==oracle(l,65,query()),"concurrent target current-token scoring changed IDs");check_rows(p.fetch(l,physical(ids)),l,physical(ids));}
        p.drain();
    }
    {
        NoSSDPolicy p(tight_config("PREFETCH",12),{});append(p,0,0,64);
        rejects([&]{p.fetch(0,{0});},"missing callback silently replaced real recomputation");
        rejects([&]{p.fetch(0,{64});},"future KV fetch accepted");
    }
    {
        std::uint64_t frequency_mock_calls=0;
        NoSSDPolicy p(tight_config("PREFETCH",12),mock_callback(frequency_mock_calls));
        append(p,2,0,64);append(p,2,64,1);auto q=query();for(auto& v:q)v=-v;
        const auto ids=p.select(2,64,q);
        check(std::find(ids[0].begin(),ids[0].end(),53)!=ids[0].end()&&
              std::find(ids[0].begin(),ids[0].end(),54)==ids[0].end(),"selection-frequency fixture geometry");
        append(p,2,65,1);check_rows(p.fetch(2,{53}),2,{53});
        check(frequency_mock_calls==0,"minimum-frequency eviction discarded an older high-count selected row");
        check_rows(p.fetch(2,{54}),2,{54});
        check(frequency_mock_calls==8,"minimum-frequency eviction did not break zero-count tie by oldest ID");
    }
    {
        NoSSDPolicy p(config("H2O"),{});append(p,1,0,64);append(p,1,64,1);
        p.set_h2o_resident(1,std::vector<std::vector<std::uint64_t>>(8));
        check(p.stats().score_key_reserved_bytes==0,"H2O allocated rehearsal score K");
        const auto ids=p.select(1,64,{});check(ids.size()==32&&ids.front().size()==65,"H2O extension must perform full attention");
        for(const auto& head:ids)for(std::size_t i=0;i<head.size();++i)check(head[i]==i,"H2O omitted historical full-attention KV");
        check_rows(p.fetch(1,{0,63,64}),1,{0,63,64});
        check(p.stats().evicted_rows==0,"all-fit H2O fabricated eviction");
        rejects([&]{p.prepare_history(1,65,query());},"H2O mislabeled as rehearsal prefetch");
    }
    {
        NoSSDPolicy p(tight_config("H2O",12),mock_callback(mock_calls));append(p,0,0,64);append(p,0,64,1);
        std::vector<std::vector<std::uint64_t>> resident(8,{0,1,2,3,4,5,6,7,8,9,64});p.set_h2o_resident(0,resident);
        check_rows(p.fetch(0,{0,10,63,64}),0,{0,10,63,64});
        check(p.stats().recomputed_rows>0,"H2O tight memory did not use MOCK recomputation");
        resident.assign(8,{0,1,2,3,4,5,6,7,8,9,10,11});
        rejects([&]{p.set_h2o_resident(0,resident);},"H2O residency failed to reserve generated pinned slot");
        resident.assign(8,{0,0});rejects([&]{p.set_h2o_resident(0,resident);},"duplicate H2O ID accepted");
    }
    {
        // Fixed APP partition, not H2O selection/frequency residency. All
        // payloads below are constructed fixtures; callback is explicitly MOCK.
        auto c=fixed_config();std::uint64_t batch_calls=0;
        std::vector<std::vector<std::uint64_t>> observed;
        const auto callback=mock_batch_callback(batch_calls,observed,c.prompt_tokens);
        rejects([&]{NoSSDPolicy missing(c,{});},"FULL_RECOMPUTE accepted no actual batch callback");
        auto bad=c;bad.offline_beta=std::numeric_limits<double>::quiet_NaN();
        rejects([&]{NoSSDPolicy unknown(bad,{},callback);},"unspecified offline beta accepted");
        bad=c;--bad.resident_kv_payload_budget_bytes;
        rejects([&]{NoSSDPolicy small(bad,{},callback);},"under-budget fixed hot slab accepted");
        bad=c;++bad.resident_kv_payload_budget_bytes;
        rejects([&]{NoSSDPolicy oversized(bad,{},callback);},"ambiguous oversized payload allowance accepted");
        NoSSDPolicy p(c,{},callback);
        check(!p.all_kv_fits()&&p.resident_rows_per_head()==132,"fixed APP hot+generated rows differ");
        check(p.stats().resident_kv_payload_reserved_bytes==c.resident_kv_payload_budget_bytes&&
              p.stats().resident_reserved_bytes==c.resident_kv_payload_budget_bytes,"payload mixed with metadata budget");
        check(p.stats().score_key_reserved_bytes==0&&p.stats().hot_prompt_tokens==128&&p.stats().cold_prompt_tokens==128,
              "FULL_RECOMPUTE allocated score K or partition differs");
        check(p.managed_bytes()==p.stats().resident_reserved_bytes+p.stats().metadata_reserved_bytes+p.stats().staging_reserved_bytes,
              "fixed memory budget components do not reconcile");
        // Append all layers to verify that the allowance covers the whole
        // model and does not accidentally grant a separate quota per head.
        for(int layer=0;layer<32;++layer)append(p,layer,0,256);
        check(p.stats().fixed_resident_records==32*128*8&&
              p.stats().cold_append_discarded_bytes==32*128*4096ULL,"cold append retained hidden history or hot slots missing");
        // Hot positions exactly match the shared runtime APP plan.
        hillserve::runtime::FullGqaRuntimeSpec spec;spec.capacity=c.capacity;spec.prompt_tokens=c.prompt_tokens;
        spec.generated_capacity=c.generated_capacity;spec.beta=c.offline_beta;
        spec.hot_kv_budget_bytes=c.resident_kv_payload_budget_bytes;spec.total_host_workspace_budget_bytes=c.total_host_budget_bytes;
        const auto plan=hillserve::runtime::plan_full_gqa_runtime(spec);
        check_rows(p.fetch(7,plan.hot_positions),7,plan.hot_positions);
        check(batch_calls==0,"shared APP hot position was recomputed");
        append(p,7,256,4);
        const auto full=p.select(7,259,{});
        check(full.size()==32,"FULL_RECOMPUTE Q head count changed");
        for(const auto& head:full){check(head.size()==260,"FULL_RECOMPUTE omitted historical positions");
            for(std::size_t i=0;i<head.size();++i)check(head[i]==i,"FULL_RECOMPUTE attention IDs changed");}
        const auto before=p.stats().fixed_resident_records;
        const std::vector<std::uint64_t> requested{255,0,128,127,256,0,259};
        check_rows(p.fetch(7,requested),7,requested);
        check(batch_calls==1,"missing heads caused more than one actual replay call");
        for(const auto& head:observed)check(head==std::vector<std::uint64_t>({0,127,0}),"batch missing order/set changed");
        check_rows(p.fetch(7,requested),7,requested);
        check(batch_calls==2&&p.stats().fixed_resident_records==before&&p.stats().restored_rows==0,
              "cold fetch persisted hidden KV or changed fixed hot capacity");
        check_rows(p.fetch(7,{256,257,258,259}),7,{256,257,258,259});
        check(batch_calls==2,"generated rows were recomputed rather than pinned");
        check_rows(p.fetch(7,full.front()),7,full.front());
        check(batch_calls==3&&p.stats().score_calls==0&&p.stats().prefetch_calls==0,"full history path performed scoring/prefetch");
        rejects([&]{append(p,7,260,1);},"generated append exceeded explicit pin reserve");
        rejects([&]{p.prepare_history(7,260,{});},"FULL_RECOMPUTE accepted rehearsal prefetch");
        rejects([&]{p.set_h2o_resident(7,std::vector<std::vector<std::uint64_t>>(8));},"FULL_RECOMPUTE accepted H2O mutation");
        p.drain();
        const auto report=std::filesystem::temp_directory_path()/
            ("hillserve-full-recompute-MOCK-contract-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".json");
        p.write_stats(report.string());std::ifstream input(report);const std::string evidence((std::istreambuf_iterator<char>(input)),{});
        input.close();std::filesystem::remove(report);
        check(evidence.find("\"cold_KV_retained_after_fetch_bytes\": 0")!=std::string::npos&&
              evidence.find("\"first_rehearsal_layer\": null")!=std::string::npos,"fixed-mode statistics misrepresent cold retention/rehearsal");
        bad=c;bad.total_host_budget_bytes=p.managed_bytes()-1;
        rejects([&]{NoSSDPolicy tight(bad,{},callback);},"fixed memory components exceeded total budget");
        bad=c;bad.staging_budget_bytes=p.stats().staging_reserved_bytes-1;
        rejects([&]{NoSSDPolicy stage(bad,{},callback);},"insufficient batch workspace accepted");
    }
    {
        // All-hot control must not manufacture misses; all-cold still pins
        // generated KV and calls exactly one batch replay for each fetch.
        std::uint64_t calls=0;std::vector<std::vector<std::uint64_t>> observed;
        auto callback=mock_batch_callback(calls,observed,256);
        auto c=fixed_config(std::numeric_limits<double>::infinity());NoSSDPolicy hot(c,{},callback);
        append(hot,0,0,256);append(hot,0,256,1);
        check_rows(hot.fetch(0,{0,255,256}),0,{0,255,256});
        check(hot.all_kv_fits()&&calls==0,"all-hot FULL_RECOMPUTE fabricated a miss");
        c=fixed_config(0);NoSSDPolicy cold(c,{},callback);append(cold,0,0,256);append(cold,0,256,1);
        check_rows(cold.fetch(0,{0,255,256}),0,{0,255,256});
        check(calls==1&&cold.stats().fixed_resident_records==8,"all-cold prompt unexpectedly became resident");
        NoSSDPolicy wrong_shape(c,{},[](int,const auto&){return std::vector<std::vector<std::uint8_t>>(7);});
        append(wrong_shape,0,0,256);rejects([&]{wrong_shape.fetch(0,{0});},"malformed actual batch output accepted");
    }
    std::cout<<"PASS no-SSD policy: exact scorer/selector, bounded prefetch, H2O full attention, fixed APP FULL_RECOMPUTE payload budget, eight-head MOCK batch misses, no cold retention, generated pins\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL no-SSD policy: "<<e.what()<<'\n';return 1;}}
