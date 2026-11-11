#include "policy/full_gqa_runtime.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "storage/segmented_store.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using namespace hillserve;
using namespace hillserve::policy;
namespace {
constexpr std::size_t P=4096,G=3,CAP=4160,LAYER=8;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// Software test only: block16 ranks first, then other cold blocks, then hot
// blocks. Distinct per-KV-head/channel values exercise the full GQA map.
std::uint16_t key(std::size_t t,std::size_t kh,std::size_t d){
    const auto block=t/64;
    const float base=block==16?.25f:(block>=17&&block<48)?.125f:.01f;
    return half_bits(base+float(kh)*.0002f+float(d)*.000001f);
}
std::vector<std::uint8_t> kv_records(std::size_t first,std::size_t n){
    std::vector<std::uint16_t> words(n*2048);
    for(std::size_t i=0;i<n;++i)for(std::size_t kh=0;kh<8;++kh)for(std::size_t d=0;d<128;++d){
        words[i*2048+kh*128+d]=key(first+i,kh,d);
        words[i*2048+1024+kh*128+d]=half_bits(.01f*float(LAYER+1)+float(first+i)*.0001f+float(kh*128+d)*.000001f);
    }
    std::vector<std::uint8_t> bytes(words.size()*2);std::memcpy(bytes.data(),words.data(),bytes.size());return bytes;
}
std::vector<float> query(std::size_t step){
    std::vector<float> q(32*128);for(std::size_t h=0;h<32;++h)for(std::size_t d=0;d<128;++d)q[h*128+d]=.05f+float(h)*.001f+float(d)*.000001f+float(step)*.0001f;return q;
}
std::vector<std::vector<std::uint64_t>> reference(std::size_t n,const std::vector<float>& q){
    std::vector<std::uint16_t> scores(32*n);
    for(std::size_t h=0;h<32;++h)for(std::size_t t=0;t<n;++t){
        volatile float sum=0;
        for(std::size_t d=0;d<128;++d){volatile float product=half_value(half_bits(q[h*128+d]))*half_value(key(t,h/4,d));sum=sum+product;}
        scores[h*n+t]=half_bits(half_value(half_bits(sum))/std::sqrt(128.f));
    }
    return select_gqa_scores(scores,32,n,n,5,.2f).ids;
}
std::vector<std::uint64_t> physical(const std::vector<std::vector<std::uint64_t>>& ids){
    std::vector<std::uint64_t> u;for(const auto& h:ids)u.insert(u.end(),h.begin(),h.end());
    std::sort(u.begin(),u.end());u.erase(std::unique(u.begin(),u.end()),u.end());return u;
}
}
int main(){
    std::filesystem::path directory;
    try{
        directory=std::filesystem::temp_directory_path()/("hillserve-hpm-runtime-test-"+std::to_string(getpid()));
        check(std::filesystem::create_directory(directory),"exclusive new software-test directory");
        std::uint64_t verified=0,callback_bytes=0,callback_count=0;
        {
            runtime::FullGqaRuntimeSpec s;s.prompt_tokens=P;s.generated_capacity=G;s.capacity=CAP;s.beta=1;
            s.hot_kv_budget_bytes=32*(P/2+G)*4096ULL;s.additional_workspace_bytes=64ULL<<20;
            s.total_host_workspace_budget_bytes=s.hot_kv_budget_bytes+s.additional_workspace_bytes;s.chunk_tokens=256;
            const auto plan=runtime::plan_full_gqa_runtime(s);
            SegmentedKVStore backing(directory/"complete-kv",711,32,CAP,4096,plan.segmented_KV_extent_bytes);
            backing.enable_direct_reads(directory/"complete-kv");
            backing.enable_direct_writes(directory/"complete-kv");
            FullGqaRuntimePolicy::Config c;c.prompt_tokens=P;c.generated_capacity=G;c.capacity=CAP;c.beta=1;
            c.hot_kv_budget_bytes=s.hot_kv_budget_bytes;c.total_host_workspace_budget_bytes=512ULL<<20;
            c.chunk_tokens=256;c.hot_threads=2;c.cold_threads=1;c.mode="HOST_MATCHED";
            c.score_file=(directory/"raw-k").string();c.hpm=true;c.hpm_swaps=1;c.verify_scores=true;
            FullGqaRuntimePolicy policy(c);
            check(policy.stats().hot_kv_cache_bytes==32*(P/2+G)*4096ULL,"all32 layers plus generated tail budget remains reserved");
            check(policy.is_hot(LAYER,0)&&!policy.is_hot(LAYER,16*64),"initial mutable hot victim and cold candidate");
            auto append=[&](std::size_t first,std::size_t n){
                auto kv=kv_records(first,n);std::vector<std::uint16_t> keys(n*1024);
                for(std::size_t i=0;i<n;++i)std::memcpy(keys.data()+i*1024,kv.data()+i*4096,2048);
                std::vector<std::uint64_t> pos(n);std::iota(pos.begin(),pos.end(),first);
                backing.append(LAYER,first,kv);policy.append(LAYER,pos,keys,kv);
            };
            for(std::size_t first=0;first<P;first+=256)append(first,256);
            auto read=[&](int layer,const std::vector<std::uint64_t>& ids){
                check(layer==int(LAYER)&&!ids.empty()&&ids.size()<=64,"bounded actual cold KV callback");
                for(auto id:ids)check(!policy.is_hot(layer,id)&&id<backing.completed(layer),"hot/future token reached cold file reader");
                auto data=backing.fetch(layer,ids,64*4096);callback_bytes+=data.size();++callback_count;return data;
            };
            auto verify_fetch=[&](const std::vector<std::uint64_t>& ids){
                auto data=policy.fetch(LAYER,ids,read);check(data.size()==ids.size()*4096,"exact fetched byte count");
                for(std::size_t i=0;i<ids.size();++i){const auto expected=kv_records(ids[i],1);check(!std::memcmp(data.data()+i*4096,expected.data(),4096),"selected K and V bytes changed after HPM slot reuse");}
                verified+=data.size();
            };
            for(std::size_t step=0;step<G;++step){
                const auto position=P+step;const auto q=query(step);
                if(step)policy.prepare_history(LAYER,position,q);
                append(position,1);
                const auto ids=policy.select(LAYER,position,q);
                check(ids==reference(position+1,q),"HPM and early history changed ordered per-head selected IDs");
                auto union_ids=physical(ids);check(std::find(union_ids.begin(),union_ids.end(),16*64)!=union_ids.end(),"fixture must really select cold block16");
                const auto before=policy.stats().layers[LAYER].hpm_migration_bytes;
                verify_fetch(union_ids);
                const auto& st=policy.stats().layers[LAYER];
                check(st.hpm_updates==step+1&&st.hpm_promoted_blocks==step+1,"each real selected union must trigger one nonempty migration");
                check(st.hpm_migration_bytes-before==64*4096,"exact one-block real KV migration bytes");
                check(policy.is_hot(LAYER,(16+step)*64)&&!policy.is_hot(LAYER,step*64),"deterministic cold promotion and reusable hot victim slots");
                check(policy.is_hot(LAYER,P-1),"recent prompt block was evicted");
                for(std::size_t tail=P;tail<=position;++tail)check(policy.is_hot(LAYER,tail),"generated tail must remain hot");
                verify_fetch({position,(16+step)*64,step*64,P-1});
                check(st.hpm_updates==step+1,"extra validation fetch must not cause another cache update");
                check(policy.hot_prompt_tokens()==P/2&&policy.cold_prompt_tokens()==P/2,"HPM changed frozen beta or capacity");
                std::size_t resident=0;for(std::size_t t=0;t<P;++t)resident+=policy.is_hot(LAYER,t);
                check(resident==P/2,"actual slot map changed resident hot token count");
                check(policy.is_hot(0,0)&&!policy.is_hot(0,16*64),"layer8 migration leaked into another layer");
            }
            const auto& st=policy.stats().layers[LAYER];
            check(st.history_prefetch_calls==2&&st.score_calls==3&&st.verification_scores>0&&st.fpga_calls==0,"sync-to-prefetch path and independent same-input scalar verification");
            check(st.hpm_migration_bytes==3*64*4096&&callback_bytes==st.cold_fetch_bytes+st.hpm_migration_bytes,"all callback traffic includes actual cache migrations");
            policy.drain();backing.drain();
            policy.write_stats((directory/"policy-statistics.json").string());
            std::cout<<"{\"status\":\"passed\",\"evidence\":\"synthetic_CPU_regular_file_HPM_correctness_not_performance\",\"budget_layers\":32,\"executed_layer\":8,\"P\":4096,\"steps\":3,\"offline_beta\":1,\"HPM_promoted_blocks\":"<<st.hpm_promoted_blocks<<",\"HPM_migration_bytes\":"<<st.hpm_migration_bytes<<",\"verified_KV_bytes\":"<<verified<<",\"cold_callback_bytes\":"<<callback_bytes<<",\"cold_callback_count\":"<<callback_count<<",\"history_prefetch_calls\":2,\"FPGA_executed\":false,\"GPU_executed\":false}\n";
        }
        std::filesystem::remove_all(directory);return 0;
    }catch(const std::exception& e){std::cerr<<"HPM test failed: "<<e.what()<<"; evidence retained at "<<directory<<'\n';return 1;}
}
