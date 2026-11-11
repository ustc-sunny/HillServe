#include "policy/full_gqa_runtime.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "storage/segmented_store.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

// Authored software fixtures only: no model quality or hardware-speed claims.
using namespace hillserve;
using namespace hillserve::policy;
using namespace hillserve::runtime;
namespace {
constexpr std::size_t P=128,G=3,CAP=192,H=32,D=128;
constexpr std::array<float,4> ALPHAS{5.f,10.f,20.f,50.f};
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f,const char* why){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,why);}

void selector_contracts(){
    check(FullGqaRuntimePolicy::Config{}.selector_alpha==5.f,"historical alpha default changed");
    // Same input scores: alpha changes count only; ranking and cap do not move.
    std::vector<std::uint16_t> scores(H*P);
    for(std::size_t h=0;h<H;++h)for(std::size_t t=0;t<P;++t)
        scores[h*P+t]=half_bits(100.f-2.f*float(t));
    const std::array<std::size_t,4> expected{3,6,11,25};
    const std::array<std::uint32_t,4> counts{3,6,11,26};
    for(std::size_t a=0;a<ALPHAS.size();++a){
        auto s=select_gqa_scores(scores,H,P,P,ALPHAS[a],.2f);
        check(s.k==expected[a],"alpha/count/20percent-cap changed on same input");
        for(std::size_t h=0;h<H;++h){
            check(s.counts[h]==counts[a],"inclusive threshold count changed");
            for(std::size_t t=0;t<s.k;++t)check(s.ids[h][t]==t,"alpha changed score ranking");
        }
    }
    // Preserve head0 broadcast, F16 threshold quantization and original-ID ties.
    for(float alpha:ALPHAS){
        std::fill(scores.begin(),scores.end(),half_bits(-1000.f));
        const auto threshold_bits=half_bits(100.f-alpha);
        const float threshold=half_value(threshold_bits);
        for(std::size_t h=0;h<H;++h){
            scores[h*P]=half_bits(100.f);
            scores[h*P+1]=threshold_bits;scores[h*P+2]=threshold_bits;
            scores[h*P+3]=std::uint16_t(threshold_bits-1); // one F16 value below
        }
        auto tied=select_gqa_scores(scores,H,P,P,alpha,.2f);
        check(tied.k==3&&tied.counts[0]==3&&tied.ids[0]==std::vector<std::uint64_t>({0,1,2}),
              "threshold equality or original-ID tie order changed");
        scores[0]=half_bits(120.f); // only head0 moves the shared threshold
        auto broadcast=select_gqa_scores(scores,H,P,P,alpha,.2f);
        const float new_threshold=half_value(half_bits(120.f-alpha));
        check(broadcast.counts[1]==std::uint32_t(100.f>=new_threshold),
              "head0 maximum was silently replaced by per-head maxima");
        check(threshold<new_threshold,"test must exercise raised head0 threshold");
    }
    // Count2049 is representable as an integer but rounds to2048 in the original
    // F16 count/mean rule. The fixed20% cap here is larger, exposing the rounding.
    constexpr std::size_t N=16384;
    std::vector<std::uint16_t> rounded_scores(H*N,half_bits(-1000.f));
    for(std::size_t h=0;h<H;++h)std::fill_n(rounded_scores.begin()+h*N,2049,half_bits(0.f));
    for(float alpha:ALPHAS){
        auto s=select_gqa_scores(rounded_scores,H,N,N,alpha,.2f);
        check(s.k==2048&&s.counts[0]==2049&&s.ids[0].back()==2047,
              "alpha variant must retain F16 count rounding and original-ID ties");
    }
}

std::uint16_t key(int layer,std::uint64_t position,std::size_t kh,std::size_t d){
    if(d)return half_bits(0.f);
    float value=position<P?float(P-position)*16.f+float(kh)*.5f:-4096.f;
    if(layer==2&&position>=P&&position!=P+1)value=4096.f;
    return half_bits(value);
}
std::vector<float> query(){
    std::vector<float> q(H*D,0.f);
    for(std::size_t h=0;h<H;++h)q[h*D]=1.f+float(h%4)/256.f;
    return q;
}
std::vector<std::uint8_t> records(int layer,std::uint64_t first,std::size_t count){
    std::vector<std::uint16_t> words(count*2048);
    for(std::size_t t=0;t<count;++t)for(std::size_t kh=0;kh<8;++kh)for(std::size_t d=0;d<D;++d){
        words[t*2048+kh*D+d]=key(layer,first+t,kh,d);
        words[t*2048+1024+kh*D+d]=half_bits(float(layer)*.03125f+float((first+t+kh+d)%31)*.015625f);
    }
    std::vector<std::uint8_t> result(words.size()*2);std::memcpy(result.data(),words.data(),result.size());return result;
}
std::vector<std::uint16_t> keys(const std::vector<std::uint8_t>& kv){
    std::vector<std::uint16_t> result(kv.size()/4);
    for(std::size_t t=0;t<kv.size()/4096;++t)std::memcpy(result.data()+t*1024,kv.data()+t*4096,2048);
    return result;
}
GQASelection oracle(int layer,std::size_t valid,const std::vector<float>& q,float alpha){
    std::vector<std::uint16_t> scores(H*valid);
    for(std::size_t h=0;h<H;++h)for(std::size_t t=0;t<valid;++t){
        volatile float sum=0;
        for(std::size_t d=0;d<D;++d){
            volatile float product=half_value(half_bits(q[h*D+d]))*half_value(key(layer,t,h/4,d));
            sum=sum+product;
        }
        scores[h*valid+t]=half_bits(half_value(half_bits(sum))/std::sqrt(128.f));
    }
    return select_gqa_scores(scores,H,valid,valid,alpha,.2f);
}
std::vector<std::uint64_t> physical(const std::vector<std::vector<std::uint64_t>>& ids){
    std::vector<std::uint64_t> result;
    for(const auto& head:ids)result.insert(result.end(),head.begin(),head.end());
    std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());return result;
}
std::string read(const std::filesystem::path& p){
    std::ifstream f(p);check(bool(f),"test statistics missing");return {std::istreambuf_iterator<char>(f),{}};
}
float number(const std::string& text,const char* field){
    const std::regex re(std::string("\"")+field+"\"\\s*:\\s*([-+0-9.eE]+)");std::smatch m;
    check(std::regex_search(text,m,re),"test statistics numeric field missing");return std::stof(m[1].str());
}
struct Evidence {std::uint64_t selections=0,cover_checks=0,verified_bytes=0;std::array<std::size_t,4> unchanged_max_k{};};
void runtime_contracts(const std::filesystem::path& root,Evidence& evidence){
    const auto q=query();
    for(std::size_t a=0;a<ALPHAS.size();++a){
        const float alpha=ALPHAS[a];const auto dir=root/("alpha-"+std::to_string(int(alpha)));
        check(std::filesystem::create_directory(dir),"exclusive alpha fixture directory");
        const auto extent=full_gqa_align_page(4096+32*CAP*4128ULL);
        SegmentedKVStore store(dir/"complete-kv",991,32,CAP,4096,extent);
        store.enable_direct_reads(dir/"complete-kv");store.enable_direct_writes(dir/"complete-kv");
        FullGqaRuntimePolicy::Config c;c.capacity=CAP;c.prompt_tokens=P;c.generated_capacity=G;
        c.selector_alpha=alpha;c.beta=1;c.chunk_tokens=64;c.hot_threads=1;c.cold_threads=1;
        c.hot_kv_budget_bytes=32*(64+G)*4096ULL;c.total_host_workspace_budget_bytes=128ULL<<20;
        c.score_file=(dir/"raw-k").string();c.verify_scores=true;c.payload_prefetch=true;
        c.full_kv_backing_path=(dir/"complete-kv").string();c.full_kv_io_fence=std::make_shared<GQAPrefetchIOFence>();
        for(float invalid:{-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
            auto bad=c;bad.selector_alpha=invalid;bad.score_file=(dir/"invalid-alpha-must-not-exist").string();
            rejects([&]{FullGqaRuntimePolicy rejected(bad);},"invalid alpha accepted");
            check(!std::filesystem::exists(bad.score_file),"invalid alpha created backing file before rejection");
        }
        // Constructed before policy so asynchronous callbacks cannot outlive it
        // during an exception: policy destructor waits before artifact teardown.
        std::shared_ptr<const GQAPrefetchArtifact> artifact;
        FullGqaRuntimePolicy policy(c);
        auto append=[&](int layer,std::uint64_t first,std::size_t count){
            auto kv=records(layer,first,count);std::vector<std::uint64_t> positions(count);std::iota(positions.begin(),positions.end(),first);
            {std::lock_guard<std::mutex> guard(c.full_kv_io_fence->file_io);store.append(layer,first,kv);}
            policy.append(layer,positions,keys(kv),kv);
        };
        for(int layer:{2,3})append(layer,0,P);
        std::uint64_t cold_callbacks=0;
        auto cold=[&](int layer,const std::vector<std::uint64_t>& ids){
            ++cold_callbacks;std::lock_guard<std::mutex> guard(c.full_kv_io_fence->file_io);return store.fetch(layer,ids,64*4096);
        };
        for(std::size_t step=0;step<G;++step)for(int layer:{2,3}){
            const auto position=P+step;artifact.reset();
            policy.prepare_history(layer,position,q,[&](std::shared_ptr<const GQAPrefetchArtifact> p){artifact=std::move(p);});
            policy.wait_pending();check(bool(artifact),"history payload did not finish before current append");
            check(artifact->position==position&&artifact->layer==layer&&artifact->per_head_ids.size()==H,
                  "history artifact has wrong identity");
            for(const auto& head:artifact->per_head_ids)for(auto id:head)check(id<position,"prefetch read current/future KV");
            append(layer,position,1);
            const auto selected=policy.select(layer,position,q);const auto expected=oracle(layer,position+1,q,alpha);
            check(selected==expected.ids,"live alpha/current selector differs from same-input scalar oracle");
            ++evidence.selections;
            for(std::size_t h=0;h<H;++h)for(auto id:selected[h])if(id<position){
                check(std::find(artifact->per_head_ids[h].begin(),artifact->per_head_ids[h].end(),id)!=artifact->per_head_ids[h].end(),
                      "alpha-aware historical cover misses a finally selected ID");++evidence.cover_checks;
            }
            if(layer==2&&step==0)check(selected[0]==std::vector<std::uint64_t>({position}),"raised current head0 threshold not applied");
            if(layer==3){
                for(const auto& head:selected)check(std::find(head.begin(),head.end(),position)==head.end(),"low current token was forced into selection");
                if(step==0)evidence.unchanged_max_k[a]=expected.k;
            }
            const auto ids=physical(selected);const auto fetched=policy.fetch(layer,ids,cold);
            for(std::size_t i=0;i<ids.size();++i){const auto kv=records(layer,ids[i],1);
                check(std::memcmp(fetched.data()+i*4096,kv.data(),4096)==0,"alpha prefetch/fetch corrupted real KV bytes");}
            evidence.verified_bytes+=fetched.size();artifact.reset();
        }
        check(cold_callbacks==0,"final fetch reread history missing from alpha-aware cover");
        for(int layer:{2,3}){const auto& st=policy.stats().layers[layer];
            check(st.payload_prefetch_calls==G&&st.current_token_score_calls==G&&st.payload_prefetch_miss_bytes==0,
                  "alpha did not preserve current completion and zero historical misses");
            check(st.verification_scores==H*(P+1+P+2+P+3),"same-input raw score oracle coverage incomplete");
            check(st.payload_prefetch_syscall_bytes>0&&st.payload_prefetch_read_calls>0,"fixture did not exercise direct cold-file prefetch");
        }
        policy.drain();store.drain();const auto stats=dir/"policy-stats.json";policy.write_stats(stats.string());
        const auto text=read(stats);
        check(number(text,"selector_alpha")==alpha&&number(text,"selector_cap_ratio")==.2f,"alpha/cap provenance differs from execution");
        const std::string label=alpha==5.f?"head0_max_minus5_inclusive":"head0_max_minus_alpha_inclusive";
        check(text.find(label)!=std::string::npos,"alpha variant mislabeled as historical policy");
        check(text.find(alpha==5.f?"v6_alpha5_cap20":"v6_explicit_alpha_variant_cap20")!=std::string::npos,
              "policy variant identity missing");
        const auto events=read(stats.string()+".events.jsonl");
        check(number(events,"selector_alpha")==alpha,"per-event alpha provenance missing");
    }
    check(std::adjacent_find(evidence.unchanged_max_k.begin(),evidence.unchanged_max_k.end(),
        [](auto a,auto b){return a>=b;})==evidence.unchanged_max_k.end(),
        "fixture must detect all four distinct alpha retention levels on identical input");
}
}
int main(int argc,char** argv){
    std::filesystem::path dir;
    try{
        const bool selector_only=argc==2&&std::string(argv[1])=="--selector-only";
        check(argc==1||selector_only,"usage: quality_alpha_tests [--selector-only]");
        selector_contracts();Evidence e;
        if(!selector_only){
            dir=std::filesystem::temp_directory_path()/("hillserve-quality-alpha-"+std::to_string(getpid()));
            check(std::filesystem::create_directory(dir),"exclusive alpha test root");runtime_contracts(dir,e);
            std::filesystem::remove_all(dir);
        }
        std::cout<<"{\"status\":\"passed\",\"evidence\":\"CPU_software_fixture_not_model_or_FPGA\",\"alphas\":[5,10,20,50],\"selector_only\":"<<(selector_only?"true":"false")
                 <<",\"live_current_selections\":"<<e.selections<<",\"history_cover_checks\":"<<e.cover_checks<<",\"verified_final_KV_bytes\":"<<e.verified_bytes<<",\"unchanged_max_k\":[";
        for(std::size_t i=0;i<e.unchanged_max_k.size();++i){if(i)std::cout<<',';std::cout<<e.unchanged_max_k[i];}
        std::cout<<"],\"GPU_executed\":false,\"FPGA_executed\":false}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"alpha test failed: "<<error.what()<<"; retained="<<dir<<'\n';return 1;}
}
