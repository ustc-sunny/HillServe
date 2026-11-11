#include "policy/full_gqa_runtime.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "storage/segmented_store.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <future>
#include <fcntl.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

// Deliberately constructed software inputs, not model/hardware measurements.
using namespace hillserve;
using namespace hillserve::policy;
using namespace hillserve::runtime;
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f,const char* why){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,why);}
std::uint16_t key(int layer,std::uint64_t pos,std::size_t head,std::size_t d){
    if(d)return half_bits(0.f);
    float value=float(pos%16)*.25f+float(layer%8)*.125f+float(head)*.03125f;
    if(pos>=128)value=(layer%2==0?512.f:-512.f)+float(pos-128);
    return half_bits(value);
}
std::vector<std::uint8_t> records(int layer,std::uint64_t first,std::size_t n){
    std::vector<std::uint16_t> words(n*2048);
    for(std::size_t t=0;t<n;++t)for(std::size_t h=0;h<8;++h)for(std::size_t d=0;d<128;++d){
        words[t*2048+h*128+d]=key(layer,first+t,h,d);
        words[t*2048+1024+h*128+d]=half_bits(float(layer+1)*.03125f+float((first+t+d)%29)*.0078125f);
    }
    std::vector<std::uint8_t> bytes(words.size()*2);std::memcpy(bytes.data(),words.data(),bytes.size());return bytes;
}
std::vector<std::uint16_t> keys(const std::vector<std::uint8_t>& kv){
    std::vector<std::uint16_t> out(kv.size()/4);
    for(std::size_t i=0;i<kv.size()/4096;++i)std::memcpy(out.data()+i*1024,kv.data()+i*4096,2048);
    return out;
}
std::vector<float> query(){std::vector<float> q(32*128,0);for(std::size_t h=0;h<32;++h)q[h*128]=1;return q;}
std::vector<std::vector<std::uint64_t>> oracle(int layer,std::size_t valid){
    std::vector<std::uint16_t> scaled(32*valid);
    for(std::size_t h=0;h<32;++h)for(std::size_t t=0;t<valid;++t){
        volatile float sum=0;
        for(std::size_t d=0;d<128;++d){volatile float product=(d==0?1.f:0.f)*half_value(key(layer,t,h/4,d));sum=sum+product;}
        scaled[h*valid+t]=half_bits(half_value(half_bits(sum))/std::sqrt(128.f));
    }
    return select_gqa_scores(scaled,32,valid,valid,5,.2f).ids;
}
std::vector<std::uint64_t> union_ids(const std::vector<std::vector<std::uint64_t>>& ids){
    std::vector<std::uint64_t> v;for(const auto& head:ids)v.insert(v.end(),head.begin(),head.end());
    std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());return v;
}

// A held payload callback represents delayed IO/CUDA completion. Selection
// must finish while it is held, while fetch must wait and surface its errors.
void test_completion(const std::filesystem::path& root,bool callback_error){
    const auto dir=root/(callback_error?"callback-error":"overlap");std::filesystem::create_directory(dir);
    constexpr std::size_t capacity=192,prompt=128;
    const auto extent=full_gqa_align_page(4096+32*capacity*4128ULL);
    SegmentedKVStore store(dir/"kv",765,32,capacity,4096,extent);store.enable_direct_writes(dir/"kv");
    FullGqaRuntimePolicy::Config c;c.capacity=capacity;c.prompt_tokens=prompt;c.generated_capacity=1;
    c.beta=1;c.hot_kv_budget_bytes=32*65*4096ULL;c.total_host_workspace_budget_bytes=128ULL<<20;
    c.chunk_tokens=64;c.hot_threads=1;c.cold_threads=1;c.score_file=(dir/"k").string();
    c.payload_prefetch=true;c.full_kv_backing_path=(dir/"kv").string();c.full_kv_io_fence=std::make_shared<GQAPrefetchIOFence>();
    FullGqaRuntimePolicy policy(c);
    auto append=[&](std::uint64_t first,std::size_t count){auto kv=records(1,first,count);std::vector<std::uint64_t> ids(count);std::iota(ids.begin(),ids.end(),first);
        {std::lock_guard<std::mutex> lock(c.full_kv_io_fence->file_io);store.append(1,first,kv);}policy.append(1,ids,keys(kv),kv);};
    append(0,prompt);
    std::promise<void> entered,release;auto entered_future=entered.get_future();auto gate=release.get_future().share();
    struct Cleanup{std::promise<void>& release;FullGqaRuntimePolicy& policy;bool released=false;
        void open(){if(!released){release.set_value();released=true;}}
        ~Cleanup(){open();policy.wait_pending();}} cleanup{release,policy};
    policy.prepare_history(1,prompt,query(),[&](auto){entered.set_value();gate.get();if(callback_error)throw std::runtime_error("deliberate payload callback failure");});
    append(prompt,1);
    check(entered_future.wait_for(std::chrono::seconds(5))==std::future_status::ready,"payload callback entered");
    auto selecting=std::async(std::launch::async,[&]{return policy.select(1,prompt,query());});
    const auto scores_ready=selecting.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    if(!scores_ready)cleanup.open(); // never deadlock a failing old implementation
    std::vector<std::vector<std::uint64_t>> selected;
    try{selected=selecting.get();}catch(...){cleanup.open();throw;}
    check(scores_ready,"selection waited for blocked payload callback");
    check(selected==oracle(1,prompt+1),"immutable history/current score copy preserves exact ordered IDs");
    const auto physical=union_ids(selected);
    auto wrong=physical;wrong.pop_back();
    rejects([&]{policy.fetch(1,wrong,{});},"wrong final selection cannot consume prepared payload");
    std::promise<void> fetch_entered;auto fetching_started=fetch_entered.get_future();
    std::atomic<unsigned> cold_callbacks{0};
    auto fetching=std::async(std::launch::async,[&]{fetch_entered.set_value();return policy.fetch(1,physical,[&](int,const auto&){++cold_callbacks;return std::vector<std::uint8_t>{};});});
    fetching_started.wait();
    const auto blocked=fetching.wait_for(std::chrono::milliseconds(30))==std::future_status::timeout;
    cleanup.open();
    if(callback_error){
        bool correct_error=false;try{fetching.get();}catch(const std::runtime_error& e){correct_error=std::string(e.what())=="deliberate payload callback failure";}
        check(correct_error,"payload failure must surface at fetch without fallback");
    }else{
        const auto fetched=fetching.get();
        for(std::size_t i=0;i<physical.size();++i){const auto expected=records(1,physical[i],1);check(std::equal(expected.begin(),expected.end(),fetched.begin()+i*4096),"overlapped fetch changed KV bytes");}
        policy.drain();policy.write_stats((dir/"stats.json").string());
    }
    check(blocked,"fetch must not return before payload callback completes");check(!cold_callbacks,"no hidden cold callback on completion/failure");
    store.drain();
}
void test_corrupt_backing(const std::filesystem::path& root,unsigned kind){
    const auto dir=root/("corrupt-"+std::to_string(kind));std::filesystem::create_directory(dir);
    constexpr std::size_t capacity=192,prompt=128;
    const auto extent=full_gqa_align_page(4096+32*capacity*4128ULL);
    SegmentedKVStore store(dir/"kv",766,32,capacity,4096,extent);store.enable_direct_writes(dir/"kv");
    FullGqaRuntimePolicy::Config c;c.capacity=capacity;c.prompt_tokens=prompt;c.generated_capacity=1;
    c.beta=1;c.hot_kv_budget_bytes=32*65*4096ULL;c.total_host_workspace_budget_bytes=128ULL<<20;
    c.chunk_tokens=64;c.hot_threads=1;c.cold_threads=1;c.score_file=(dir/"k").string();
    c.payload_prefetch=true;c.full_kv_backing_path=(dir/"kv").string();c.full_kv_io_fence=std::make_shared<GQAPrefetchIOFence>();
    FullGqaRuntimePolicy policy(c);
    auto append=[&](std::uint64_t first,std::size_t count){auto kv=records(1,first,count);std::vector<std::uint64_t> ids(count);std::iota(ids.begin(),ids.end(),first);
        {std::lock_guard<std::mutex> lock(c.full_kv_io_fence->file_io);store.append(1,first,kv);}policy.append(1,ids,keys(kv),kv);};
    append(0,prompt);
    check(!policy.is_hot(1,15),"corruption fixture must target a selected cold record");
    const auto fd=open((dir/"kv").c_str(),O_RDWR|O_CLOEXEC);check(fd>=0,"open local corruption fixture");
    bool mutated=false;
    if(kind==2)mutated=ftruncate(fd,extent-4096)==0;
    else{const auto offset=4096+(capacity+15)*4128ULL+(kind==0?32:8);unsigned char byte=0;
        mutated=pread(fd,&byte,1,offset)==1;byte^=0x40;mutated=mutated&&pwrite(fd,&byte,1,offset)==1;}
    const auto synced=fdatasync(fd);close(fd);check(mutated&&synced==0,"mutate only owned /tmp fixture");
    bool rejected=false;
    try{policy.prepare_history(1,prompt,query());append(prompt,1);const auto ids=union_ids(policy.select(1,prompt,query()));policy.fetch(1,ids,{});}
    catch(const std::runtime_error& e){const std::string reason=e.what();
        rejected=reason.find(kind==0?"checksum mismatch":kind==1?"record layer/position/request mismatch":"extent")!=std::string::npos;}
    policy.wait_pending();check(rejected,"corruption/identity/truncation must be rejected, without fallback");
}
}
int main(){
    std::filesystem::path dir;
    try{
        // The upper bound depends on the *existing* F16 count representation.
        // Check monotonicity through every supported integer count, including
        // tie-to-even steps above2048 and the finite65504 endpoint.
        float previous=0;
        for(std::size_t n=0;n<=65504;++n){const auto rounded=half_value(half_bits(float(n)));
            check(std::isfinite(rounded)&&rounded>=previous,"F16 count rounding must stay finite and monotone");previous=rounded;}
        check(half_value(half_bits(2049.f))==2048.f&&half_value(half_bits(2051.f))==2052.f&&previous==65504.f,
              "bound must preserve F16 tie-to-even and finite-count boundary");
        dir=std::filesystem::temp_directory_path()/("hillserve-payload-prefetch-"+std::to_string(getpid()));
        check(std::filesystem::create_directory(dir),"exclusive software test directory");
        {
            constexpr std::size_t capacity=192,prompt=128,steps=3;
            const auto extent=full_gqa_align_page(4096+32*capacity*4128ULL);
            SegmentedKVStore store(dir/"complete-kv",991,32,capacity,4096,extent);
            // The tested reader always uses O_DIRECT. The writer also uses
            // page RMW, with the same fence as the production executor.
            store.enable_direct_writes(dir/"complete-kv");
            FullGqaRuntimePolicy::Config c;c.capacity=capacity;c.prompt_tokens=prompt;c.generated_capacity=steps;
            c.beta=1;c.hot_kv_budget_bytes=32*(64+steps)*4096ULL;c.total_host_workspace_budget_bytes=128ULL<<20;
            c.chunk_tokens=64;c.hot_threads=1;c.cold_threads=1;c.score_file=(dir/"raw-k").string();c.verify_scores=true;
            c.payload_prefetch=true;c.full_kv_backing_path=(dir/"complete-kv").string();c.full_kv_io_fence=std::make_shared<GQAPrefetchIOFence>();
            c.payload_read_limits.window_bytes=16384; // force multiple QD4 batches with the small fixture
            auto invalid_depth=c;invalid_depth.payload_read_queue_depth=5;
            rejects([&]{FullGqaRuntimePolicy invalid(invalid_depth);},"read depth cannot bypass fixed four-window reservation");
            auto bad=c;bad.score_file=(dir/"no-fence").string();bad.full_kv_io_fence.reset();
            rejects([&]{FullGqaRuntimePolicy invalid(bad);},"prefetch cannot run without writer fence");
            FullGqaRuntimePolicy policy(c);
            check(policy.stats().payload_prefetch_reserved_bytes>=2*capacity*4096ULL,"two complete bounded payloads reserved");
            bad=c;bad.score_file=(dir/"reject-budget").string();bad.total_host_workspace_budget_bytes=policy.managed_bytes()-1;
            rejects([&]{FullGqaRuntimePolicy invalid(bad);},"extra payload memory cannot bypass combined host budget");
            check(!std::filesystem::exists(bad.score_file),"budget rejection precedes score file creation");
            auto append=[&](int l,std::uint64_t first,std::size_t n){
                auto kv=records(l,first,n);std::vector<std::uint64_t> positions(n);std::iota(positions.begin(),positions.end(),first);
                {std::lock_guard<std::mutex> guard(c.full_kv_io_fence->file_io);store.append(l,first,kv);}
                policy.append(l,positions,keys(kv),kv);
            };
            for(int l=0;l<32;++l)append(l,0,prompt);
            // Production dense prefill and layers0/1 fetch without any history
            // selection. Enabling later-layer prefetch must not reject them.
            const std::vector<std::uint64_t> dense_ids{127,0,64,63};
            const auto dense=policy.fetch(0,dense_ids,{});
            check(policy.stats().layers[0].dense_fetch_read_calls>0,"dense layer uses bounded direct reader without old callbacks");
            for(std::size_t i=0;i<dense_ids.size();++i){const auto expected=records(0,dense_ids[i],1);
                check(std::equal(expected.begin(),expected.end(),dense.begin()+i*4096),"unselected dense-layer hot/cold fetch preserved");}
            std::array<std::shared_ptr<const GQAPrefetchArtifact>,32> artifacts{};
            std::atomic<std::size_t> callbacks{0};
            std::atomic<std::size_t> reduced_cover_cases{0};
            struct WaitBeforeCallbackStateDestruction{FullGqaRuntimePolicy& p;~WaitBeforeCallbackStateDestruction(){p.wait_pending();}} wait_guard{policy};
            std::uint64_t checked_bytes=0,cold_callbacks=0;
            auto callback=[&](int layer,std::uint64_t position){return [&,layer,position](std::shared_ptr<const GQAPrefetchArtifact> p){
                check(p&&p->request_id==991&&p->layer==layer&&p->position==position,"callback request/layer/position binding");
                check(p->per_head_ids.size()==32&&p->kv_bytes.size()==p->physical_ids.size()*4096,"immutable artifact layout");
                check(p->score_ready_ns<=p->read_started_ns&&p->read_started_ns<=p->read_finished_ns,"actual score/read event order");
                check(p->hot_bytes+p->cold_bytes==p->kv_bytes.size(),"hot and cold source byte conservation");
                check(p->required_page_bytes+p->gap_read_bytes==p->cold_syscall_bytes,"planned read bytes include every gap page");
                check(p->cold_syscall_bytes*1000<=p->required_page_bytes*1500,"actual read amplification bounded");
                check(p->extent_checks==(p->cold_read_calls?2:0),"extent validated at both batch boundaries");
                check(p->plan_ns+p->hot_copy_ns+p->io_lock_wait_ns+p->extent_check_ns+p->pread_ns+p->validate_copy_ns+p->checksum_ns<=p->read_finished_ns-p->read_started_ns,"measured read subintervals do not overlap");
                check(std::is_sorted(p->physical_ids.begin(),p->physical_ids.end()),"physical cover sorted");
                const auto cap=std::size_t(std::floor(double(position+1)*double(.2f)));
                const auto expected_bound=layer%2==0&&position>128?std::min(cap,std::size_t(position-128+1)):cap;
                for(const auto& ids:p->per_head_ids){
                    check(ids.size()==expected_bound,"F16 threshold-count upper bound includes unknown current contribution");
                    for(auto id:ids)check(id<position,"prefetch never contains current or future KV");
                }
                if(expected_bound<cap){check(expected_bound*4<cap,"constructed sparse threshold significantly reduces old full-cap cover");++reduced_cover_cases;}
                for(std::size_t i=0;i<p->physical_ids.size();++i){auto expected=records(layer,p->physical_ids[i],1);
                    check(std::equal(expected.begin(),expected.end(),p->kv_bytes.begin()+i*4096),"prefetched actual file/hot KV bytes preserve layer and position");}
                artifacts[layer]=std::move(p);++callbacks;
            };};
            auto cold=[&](int l,const std::vector<std::uint64_t>& ids){
                ++cold_callbacks;std::lock_guard<std::mutex> guard(c.full_kv_io_fence->file_io);
                return store.fetch(l,ids,64*4096);
            };
            auto q=query();
            for(std::size_t step=0;step<steps;++step){
                const auto pos=prompt+step;
                policy.prepare_history(0,pos,q,callback(0,pos));
                rejects([&]{policy.prepare_history(0,pos,q);},"same layer/step duplicate preparation rejected");
                policy.prepare_history(1,pos,q,callback(1,pos));
                rejects([&]{policy.prepare_history(2,pos,q);},"third in-flight payload rejected");
                if(!step){policy.wait_pending();check(callbacks==2,"real historical payload can finish before current token exists");}
                for(int l=0;l<32;++l){
                    if(l>=2)policy.prepare_history(l,pos,q,callback(l,pos));
                    append(l,pos,1);
                    rejects([&]{auto different=q;different[0]=2;policy.select(l,pos,different);},"stale/different query cannot consume prepared payload");
                    auto selected=policy.select(l,pos,q);
                    check(selected==oracle(l,pos+1),"final scores/selector unchanged after current append");
                    if(!step&&l%2==0)check(selected[0].size()==1&&selected[0][0]==pos,"current token raises head0 threshold and changes final k");
                    if(l%2)for(const auto& head:selected)check(std::find(head.begin(),head.end(),pos)==head.end(),"current token is not forced into final IDs");
                    const auto physical=union_ids(selected);auto fetched=policy.fetch(l,physical,cold);
                    const auto& a=artifacts[l];check(bool(a),"callback completed before fetch returns");
                    for(std::size_t h=0;h<32;++h)for(auto id:selected[h])if(id<pos)
                        check(std::find(a->per_head_ids[h].begin(),a->per_head_ids[h].end(),id)!=a->per_head_ids[h].end(),"threshold-count bounded history covers final ordered selection");
                    for(std::size_t i=0;i<physical.size();++i){auto expected=records(l,physical[i],1);
                        check(std::equal(expected.begin(),expected.end(),fetched.begin()+i*4096),"prefetch and current completion preserve final physical KV bytes");}
                    checked_bytes+=fetched.size();
                    rejects([&]{policy.select(l,pos,q);},"same step cannot select twice");
                    artifacts[l].reset(); // shared immutable ownership bounded
                }
            }
            std::uint64_t useful=0,wasted=0,hot=0,cold_bytes=0;
            for(const auto& st:policy.stats().layers){
                check(st.payload_prefetch_calls==steps&&st.payload_prefetch_miss_bytes==0,"all layer/step final history covered without missed cold IO");
                check(st.payload_prefetch_useful_bytes+st.payload_prefetch_wasted_bytes==st.payload_prefetch_bytes,"prefetch useful/wasted conservation");
                check(st.payload_prefetch_syscall_bytes>=st.payload_prefetch_cold_bytes&&st.payload_prefetch_read_calls>0,"actual direct cold IO accounted");
                useful+=st.payload_prefetch_useful_bytes;wasted+=st.payload_prefetch_wasted_bytes;hot+=st.payload_prefetch_hot_bytes;cold_bytes+=st.payload_prefetch_cold_bytes;
            }
            check(callbacks==32*steps&&hot&&cold_bytes&&useful&&wasted,"test exercises all layers, both storage tiers and useful/wasted prefetch");
            check(reduced_cover_cases==32,"sparse-threshold tests exercise32 genuinely smaller covers");
            check(policy.stats().payload_prefetch_setup_read_bytes==4096&&policy.stats().payload_prefetch_setup_read_calls==1,
                  "one existing-file header read accounted only as setup, not per-layer payload IO");
            check(cold_callbacks==0,"final historical fetch must reuse fully covered real-file prefetch");
            policy.drain();store.drain();policy.write_stats((dir/"stats.json").string());
            std::cout<<"{\"status\":\"passed\",\"evidence\":\"CPU_software_O_DIRECT_regular_file_not_model_or_FPGA\",\"layers\":32,\"steps\":3,\"callbacks\":"<<callbacks
                     <<",\"verified_final_KV_bytes\":"<<checked_bytes<<",\"prefetch_hot_bytes\":"<<hot<<",\"prefetch_cold_bytes\":"<<cold_bytes
                     <<",\"useful_bytes\":"<<useful<<",\"wasted_bytes\":"<<wasted<<",\"threshold_count_reduced_cover_cases\":"<<reduced_cover_cases
                     <<",\"setup_read_bytes\":4096,\"historical_miss_bytes\":0,\"two_pending_jobs_verified\":true,\"current_token_changes_final_k\":true,\"GPU_executed\":false,\"FPGA_executed\":false}\n";
        }
        test_completion(dir,false);test_completion(dir,true);
        for(unsigned kind=0;kind<3;++kind)test_corrupt_backing(dir,kind);
        std::cout<<"{\"completion_overlap\":\"passed\",\"callback_error_propagation\":\"passed\",\"corrupt_checksum_identity_extent\":\"passed\",\"hardware_measurement\":false}\n";
        std::filesystem::remove_all(dir);return 0;
    }catch(const std::exception& e){if(!dir.empty())std::filesystem::remove_all(dir);std::cerr<<"payload prefetch test failed: "<<e.what()<<'\n';return 1;}
}
