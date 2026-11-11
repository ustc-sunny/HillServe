#include "policy/full_gqa_runtime.hpp"
#include "policy/full_gqa_cpu.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "policy/half_order.hpp"
#include "runtime/full_gqa_contract.hpp"
#include "core/hkm.hpp"
#include "storage/file_store.hpp"
#include "storage/direct_read_batch.hpp"
#include "storage/kv_checksum.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hillserve::policy {
namespace {
using Clock=std::chrono::steady_clock;
using json=nlohmann::ordered_json;
constexpr std::size_t H=32, KH=8, D=128, KEY_WORDS=1024;
constexpr std::size_t KEY_BYTES=2048, KV_BYTES=4096, UNIT=64;
constexpr const char* V3="HS_FULL_GQA_PIPELINE_F16_V3";
constexpr const char* SHA="679e85f964bf644cb1fe3bd5752b5a2452d288106e8139b46d702aafbdb2dcd6";
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::uint64_t stamp(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
std::size_t add(std::size_t a,std::size_t b){return runtime::full_gqa_checked_add(a,b);}
std::size_t mul(std::size_t a,std::size_t b){return runtime::full_gqa_checked_mul(a,b);}
std::uint64_t hash_bytes(const void* data,std::size_t n){auto* p=static_cast<const unsigned char*>(data);std::uint64_t h=1469598103934665603ULL;for(std::size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}
struct FD {
    int value=-1;
    ~FD(){if(value>=0)close(value);}
};
struct Aligned {
    void* data=nullptr;std::size_t bytes=0;
    ~Aligned(){free(data);}
    void allocate(std::size_t n){need(!data&&n&&n%4096==0,"full GQA aligned allocation extent");need(posix_memalign(&data,4096,n)==0,"full GQA aligned allocation failed");bytes=n;}
};
void io_exact(int fd,void* data,std::size_t bytes,std::uint64_t offset,bool write){
    need(bytes&&bytes%4096==0&&offset%4096==0&&reinterpret_cast<std::uintptr_t>(data)%4096==0,"full GQA direct IO alignment");
    ssize_t n;do{n=write?pwrite(fd,data,bytes,static_cast<off_t>(offset)):pread(fd,data,bytes,static_cast<off_t>(offset));}while(n<0&&errno==EINTR);
    need(n==static_cast<ssize_t>(bytes),std::string("full GQA ")+(write?"pwrite":"pread")+" short/error: "+std::strerror(errno));
}
std::vector<std::uint16_t> query_f16(const std::vector<float>& input){
    need(input.size()==H*D,"full GQA actual rehearsal must be32x128");std::vector<std::uint16_t> q(input.size());
    for(std::size_t i=0;i<input.size();++i){need(std::isfinite(input[i]),"full GQA rehearsal nonfinite");q[i]=half_bits(input[i]);need(std::isfinite(half_value(q[i])),"full GQA rehearsal F16 overflow");}return q;
}
json chunk_json(const FileScoreChunkInfo& c){return {{"token_offset",c.token_offset},{"tokens",c.tokens},{"read_started_ns",c.read_started_ns},{"read_finished_ns",c.read_finished_ns},{"kernel_started_ns",c.kernel_started_ns},{"kernel_finished_ns",c.kernel_finished_ns},{"scores_CPU_ready_ns",c.scores_CPU_ready_ns},{"callback_started_ns",c.callback_started_ns},{"callback_finished_ns",c.callback_finished_ns},{"file_read_bytes",c.file_read_bytes},{"score_d2h_bytes",c.score_d2h_bytes}};}
}

struct FullGqaRuntimePolicy::Impl {
    struct Work {
        std::vector<std::uint16_t> raw;
        std::uint64_t started=0,hot_start=0,hot_end=0,cold_start=0,cold_end=0,ended=0;
        std::uint64_t cold_bytes=0,hot_packed_key_bytes=0,hot_cpu_calls=0,cold_cpu_calls=0;
        FileScoreStreamReport device;
    };
    struct Pending {
        std::uint64_t position=0,submitted=0;
        std::vector<std::uint16_t> query;
        std::shared_future<std::shared_ptr<const Work>> scores;
        std::future<std::shared_ptr<const runtime::GQAPrefetchArtifact>> payload_future;
    };
    struct Event {
        int layer=0;std::uint64_t position=0,started=0,joined=0,scores_ready=0,ids_ready=0,ended=0;
        bool prefetched=false;
        std::uint64_t history_submitted=0,score_hash=0,id_hash=0;
        std::uint64_t current_score_started=0,current_score_finished=0;
        std::uint64_t hot_start=0,hot_end=0,cold_start=0,cold_end=0,work_started=0,work_ended=0;
        std::uint64_t cold_bytes=0,qbytes=0,sbytes=0;
        std::uint64_t hot_packed_key_bytes=0,hot_cpu_calls=0,cold_cpu_calls=0;
        std::size_t k=0,physical_count=0,fpga_calls=0;
        std::uint64_t physical_hash=0,fetch_started=0,fetch_finished=0;
        std::uint64_t hot_fetch_bytes=0,cold_fetch_bytes=0,hpm_migration_bytes=0;
        double hpm_ms=0;
        std::vector<std::uint64_t> promote,demote;
        std::uint64_t payload_started=0,payload_ready=0,payload_bytes=0;
        std::uint64_t payload_join_started=0,payload_join_finished=0;
        std::uint64_t plan_ns=0,hot_copy_ns=0,io_lock_wait_ns=0,extent_check_ns=0;
        std::uint64_t pread_ns=0,validate_copy_ns=0,checksum_ns=0;
        std::uint64_t required_page_bytes=0,gap_read_bytes=0,extent_checks=0;
        std::uint64_t payload_hot=0,payload_cold=0,payload_syscall=0,payload_reads=0;
        std::uint64_t payload_useful=0,payload_wasted=0,payload_miss=0,payload_current=0;
        std::array<std::uint32_t,H> counts{};
        double verification_ms=0;
        std::vector<FileScoreChunkInfo> chunks;
    };
    Config config;Stats statistics;runtime::FullGqaRuntimePlan plan;
    FD file;Aligned append_buffer,read_buffer;
    FD full_kv_file;
    std::unique_ptr<storage::DirectReadBatch> payload_reader;
    std::uint64_t full_kv_request=0;
    std::vector<std::shared_ptr<const runtime::GQAPrefetchArtifact>> ready_payloads;
    std::vector<std::vector<std::uint16_t>> hot_kv;
    std::vector<std::vector<std::uint64_t>> cold_positions;
    std::vector<std::vector<std::int64_t>> slot_maps;
    std::vector<std::unique_ptr<ApproxHkm>> placements;
    std::vector<std::uint64_t> last_selected,last_hpm;
    std::vector<Event> events;
    std::map<int,Pending> pending;
    std::unique_ptr<FullGqaCpuScorer> hot_cpu,cold_cpu,scalar;
    std::vector<std::uint16_t> hot_scores,cold_scores;
    std::unique_ptr<FpgaScoreBackend> fpga;
    // All public entry points run on the executor thread. engine serializes
    // up to two history tasks across one XRT. Current-token completion uses
    // an independent scalar path so a queued next-layer scan cannot block it.
    std::mutex engine;
    std::size_t effective_chunk=0,event_limit=0;
    bool drained=false;

    explicit Impl(const Config& c):config(c){
        need(c.mode=="HOST_MATCHED"||c.mode=="FPGA_SERIAL"||c.mode=="FPGA_CSS","explicit full GQA runtime mode required");
        need(c.cpu_implementation=="auto"||c.cpu_implementation=="legacy","cpu_implementation must be auto or legacy");
        need(std::isfinite(c.selector_alpha)&&c.selector_alpha>=0,
             "full GQA selector_alpha must be finite and nonnegative");
        statistics.cpu_implementation_requested=c.cpu_implementation;
        need(c.hpm_swaps<=1,"live HPM permits at most one64-token migration per layer/step");
        need(c.hot_threads>=1&&c.hot_threads<=32&&c.cold_threads>=1&&c.cold_threads<=32,"full GQA thread budget1..32");
        need(c.fp32_selection_counts||c.prompt_tokens+c.generated_capacity<=65504,"full GQA F16 selector count range exceeded; no long-count substitution");
        need(c.chunk_tokens>=UNIT&&c.chunk_tokens<=32768&&c.chunk_tokens%UNIT==0,"V3 chunk must align64 within32768");
        need(!c.score_file.empty(),"full GQA explicit new rawK file required");
        need(!c.payload_prefetch||(!c.full_kv_backing_path.empty()&&bool(c.full_kv_io_fence)),
             "payload prefetch requires existing complete KV path and shared writer IO fence");
        storage::validate_kv_read_limits(c.payload_read_limits);
        need(c.payload_read_queue_depth>=1&&c.payload_read_queue_depth<=4,"complete KV read depth must be 1..4");
        const auto hot_blocks=runtime::full_gqa_hot_blocks(c.prompt_tokens,c.beta,c.allow_partial_prompt);
        const auto cold=c.prompt_tokens-hot_blocks*UNIT-c.prompt_tokens%UNIT;
        need(!c.hpm || c.prompt_tokens%UNIT==0,"HPM partial prompt blocks not supported; keep tail pinned and HPM disabled");
        effective_chunk=c.mode=="FPGA_SERIAL"&&cold?cold:c.chunk_tokens;
        need(effective_chunk<=32768,"FPGA_SERIAL cold set exceeds one V3 call; use explicit CSS mode");
        event_limit=mul(c.layers,c.generated_capacity);
        // Conservative simultaneous application buffers, excluding model/GPU
        // allocator and thread stacks (reported separately by process RSS).
        // Three score vectors allow two prepared histories plus normalization;
        // another three cover hot/cold outputs and independent verification.
        std::size_t workspace=mul(mul(H,c.capacity),2*6+sizeof(std::uint64_t));
        // select copies immutable history before writing the current score;
        // the producer may still read that history while building its cover.
        workspace=add(workspace,mul(mul(H,c.capacity),sizeof(std::uint16_t)));
        workspace=add(workspace,mul(c.capacity,KV_BYTES)); // returned selected union
        workspace=add(workspace,UNIT*KV_BYTES); // bounded cold-fetch callback
        workspace=add(workspace,mul(std::max(c.chunk_tokens,effective_chunk),KEY_BYTES));
        workspace=add(workspace,UNIT*KEY_BYTES); // direct append RMW
        workspace=add(workspace,mul(c.capacity,128)); // indexes, views, jobs, vector overhead
        const auto chunks=cold?(cold+effective_chunk-1)/effective_chunk:0;
        workspace=add(workspace,mul(event_limit,add(sizeof(Event),mul(chunks,sizeof(FileScoreChunkInfo)))));
        workspace=add(workspace,mul(mul(c.layers,c.capacity),16)); // layer slot maps and cold IDs
        workspace=add(workspace,mul(mul(c.layers,c.prompt_tokens/UNIT),64)); // bounded HPM metadata/scratch
        workspace=add(workspace,mul(event_limit,2*sizeof(std::uint64_t))); // at most one swap/event
        workspace=add(workspace,2*1024*1024); // fixed scorer/query/JSON/IO bookkeeping
        if(c.cpu_implementation=="legacy"){
            // Match the original replay's hot_keys preparation. Include the
            // packed full K and both returned score vectors while assignment
            // still owns the previous hot/cold results; no hidden allocation.
            statistics.legacy_cpu_workspace_bytes=add(mul(c.capacity,KEY_BYTES),mul(mul(H,c.capacity),2*2));
            workspace=add(workspace,statistics.legacy_cpu_workspace_bytes);
        }
        if(c.payload_prefetch){
            // Two immutable payloads, per-head cap ID vectors, physical IDs,
            // copied placement maps, sorting scratch and direct read windows.
            // Callback references share these bytes, never duplicate them.
            const auto cap=std::size_t(std::floor(double(c.prompt_tokens+c.generated_capacity)*double(.2f)));
            auto per=add(mul(c.capacity,KV_BYTES+4*sizeof(std::uint64_t)),mul(H*cap,sizeof(std::uint64_t)));
            per=add(per,mul(H*c.capacity,sizeof(std::uint16_t))); // normalization scratch
            per=add(per,mul(c.capacity,sizeof(storage::KVReadRecord)+sizeof(storage::KVReadWindow)));
            per=add(per,mul(c.payload_read_limits.window_bytes,c.payload_read_queue_depth)+H*sizeof(std::vector<std::uint64_t>)+sizeof(runtime::GQAPrefetchArtifact));
            statistics.payload_prefetch_reserved_bytes=mul(2,per);
            // A dense layer can run beside prepared history. Its returned KV
            // was reserved above; count the additional IO and planner buffers.
            statistics.payload_prefetch_reserved_bytes=add(statistics.payload_prefetch_reserved_bytes,
                add(mul(c.payload_read_limits.window_bytes,c.payload_read_queue_depth),
                    mul(c.capacity,sizeof(storage::KVReadRecord)+sizeof(storage::KVReadWindow))));
            workspace=add(workspace,statistics.payload_prefetch_reserved_bytes);
        }
        runtime::FullGqaRuntimeSpec spec;
        spec.layers=c.layers;spec.capacity=c.capacity;spec.prompt_tokens=c.prompt_tokens;
        spec.generated_capacity=c.generated_capacity;spec.beta=c.beta;spec.allow_partial_prompt=c.allow_partial_prompt;
        spec.fp32_selection_counts=c.fp32_selection_counts;
        spec.hot_kv_budget_bytes=c.hot_kv_budget_bytes;
        spec.total_host_workspace_budget_bytes=c.total_host_workspace_budget_bytes;
        spec.additional_workspace_bytes=workspace;spec.chunk_tokens=effective_chunk;
        spec.bo_budget_bytes=c.hardware.buffer_budget_bytes;
        plan=runtime::plan_full_gqa_runtime(spec);
        statistics.hot_prompt_tokens=plan.prompt_hot_tokens;
        statistics.cold_prompt_tokens=plan.prompt_cold_tokens;
        statistics.hot_kv_cache_bytes=plan.reserved_all_layers_hot_bytes;
        statistics.workspace_reserved_bytes=workspace;
        statistics.managed_reserved_bytes=plan.managed_total;
        statistics.score_file_extent_bytes=plan.raw_K_extent_bytes;
        statistics.layers.resize(c.layers);
        if(c.mode!="HOST_MATCHED"){
            need(c.hardware.abi==V3&&c.hardware.image_sha256==SHA&&c.hardware.image_execute_authorized,"runtime requires the specifically authorized V3 image");
            need(c.hardware.direct_read_queue_depth==1,"runtime uses explicitly validated QD1 only");
            auto hw=c.hardware;hw.reuse_file_buffers=true;
            // An empty cold set is an explicit all-hot capacity control.
            if(cold)fpga=make_fpga_backend(HardwareMode::fpga_storage_streaming,hw);
        }
#ifndef O_DIRECT
        throw std::runtime_error("full GQA runtime requires O_DIRECT; no buffered fallback");
#else
        file.value=open(c.score_file.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC|O_NOFOLLOW|O_DIRECT,0600);
        need(file.value>=0,"full GQA requires new exclusive direct rawK file: "+std::string(std::strerror(errno)));
        struct stat st{};need(fstat(file.value,&st)==0&&S_ISREG(st.st_mode),"rawK backing must be a regular file");
        need(ftruncate(file.value,static_cast<off_t>(plan.raw_K_extent_bytes))==0,"rawK extent creation failed");
#endif
        const auto slots=plan.prompt_hot_tokens+c.generated_capacity;
        hot_kv.resize(c.layers);
        for(auto& layer:hot_kv)layer.resize(mul(slots,KEY_WORDS*2));
        last_selected.assign(c.layers,std::numeric_limits<std::uint64_t>::max());
        ready_payloads.resize(c.layers);
        slot_maps.assign(c.layers,plan.hot_slot_by_position);
        cold_positions.resize(c.layers);placements.resize(c.layers);
        last_hpm.assign(c.layers,std::numeric_limits<std::uint64_t>::max());
        for(std::size_t l=0;l<c.layers;++l){
            for(std::size_t t=0;t<c.prompt_tokens;++t)if(plan.hot_slot_by_position[t]<0)cold_positions[l].push_back(t);
            if(c.hpm)placements[l]=std::make_unique<ApproxHkm>(HkmConfig{c.prompt_tokens/UNIT,plan.hot_prompt_blocks,std::min<std::size_t>(16,plan.hot_prompt_blocks),std::min(c.hpm_swaps,std::size_t(plan.hot_prompt_blocks)),8},c.prompt_tokens/UNIT-1);
        }
        events.reserve(event_limit);
        append_buffer.allocate(UNIT*KEY_BYTES);
        read_buffer.allocate(mul(std::max(c.chunk_tokens,effective_chunk),KEY_BYTES));
        if(c.cpu_implementation=="auto"){
            hot_cpu=std::make_unique<FullGqaCpuScorer>(c.hot_threads);
            cold_cpu=std::make_unique<FullGqaCpuScorer>(c.cold_threads);
            statistics.cpu_implementation_actual=hot_cpu->implementation_name();
        }else statistics.cpu_implementation_actual=full_gqa_legacy_cpu_name;
        if(c.verify_scores)scalar=std::make_unique<FullGqaCpuScorer>(1,FullGqaCpuImplementation::scalar);
        if(c.payload_prefetch)payload_reader=std::make_unique<storage::DirectReadBatch>(c.payload_read_queue_depth);
    }
    ~Impl(){for(auto& j:pending)if(j.second.payload_future.valid())j.second.payload_future.wait();}
    void layer_check(int l)const{need(l>=0&&std::size_t(l)<config.layers,"full GQA layer out of bounds");}
    bool hot(int l,std::uint64_t t)const{return t<slot_maps[l].size()&&slot_maps[l][t]>=0;}
    const std::uint16_t* cached(int l,std::uint64_t t)const{
        need(hot(l,t),"full GQA hot lookup on cold/unreserved token");
        return hot_kv[l].data()+std::size_t(slot_maps[l][t])*KEY_WORDS*2;
    }
    std::uint64_t file_offset(int l,std::uint64_t t)const{return std::uint64_t(l)*plan.raw_K_layer_stride_bytes+t*KEY_BYTES;}
    std::vector<std::uint64_t> hot_ids(int l,std::size_t valid)const{
        std::vector<std::uint64_t> ids;ids.reserve(plan.prompt_hot_tokens+config.generated_capacity);
        for(std::size_t t=0;t<config.prompt_tokens;++t)if(hot(l,t))ids.push_back(t);
        for(std::size_t t=config.prompt_tokens;t<valid;++t)ids.push_back(t);
        return ids;
    }
    std::vector<FullGqaKeyView> views(int l,const std::vector<std::uint64_t>& ids)const{
        std::vector<FullGqaKeyView> v;
        for(std::size_t j=0;j<ids.size();){std::size_t end=j+1;
            while(end<ids.size()&&ids[end]==ids[end-1]+1&&slot_maps[l][ids[end]]==slot_maps[l][ids[end-1]]+1)++end;
            v.push_back({cached(l,ids[j]),end-j,KEY_WORDS*2});j=end;
        }return v;
    }
    void merge(std::vector<std::uint16_t>& raw,const std::vector<std::uint16_t>& values,
               const std::vector<std::uint64_t>& ids,std::size_t offset,std::size_t n)const{
        need(values.size()==H*n&&offset+n<=ids.size(),"full GQA score merge shape");
        for(std::size_t h=0;h<H;++h)for(std::size_t j=0;j<n;++j){const auto x=values[h*n+j];need(std::isfinite(half_value(x)),"full GQA nonfinite score");raw[h*config.capacity+ids[offset+j]]=x;}
    }
    std::vector<ScoreFileRegion> regions(int l)const{
        std::vector<ScoreFileRegion> r;
        for(auto t:cold_positions[l]){const auto off=file_offset(l,t);if(!r.empty()&&r.back().offset+r.back().bytes==off)r.back().bytes+=KEY_BYTES;else r.push_back({file.value,off,KEY_BYTES,false});}
        for(const auto& x:r)need(x.offset%4096==0&&x.bytes%(UNIT*KEY_BYTES)==0,"cold file region not64token aligned");
        return r;
    }
    void open_full_kv(){
        if(full_kv_file.value>=0)return;
#ifndef O_DIRECT
        throw std::runtime_error("payload prefetch requires O_DIRECT");
#else
        std::lock_guard<std::mutex> io(config.full_kv_io_fence->file_io);
        FD opened;opened.value=open(config.full_kv_backing_path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_DIRECT);
        need(opened.value>=0,"payload prefetch cannot open existing full KV O_DIRECT: "+std::string(std::strerror(errno)));
        struct stat st{};need(fstat(opened.value,&st)==0&&S_ISREG(st.st_mode)&&
            std::uint64_t(st.st_size)==plan.segmented_KV_extent_bytes,"payload prefetch complete KV identity/extent mismatch");
        Aligned header;header.allocate(4096);const auto header_started=Clock::now();
        io_exact(opened.value,header.data,4096,0,false);
        statistics.payload_prefetch_setup_read_ms+=elapsed(header_started);
        statistics.payload_prefetch_setup_read_bytes+=4096;++statistics.payload_prefetch_setup_read_calls;
        const auto* h=static_cast<const std::uint64_t*>(header.data);
        need(h[0]==0x48534b5653454731ULL&&h[1]==1&&h[3]==config.layers&&h[4]==config.capacity&&
             h[5]==KV_BYTES&&h[6]==KV_BYTES+32&&h[7]==plan.segmented_KV_extent_bytes,
             "payload prefetch requires exact existing SegmentedKVStore v1 layout");
        full_kv_request=h[2];full_kv_file.value=opened.value;opened.value=-1;
#endif
    }
    std::shared_ptr<const runtime::GQAPrefetchArtifact> prefetch_payload(
            int l,std::uint64_t position,const Work& work,const std::vector<std::int64_t>& slots){
        auto p=std::make_shared<runtime::GQAPrefetchArtifact>();
        p->request_id=full_kv_request;p->layer=l;p->position=position;p->score_ready_ns=work.ended;
        // Same F16 scaling/order as the final selector. Adding current head0
        // can only increase its maximum and the rounded max-alpha threshold.
        // Alpha is fixed for this request and shared with final selection.
        // Thus each final count <= historical count at the old threshold +1.
        // F16 rounding is monotone. Counts stay <=65504; the sum of32 rounded
        // integer counts is exact in FP32 (<2^24), so the rounded mean is also
        // an upper bound. Current K cannot reorder historical scores. Taking
        // historical top(bound) therefore covers every final historical ID,
        // without paying the full20% cap when the threshold admits few tokens.
        const auto cap=std::min<std::size_t>(position,std::floor(double(position+1)*double(.2f)));
        need(position&&position<config.capacity&&slots.size()==config.capacity&&cap,
             "payload prefetch historical cap/range mismatch");
        std::vector<std::uint16_t> scaled(work.raw.size());
        for(std::size_t i=0;i<work.raw.size();++i)scaled[i]=half_bits(half_value(work.raw[i])/std::sqrt(128.f));
        float maximum=-std::numeric_limits<float>::infinity();
        for(std::size_t t=0;t<position;++t)maximum=std::max(maximum,half_value(scaled[t]));
        const auto threshold=half_value(half_bits(maximum-config.selector_alpha));
        need(std::isfinite(threshold),"prefetch threshold upper-bound requires finite historical threshold");
        float count_sum=0;
        for(std::size_t h=0;h<H;++h){
            std::uint64_t count=1; // at most one newly appended current token
            for(std::size_t t=0;t<position;++t)count+=half_value(scaled[h*config.capacity+t])>=threshold;
            const auto rounded=config.fp32_selection_counts?float(count):half_value(half_bits(float(count)));
            need(std::isfinite(rounded),"prefetch count upper-bound exceeds original F16 range");count_sum+=rounded;
        }
        const auto mean_bound=config.fp32_selection_counts?count_sum/float(H):half_value(half_bits(count_sum/float(H)));
        need(std::isfinite(mean_bound)&&mean_bound>=1,"prefetch mean upper-bound invalid");
        const auto bound=std::min(cap,std::size_t(mean_bound));
        std::vector<std::uint64_t> ranked(position);
        std::vector<unsigned char> used(position,0);p->per_head_ids.resize(H);
        for(std::size_t h=0;h<H;++h){
            std::iota(ranked.begin(),ranked.end(),0);
            auto compare=[&](std::uint64_t a,std::uint64_t b){const auto av=finite_half_order_key(scaled[h*config.capacity+a]),bv=finite_half_order_key(scaled[h*config.capacity+b]);return av!=bv?av>bv:a<b;};
            std::partial_sort(ranked.begin(),ranked.begin()+bound,ranked.end(),compare);
            p->per_head_ids[h].assign(ranked.begin(),ranked.begin()+bound);
            for(auto id:p->per_head_ids[h])used[id]=1;
        }
        p->physical_ids.reserve(position);
        for(std::size_t t=0;t<position;++t)if(used[t])p->physical_ids.push_back(t);
        read_payload_rows(*p,slots);
        return p;
    }
    // Shared by selected history and the two dense decode layers. It reads
    // exactly the caller's rows, using the same bounded windows and integrity
    // checks. Gap overread is counted and is never admitted to the hot cache.
    void read_payload_rows(runtime::GQAPrefetchArtifact& artifact,
                           const std::vector<std::int64_t>& slots){
        auto* p=&artifact;const auto l=p->layer;
        p->kv_bytes.resize(mul(p->physical_ids.size(),KV_BYTES));
        p->read_started_ns=stamp();
        struct Header{std::uint64_t digest,position,request;std::uint32_t layer,version;};
        static_assert(sizeof(Header)==32,"existing complete KV record header");
        Aligned buffer;buffer.allocate(mul(config.payload_read_limits.window_bytes,config.payload_read_queue_depth));
        auto offset=[&](std::uint64_t t){return std::uint64_t(4096)+(std::uint64_t(l)*config.capacity+t)*(KV_BYTES+32);};
        std::vector<storage::KVReadRecord> cold;cold.reserve(p->physical_ids.size());
        auto begin=stamp();
        for(std::size_t i=0;i<p->physical_ids.size();++i){
            const auto id=p->physical_ids[i];
            if(slots[id]<0){cold.push_back({offset(id),i});continue;}
            std::memcpy(p->kv_bytes.data()+i*KV_BYTES,hot_kv[l].data()+std::size_t(slots[id])*KEY_WORDS*2,KV_BYTES);
            p->hot_bytes+=KV_BYTES;
        }
        p->hot_copy_ns=stamp()-begin;
        begin=stamp();
        // Dense callers may request a non-sorted physical order; the output_row
        // mapping preserves it while file IO stays ordered and deduplicated.
        std::sort(cold.begin(),cold.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
        const auto windows=storage::plan_kv_reads(cold,KV_BYTES+32,plan.segmented_KV_extent_bytes,config.payload_read_limits);
        p->plan_ns=stamp()-begin;
        auto check_extent=[&]{
            const auto waiting=stamp();std::lock_guard<std::mutex> io(config.full_kv_io_fence->file_io);
            const auto locked=stamp();p->io_lock_wait_ns+=locked-waiting;
            struct stat st{};need(fstat(full_kv_file.value,&st)==0&&S_ISREG(st.st_mode)&&
                std::uint64_t(st.st_size)==plan.segmented_KV_extent_bytes,"prefetch complete KV extent changed");
            p->extent_check_ns+=stamp()-locked;++p->extent_checks;
        };
        if(!windows.empty())check_extent();
        for(std::size_t first=0;first<windows.size();first+=config.payload_read_queue_depth){
            const auto n=std::min<std::size_t>(config.payload_read_queue_depth,windows.size()-first);
            std::array<storage::DirectReadBatch::Request,4> requests{};
            for(std::size_t i=0;i<n;++i){const auto& w=windows[first+i];
                requests[i]={full_kv_file.value,static_cast<unsigned char*>(buffer.data)+i*config.payload_read_limits.window_bytes,
                             std::size_t(w.end-w.begin),w.begin};
            }
            // All reads finish before releasing the RMW fence, including an
            // IO error. CPU validation runs outside it, allowing the writer.
            {const auto waiting=stamp();std::lock_guard<std::mutex> io(config.full_kv_io_fence->file_io);
             const auto locked=stamp();p->io_lock_wait_ns+=locked-waiting;
             payload_reader->read_exact(requests,n);p->pread_ns+=stamp()-locked;}
            for(std::size_t i=0;i<n;++i){const auto& w=windows[first+i];
                p->cold_syscall_bytes+=w.end-w.begin;++p->cold_read_calls;
                p->required_page_bytes+=w.required_page_bytes;
                p->gap_read_bytes+=w.end-w.begin-w.required_page_bytes;
                const auto* raw=static_cast<const unsigned char*>(requests[i].data);
                for(auto j=w.first_record;j<w.end_record;j+=4){
                    const auto count=std::min<std::size_t>(4,w.end_record-j);const auto start=stamp();
                    std::array<Header,4> headers{};std::array<const unsigned char*,4> rows{};
                    for(std::size_t k=0;k<count;++k){const auto& r=cold[j+k];auto& h=headers[k];
                        std::memcpy(&h,raw+r.offset-w.begin,sizeof h);
                        need(h.position==p->physical_ids[r.output_row]&&h.request==p->request_id&&h.layer==std::uint32_t(l)&&h.version==1,
                             "prefetch complete KV record layer/position/request mismatch");
                        auto* dst=p->kv_bytes.data()+r.output_row*KV_BYTES;
                        std::memcpy(dst,raw+r.offset-w.begin+sizeof h,KV_BYTES);rows[k]=dst;
                    }
                    const auto copied=stamp();p->validate_copy_ns+=copied-start;
                    const auto hashes=storage::checksum_four(rows,KV_BYTES,count);
                    for(std::size_t k=0;k<count;++k)need(hashes[k]==headers[k].digest,"prefetch complete KV checksum mismatch");
                    p->checksum_ns+=stamp()-copied;p->cold_bytes+=count*KV_BYTES;
                }
            }
        }
        if(!windows.empty())check_extent();
        p->read_finished_ns=stamp();
    }
    // Called with engine held; cold set is fixed and fully persisted at prefill.
    Work score_history(int l,std::size_t valid,const std::vector<std::uint16_t>& q){
        std::lock_guard<std::mutex> exclusive(engine);
        Work w;w.started=stamp();w.raw.resize(H*config.capacity,0);
        auto hi=hot_ids(l,valid);
        const auto& cold=cold_positions[l];auto vv=views(l,hi);
        auto hot_task=std::async(std::launch::async,[&]{
            w.hot_start=stamp();
            if(!hi.empty()){
                if(hot_cpu)hot_cpu->score(q,vv,hot_scores);
                else{
                    // Same preparation as hkm_app_replay::score_hot legacy:
                    // gather physical K once, preserving caller/view order.
                    std::vector<std::uint16_t> packed(hi.size()*KEY_WORDS);std::size_t at=0;
                    for(const auto& view:vv)for(std::size_t t=0;t<view.tokens;++t){
                        std::memcpy(packed.data()+at*KEY_WORDS,view.data+t*view.token_stride_words,KEY_BYTES);++at;
                    }
                    need(at==hi.size(),"legacy hot packed K extent differs from candidate IDs");
                    w.hot_packed_key_bytes=packed.size()*sizeof(std::uint16_t);
                    hot_scores=score_full_gqa_legacy_cpu(q,packed.data(),hi.size(),KEY_WORDS,config.hot_threads);
                }
                ++w.hot_cpu_calls;merge(w.raw,hot_scores,hi,0,hi.size());
            }
            w.hot_end=stamp();
        });
        w.cold_start=stamp();
        if(!cold.empty()){
            if(config.mode=="HOST_MATCHED"){
                // Cold positions form aligned runs. Chunking bounds the CPU
                // read buffer without changing input order or candidate set.
                for(std::size_t start=0;start<cold.size();){
                    const auto n=std::min(config.chunk_tokens,cold.size()-start);
                    std::size_t copied=0;
                    while(copied<n){std::size_t end=copied+1;
                        while(end<n&&cold[start+end]==cold[start+end-1]+1)++end;
                        const auto bytes=(end-copied)*KEY_BYTES;
                        io_exact(file.value,static_cast<unsigned char*>(read_buffer.data)+copied*KEY_BYTES,bytes,file_offset(l,cold[start+copied]),false);
                        copied=end;w.cold_bytes+=bytes;
                    }
                    if(cold_cpu)cold_cpu->score(q,{{static_cast<const std::uint16_t*>(read_buffer.data),n,KEY_WORDS}},cold_scores);
                    else cold_scores=score_full_gqa_legacy_cpu(q,static_cast<const std::uint16_t*>(read_buffer.data),n,KEY_WORDS,config.cold_threads);
                    ++w.cold_cpu_calls;
                    merge(w.raw,cold_scores,cold,start,n);start+=n;
                }
            }else{
                need(bool(fpga),"full GQA FPGA backend absent");
                w.device=fpga->score_file_regions_raw_gqa_stream(q,regions(l),H,KH,D,effective_chunk,
                    [&](const FileScoreChunk& c){merge(w.raw,c.values,cold,c.info.token_offset,c.info.tokens);});
                need(w.device.p2p_allocation_verified&&w.device.file_read_bytes==cold.size()*KEY_BYTES,
                     "full GQA P2P file path/read volume not verified");
                w.cold_bytes=w.device.file_read_bytes;
            }
        }
        w.cold_end=stamp();hot_task.get();w.ended=stamp();return w;
    }
    void append(int l,const std::vector<std::uint64_t>& positions,const std::vector<std::uint16_t>& keys,const std::vector<std::uint8_t>& kv){
        layer_check(l);need(!drained&&!positions.empty(),"append after drain or empty append");auto& st=statistics.layers[l];
        need(positions.front()==st.completed_tokens&&positions.size()<=config.prompt_tokens+config.generated_capacity-st.completed_tokens,"full GQA incremental append range");
        need(keys.size()==positions.size()*KEY_WORDS&&kv.size()==positions.size()*KV_BYTES,"full GQA append layout");
        const auto job=pending.find(l);if(job!=pending.end())need(positions.size()==1&&positions.front()==job->second.position,"prefetched history permits only its current-token append");
        for(std::size_t i=0;i<positions.size();++i){need(positions[i]==positions.front()+i,"nonconsecutive full GQA append");need(!std::memcmp(keys.data()+i*KEY_WORDS,kv.data()+i*KV_BYTES,KEY_BYTES),"full GQA scoring K differs from complete KV backing");}
        // Bounded page-preserving direct writes. No per-layer fsync; completed
        // advances only after every write and hot-copy succeeds.
        std::size_t done=0;
        while(done<positions.size()){
            const auto token=positions[done];const auto block_first=token/2*2;
            const auto count=std::min(positions.size()-done,UNIT-std::size_t(token%UNIT));
            const auto off=file_offset(l,block_first);const auto nbytes=((token-block_first+count)*KEY_BYTES+4095)/4096*4096;
            if(token!=block_first||count*KEY_BYTES!=nbytes){io_exact(file.value,append_buffer.data,nbytes,off,false);st.append_rmw_read_bytes+=nbytes;}
            std::memcpy(static_cast<unsigned char*>(append_buffer.data)+(token-block_first)*KEY_BYTES,keys.data()+done*KEY_WORDS,count*KEY_BYTES);
            io_exact(file.value,append_buffer.data,nbytes,off,true);st.append_syscall_bytes+=nbytes;
            done+=count;
        }
        for(std::size_t i=0;i<positions.size();++i)if(hot(l,positions[i]))std::memcpy(hot_kv[l].data()+std::size_t(slot_maps[l][positions[i]])*KEY_WORDS*2,kv.data()+i*KV_BYTES,KV_BYTES);
        st.completed_tokens+=positions.size();st.append_key_bytes+=keys.size()*2;++st.append_calls;
    }
    void prepare(int l,std::uint64_t position,const std::vector<float>& rehearsal,runtime::GQAPrefetchReady ready){
        layer_check(l);need(!drained,"prepare after request drain");
        need(position>=config.prompt_tokens&&position<config.prompt_tokens+config.generated_capacity&&statistics.layers[l].completed_tokens==position,"history prefetch requires exactly completed past positions");
        need(pending.size()<2&&pending.count(l)==0,"at most two distinct full GQA history preparations may be outstanding");
        const auto retained=std::count_if(ready_payloads.begin(),ready_payloads.end(),[](const auto& x){return bool(x);});
        need(!ready_payloads[l]&&pending.size()+std::size_t(retained)<2,"at most two outstanding/selected-but-unconsumed payloads");
        need(config.payload_prefetch||!ready,"payload callback requires explicitly enabled payload prefetch");
        if(config.payload_prefetch)open_full_kv();
        Pending p;p.position=position;p.submitted=stamp();p.query=query_f16(rehearsal);
        const auto q=p.query;
        auto slots=config.payload_prefetch?slot_maps[l]:std::vector<std::int64_t>{};
        std::promise<std::shared_ptr<const Work>> scores;
        p.scores=scores.get_future().share();
        p.payload_future=std::async(std::launch::async,[this,l,position,q,slots=std::move(slots),ready=std::move(ready),scores=std::move(scores)]() mutable {
            std::shared_ptr<const Work> work;
            try{work=std::make_shared<const Work>(score_history(l,position,q));}
            catch(...){scores.set_exception(std::current_exception());throw;}
            scores.set_value(work); // final selection need not wait for KV IO/CUDA
            std::shared_ptr<const runtime::GQAPrefetchArtifact> payload;
            if(config.payload_prefetch){payload=prefetch_payload(l,position,*work,slots);if(ready)ready(payload);}
            return payload; // callback errors propagate at fetch(), never fallback
        });
        pending.emplace(l,std::move(p));++statistics.layers[l].history_prefetch_calls;
    }
    void verify(int l,std::size_t valid,const std::vector<std::uint16_t>& q,const std::vector<std::uint16_t>& raw){
        if(!scalar)return;
        std::lock_guard<std::mutex> exclusive(engine);auto& st=statistics.layers[l];std::vector<std::uint16_t> reference;
        for(std::size_t first=0;first<valid;){auto n=std::min(config.chunk_tokens,valid-first);auto bytes=((n*KEY_BYTES+4095)/4096)*4096;
            io_exact(file.value,read_buffer.data,bytes,file_offset(l,first),false);st.verification_read_bytes+=bytes;
            scalar->score(q,{{static_cast<const std::uint16_t*>(read_buffer.data),n,KEY_WORDS}},reference);
            for(std::size_t h=0;h<H;++h)for(std::size_t t=0;t<n;++t)need(reference[h*n+t]==raw[h*config.capacity+first+t],"full GQA live raw score differs from independent scalar same-input oracle");
            st.verification_scores+=H*n;first+=n;
        }
    }
    // The executor thread alone owns this operation. It touches no history
    // scorer, shared output vector, read buffer, or XRT state. History was
    // joined before this call; another layer may still hold engine. This one
    // ordinary candidate must not wait behind that layer's whole cold scan.
    void score_current(int l,std::uint64_t position,const std::vector<std::uint16_t>& q,
                       std::vector<std::uint16_t>& raw){
        need(position>=config.prompt_tokens&&position<statistics.layers[l].completed_tokens&&
             q.size()==H*D&&raw.size()==H*config.capacity,
             "current-token completion requires appended hot tail and unchanged query shape");
        need(std::fegetround()==FE_TONEAREST,"current-token score requires round-to-nearest FP32 mode");
        const auto* key=cached(l,position);
        for(std::size_t h=0;h<H;++h){
            volatile float sum=0;
            for(std::size_t d=0;d<D;++d){
                volatile float product=half_value(q[h*D+d])*half_value(key[(h/4)*D+d]);
                sum=sum+product;
            }
            const auto bits=half_bits(sum);
            need(std::isfinite(half_value(bits)),"current-token full GQA nonfinite score");
            raw[h*config.capacity+position]=bits;
        }
    }
    std::vector<std::vector<std::uint64_t>> select(int l,std::uint64_t position,const std::vector<float>& rehearsal){
        layer_check(l);need(!drained,"select after drain");auto& st=statistics.layers[l];
        need(position>=config.prompt_tokens&&position<config.prompt_tokens+config.generated_capacity&&st.completed_tokens==position+1,"full GQA selection requires current KV append and valid decode range");
        need(last_selected[l]==std::numeric_limits<std::uint64_t>::max()||position>last_selected[l],"full GQA duplicate/backwards layer selection");
        need(events.size()<event_limit,"full GQA event budget exceeded");
        Event e;e.layer=l;e.position=position;e.started=stamp();auto q=query_f16(rehearsal);Work work;
        auto job=pending.find(l);
        need(!config.payload_prefetch||job!=pending.end(),"enabled payload prefetch requires prior same-layer history preparation");
        if(job!=pending.end()){
            need(job->second.position==position&&job->second.query==q,"history Q/position differs from current same-query selection");
            e.prefetched=true;e.history_submitted=job->second.submitted;
            work=*job->second.scores.get();e.joined=stamp();
            if(!config.payload_prefetch){job->second.payload_future.get();pending.erase(job);}
            // Retain the async future until fetch: its destruction here would
            // block on payload IO and silently serialize the stages again.
            // Current token is an ordinary candidate. Its small independent
            // scalar completion never queues behind the next layer's history.
            e.current_score_started=stamp();score_current(l,position,q,work.raw);e.current_score_finished=stamp();
            ++st.current_token_score_calls;
            st.current_token_score_ms+=double(e.current_score_finished-e.current_score_started)/1e6;
        }else{work=score_history(l,position+1,q);e.joined=stamp();}
        e.scores_ready=stamp();
        auto scaled=work.raw;for(auto& x:scaled)x=half_bits(half_value(x)/std::sqrt(128.f));
        auto selected=select_gqa_scores(scaled,H,config.capacity,position+1,config.selector_alpha,.2f,config.fp32_selection_counts);
        need(selected.k>0,"full GQA zero selection; no dense/current-token fallback");e.ids_ready=stamp();
        e.k=selected.k;std::copy(selected.counts.begin(),selected.counts.end(),e.counts.begin());
        e.score_hash=hash_bytes(work.raw.data(),work.raw.size()*2);
        std::vector<std::uint64_t> ids;ids.reserve(selected.k*H);for(const auto& h:selected.ids)ids.insert(ids.end(),h.begin(),h.end());
        e.id_hash=hash_bytes(ids.data(),ids.size()*sizeof(std::uint64_t));std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());e.physical_count=ids.size();e.physical_hash=hash_bytes(ids.data(),ids.size()*sizeof(std::uint64_t));
        const auto begin=Clock::now();verify(l,position+1,q,work.raw);e.verification_ms=elapsed(begin);
        e.hot_start=work.hot_start;e.hot_end=work.hot_end;e.cold_start=work.cold_start;e.cold_end=work.cold_end;e.work_started=work.started;e.work_ended=work.ended;
        e.cold_bytes=work.cold_bytes;e.qbytes=work.device.query_h2d_bytes;e.sbytes=work.device.score_d2h_bytes;e.fpga_calls=work.device.chunks.size();e.chunks=std::move(work.device.chunks);e.ended=stamp();
        e.hot_packed_key_bytes=work.hot_packed_key_bytes;e.hot_cpu_calls=work.hot_cpu_calls;e.cold_cpu_calls=work.cold_cpu_calls;
        st.hot_cpu_packed_key_bytes+=work.hot_packed_key_bytes;st.hot_cpu_calls+=work.hot_cpu_calls;st.cold_cpu_calls+=work.cold_cpu_calls;
        ++st.score_calls;st.scored_candidates+=position+1;st.cold_read_bytes+=e.cold_bytes;st.fpga_calls+=e.fpga_calls;st.query_h2d_bytes+=e.qbytes;st.score_d2h_bytes+=e.sbytes;
        last_selected[l]=position;events.push_back(std::move(e));return selected.ids;
    }
    std::vector<std::uint8_t> fetch(int l,const std::vector<std::uint64_t>& ids,const ColdFetch& reader){
        layer_check(l);need(!drained,"fetch after drain");auto& st=statistics.layers[l];
        need(ids.size()<=config.capacity,"full GQA fetch exceeds admitted union");
        for(auto id:ids)plan.require_visible(l,id,st.completed_tokens);
        auto sorted=ids;std::sort(sorted.begin(),sorted.end());need(std::adjacent_find(sorted.begin(),sorted.end())==sorted.end(),"full GQA physical fetch IDs must be unique");
        Event* event=nullptr;
        if(last_selected[l]!=std::numeric_limits<std::uint64_t>::max()){
            auto it=std::find_if(events.rbegin(),events.rend(),[&](const Event& e){return e.layer==l&&e.position==last_selected[l];});
            if(it!=events.rend()&&it->physical_count==sorted.size()&&it->physical_hash==hash_bytes(sorted.data(),sorted.size()*sizeof(std::uint64_t)))event=&*it;
        }
        // The dense path shares the selected-payload reader instead of
        // fragmenting full history into 64-token callback batches.
        if(config.payload_prefetch&&!pending.count(l)&&last_selected[l]==std::numeric_limits<std::uint64_t>::max()){
            open_full_kv();runtime::GQAPrefetchArtifact dense;
            dense.request_id=full_kv_request;dense.layer=l;dense.physical_ids=ids;
            read_payload_rows(dense,slot_maps[l]);
            st.hot_fetch_bytes+=dense.hot_bytes;st.cold_fetch_bytes+=dense.cold_bytes;++st.fetch_calls;
            st.dense_fetch_read_calls+=dense.cold_read_calls;st.dense_fetch_syscall_bytes+=dense.cold_syscall_bytes;
            st.dense_fetch_pread_ns+=dense.pread_ns;st.dense_fetch_checksum_ns+=dense.checksum_ns;
            return std::move(dense.kv_bytes);
        }
        // A sparse layer must join its exact selected history and GPU callback.
        if(config.payload_prefetch&&(pending.count(l)||last_selected[l]!=std::numeric_limits<std::uint64_t>::max())){
            need(event,"prefetch consumption requires the exact final physical selection");
            auto job=pending.find(l);
            need(job!=pending.end()&&job->second.position==last_selected[l],"prefetch fetch requires unconsumed selected history");
            event->payload_join_started=stamp();
            auto payload=job->second.payload_future.get();
            event->payload_join_finished=stamp();
            need(payload&&payload->layer==l&&payload->position==last_selected[l],"prefetch consumption layer/step mismatch");
            pending.erase(job);ready_payloads[l]=payload;
            const auto& p=*payload;
            event->payload_started=p.read_started_ns;event->payload_ready=p.read_finished_ns;event->payload_bytes=p.kv_bytes.size();
            event->payload_hot=p.hot_bytes;event->payload_cold=p.cold_bytes;event->payload_syscall=p.cold_syscall_bytes;event->payload_reads=p.cold_read_calls;
            event->plan_ns=p.plan_ns;event->hot_copy_ns=p.hot_copy_ns;event->io_lock_wait_ns=p.io_lock_wait_ns;
            event->extent_check_ns=p.extent_check_ns;event->pread_ns=p.pread_ns;
            event->validate_copy_ns=p.validate_copy_ns;event->checksum_ns=p.checksum_ns;
            event->required_page_bytes=p.required_page_bytes;event->gap_read_bytes=p.gap_read_bytes;event->extent_checks=p.extent_checks;
            ++st.payload_prefetch_calls;st.payload_prefetch_bytes+=p.kv_bytes.size();
            st.payload_prefetch_hot_bytes+=p.hot_bytes;st.payload_prefetch_cold_bytes+=p.cold_bytes;
            st.payload_prefetch_syscall_bytes+=p.cold_syscall_bytes;st.payload_prefetch_read_calls+=p.cold_read_calls;
        }
        auto payload=ready_payloads[l];
        auto prefetched_index=[&](std::uint64_t id)->std::size_t{
            if(!payload)return std::numeric_limits<std::size_t>::max();
            auto at=std::lower_bound(payload->physical_ids.begin(),payload->physical_ids.end(),id);
            return at!=payload->physical_ids.end()&&*at==id?std::size_t(at-payload->physical_ids.begin()):std::numeric_limits<std::size_t>::max();
        };
        std::uint64_t useful=0,miss=0,current=0;
        const auto fetch_started=stamp(),hot_before=st.hot_fetch_bytes,cold_before=st.cold_fetch_bytes;
        std::vector<std::uint8_t> out(ids.size()*KV_BYTES);
        for(std::size_t i=0;i<ids.size();){
            const auto at=prefetched_index(ids[i]);
            if(at!=std::numeric_limits<std::size_t>::max()){
                std::memcpy(out.data()+i*KV_BYTES,payload->kv_bytes.data()+at*KV_BYTES,KV_BYTES);
                useful+=KV_BYTES;if(hot(l,ids[i]))st.hot_fetch_bytes+=KV_BYTES;else st.cold_fetch_bytes+=KV_BYTES;
                ++i;continue;
            }
            if(payload){if(ids[i]==payload->position)current+=KV_BYTES;else miss+=KV_BYTES;}
            if(hot(l,ids[i])){std::memcpy(out.data()+i*KV_BYTES,cached(l,ids[i]),KV_BYTES);st.hot_fetch_bytes+=KV_BYTES;++i;continue;}
            need(bool(reader),"cold selected KV requires complete backing-store callback");std::size_t end=i;
            while(end<ids.size()&&!hot(l,ids[end])&&end-i<UNIT&&prefetched_index(ids[end])==std::numeric_limits<std::size_t>::max())++end;
            if(payload&&end>i+1)miss+=(end-i-1)*KV_BYTES;
            std::vector<std::uint64_t> batch(ids.begin()+i,ids.begin()+end);auto bytes=reader(l,batch);
            need(bytes.size()==batch.size()*KV_BYTES,"cold backing callback returned wrong KV byte count");
            std::memcpy(out.data()+i*KV_BYTES,bytes.data(),bytes.size());st.cold_fetch_bytes+=bytes.size();i=end;
        }
        ++st.fetch_calls;
        if(config.hpm&&event&&last_hpm[l]!=last_selected[l]){
            need(pending.count(l)==0,"HPM cannot mutate cache used by an outstanding same-layer history job");
            const auto begin=Clock::now();
            std::vector<std::uint16_t> counts(config.prompt_tokens/UNIT,0);
            for(auto id:sorted)if(id<config.prompt_tokens)++counts[id/UNIT];
            std::vector<std::pair<std::uint64_t,std::uint16_t>> weighted;
            for(std::size_t b=0;b<counts.size();++b)if(counts[b])weighted.emplace_back(b,counts[b]);
            const auto move=placements[l]->update_weighted(weighted,config.prompt_tokens/UNIT-1);
            need(move.promote.size()==move.demote.size()&&move.promote.size()<=config.hpm_swaps,"HPM paired bounded migration contract");
            for(std::size_t m=0;m<move.promote.size();++m){
                const auto source=move.promote[m]*UNIT,victim=move.demote[m]*UNIT;
                need(!hot(l,source)&&hot(l,victim)&&reader,"HPM requires real cold backing and hot victim");
                std::vector<std::uint64_t> batch(UNIT);for(std::size_t j=0;j<UNIT;++j)batch[j]=source+j;
                auto data=reader(l,batch);need(data.size()==UNIT*KV_BYTES,"HPM promotion returned incomplete KV block");
                const auto slot=slot_maps[l][victim];
                for(std::size_t j=0;j<UNIT;++j)need(slot_maps[l][victim+j]==slot+std::int64_t(j)&&!hot(l,source+j),"HPM block slot layout corrupted");
                std::memcpy(hot_kv[l].data()+std::size_t(slot)*KEY_WORDS*2,data.data(),data.size());
                for(std::size_t j=0;j<UNIT;++j){slot_maps[l][victim+j]=-1;slot_maps[l][source+j]=slot+std::int64_t(j);}
                st.hpm_migration_bytes+=data.size();event->hpm_migration_bytes+=data.size();
            }
            cold_positions[l].clear();
            for(std::size_t t=0;t<config.prompt_tokens;++t)if(!hot(l,t))cold_positions[l].push_back(t);
            need(cold_positions[l].size()==plan.prompt_cold_tokens,"HPM changed fixed beta capacity");
            event->promote=move.promote;event->demote=move.demote;event->hpm_ms=elapsed(begin);
            st.hpm_ms+=event->hpm_ms;st.hpm_promoted_blocks+=move.promote.size();++st.hpm_updates;last_hpm[l]=last_selected[l];
        }
        if(payload){
            need(useful<=payload->kv_bytes.size(),"prefetch useful payload accounting overflow");
            const auto waste=payload->kv_bytes.size()-useful;
            st.payload_prefetch_useful_bytes+=useful;st.payload_prefetch_wasted_bytes+=waste;
            st.payload_prefetch_miss_bytes+=miss;st.payload_prefetch_current_bytes+=current;
            if(event){event->payload_useful=useful;event->payload_wasted=waste;event->payload_miss=miss;event->payload_current=current;}
            ready_payloads[l].reset();
        }
        if(event){event->fetch_started=fetch_started;event->fetch_finished=stamp();event->hot_fetch_bytes=st.hot_fetch_bytes-hot_before;event->cold_fetch_bytes=st.cold_fetch_bytes-cold_before;}
        return out;
    }
    void drain(){
        need(pending.empty(),"cannot drain while prepared history is unconsumed");
        need(std::none_of(ready_payloads.begin(),ready_payloads.end(),[](const auto& p){return bool(p);}),"cannot drain while selected prefetched payload is unconsumed");
        if(drained)return;
        const auto t=Clock::now();
#ifdef __linux__
        need(fdatasync(file.value)==0,"full GQA end-of-request rawK fdatasync failed");
#else
        // Constructor rejects non-O_DIRECT platforms; keep this TU portable.
        need(fsync(file.value)==0,"full GQA request-end sync failed");
#endif
        ++statistics.drain_calls;statistics.drain_ms+=elapsed(t);drained=true;
    }
    void write_stats(const std::string& path)const{
        json layers=json::array();for(std::size_t l=0;l<statistics.layers.size();++l){const auto& s=statistics.layers[l];layers.push_back({{"layer",l},{"completed_tokens",s.completed_tokens},{"append_calls",s.append_calls},{"append_key_bytes",s.append_key_bytes},{"append_syscall_bytes",s.append_syscall_bytes},{"append_rmw_read_bytes",s.append_rmw_read_bytes},{"score_calls",s.score_calls},{"fpga_calls",s.fpga_calls},{"scored_candidates",s.scored_candidates},{"cold_read_bytes",s.cold_read_bytes},{"query_h2d_bytes",s.query_h2d_bytes},{"score_d2h_bytes",s.score_d2h_bytes},{"hot_fetch_bytes",s.hot_fetch_bytes},{"cold_fetch_bytes",s.cold_fetch_bytes},{"fetch_calls",s.fetch_calls},{"verification_scores",s.verification_scores},{"verification_read_bytes",s.verification_read_bytes},{"history_prefetch_calls",s.history_prefetch_calls},{"current_token_score_calls",s.current_token_score_calls},{"current_token_score_ms",s.current_token_score_ms},{"hot_cpu_packed_key_bytes",s.hot_cpu_packed_key_bytes},{"hot_cpu_calls",s.hot_cpu_calls},{"cold_cpu_calls",s.cold_cpu_calls},{"hpm_updates",s.hpm_updates},{"hpm_promoted_blocks",s.hpm_promoted_blocks},{"hpm_migration_bytes",s.hpm_migration_bytes},{"hpm_ms",s.hpm_ms},{"payload_prefetch_calls",s.payload_prefetch_calls},{"payload_prefetch_bytes",s.payload_prefetch_bytes},{"payload_prefetch_hot_bytes",s.payload_prefetch_hot_bytes},{"payload_prefetch_cold_bytes",s.payload_prefetch_cold_bytes},{"payload_prefetch_syscall_bytes",s.payload_prefetch_syscall_bytes},{"payload_prefetch_read_calls",s.payload_prefetch_read_calls},{"payload_prefetch_useful_bytes",s.payload_prefetch_useful_bytes},{"payload_prefetch_wasted_bytes",s.payload_prefetch_wasted_bytes},{"payload_prefetch_miss_bytes",s.payload_prefetch_miss_bytes},{"payload_prefetch_current_bytes",s.payload_prefetch_current_bytes},{"dense_fetch_read_calls",s.dense_fetch_read_calls},{"dense_fetch_syscall_bytes",s.dense_fetch_syscall_bytes},{"dense_fetch_pread_ns",s.dense_fetch_pread_ns},{"dense_fetch_checksum_ns",s.dense_fetch_checksum_ns}});}
        std::ofstream f(path);need(bool(f),"full GQA statistics output unavailable");
        const char* selector=config.selector_alpha==5.f
            ?"F16_div_sqrt128_head0_max_minus5_inclusive_F16_counts_mean_cap0.2_per_head_IDs"
            :"F16_div_sqrt128_head0_max_minus_alpha_inclusive_F16_counts_mean_cap0.2_per_head_IDs";
        f<<json({{"schema","hillserve-live-full128-runtime-v1"},{"mode",config.mode},{"query_source","live_previous_layer_RMSNorm_target_Wq_target_RoPE_rehearsal_F32_to_F16"},{"raw_score","ordered_c0_to127_FP32_mul_add_no_FMA_final_F16"},{"selector",selector},{"selector_alpha",config.selector_alpha},{ "selector_count_precision",config.fp32_selection_counts?"FP32":"F16"},{"selector_cap_ratio",.2f},{"policy_variant",config.selector_alpha==5.f?"v6_alpha5_cap20":"v6_explicit_alpha_variant_cap20"},{"current_token_forced",false},{"requested_beta",std::isfinite(config.beta)?json(config.beta):json("all_hot")},{"actual_prompt_beta",plan.prompt_cold_tokens?json(double(plan.prompt_hot_tokens)/plan.prompt_cold_tokens):json(nullptr)},{"actual_prompt_rho",double(plan.prompt_hot_tokens)/config.prompt_tokens},{"partial_prompt_enabled",config.allow_partial_prompt},{"prompt_tail_hot_tokens",plan.prompt_tail_hot_tokens},{"hot_prompt_tokens",plan.prompt_hot_tokens},{"cold_prompt_tokens",plan.prompt_cold_tokens},{"recent_pin_tokens",plan.recent_pin_tokens},{"generated_reserved_always_hot",config.generated_capacity},{"capacity",config.capacity},{"layers",layers},{"hot_KV_all_layers_reserved_bytes",statistics.hot_kv_cache_bytes},{"workspace_reserved_bytes",statistics.workspace_reserved_bytes},{"managed_host_reserved_bytes",managed_bytes()},{"host_budget_bytes",config.total_host_workspace_budget_bytes},{"score_file_extent_bytes",statistics.score_file_extent_bytes},{"drain_calls",statistics.drain_calls},{"drain_ms",statistics.drain_ms},{"durability","volatile_query_kv+end_of_request_sync"},{"chunk_tokens",effective_chunk},{"CPU_hot_threads",config.hot_threads},{"CPU_cold_threads",config.cold_threads},{"cpu_implementation_requested",statistics.cpu_implementation_requested},{"CPU_scorer",statistics.cpu_implementation_actual},{"CPU_hot_scorer",statistics.cpu_implementation_actual},{"CPU_cold_scorer",config.mode=="HOST_MATCHED"?statistics.cpu_implementation_actual:"not_used_FPGA"},{"legacy_per_call_thread_creation_included",config.cpu_implementation=="legacy"},{"legacy_hot_full_K_packing_included",config.cpu_implementation=="legacy"},{"legacy_cpu_workspace_reserved_bytes",statistics.legacy_cpu_workspace_bytes},{"verify_scores",config.verify_scores},{"dynamic_HPM",config.hpm},{"HPM_max_swaps",config.hpm_swaps},{"HPM_granularity_tokens",64},{"HPM_decay_updates",8},{"online_beta",false},{"historical_score_prefetch_supported",true},{"payload_prefetch_enabled",config.payload_prefetch},{"score_join_scope","historical_scores_only"},{"payload_join_scope","fetch_waits_for_KV_and_callback_including_GPU_stage"},{"payload_read_window_bytes",config.payload_read_limits.window_bytes},{"payload_read_queue_depth",config.payload_read_queue_depth},{"payload_reader","persistent_bounded_pread_workers_with_page_RMW_fence"},{"payload_checksum","unchanged_FNV1a_interleaved_four_records"},{"dense_layers_use_payload_reader",config.payload_prefetch},{"payload_read_max_gap_bytes",config.payload_read_limits.gap_bytes},{"payload_read_max_amplification_permille",config.payload_read_limits.max_amplification_permille},{"payload_prefetch_reserved_bytes",statistics.payload_prefetch_reserved_bytes},{"payload_prefetch_setup_read_bytes",statistics.payload_prefetch_setup_read_bytes},{"payload_prefetch_setup_read_calls",statistics.payload_prefetch_setup_read_calls},{"payload_prefetch_setup_read_ms",statistics.payload_prefetch_setup_read_ms},{"payload_prefetch_cover",config.fp32_selection_counts?"FP32_count_upper_bound_F16_scores_per_head_history_topk":"F16_threshold_count_upper_bound_per_head_history_topk"},{"payload_prefetch_backing","existing_segmented_v1_complete_KV_O_DIRECT"},{"max_outstanding_history_jobs",2},{"FPGA_calls_serialized",true},{"current_token_completion","independent_ordered_scalar_without_history_engine_lock"},{"GPU_completion_accounted_by_executor",true},{"process_RSS_and_thread_stacks_not_managed_bytes",true},{"physical_NAND_bytes",nullptr}}).dump(2)<<'\n';need(bool(f),"full GQA statistics write failed");
        std::ofstream ef(path+".events.jsonl");need(bool(ef),"full GQA event output unavailable");
        for(const auto& e:events){json chunks=json::array();for(const auto& c:e.chunks)chunks.push_back(chunk_json(c));ef<<json({{"layer",e.layer},{"position",e.position},{"selector_alpha",config.selector_alpha},{ "selector_count_precision",config.fp32_selection_counts?"FP32":"F16"},{"selector_cap_ratio",.2f},{"selection_entered_ns",e.started},{"history_prefetched",e.prefetched},{"history_submitted_ns",e.history_submitted},{"history_joined_ns",e.joined},{"current_score_started_ns",e.current_score_started},{"current_score_finished_ns",e.current_score_finished},{"score_work_started_ns",e.work_started},{"score_work_finished_ns",e.work_ended},{"hot_started_ns",e.hot_start},{"hot_finished_ns",e.hot_end},{"cold_started_ns",e.cold_start},{"cold_finished_ns",e.cold_end},{"all_scores_ready_ns",e.scores_ready},{"IDs_ready_ns",e.ids_ready},{"validation_finished_ns",e.ended},{"cold_K_read_bytes",e.cold_bytes},{"CPU_scorer",statistics.cpu_implementation_actual},{"hot_cpu_packed_key_bytes",e.hot_packed_key_bytes},{"hot_cpu_calls",e.hot_cpu_calls},{"cold_cpu_calls",e.cold_cpu_calls},{"query_H2D_bytes",e.qbytes},{"scores_D2H_bytes",e.sbytes},{"FPGA_calls",e.fpga_calls},{"selected_per_head",e.k},{"physical_union_tokens",e.physical_count},{"threshold_counts",e.counts},{"raw_F16_score_checksum",std::to_string(e.score_hash)},{"ordered_per_head_ID_checksum",std::to_string(e.id_hash)},{"checksum_kind","fnv64_integrity_not_cryptographic"},{"verification_ms",e.verification_ms},{"fetch_started_ns",e.fetch_started},{"fetch_finished_ns",e.fetch_finished},{"hot_fetch_bytes",e.hot_fetch_bytes},{"cold_fetch_bytes",e.cold_fetch_bytes},{"HPM_ms",e.hpm_ms},{"HPM_promote_blocks",e.promote},{"HPM_demote_blocks",e.demote},{"HPM_migration_bytes",e.hpm_migration_bytes},{"payload_prefetch_started_ns",e.payload_started},{"payload_prefetch_ready_ns",e.payload_ready},{"payload_join_started_ns",e.payload_join_started},{"payload_join_finished_ns",e.payload_join_finished},{"payload_read_plan_ns",e.plan_ns},{"payload_hot_copy_and_classify_ns",e.hot_copy_ns},{"payload_IO_lock_wait_ns",e.io_lock_wait_ns},{"payload_extent_check_ns",e.extent_check_ns},{"payload_pread_ns",e.pread_ns},{"payload_validate_copy_ns",e.validate_copy_ns},{"payload_checksum_ns",e.checksum_ns},{"payload_required_page_bytes",e.required_page_bytes},{"payload_gap_read_bytes",e.gap_read_bytes},{"payload_extent_checks",e.extent_checks},{"payload_prefetch_bytes",e.payload_bytes},{"payload_prefetch_hot_bytes",e.payload_hot},{"payload_prefetch_cold_bytes",e.payload_cold},{"payload_prefetch_syscall_bytes",e.payload_syscall},{"payload_prefetch_read_calls",e.payload_reads},{"payload_prefetch_useful_bytes",e.payload_useful},{"payload_prefetch_wasted_bytes",e.payload_wasted},{"payload_prefetch_miss_bytes",e.payload_miss},{"payload_prefetch_current_bytes",e.payload_current},{"chunks",chunks}}).dump()<<'\n';}need(bool(ef),"full GQA event write failed");
    }
    std::size_t managed_bytes()const{return statistics.managed_reserved_bytes;}
};
FullGqaRuntimePolicy::FullGqaRuntimePolicy(const Config& c):impl_(std::make_unique<Impl>(c)){}
FullGqaRuntimePolicy::~FullGqaRuntimePolicy()=default;
void FullGqaRuntimePolicy::append(int l,const std::vector<std::uint64_t>& p,const std::vector<std::uint16_t>& k,const std::vector<std::uint8_t>& v){impl_->append(l,p,k,v);}
void FullGqaRuntimePolicy::prepare_history(int l,std::uint64_t p,const std::vector<float>& q,runtime::GQAPrefetchReady ready){impl_->prepare(l,p,q,std::move(ready));}
std::vector<std::vector<std::uint64_t>> FullGqaRuntimePolicy::select(int l,std::uint64_t p,const std::vector<float>& q){return impl_->select(l,p,q);}
std::vector<std::uint8_t> FullGqaRuntimePolicy::fetch(int l,const std::vector<std::uint64_t>& ids,const ColdFetch& f){return impl_->fetch(l,ids,f);}
void FullGqaRuntimePolicy::drain(){impl_->drain();}
void FullGqaRuntimePolicy::wait_pending()noexcept{for(auto& j:impl_->pending)if(j.second.payload_future.valid())j.second.payload_future.wait();}
std::size_t FullGqaRuntimePolicy::managed_bytes()const{return impl_->managed_bytes();}
std::size_t FullGqaRuntimePolicy::hot_prompt_tokens()const{return impl_->statistics.hot_prompt_tokens;}
std::size_t FullGqaRuntimePolicy::cold_prompt_tokens()const{return impl_->statistics.cold_prompt_tokens;}
bool FullGqaRuntimePolicy::is_hot(std::uint64_t p)const{need(!impl_->config.hpm,"dynamic HPM requires layer-specific hot lookup");return impl_->hot(0,p);}
bool FullGqaRuntimePolicy::is_hot(int l,std::uint64_t p)const{impl_->layer_check(l);return impl_->hot(l,p);}
const FullGqaRuntimePolicy::Stats& FullGqaRuntimePolicy::stats()const{return impl_->statistics;}
void FullGqaRuntimePolicy::write_stats(const std::string& p)const{impl_->write_stats(p);}
}
