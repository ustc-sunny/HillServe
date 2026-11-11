#include "runtime/batch_delivery.hpp"
#include "smartssd/adapter.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <cerrno>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <set>
#include <thread>

// Deliberately separate from the historical 49-channel experiment.  This runner
// reads physical GQA K [token,8,128], never a 32-head transformed key index.
using namespace hillserve;
using namespace hillserve::replay;
namespace {
constexpr std::size_t B=8,P=32768,N=P+1,COLD=P/2,H=32,KH=8,D=128;
constexpr std::size_t KEY_WORDS=KH*D,KEY_BYTES=KEY_WORDS*2,KV_BYTES=KEY_BYTES*2;
constexpr const char* REPRESENTATION="raw_gqa_f16_128";
constexpr const char* QUERY_SOURCE="rehearsal-query-f32.bin_before_skew_mask_round_to_F16";
constexpr const char* POLICY_ID="full_raw_gqa128_head0_threshold_v1";
constexpr const char* V2_ABI="HS_FULL_GQA_F16_V2";
constexpr const char* V3_ABI="HS_FULL_GQA_PIPELINE_F16_V3";

std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
std::size_t slots_for(std::size_t valid){return (std::size_t(std::floor(double(valid)*double(.2f)))+31)/32*32;}

struct DirectFile {
    int fd=-1;
    void* buffer=nullptr;
    std::size_t capacity=0,extent=0;
    std::uint64_t bytes=0,calls=0;
    DirectFile(const std::string& path,std::size_t cap):capacity((cap+4095)/4096*4096){
        fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);
        need(fd>=0,"O_DIRECT existing input required: "+path);
        struct stat st{};
        if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0){close(fd);fd=-1;throw std::runtime_error("regular direct input required");}
        extent=std::size_t(st.st_size);
        if(posix_memalign(&buffer,4096,capacity)){close(fd);fd=-1;throw std::runtime_error("bounded aligned allocation failed");}
    }
    ~DirectFile(){if(fd>=0)close(fd);free(buffer);}
    DirectFile(const DirectFile&)=delete;
    DirectFile& operator=(const DirectFile&)=delete;
    const std::uint8_t* read(std::size_t offset,std::size_t size){
        need(offset%4096==0&&size%4096==0&&offset<=extent&&size<=extent-offset&&size<=capacity,"aligned direct read bounds");
        ssize_t got;
        do{got=pread(fd,buffer,size,offset);}while(got<0&&errno==EINTR);
        need(got==ssize_t(size),"short O_DIRECT read; no buffered fallback");
        bytes+=std::uint64_t(got);++calls;
        return static_cast<const std::uint8_t*>(buffer);
    }
};

// Compile this translation unit with -ffp-contract=off (as the v2 HLS contract
// requires): ascending c, FP32 product/add, one final binary16 rounding.
std::vector<std::uint16_t> cpu_score_raw(const std::vector<std::uint16_t>& q,
        const std::uint16_t* keys,std::size_t n,unsigned threads){
    need(q.size()==H*D&&keys&&n>0&&threads>=1&&threads<=3,"raw GQA CPU score shape/thread budget");
    std::vector<std::uint16_t> out(H*n);
    constexpr std::size_t GROUP=H/KH;
    // Adjacent lanes are independent Q heads, never terms in one reduction.
    // Sharing each decoded physical K across these lanes keeps the ascending
    // channel accumulation order of every score unchanged.
    alignas(16) std::array<std::array<std::array<float,GROUP>,D>,KH> query{};
    for(std::size_t kh=0;kh<KH;++kh)for(std::size_t c=0;c<D;++c)
        for(std::size_t group=0;group<GROUP;++group)
            query[kh][c][group]=half_value(q[(kh*GROUP+group)*D+c]);
    auto worker=[&](unsigned index){
        // Token x physical-head work balances exactly to within one item.
        for(std::size_t item=index;item<n*KH;item+=threads){
            const auto t=item/KH,kh=item%KH;
            const auto* key=keys+t*KEY_WORDS+kh*D;
            alignas(16) std::array<float,GROUP> sum{};
            for(std::size_t c=0;c<D;++c){
                const float value=half_value(key[c]);
                for(std::size_t group=0;group<GROUP;++group)
                    sum[group]+=query[kh][c][group]*value;
            }
            for(std::size_t group=0;group<GROUP;++group)
                out[(kh*GROUP+group)*n+t]=half_bits(sum[group]);
        }
    };
    std::vector<std::thread> jobs;
    try{for(unsigned i=1;i<threads;++i)jobs.emplace_back(worker,i);worker(0);}
    catch(...){for(auto& job:jobs)job.join();throw;}
    for(auto& job:jobs)job.join();
    return out;
}

// Independent scalar oracle walks token -> physical KV head -> grouped Q head.
// Volatile rounding points prevent contraction even if build flags regress.
// It consumes the original physical capture, not the timed score file or any
// old score/ID artifact. It is deliberately excluded from delivery timing.
std::array<std::uint16_t,H> oracle_token(const std::vector<std::uint16_t>& q,
        const std::uint16_t* physical_key){
    std::array<std::uint16_t,H> result{};
    for(std::size_t kh=0;kh<KH;++kh){
        for(std::size_t group=0;group<H/KH;++group){
            const auto h=kh*(H/KH)+group;
            volatile float accumulator=0;
            for(std::size_t c=0;c<D;++c){
                volatile float product=half_value(q[h*D+c])*half_value(physical_key[kh*D+c]);
                accumulator=accumulator+product;
            }
            const auto score=half_bits(accumulator);
            need(std::isfinite(half_value(score)),"full128 oracle score overflow/nonfinite");
            result[h]=score;
        }
    }
    return result;
}

struct Sequence {
    std::filesystem::path trace;
    std::vector<std::uint16_t> q,oracle,hot_keys,raw;
    std::vector<std::uint64_t> expected_ids,physical;
    std::vector<std::uint8_t> hot_kv,payload;
    std::unique_ptr<DirectFile> score,kv;
    policy::GQASelection selection;
    std::uint64_t hot_start=0,hot_end=0,cold_payload=0,hot_payload=0;
};

// Conservative managed reservation, including maximum per-head selected ID
// capacity, full physical payload union, all outstanding hot results, validation
// copies and both batched and independent-B1 host packing workspaces.  This is
// an admission bound, not a claim that all these bytes are resident/useful KV.
std::uint64_t host_reservation(std::size_t slots){
    const auto per_sequence=(N-COLD)*KV_BYTES+(N-COLD)*KEY_BYTES+COLD*KEY_BYTES+
        N*KV_BYTES+H*N*2ULL*3+H*N*8ULL+H*slots*8ULL+N*8ULL+
        H*(N-COLD)*2ULL+H*COLD*2ULL+65536ULL+H*D*2ULL;
    return B*per_sequence+2*slots*24576ULL+128ULL*1024*1024;
}

json prepare_oracle(Sequence& s,const json& entry,std::size_t sequence,
        const std::filesystem::path& out){
    auto t=Clock::now();
    const auto rehearsal=load<float>(s.trace/"rehearsal-query-f32.bin",H*D);
    s.q.resize(H*D);
    for(std::size_t i=0;i<s.q.size();++i){
        need(std::isfinite(rehearsal[i]),"nonfinite untransformed rehearsal query");
        s.q[i]=half_bits(rehearsal[i]);
        need(std::isfinite(half_value(s.q[i])),"query F16 overflow");
    }
    s.hot_kv.resize((N-COLD)*KV_BYTES);
    std::ifstream backing(entry.at("full_KV_file").get<std::string>(),std::ios::binary);
    backing.seekg(COLD*KV_BYTES);
    backing.read(reinterpret_cast<char*>(s.hot_kv.data()),s.hot_kv.size());
    need(bool(backing),"hot KV cache from complete SSD backing");
    s.hot_keys.resize((N-COLD)*KEY_WORDS);
    for(std::size_t tkn=0;tkn<N-COLD;++tkn){
        std::memcpy(s.hot_keys.data()+tkn*KEY_WORDS,s.hot_kv.data()+tkn*KV_BYTES,KEY_BYTES);
    }
    s.score=std::make_unique<DirectFile>(entry.at("raw_cold_K_file"),COLD*KEY_BYTES);
    s.kv=std::make_unique<DirectFile>(entry.at("full_KV_file"),65536);
    need(s.score->extent==COLD*KEY_BYTES&&s.kv->extent==N*KV_BYTES,"raw cold K / complete KV extents");
    const auto* cold_keys=s.score->read(0,COLD*KEY_BYTES);
    need(std::filesystem::file_size(s.trace/"kv-physical.bin")==N*KV_BYTES,"physical oracle capture extent");
    std::ifstream original(s.trace/"kv-physical.bin",std::ios::binary);
    std::array<std::uint16_t,KEY_WORDS*2> token{};
    s.oracle.resize(H*N);
    for(std::size_t tkn=0;tkn<N;++tkn){
        original.read(reinterpret_cast<char*>(token.data()),KV_BYTES);
        need(bool(original),"short physical oracle capture");
        const auto* bytes=reinterpret_cast<const std::uint8_t*>(token.data());
        if(tkn<COLD)need(std::memcmp(bytes,cold_keys+tkn*KEY_BYTES,KEY_BYTES)==0,"raw cold K differs bitwise from physical capture");
        else need(std::memcmp(bytes,s.hot_kv.data()+(tkn-COLD)*KV_BYTES,KV_BYTES)==0,"hot KV differs bitwise from physical capture");
        const auto scores=oracle_token(s.q,token.data());
        for(std::size_t h=0;h<H;++h)s.oracle[h*N+tkn]=scores[h];
    }
    // Discard the temporary validation copy. Every timed CPU round overwrites
    // this buffer through O_DIRECT; the FPGA path reads into its P2P BO instead.
    // Keep the same admitted allocation in both methods, without retained K.
    std::memset(s.score->buffer,0,s.score->capacity);
    const auto selected=policy::select_gqa_scores(scaled(s.oracle),H,N,N,5,.2f);
    need(selected.k>0&&selected.k<=slots_for(N),"new full128 oracle selection outside GPU slot bound");
    s.expected_ids=flatten(selected);
    s.raw.reserve(H*N);s.physical.reserve(N);s.payload.reserve(N*KV_BYTES);
    const auto prefix="full-oracle-seq"+std::to_string(sequence);
    save(out/(prefix+"-query-f16.bin"),s.q);
    save(out/(prefix+"-score-f16.bin"),s.oracle);
    save(out/(prefix+"-selected-ids.bin"),s.expected_ids);
    return {{"sequence",sequence},{"status","passed_independent_raw128_oracle"},
        {"query_source",QUERY_SOURCE},{"query_F16_checksum",std::to_string(checksum(s.q.data(),s.q.size()*2))},
        {"raw_score_checksum",std::to_string(checksum(s.oracle.data(),s.oracle.size()*2))},
        {"ordered_ID_checksum",std::to_string(checksum(s.expected_ids.data(),s.expected_ids.size()*8))},
        {"selected_per_head",selected.k},{"cold_K_bit_errors",0},{"hot_KV_bit_errors",0},
        {"cold_K_bytes_checked",COLD*KEY_BYTES},{"hot_KV_bytes_checked",(N-COLD)*KV_BYTES},
        {"cold_K_validation_buffer_bytes",s.score->capacity},
        {"cold_K_validation_buffer_zeroed_after_oracle",true},
        {"cold_K_validation_buffer_role","temporary_validation_then_CPU_O_DIRECT_destination_not_a_cache"},
        {"oracle_prepare_ms_excluded",milliseconds(t)},{"old_partial_scores_or_IDs_used",false}};
}

void gather(Sequence& s){
    std::set<std::uint64_t> ids;
    for(const auto& head:s.selection.ids)ids.insert(head.begin(),head.end());
    need(ids.size()<=N,"physical selected union bound");
    s.physical.assign(ids.begin(),ids.end());s.payload.resize(s.physical.size()*KV_BYTES);
    s.cold_payload=s.hot_payload=0;
    for(std::size_t i=0;i<s.physical.size();){
        const auto id=s.physical[i];
        need(id<N,"selected physical position out of range");
        if(id>=COLD){
            std::memcpy(s.payload.data()+i*KV_BYTES,s.hot_kv.data()+(id-COLD)*KV_BYTES,KV_BYTES);
            s.hot_payload+=KV_BYTES;++i;continue;
        }
        auto j=i+1;
        while(j<s.physical.size()&&s.physical[j]<COLD&&(s.physical[j]-id+1)*KV_BYTES<=65536)++j;
        const auto* bytes=s.kv->read(id*KV_BYTES,(s.physical[j-1]-id+1)*KV_BYTES);
        for(auto k=i;k<j;++k){
            std::memcpy(s.payload.data()+k*KV_BYTES,bytes+(s.physical[k]-id)*KV_BYTES,KV_BYTES);
            s.cold_payload+=KV_BYTES;
        }
        i=j;
    }
}

void software_check(){
    // The delivery endpoint must apply the selected ABI before loading any
    // image. In particular V3 never pads the candidate set to its tile size.
    validate_raw_gqa_shape(V2_ABI,H,KH,1,D);
    validate_raw_gqa_shape(V2_ABI,H,KH,COLD,D);
    validate_raw_gqa_shape(V3_ABI,H,KH,64,D);
    validate_raw_gqa_shape(V3_ABI,H,KH,COLD,D);
    validate_raw_gqa_shape(V3_ABI,H,KH,32768,D);
    std::size_t rejected_shapes=0;
    for(const auto n:{std::size_t(0),std::size_t(1),std::size_t(63),std::size_t(65),std::size_t(32769)}){
        try{validate_raw_gqa_shape(V3_ABI,H,KH,n,D);}
        catch(const std::invalid_argument&){++rejected_shapes;}
    }
    need(rejected_shapes==5,"V3 delivery shape gate accepted unsupported candidates");
    constexpr std::size_t n=64,cold=32;
    std::vector<std::uint16_t> q(H*D),keys(n*KEY_WORDS),reference(H*n);
    for(std::size_t i=0;i<q.size();++i)q[i]=half_bits(float(int(i%29)-14)/16.f);
    for(std::size_t i=0;i<keys.size();++i)keys[i]=half_bits(float(int((i*7+i/KEY_WORDS)%43)-21)/32.f);
    for(std::size_t t=0;t<n;++t){auto row=oracle_token(q,keys.data()+t*KEY_WORDS);for(std::size_t h=0;h<H;++h)reference[h*n+t]=row[h];}
    const auto whole=cpu_score_raw(q,keys.data(),n,3);
    const auto a=cpu_score_raw(q,keys.data(),cold,3),b=cpu_score_raw(q,keys.data()+cold*KEY_WORDS,n-cold,1);
    need(whole==reference,"64-token raw128 CPU differs from independent scalar oracle");
    for(std::size_t h=0;h<H;++h)for(std::size_t t=0;t<n;++t)need(reference[h*n+t]==(t<cold?a[h*cold+t]:b[h*(n-cold)+t-cold]),"hot/cold global score placement mismatch");
    const auto choice=policy::select_gqa_scores(scaled(reference),H,n,n,5,.2f);
    need(choice.k>0&&choice.k<=slots_for(n)&&slots_for(N)==6560&&slots_for(N)<=6592,"64/32K GPU slot bound");
    need(host_reservation(slots_for(N))<=4ULL*1024*1024*1024,"full128 B8 host reservation over4GiB");
    std::cout<<json({{"status","passed"},{"evidence_class","software_fixture_test_not_hardware_result"},
        {"fixture_tokens",n},{"score_comparisons",H*n},{"raw128_bit_errors",0},
        {"V3_invalid_shapes_rejected",rejected_shapes},
        {"cold_hot_merge_bit_errors",0},{"B8_32K_slots",slots_for(N)},
        {"B8_32K_host_reservation_bytes",host_reservation(slots_for(N))}}).dump()<<'\n';
}
}

int main(int argc,char** argv){
    if(argc==2&&std::string(argv[1])=="--software-check"){
        try{software_check();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc!=4){std::cerr<<"MANIFEST CPU_RETURN|HILLSERVE|FPGA_STAGED NEW_OUTPUT | --software-check\n";return 2;}
    const std::filesystem::path out=argv[3];
    if(!std::filesystem::create_directory(out))return 2;
    try{
        json cfg;std::ifstream(argv[1])>>cfg;
        const std::string method=argv[2];
        const bool direct=method=="HILLSERVE",staged=method=="FPGA_STAGED",fpga=direct||staged;
        need(fpga||method=="CPU_RETURN","unknown method");
        // Old manifests retain V2. An explicit image ABI may corroborate but
        // must never silently override the top-level comparison contract.
        const std::string fpga_abi=cfg.value("fpga_abi",std::string(V2_ABI));
        validate_raw_gqa_shape(fpga_abi,H,KH,COLD,D);
        if(cfg.contains("fpga_image")&&cfg["fpga_image"].contains("abi"))
            need(cfg["fpga_image"]["abi"]==fpga_abi,"top-level and image FPGA ABI disagree");
        need(cfg.at("B")==B&&cfg.at("P")==P&&cfg.at("cold_prompt_positions")==COLD&&cfg.at("sequences").size()==B,"explicit actual B8 32K half-cold manifest");
        need(cfg.at("score_representation")==REPRESENTATION,"explicit raw_gqa_f16_128 manifest required");
        need(cfg.at("score_dimensions")==D&&cfg.at("scoring_Q_heads")==H&&cfg.at("physical_KV_heads")==KH&&cfg.at("query_source")==QUERY_SOURCE,"explicit raw128 GQA geometry/query source required");
        need(cfg.at("policy_id")==POLICY_ID&&cfg.at("policy_alpha")==5&&cfg.at("selection_cap_ratio")==.2,"full128 runner keeps the declared alpha5 cap0.2 selector");
        need(cfg.at("first_decode_new_token_placement")=="host_hot","explicit first decode hot placement required");
        need(cfg.at("CPU_thread_budget")==4,"same four-CPU budget required");
        const std::uint64_t host_budget=cfg.at("pilot_host_budget_bytes"),gpu_budget=cfg.at("GPU_workspace_budget_bytes");
        const auto slots=slots_for(N);const auto reserved=host_reservation(slots);
        need(host_budget<=4ULL*1024*1024*1024&&reserved<=host_budget,"bounded full128 host admission");
        need(gpu_budget<=2ULL*1024*1024*1024&&B*slots*24576ULL+16*1024*1024<=gpu_budget,"bounded full128 GPU admission");
        std::unique_ptr<FpgaScoreBackend> device;
        if(fpga){
            need(cfg.contains("fpga_image")&&cfg["fpga_image"].value("execute_authorized",false),"new raw128 FPGA image is not authorized; no old-image/CPU fallback");
            const auto& image=cfg["fpga_image"];HardwareContract hc;
            hc.abi=fpga_abi;hc.image_execute_authorized=true;
            hc.image_path=image.at("path").get<std::string>();hc.image_sha256=image.at("sha256").get<std::string>();
            hc.reuse_file_buffers=direct;
            device=make_fpga_backend(direct?HardwareMode::fpga_storage_streaming:HardwareMode::fpga_host_staged,hc);
        }
        std::array<Sequence,B> seq;std::vector<float> actual;json oracles=json::array();
        for(std::size_t i=0;i<B;++i){
            const auto& entry=cfg["sequences"][i];need(entry.at("sequence")==i,"manifest sequence identity/order");
            auto& s=seq[i];s.trace=entry.at("trace").get<std::string>();json capture;std::ifstream(s.trace/"manifest.json")>>capture;
            need(capture.at("status")=="passed"&&capture.at("P")==P&&capture.at("valid_candidates")==N&&capture.at("layer")==8&&capture.at("physical_KV_heads")==KH&&capture.at("head_dim")==D,"verified original Gradient32K layer8 checkpoint required");
            auto q=load<float>(s.trace/"actual-attention-query-f32.bin",H*D);actual.insert(actual.end(),q.begin(),q.end());
            oracles.push_back(prepare_oracle(s,entry,i,out));
        }
        std::ofstream(out/"oracle-manifest.json")<<json({{"status","passed"},{"score_representation",REPRESENTATION},{"query_source",QUERY_SOURCE},{"policy_id",POLICY_ID},{"score_accumulation","ascending_c_FP32_mul_add_then_F16"},{"normalization","raw_F16_div_sqrt128_then_F16"},{"selection","existing_head0_max_inclusive_threshold_F16_counts_mean_cap20pct_position_ties"},{"preparation_excluded_from_delivery",true},{"sequences",oracles}}).dump(2)<<'\n';
        GQABatchDeliveryWorkspace gpu(B,slots,actual,gpu_budget);
        std::ofstream records(out/"runs.jsonl");need(bool(records),"cannot create new result stream");
        for(int round=0;round<2;++round){
            gpu.begin_round();std::array<std::promise<std::vector<std::uint16_t>>,B> promises;
            std::array<std::future<std::vector<std::uint16_t>>,B> futures;
            for(std::size_t i=0;i<B;++i)futures[i]=promises[i].get_future();
            const auto ready=now_ns();
            auto hot_job=std::async(std::launch::async,[&]{
                for(std::size_t i=0;i<B;++i){try{
                    auto& s=seq[i];s.hot_start=now_ns();auto scores=cpu_score_raw(s.q,s.hot_keys.data(),N-COLD,1);s.hot_end=now_ns();promises[i].set_value(std::move(scores));
                }catch(...){promises[i].set_exception(std::current_exception());}}
            });
            json rows=json::array();
            for(std::size_t i=0;i<B;++i){
                auto& s=seq[i];const auto kv_before=s.kv->bytes;std::vector<std::uint16_t> scores;
                FileDeviceScores hw;double read_ms=0,cpu_ms=0,staged_copy_ms=0;
                std::uint64_t read_bytes=0,staged_copy_bytes=0,key_h2d_bytes=0;
                const auto cold_start=now_ns();
                if(direct){
                    hw=device->score_file_raw_gqa(s.q,{s.score->fd,0,COLD*KEY_BYTES,false},H,KH,COLD,D);
                    need(hw.kernel_completed&&hw.p2p_allocation_verified&&!hw.resident_keys
                         &&hw.file_read_bytes==COLD*KEY_BYTES&&hw.key_host_staging_bytes==0
                         &&hw.query_h2d_bytes==H*D*2&&hw.timings.d2h_bytes==H*COLD*2,
                         "raw128 FPGA must perform complete same-file direct read and hardware score");
                    scores=std::move(hw.values);read_ms=hw.file_read_ms;read_bytes=hw.file_read_bytes;
                }else{
                    const auto before=s.score->bytes;auto t=Clock::now();
                    const auto* keys=reinterpret_cast<const std::uint16_t*>(s.score->read(0,COLD*KEY_BYTES));read_ms=milliseconds(t);read_bytes=s.score->bytes-before;
                    if(staged){
                        // Explicit diagnostic control: same cold file and
                        // exact FPGA image, with a real host K staging vector.
                        // Its allocation/copy and backend H2D are timed, never
                        // reclassified as direct or resident scoring.
                        t=Clock::now();std::vector<std::uint16_t> key_copy(keys,keys+COLD*KEY_WORDS);
                        staged_copy_ms=milliseconds(t);staged_copy_bytes=COLD*KEY_BYTES;
                        auto result=device->score_raw_gqa(s.q,key_copy,H,KH,COLD,D);
                        need(result.timings.h2d_bytes==H*D*2+COLD*KEY_BYTES
                             &&result.timings.d2h_bytes==H*COLD*2,"staged FPGA transfer byte contract");
                        scores=std::move(result.values);hw.timings=result.timings;
                        // The backend returns only after ERT completion and
                        // score sync. It does not reuse staged BOs today.
                        hw.kernel_completed=true;hw.key_host_staging_bytes=COLD*KEY_BYTES;
                        hw.query_h2d_bytes=H*D*2;key_h2d_bytes=COLD*KEY_BYTES;
                    }else{
                        t=Clock::now();scores=cpu_score_raw(s.q,keys,COLD,3);cpu_ms=milliseconds(t);
                    }
                }
                const auto cold_end=now_ns();const auto hot_wait_begin=Clock::now();auto hot=futures[i].get();
                const auto hot_wait_ms=milliseconds(hot_wait_begin);const auto all_scores_ready=now_ns();
                need(scores.size()==H*COLD&&hot.size()==H*(N-COLD),"full128 score output shape");
                s.raw.resize(H*N);auto t=Clock::now();
                for(std::size_t h=0;h<H;++h){std::copy_n(scores.data()+h*COLD,COLD,s.raw.data()+h*N);std::copy_n(hot.data()+h*(N-COLD),N-COLD,s.raw.data()+h*N+COLD);}
                auto normalized=scaled(s.raw);const auto merge_ms=milliseconds(t);t=Clock::now();
                s.selection=policy::select_gqa_scores(normalized,H,N,N,5,.2f);
                need(s.selection.k>0&&s.selection.k<=slots,"full128 selection exceeds admitted workspace or is empty; no fallback");
                const auto selection_ms=milliseconds(t);const auto id_ready=now_ns();
                t=Clock::now();gather(s);const auto gather_ms=milliseconds(t);const auto host_ready=now_ns();
                const auto delivered=gpu.deliver(i,s.selection.ids,P,s.physical,s.payload);
                rows.push_back({{"sequence",i},{"fpga_abi",fpga_abi},
                    {"query_ready_ns",ready},{"cold_started_ns",cold_start},{"cold_finished_ns",cold_end},
                    {"hot_started_ns",s.hot_start},{"hot_finished_ns",s.hot_end},
                    {"all_required_scores_CPU_ready_ns",all_scores_ready},
                    {"global_ID_ready_ns",id_ready},{"KV_host_ready_ns",host_ready},
                    {"KV_mask_GPU_ready_ns",delivered.gpu_ready_ns},
                    {"sequence_T_delivery_ms",(delivered.gpu_ready_ns-ready)/1e6},
                    {"cold_file_read_ms",read_ms},{"CPU_cold_score_ms",cpu_ms},
                    {"CPU_hot_score_ms",double(s.hot_end-s.hot_start)/1e6},{"wait_for_hot_score_ms",hot_wait_ms},
                    {"staged_key_vector_copy_ms",staged_copy_ms},
                    {"XRT_initialization_ms_excluded",hw.timings.initialization_ms},
                    {"XRT_allocation_ms",hw.timings.allocation_ms},{"XRT_pack_ms",hw.timings.pack_ms},
                    {"XRT_H2D_ms",hw.timings.h2d_ms},
                    {"XRT_H2D_scope",direct?"query_only":staged?"query_and_cold_K":"none"},
                    {"XRT_submission_ms",hw.timings.submission_ms},{"XRT_host_wait_ms",hw.timings.host_wait_ms},
                    {"FPGA_launch_wait_ms",hw.timings.launch_wait_ms},
                    {"FPGA_launch_started_ns",hw.timings.launch_started_ns},
                    {"FPGA_launch_finished_ns",hw.timings.launch_finished_ns},
                    {"FPGA_score_return_ms",hw.timings.d2h_ms},{"XRT_release_ms",hw.timings.release_ms},
                    {"XRT_total_ms",hw.timings.total_ms},
                    {"merge_ms",merge_ms},{"selection_ms",selection_ms},{"KV_gather_ms",gather_ms},
                    {"GPU_pack_ms",delivered.pack_ms},{"GPU_upload_ms",delivered.upload_wall_ms},
                    {"GPU_copy_stream_interval_ms",delivered.cuda_copy_stream_interval_ms},
                    {"cold_key_read_bytes",read_bytes},{"cold_key_SSD_to_host_bytes",direct?0:read_bytes},
                    {"cold_key_host_to_FPGA_bytes",key_h2d_bytes},
                    {"cold_key_SSD_to_P2P_bytes",direct?read_bytes:0},
                    {"staged_key_vector_copy_bytes",staged_copy_bytes},
                    {"key_host_staging_bytes",hw.key_host_staging_bytes},
                    {"scores_FPGA_to_host_bytes",hw.timings.d2h_bytes},
                    {"query_host_to_FPGA_bytes",hw.query_h2d_bytes},{"XRT_H2D_bytes",hw.timings.h2d_bytes},
                    {"selected_cold_KV_bytes",s.cold_payload},{"selected_hot_KV_bytes",s.hot_payload},
                    {"selected_KV_SSD_read_bytes",s.kv->bytes-kv_before},
                    {"GPU_upload_bytes",delivered.upload_bytes},{"selected_per_head",s.selection.k},
                    {"P2P_BO_verified",hw.p2p_allocation_verified},{"FPGA_kernel_completed",hw.kernel_completed},
                    {"BO_reused",hw.bo_reused},{"resident_keys",hw.resident_keys},
                    {"BO_flags",hw.bo_flags},{"BO_memory_group",hw.bo_memory_group},
                    {"kernel_key_bank",hw.kernel_key_bank}});
            }
            const auto end=gpu.batch_ready_ns();hot_job.get();json verification=json::array();
            for(std::size_t i=0;i<B;++i){
                auto& s=seq[i];need(s.raw==s.oracle,"full128 scores differ from independent original-capture oracle");
                need(flatten(s.selection)==s.expected_ids,"full128 ordered IDs differ from new oracle selection");
                std::ifstream original(s.trace/"kv-physical.bin",std::ios::binary);std::array<std::uint8_t,KV_BYTES> token{};
                for(std::size_t j=0;j<s.physical.size();++j){original.seekg(s.physical[j]*KV_BYTES);original.read(reinterpret_cast<char*>(token.data()),KV_BYTES);need(bool(original)&&std::memcmp(token.data(),s.payload.data()+j*KV_BYTES,KV_BYTES)==0,"selected physical KV byte mismatch");}
                auto v=gpu.verify_sequence(i,s.selection.ids,P,s.physical,s.payload);
                v["score_bit_errors"]=0;v["ordered_ID_errors"]=0;v["KV_byte_errors"]=0;
                v["oracle_kind"]="independent_raw128_original_physical_capture";
                v["raw_score_checksum"]=std::to_string(checksum(s.raw.data(),s.raw.size()*2));
                v["ordered_ID_checksum"]=std::to_string(checksum(s.expected_ids.data(),s.expected_ids.size()*8));
                v["physical_ID_checksum"]=std::to_string(checksum(s.physical.data(),s.physical.size()*8));
                v["selected_KV_payload_checksum"]=std::to_string(checksum(s.payload.data(),s.payload.size()));
                v["scores_checked"]=s.raw.size();v["selected_KV_bytes_checked"]=s.payload.size();
                verification.push_back(v);
            }
            auto attention=gpu.verify_and_consume();json independent=json::array();
            if(round==0){
                for(std::size_t i=0;i<B;++i){std::vector<float> q(actual.begin()+i*H*D,actual.begin()+(i+1)*H*D);GQADeliveryWorkspace single(slots,q);auto& s=seq[i];single.deliver(s.selection.ids,P,s.physical,s.payload);auto check=single.verify_and_consume();need(check["actual_attention_output_checksum"]==attention["per_sequence_attention_output_checksums"][i],"full128 B8 attention differs from independent B1 graph");independent.push_back({{"sequence",i},{"status","passed_bit_identical_to_B1_graph"},{"attention_output_checksum",check["actual_attention_output_checksum"]}});}
            }
            struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
            json result={{"status","passed"},{"phase",round?"measured":"warmup"},{"method",method},{"classification","actual_B8_single_layer_retrieval_and_batched_GPU_attention"},{"score_representation",REPRESENTATION},{"score_dimensions",D},{"scoring_Q_heads",H},{"physical_KV_heads",KH},{"query_source",QUERY_SOURCE},{"policy_id",POLICY_ID},{"policy_alpha",5},{"selection_cap_ratio",.2},{"score_accumulation","ascending_c_FP32_mul_add_then_F16"},{"normalization","raw_F16_div_sqrt128_then_F16"},{"head0_max_broadcast",true},{"inclusive_threshold",true},{"B",B},{"P",P},{"layer",8},{"valid_candidates_per_sequence",N},{"cold_positions_per_sequence",COLD},{"hot_positions_per_sequence",N-COLD},{"cold_fraction_prompt",.5},{"cold_fraction_valid",double(COLD)/N},{"query_ready_ns",ready},{"all_sequences_GPU_ready_ns",end},{"T_batch_delivery_ms",(end-ready)/1e6},{"sequence_events",rows},{"verification",verification},{"GPU_attention",attention},{"independent_B1_graph_gate",independent},{"host_reservation_bytes",reserved},{"GPU_workspace_bytes",gpu.gpu_bytes()},{"GPU_padded_slots_per_sequence",slots},{"host_peak_RSS_KiB",usage.ru_maxrss},{"preprocessing_and_oracle_excluded_from_delivery",true},{"old_partial_scores_or_IDs_used",false},{"whole_model_feedback",false},{"TPOT_ms",nullptr},{"GPU_actual_wait_ms",nullptr}};
            // This image identity is the comparison contract for all methods;
            // FPGA_executed distinguishes CPU control from actual board work.
            result["fpga_abi"]=fpga_abi;result["FPGA_executed"]=fpga;
            result["fpga_image_sha256"]=(cfg.contains("fpga_image")&&cfg["fpga_image"].contains("sha256")
                &&cfg["fpga_image"]["sha256"].is_string())?cfg["fpga_image"]["sha256"].get<std::string>():std::string();
            result["cold_scoring_path"]=direct?"file_to_P2P_BO_to_FPGA":staged?"file_to_host_to_FPGA":"file_to_host_CPU";
            result["FPGA_BO_reuse_policy"]=direct?"reuse_matching_file_buffers":staged?"allocate_staged_BOs_per_call":"not_applicable";
            result["CPU_hot_scoring_threads"]=1;result["CPU_cold_scoring_threads"]=fpga?0:3;
            result["CPU_thread_budget"]=4;
            result["pilot_host_budget_bytes"]=host_budget;result["GPU_workspace_budget_bytes"]=gpu_budget;
            result["GPU_host_staging_bytes"]=gpu.host_staging_bytes();
            result["timing_clock"]="std::chrono::steady_clock_nanoseconds";
            result["delivery_endpoint"]="selected_K_V_and_mask_uploaded_CUDA_completion_event_synchronized";
            result["physical_NVMe_DMA_without_host_bounce_verified"]=false;
#ifdef __APPLE__
            const auto rss_bytes=std::uint64_t(usage.ru_maxrss);
#else
            const auto rss_bytes=std::uint64_t(usage.ru_maxrss)*1024;
#endif
            result["host_peak_RSS_bytes"]=rss_bytes;
            need(rss_bytes<=host_budget,"actual full128 delivery host RSS exceeds the shared budget");
            records<<result.dump()<<'\n';records.flush();need(bool(records),"raw128 result write failed");std::cout<<result.dump()<<'\n';
        }
        return 0;
    }catch(const std::exception& e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
