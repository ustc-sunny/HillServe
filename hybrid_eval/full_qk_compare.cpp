#include "policy/host_policy.hpp"
#include "smartssd/adapter.hpp"
#include "storage/file_store.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace hillserve;
namespace {
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
constexpr std::size_t B=8,P=32768,COLD=16384,H=32,KH=8,D=128,GROUP=H/KH;
constexpr std::size_t KEY_WORDS=KH*D,KEY_BYTES=KEY_WORDS*2,KV_BYTES=KEY_BYTES*2;
constexpr unsigned CPU_THREADS=4;
constexpr std::uint64_t HOST_LIMIT=1ULL<<30,BO_LIMIT=256ULL<<20;
constexpr std::size_t INPUT_BYTES=COLD*KV_BYTES,SCORE_BYTES=H*COLD*2;
constexpr const char* QUERY_SOURCE="rehearsal-query-f32.bin_before_skew_mask_round_to_F16";
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
json read_json(const std::filesystem::path& path){std::ifstream file(path);need(bool(file),"missing JSON: "+path.string());json result;file>>result;return result;}
template<class T> std::vector<T> load(const std::filesystem::path& path,std::size_t count){
    need(std::filesystem::file_size(path)==count*sizeof(T),"actual capture extent mismatch: "+path.string());
    std::ifstream file(path,std::ios::binary);std::vector<T> data(count);
    file.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(T));need(bool(file),"short actual capture");return data;
}
template<class T> void save(const std::filesystem::path& path,const std::vector<T>& data){
    need(!std::filesystem::exists(path),"new output only: "+path.string());
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    need(bool(file),"private score evidence write failed");
}
struct InputBuffer {
    void* data=nullptr;
    InputBuffer(){need(posix_memalign(&data,4096,INPUT_BYTES)==0,"bounded64MiB aligned buffer allocation");std::memset(data,0,INPUT_BYTES);}
    ~InputBuffer(){free(data);}
    InputBuffer(const InputBuffer&)=delete;
    InputBuffer& operator=(const InputBuffer&)=delete;
};
struct DirectInput {
    int fd=-1;
    std::uint64_t extent=0,inode=0,device=0;
    std::string path;
    explicit DirectInput(const std::string& file,std::uint64_t expected):path(file){
#ifdef O_DIRECT
        fd=open(file.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_DIRECT);
#else
        throw std::runtime_error("O_DIRECT unavailable; no buffered fallback");
#endif
        need(fd>=0,"existing O_DIRECT file required: "+file);
        struct stat st{};
        if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||std::uint64_t(st.st_size)!=expected){close(fd);fd=-1;throw std::runtime_error("direct file identity/extent mismatch: "+file);}
        extent=st.st_size;inode=st.st_ino;device=st.st_dev;
    }
    ~DirectInput(){if(fd>=0)close(fd);}
    DirectInput(const DirectInput&)=delete;
    DirectInput& operator=(const DirectInput&)=delete;
    std::uint64_t read(void* destination,std::size_t bytes){
        need(bytes%4096==0&&bytes<=extent&&bytes<=INPUT_BYTES&&reinterpret_cast<std::uintptr_t>(destination)%4096==0,"bounded aligned direct input read");
        ssize_t got;do{got=pread(fd,destination,bytes,0);}while(got<0&&errno==EINTR);
        need(got==ssize_t(bytes),"short O_DIRECT read; no fallback");return std::uint64_t(got);
    }
    json identity()const{return {{"path",path},{"bytes",extent},{"inode",inode},{"st_dev",device},{"O_DIRECT",true},{"open_mode","read_only_existing_file"}};}
};

// Identical ordered FP32 arithmetic to the current grouped CPU baseline. Only
// the physical token stride varies: K-only1024 half words, interleaved K/V2048.
// Four independent heads are vectorizable lanes, never a reordered reduction.
std::vector<std::uint16_t> score_cpu(const std::vector<std::uint16_t>& q,
        const std::uint16_t* input,std::size_t n,std::size_t token_stride,unsigned threads){
    need(q.size()==H*D&&input&&n>0&&(token_stride==KEY_WORDS||token_stride==2*KEY_WORDS)
         &&threads>=1&&threads<=CPU_THREADS,"raw128 physical GQA CPU shape/stride/thread bound");
    alignas(16) std::array<std::array<std::array<float,GROUP>,D>,KH> query{};
    for(std::size_t kh=0;kh<KH;++kh)for(std::size_t c=0;c<D;++c)for(std::size_t group=0;group<GROUP;++group)
        query[kh][c][group]=half_value(q[(kh*GROUP+group)*D+c]);
    std::vector<std::uint16_t> result(H*n);
    auto worker=[&](unsigned index){
        for(std::size_t item=index;item<n*KH;item+=threads){
            const auto t=item/KH,kh=item%KH;const auto* key=input+t*token_stride+kh*D;
            alignas(16) std::array<float,GROUP> sum{};
            for(std::size_t c=0;c<D;++c){const float value=half_value(key[c]);
                for(std::size_t group=0;group<GROUP;++group)sum[group]+=query[kh][c][group]*value;
            }
            for(std::size_t group=0;group<GROUP;++group)result[(kh*GROUP+group)*n+t]=half_bits(sum[group]);
        }
    };
    std::vector<std::thread> workers;
    try{for(unsigned i=1;i<threads;++i)workers.emplace_back(worker,i);worker(0);}
    catch(...){for(auto& thread:workers)thread.join();throw;}
    for(auto& thread:workers)thread.join();
    return result;
}

// Independent token-first scalar oracle; volatile FP32 rounding points also
// prevent accidental fused operations in a build with changed compiler flags.
std::array<std::uint16_t,H> oracle_token(const std::vector<std::uint16_t>& q,const std::uint16_t* key){
    std::array<std::uint16_t,H> result{};
    for(std::size_t kh=0;kh<KH;++kh)for(std::size_t group=0;group<GROUP;++group){
        const auto h=kh*GROUP+group;volatile float sum=0;
        for(std::size_t c=0;c<D;++c){volatile float product=half_value(q[h*D+c])*half_value(key[kh*D+c]);sum=sum+product;}
        const auto half=half_bits(sum);need(std::isfinite(half_value(half)),"independent raw128 score overflow");result[h]=half;
    }
    return result;
}
struct Sequence {
    std::vector<std::uint16_t> q,oracle;
    std::unique_ptr<DirectInput> raw,full;
    std::string trace;
};
constexpr std::uint64_t reservation(){
    // Shared64MiB input, eight oracles, eight retained outputs, temporary output,
    // queries, optional XRT mapped BO upper bound, and64MiB runtime/metadata slack.
    return INPUT_BYTES+2ULL*B*SCORE_BYTES+SCORE_BYTES+B*H*D*2ULL+BO_LIMIT+(64ULL<<20);
}
json prepare(Sequence& seq,const json& entry,std::size_t index,const std::filesystem::path& out){
    auto begin=Clock::now();seq.trace=entry.at("trace").get<std::string>();
    const std::filesystem::path trace(seq.trace);const auto captured=read_json(trace/"manifest.json");
    need(captured.at("status")=="passed"&&captured.at("P")==P&&captured.at("layer")==8
         &&captured.at("valid_candidates")==P+1&&captured.at("heads")==H
         &&captured.at("physical_KV_heads")==KH&&captured.at("head_dim")==D,"verified real Gradient8B capture required");
    const auto source=load<float>(trace/"rehearsal-query-f32.bin",H*D);seq.q.resize(H*D);
    for(std::size_t i=0;i<source.size();++i){need(std::isfinite(source[i]),"nonfinite actual query");seq.q[i]=half_bits(source[i]);need(std::isfinite(half_value(seq.q[i])),"actual query F16 overflow");}
    seq.raw=std::make_unique<DirectInput>(entry.at("raw_cold_K_file"),COLD*KEY_BYTES);
    seq.full=std::make_unique<DirectInput>(entry.at("full_KV_file"),(P+1)*KV_BYTES);
    need(seq.raw->device==seq.full->device,"raw K and full KV must reside on the same device");
    struct stat destination{};
    need(stat(out.c_str(),&destination)==0&&std::uint64_t(destination.st_dev)!=seq.raw->device,
         "component evidence must not be written to the input SmartSSD device");
    need(std::filesystem::file_size(trace/"kv-physical.bin")==std::uint64_t(P+1)*KV_BYTES,"original physical KV extent");
    std::ifstream original(trace/"kv-physical.bin",std::ios::binary);
    std::array<std::uint16_t,KEY_WORDS*2> row{};seq.oracle.resize(H*COLD);
    for(std::size_t t=0;t<COLD;++t){original.read(reinterpret_cast<char*>(row.data()),KV_BYTES);need(bool(original),"short original physical capture");
        auto reference=oracle_token(seq.q,row.data());for(std::size_t h=0;h<H;++h)seq.oracle[h*COLD+t]=reference[h];}
    const auto prefix="seq"+std::to_string(index);
    save(out/(prefix+"-query-f16.bin"),seq.q);save(out/(prefix+"-oracle-raw-score-f16.bin"),seq.oracle);
    return {{"sequence",index},{"trace",seq.trace},{"query_source",QUERY_SOURCE},
        {"query_checksum",std::to_string(checksum(seq.q.data(),seq.q.size()*2))},
        {"oracle_checksum",std::to_string(checksum(seq.oracle.data(),seq.oracle.size()*2))},
        {"oracle_score_count",seq.oracle.size()},{"raw_K_file",seq.raw->identity()},{"full_KV_file",seq.full->identity()},
        {"preparation_ms_excluded",elapsed(begin)},{"old_partial_score_oracle_used",false}};
}
struct Event {
    std::uint64_t start=0,ready=0,read_bytes=0;
    double read_ms=0,cpu_ms=0;
    ScoreDeviceTimings xrt;
    std::uint64_t query_bytes=0,flags=0,address=0,staging_bytes=0;
    unsigned bank=0;
    bool p2p=false,reused=false,completed=false,resident=false;
};

void software_check(){
    std::uint64_t comparisons=0;
    const std::array<std::uint16_t,12> values={0x0001,0x8001,0x0400,0x8400,0x3555,0xb555,
                                              0x3c01,0xbc01,0x43ff,0xc3ff,0x4800,0xc800};
    for(std::size_t n:{1U,63U,64U,67U,259U}){
        std::vector<std::uint16_t> q(H*D),raw(n*KEY_WORDS),kv(n*KEY_WORDS*2),oracle(H*n);
        for(std::size_t i=0;i<q.size();++i)q[i]=values[(i*7+i/D)%values.size()];
        for(std::size_t t=0;t<n;++t)for(std::size_t i=0;i<KEY_WORDS;++i){
            const auto value=values[(t*17+i*7+i/D)%values.size()];
            raw[t*KEY_WORDS+i]=value;kv[t*KEY_WORDS*2+i]=value;
            kv[t*KEY_WORDS*2+KEY_WORDS+i]=0x7e00; // NaN V must never participate in QK.
        }
        for(std::size_t t=0;t<n;++t){auto reference=oracle_token(q,raw.data()+t*KEY_WORDS);for(std::size_t h=0;h<H;++h)oracle[h*n+t]=reference[h];}
        for(unsigned threads=1;threads<=CPU_THREADS;++threads){
            need(score_cpu(q,raw.data(),n,KEY_WORDS,threads)==oracle,"K-only grouped score differs from independent oracle");
            need(score_cpu(q,kv.data(),n,2*KEY_WORDS,threads)==oracle,"interleaved KV stride/head/tail oracle mismatch");
            comparisons+=2*H*n;
        }
    }
    need(reservation()<=HOST_LIMIT,"component managed host budget exceeds1GiB");
    std::cout<<json({{"status","passed"},{"evidence_class","software_fixture_not_hardware_measurement"},
        {"bitwise_score_comparisons",comparisons},{"score_bit_errors",0},{"CPU_scoring_threads_max",CPU_THREADS},
        {"NaN_V_ignored",true},{"managed_host_reservation_bytes",reservation()}}).dump()<<'\n';
}
}

int main(int argc,char** argv){
    if(argc==2&&std::string(argv[1])=="--software-check"){
        try{software_check();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc!=4){std::cerr<<"MANIFEST CPU_K_ONLY|CPU_FULL_KV|FPGA_DIRECT NEW_OUTPUT | --software-check\n";return 2;}
    const std::filesystem::path out=argv[3];
    const auto output_path=std::filesystem::weakly_canonical(out).string();
    if(output_path=="/mnt/smartssd"||output_path.rfind("/mnt/smartssd/",0)==0){std::cerr<<"no component evidence writes on SmartSSD\n";return 2;}
    if(!std::filesystem::create_directory(out))return 2;
    try{
        const auto cfg=read_json(argv[1]);const std::string mode=argv[2];
        const bool fpga=mode=="FPGA_DIRECT",full=mode=="CPU_FULL_KV";
        need(fpga||full||mode=="CPU_K_ONLY","unknown cold-QK mode; no fallback");
        need(cfg.at("B")==B&&cfg.at("P")==P&&cfg.at("cold_prompt_positions")==COLD
             &&cfg.at("sequences").size()==B&&cfg.at("score_representation")=="raw_gqa_f16_128"
             &&cfg.at("query_source")==QUERY_SOURCE,"fixed actual B8/32K/raw128 inputs required");
        need(cfg.at("CPU_thread_budget")==CPU_THREADS,"component uses all four admitted CPUs for scoring");
        need(reservation()<=HOST_LIMIT&&KH*COLD*D*2ULL+H*D*2ULL+SCORE_BYTES<=BO_LIMIT,"host/BO admission");
        InputBuffer input;std::array<Sequence,B> seq;json preparation=json::array();
        for(std::size_t i=0;i<B;++i){const auto& entry=cfg["sequences"][i];need(entry.at("sequence")==i,"sequence identity/order mismatch");preparation.push_back(prepare(seq[i],entry,i,out));}
        std::unique_ptr<FpgaScoreBackend> device;double initialization_ms=0;json image=nullptr;std::string fpga_abi;
        if(fpga){
            fpga_abi=cfg.at("fpga_abi").get<std::string>();
            validate_raw_gqa_shape(fpga_abi,H,KH,COLD,D);
            need(cfg.contains("fpga_image"),"explicit raw128 FPGA image contract required");
            image=cfg["fpga_image"];
            need(image.at("execute_authorized").is_boolean()&&image.at("execute_authorized").get<bool>(),"new image execution not authorized; no old-image/CPU fallback");
            HardwareContract contract;contract.abi=fpga_abi;contract.image_execute_authorized=true;
            contract.image_path=image.at("path").get<std::string>();contract.image_sha256=image.at("sha256").get<std::string>();
            need(contract.image_sha256!="c48f86d51667edf8f9142e98e6c99c8c108aaa40acb05cba0ab4da03c6f91c35","historical partial v1 image forbidden");
            contract.reuse_file_buffers=true;contract.buffer_budget_bytes=BO_LIMIT;
            auto begin=Clock::now();device=make_fpga_backend(HardwareMode::fpga_storage_streaming,contract);initialization_ms=elapsed(begin);
        }
        std::ofstream(out/"preparation.json")<<json({{"status","passed"},{"classification","actual_cold_raw_QK_component"},
            {"sequences",preparation},{"FPGA_image",image},{"fpga_abi",fpga_abi},{"device_initialization_ms_excluded",initialization_ms},
            {"managed_host_reservation_bytes",reservation()},{"host_budget_bytes",HOST_LIMIT},{"BO_budget_bytes",BO_LIMIT},
            {"shared_input_buffer_bytes",INPUT_BYTES},{"oracle_bytes",B*SCORE_BYTES},
            {"initial_BO_allocation","inside_separate_warmup_reported_per_call"},
            {"private_binary_evidence",true},{"no_SSD_files_created",true}}).dump(2)<<'\n';
        std::ofstream records(out/"runs.jsonl");need(bool(records),"new runs stream required");
        std::array<std::vector<std::uint16_t>,B> actual;
        for(unsigned round=0;round<2;++round){
            std::array<Event,B> events{};const auto ready=now_ns();
            for(std::size_t i=0;i<B;++i){auto& e=events[i];e.start=now_ns();
                if(fpga){
                    auto result=device->score_file_raw_gqa(seq[i].q,{seq[i].raw->fd,0,COLD*KEY_BYTES,false},H,KH,COLD,D);
                    actual[i]=std::move(result.values);e.read_ms=result.file_read_ms;e.read_bytes=result.file_read_bytes;
                    e.xrt=result.timings;e.query_bytes=result.query_h2d_bytes;e.flags=result.bo_flags;e.address=result.bo_device_address;
                    e.bank=result.bo_memory_group;e.staging_bytes=result.key_host_staging_bytes;e.p2p=result.p2p_allocation_verified;
                    e.reused=result.bo_reused;e.completed=result.kernel_completed;e.resident=result.resident_keys;
                }else{
                    const auto bytes=COLD*(full?KV_BYTES:KEY_BYTES);auto begin=Clock::now();
                    e.read_bytes=(full?seq[i].full:seq[i].raw)->read(input.data,bytes);e.read_ms=elapsed(begin);
                    begin=Clock::now();actual[i]=score_cpu(seq[i].q,static_cast<const std::uint16_t*>(input.data),COLD,full?KEY_WORDS*2:KEY_WORDS,CPU_THREADS);e.cpu_ms=elapsed(begin);
                }
                e.ready=now_ns();
            }
            const auto end=events.back().ready;json rows=json::array();std::uint64_t comparisons=0;
            for(std::size_t i=0;i<B;++i){const auto& e=events[i];
                need(actual[i].size()==H*COLD&&actual[i]==seq[i].oracle,"actual raw128 cold QK differs bitwise from independent original-capture oracle");
                need(e.read_bytes==COLD*(full?KV_BYTES:KEY_BYTES),"actual cold file read bytes differ from mode");
                if(fpga)need(e.p2p&&e.completed&&!e.resident&&e.staging_bytes==0&&e.query_bytes==H*D*2
                    &&e.xrt.d2h_bytes==SCORE_BYTES,"actual direct-file FPGA read/QK/return contract failed; no fallback");
                comparisons+=actual[i].size();const auto prefix=(round?"measured":"warmup")+std::string("-seq")+std::to_string(i);
                save(out/(prefix+"-raw-score-f16.bin"),actual[i]);
                rows.push_back({{"sequence",i},{"query_ready_ns",ready},{"cold_started_ns",e.start},{"scores_CPU_ready_ns",e.ready},
                    {"sequence_cold_QK_ms",double(e.ready-e.start)/1e6},{"sequence_completion_from_batch_start_ms",double(e.ready-ready)/1e6},
                    {"cold_file_read_ms",e.read_ms},{"CPU_cold_score_ms",e.cpu_ms},
                    {"XRT_initialization_ms_excluded",e.xrt.initialization_ms},{"XRT_allocation_ms",e.xrt.allocation_ms},
                    {"XRT_pack_ms",e.xrt.pack_ms},{"XRT_query_H2D_ms",e.xrt.h2d_ms},{"FPGA_launch_wait_ms",e.xrt.launch_wait_ms},
                    {"XRT_submission_ms",e.xrt.submission_ms},{"XRT_host_wait_ms",e.xrt.host_wait_ms},
                    {"FPGA_score_return_ms",e.xrt.d2h_ms},{"XRT_release_ms",e.xrt.release_ms},{"XRT_total_ms",e.xrt.total_ms},
                    {"FPGA_launch_started_ns",e.xrt.launch_started_ns},{"FPGA_launch_finished_ns",e.xrt.launch_finished_ns},
                    {"cold_file_read_bytes",e.read_bytes},{"SSD_to_host_K_bytes",fpga?0:COLD*KEY_BYTES},
                    {"SSD_to_host_V_bytes",full?COLD*KEY_BYTES:0},{"SSD_to_P2P_K_bytes",fpga?COLD*KEY_BYTES:0},
                    {"query_host_to_FPGA_bytes",e.query_bytes},{"scores_FPGA_to_host_bytes",e.xrt.d2h_bytes},
                    {"CPU_produced_score_bytes",fpga?0:SCORE_BYTES},{"scores_CPU_available_bytes",SCORE_BYTES},
                    {"P2P_BO_verified",e.p2p},{"BO_flags",e.flags},{"BO_device_address",e.address},{"BO_memory_group",e.bank},
                    {"BO_reused",e.reused},{"resident_keys",e.resident},{"key_host_staging_bytes",e.staging_bytes},
                    {"raw_score_checksum",std::to_string(checksum(actual[i].data(),actual[i].size()*2))},
                    {"score_bit_errors",0},{"scores_checked",actual[i].size()},{"raw_score_file",(prefix+"-raw-score-f16.bin")}});
            }
            struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
#ifdef __APPLE__
            const auto rss_bytes=std::uint64_t(usage.ru_maxrss);
#else
            const auto rss_bytes=std::uint64_t(usage.ru_maxrss)*1024;
#endif
            need(rss_bytes<=HOST_LIMIT,"actual process peak RSS exceeds component1GiB budget");
            json result={{"status","passed"},{"phase",round?"measured":"warmup"},{"method",mode},
                {"classification","actual_cold_raw_QK_component"},{"fpga_abi",fpga_abi},{"B",B},{"P",P},{"layer",8},{"cold_positions_per_sequence",COLD},
                {"scoring_Q_heads",H},{"physical_KV_heads",KH},{"score_dimensions",D},{"query_source",QUERY_SOURCE},
                {"score_representation","raw_gqa_f16_128"},{"score_accumulation","ascending_c_FP32_mul_add_then_F16"},
                {"normalization","not_applied_raw_QK_component"},{"CPU_scoring_threads",fpga?0:CPU_THREADS},{"CPU_affinity_budget",CPU_THREADS},
                {"query_ready_ns",ready},{"all_scores_CPU_ready_ns",end},{"T_batch_cold_QK_ms",double(end-ready)/1e6},
                {"sequence_events",rows},{"score_bit_errors",0},{"scores_checked",comparisons},
                {"input_token_stride_bytes",full?KV_BYTES:KEY_BYTES},{"total_file_read_bytes",B*COLD*(full?KV_BYTES:KEY_BYTES)},
                {"total_scores_CPU_available_bytes",B*SCORE_BYTES},{"managed_host_reservation_bytes",reservation()},
                {"host_peak_RSS_KiB",rss_bytes/1024},{"host_peak_RSS_bytes",rss_bytes},
                {"host_budget_bytes",HOST_LIMIT},{"host_budget_scope","managed_allocations_and_process_RSS_excludes_OS_file_cache"},{"BO_budget_bytes",BO_LIMIT},
                {"device_initialization_ms_excluded",initialization_ms},{"selector_executed",false},{"hot_scoring_executed",false},
                {"GPU_attention_executed",false},{"whole_model_feedback",false},{"TPOT_ms",nullptr},
                {"timing_differs_from_full_retrieval_chain",true},{"physical_host_bounce_verified",false}};
            records<<result.dump()<<'\n';records.flush();need(bool(records),"actual run result write failed");std::cout<<result.dump()<<'\n';
        }
        return 0;
    }catch(const std::exception& e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
