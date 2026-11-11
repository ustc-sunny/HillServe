// Real V3 board replay for numerical/ranking checks. Host-staged input is
// explicit: this program makes no SSD-direct or serving-performance claim.
#include "hybrid_eval/gqa_frozen.hpp"
#include "policy/full_gqa_cpu.hpp"
#include "smartssd/adapter.hpp"
#include <iostream>
#include <thread>
#include <sys/resource.h>

using namespace hillserve;
using namespace hillserve::replay;
namespace {
json read_json(const std::filesystem::path& p){std::ifstream f(p);need(bool(f),"missing JSON");json j;f>>j;return j;}
std::vector<float> fp32_scores(const std::vector<float>& q,const std::vector<std::uint16_t>& keys,std::size_t n) {
    need(q.size()==4096&&keys.size()>=n*1024,"FP32 reference shape");
    for(auto x:q)need(std::isfinite(x),"FP32 reference query nonfinite");
    std::vector<float> out(32*n);
    auto worker=[&](std::size_t worker_id){
        for(std::size_t row=worker_id;row<n*8;row+=4) {
            const auto t=row/8,kh=row%8;float sums[4]={};
            for(std::size_t c=0;c<128;++c){const float k=half_value(keys[t*1024+kh*128+c]);
                for(std::size_t h=0;h<4;++h){const float product=q[(kh*4+h)*128+c]*k;sums[h]=sums[h]+product;}}
            for(std::size_t h=0;h<4;++h)out[(kh*4+h)*n+t]=sums[h]/std::sqrt(128.f);
        }
    };
    std::vector<std::thread> threads;for(std::size_t i=1;i<4;++i)threads.emplace_back(worker,i);worker(0);
    for(auto& t:threads)t.join();for(float x:out)need(std::isfinite(x),"FP32 score nonfinite");return out;
}
void self_test() {
    const std::size_t n=65;std::vector<float> q(4096,0);std::vector<std::uint16_t> k(n*1024,half_bits(1));
    for(std::size_t h=0;h<32;++h){q[h*128]=1e20f;q[h*128+1]=-1e20f;q[h*128+2]=3.f+float(h);}
    const auto s=fp32_scores(q,k,n);
    for(std::size_t h=0;h<32;++h)for(std::size_t t=0;t<n;++t)
        need(s[h*n+t]==(3.f+float(h))/std::sqrt(128.f),"FP32 ordered-reduction self-test");
    std::vector<std::uint16_t> q16(4096,half_bits(.5f));
    const auto full=score_full_gqa_legacy_cpu(q16,k.data(),n,1024,4);
    auto cold=score_full_gqa_legacy_cpu(q16,k.data(),64,1024,1);
    auto hot=score_full_gqa_legacy_cpu(q16,k.data()+64*1024,1,1024,3);
    std::vector<std::uint16_t> mix(32*n);
    for(std::size_t h=0;h<32;++h){std::copy_n(cold.begin()+h*64,64,mix.begin()+h*n);mix[h*n+64]=hot[h];}
    need(mix==full,"partition scatter self-test");
    const auto a=policy::select_gqa_scores(scaled(full),32,n,n,10,.2f);
    const auto b=policy::select_gqa_scores(scaled(mix),32,n,n,10,.2f);
    need(a.k==13&&a.ids==b.ids&&a.counts==b.counts,"global selector self-test");
    std::cout<<"passed CPU-only reference/partition/selector self-tests\n";
}
}
int main(int argc,char** argv) {
    if(argc==2&&std::string(argv[1])=="--self-test"){try{self_test();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
    if(argc!=7){std::cerr<<"CAPTURE NEW_OUT IMAGE IMAGE_SHA EXECUTE_AUTHORIZED COLD_PROMPT\n";return 2;}
    const std::filesystem::path trace=argv[1],out=argv[2];
    if(!std::filesystem::create_directory(out)){std::cerr<<"new result directory required\n";return 2;}
    try {
        const auto start=Clock::now();const auto manifest=read_json(trace/"manifest.json");
        need(manifest.at("status")=="passed","verified native capture required");
        const std::size_t P=manifest.at("P"),cold=std::stoul(argv[6]);
        need(P<=32768&&cold>0&&cold<P&&cold%64==0,"aligned fixed prompt partition");
        HardwareContract contract;contract.abi="HS_FULL_GQA_PIPELINE_F16_V3";
        contract.image_path=argv[3];contract.image_sha256=argv[4];contract.image_execute_authorized=std::string(argv[5])=="true";
        need(contract.image_sha256=="679e85f964bf644cb1fe3bd5752b5a2452d288106e8139b46d702aafbdb2dcd6","only existing authorized V3 image");
        contract.buffer_budget_bytes=256ULL<<20;
        auto fpga=make_fpga_backend(HardwareMode::fpga_host_staged,contract);
        json rows=json::array();std::ofstream journal(out/"checks.jsonl");
        for(const auto& point:manifest.at("checkpoints")) {
            const std::size_t n=point.at("valid_candidates");const int l=point.at("layer_zero_based");
            const std::string prefix=point.at("prefix");need(n<=P+8&&n>P,"valid causal prefix");
            auto keys=load<std::uint16_t>(trace/("layer-"+std::to_string(l)+"-K-f16.bin"),(P+8)*1024);
            keys.resize(n*1024); // Future rows are excluded from every score and selection.
            const auto qactual=load<float>(trace/(prefix+"-actual-Q-f32.bin"),4096);
            const auto qrehearsal=load<float>(trace/(prefix+"-rehearsal-Q-f32.bin"),4096);
            const auto actual_ref=fp32_scores(qactual,keys,n);
            const auto rehearsal_ref=fp32_scores(qrehearsal,keys,n);
            std::vector<std::uint16_t> q16(4096);
            for(std::size_t i=0;i<4096;++i){q16[i]=half_bits(qrehearsal[i]);need(std::isfinite(half_value(q16[i])),"query F16 overflow");}
            const auto cpu=score_full_gqa_legacy_cpu(q16,keys.data(),n,1024,4);
            const auto hot=score_full_gqa_legacy_cpu(q16,keys.data()+cold*1024,n-cold,1024,3);
            std::vector<std::uint16_t> mixed(32*n);std::vector<unsigned char> coverage(n,0);
            for(std::size_t h=0;h<32;++h)std::copy_n(hot.begin()+h*(n-cold),n-cold,mixed.begin()+h*n+cold);
            std::fill(coverage.begin()+cold,coverage.end(),1);
            std::uint64_t calls=0,h2d=0,d2h=0;double board_ms=0;
            for(std::size_t off=0;off<cold;off+=4096) {
                const auto count=std::min(std::size_t(4096),cold-off);
                std::vector<std::uint16_t> chunk(keys.begin()+off*1024,keys.begin()+(off+count)*1024);
                const auto result=fpga->score_raw_gqa(q16,chunk,32,8,count,128);
                need(result.values.size()==32*count,"actual board output extent");
                for(std::size_t h=0;h<32;++h)std::copy_n(result.values.begin()+h*count,count,mixed.begin()+h*n+off);
                for(std::size_t t=off;t<off+count;++t)++coverage[t];
                ++calls;h2d+=result.timings.h2d_bytes;d2h+=result.timings.d2h_bytes;board_ms+=result.timings.total_ms;
            }
            need(std::all_of(coverage.begin(),coverage.end(),[](auto x){return x==1;}),"hot/cold union has gap/duplicate");
            std::uint64_t errors=0;for(std::size_t i=0;i<cpu.size();++i)errors+=cpu[i]!=mixed[i];
            const auto cpu_scaled=scaled(cpu),mixed_scaled=scaled(mixed);
            const auto ac=policy::select_gqa_scores(cpu_scaled,32,n,n,10,.2f);
            const auto bc=policy::select_gqa_scores(mixed_scaled,32,n,n,10,.2f);
            const bool same=ac.k==bc.k&&ac.counts==bc.counts&&ac.ids==bc.ids;
            save(out/(prefix+"-actual-score-f32.bin"),actual_ref);
            save(out/(prefix+"-rehearsal-score-f32.bin"),rehearsal_ref);
            save(out/(prefix+"-CPU-raw-f16.bin"),cpu);save(out/(prefix+"-mixed-raw-f16.bin"),mixed);
            save(out/(prefix+"-policy-scaled-f16.bin"),cpu_scaled);save(out/(prefix+"-policy-ids-u64.bin"),flatten(ac));
            struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
            json row=point;row.update({{"status",errors==0&&same?"passed":"failed"},{"cold_prompt_tokens",cold},
                {"hot_valid_tokens",n-cold},{"raw_score_bit_errors",errors},{"scaled_score_bit_equal",cpu_scaled==mixed_scaled},
                {"global_ordered_selection_equal",same},{"selected_per_head",ac.k},{"threshold_counts",ac.counts},
                {"boundary_tied_heads",ac.boundary_tied_heads},{"FPGA_calls",calls},{"h2d_bytes",h2d},{"d2h_bytes",d2h},
                {"query_h2d_bytes",calls*4096*2},{"candidate_key_h2d_bytes",cold*2048},
                {"backend","actual_FPGA_host_staged"},{"new_SSD_direct_claim",false},
                {"board_call_diagnostic_ms",board_ms},{"process_peak_RSS_KiB",usage.ru_maxrss},
                {"workspace_bound_bytes",1ULL<<30},{"selector_margin",10},{"retention_cap",.2}});
            rows.push_back(row);journal<<row.dump()<<'\n';journal.flush();
            need(errors==0&&same&&cpu_scaled==mixed_scaled,"actual FPGA/mixed score or selector mismatch");
            std::cout<<"verified "<<prefix<<" N="<<n<<" k="<<ac.k<<'\n'<<std::flush;
        }
        std::ofstream(out/"summary.json")<<json({{"status","passed"},{"P",P},{"checkpoints",rows},
            {"evidence_class","hardware_replay_host_staged_no_storage_timing_claim"},
            {"reference","ordered_FP32_QK_on_same_F16_K_not_a_full_FP32_model"},
            {"kernel_ABI",contract.abi},{"image_sha256",contract.image_sha256},{"total_ms",milliseconds(start)}}).dump(2)<<'\n';
        return 0;
    }catch(const std::exception& e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
