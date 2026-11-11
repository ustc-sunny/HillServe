// Bounded diagnostic components. Deliberately not a paper policy or FPGA substitute.
#include "core/core.hpp"
#include "storage/file_store.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <future>
#include <iomanip>
#include <sys/resource.h>
#include <sys/stat.h>
using namespace hillserve;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: hillserve-replay fixed_request_ids|recompute_scores NEW_DIR N");
 std::string mode=argv[1];if(mode!="fixed_request_ids"){std::cerr<<"BLOCKED: paper_policy_host unbound; FPGA unauthorized/incompatible; no fallback\n";return 2;}
 auto dir=std::filesystem::path(argv[2]);int n=std::stoi(argv[3]);if(n<32||n>4096)throw std::runtime_error("bounded N range");
 if(!std::filesystem::create_directory(dir))throw std::runtime_error("require new directory");if(std::filesystem::space(dir).available<1024ULL*1024*1024)throw std::runtime_error("space reserve");
 struct stat target{},workspace{};if(stat(dir.c_str(),&target)||stat(".",&workspace)||target.st_dev!=workspace.st_dev)throw std::runtime_error("diagnostic only on workspace filesystem; SmartSSD needs separate authorized path");
 std::ofstream f(dir/"components.csv");f<<"component,N,dtype,threads,cache_state,logical_bytes,elapsed_ms\n"<<std::setprecision(12);
 auto measure=[&](const char*name,std::uint64_t bytes,auto fn){auto t=Clock::now();fn();double ms=elapsed(t);f<<name<<','<<n<<",fixture_f32_f16_payload,1,buffered_not_cold,"<<bytes<<','<<ms<<'\n';return ms;};
 IndexedKVStore store(dir/"payload.bin",32*1024*1024,8192),writer(dir/"append.bin",32*1024*1024,8192);
 std::vector<std::uint8_t> payload(2048);for(int i=0;i<2048;++i)payload[i]=(i*7)%251;
 auto prep=Clock::now();store.begin_append_batch();for(int i=0;i<n;++i)store.append_kv(1,0,i,payload);store.end_append_batch();double prepms=elapsed(prep);
 std::vector<std::uint64_t> all(n),selected;std::iota(all.begin(),all.end(),0);for(int i=n-1;i>=0;i-=7)selected.push_back(i);
 auto gather=[&](const auto&ids){auto v=store.fetch_selected(1,0,ids,16*1024*1024);for(auto&p:v)if(p.bytes!=payload)throw std::runtime_error("byte mismatch");};
 // Legal math primitive only: no claim this is the paper's score, heads or selector.
 std::vector<float>q(128,.125f),k(n*128);for(std::size_t i=0;i<k.size();++i)k[i]=float(int(i%31)-15)/32;
 std::vector<double> scores;measure("diagnostic_dot",k.size()*4,[&]{scores=diagnostic_full_dot_product(q,k);});
 std::vector<Score> merged;measure("diagnostic_merge",n*sizeof(Score),[&]{Scope scope{1,0,0,0,0};ScoreMerge m(scope,"diagnostic-f64",{{"host",all}},all);ScoreBlock b{scope,"diagnostic-f64","host",{}};for(int i=0;i<n;++i)b.scores.push_back({std::uint64_t(i),scores[i]});m.accept(b);merged=m.finish();});
 measure("diagnostic_select_topk_not_paper",n*sizeof(Score),[&]{auto ids=diagnostic_top_k(merged,32);if(ids.size()!=32)throw std::runtime_error("selector");});
 measure("candidate_read",n*2048,[&]{gather(all);});measure("selected_read",selected.size()*2048,[&]{gather(selected);});
 measure("read_plus_selected_serial",(n+selected.size())*2048,[&]{gather(all);gather(selected);});
 measure("read_plus_selected_concurrent",(n+selected.size())*2048,[&]{auto secondary=std::async(std::launch::async,[&]{std::ifstream in(dir/"payload.bin",std::ios::binary);std::vector<std::uint8_t> b(2048);for(auto id:selected){in.seekg(id*2048);in.read(reinterpret_cast<char*>(b.data()),2048);if(!in||b!=payload)throw std::runtime_error("secondary read failed");}});gather(all);secondary.get();});
 // Concurrent secondary reads use their own descriptor and do not race store counters.
 auto append=[&]{writer.begin_append_batch();for(int i=0;i<32;++i)writer.append_kv(1,0,i,payload);};
 measure("append_unsynced_write",32*2048,append);measure("append_fdatasync",0,[&]{writer.end_append_batch();});
 measure("read_plus_append_interference",n*2048+32*2048,[&]{auto w=std::async(std::launch::async,[&]{writer.begin_append_batch();for(int i=32;i<64;++i)writer.append_kv(1,0,i,payload);writer.end_append_batch();});gather(all);w.get();});
 // Stateful fixed-ID replay: eight appends, current token explicitly requested after completion.
 std::ofstream events(dir/"replay-events.jsonl");auto chain=Clock::now();for(int step=0;step<8;++step){auto t=Clock::now();store.append_kv(1,0,n+step,payload);auto got=store.fetch_selected(1,0,{std::uint64_t(n+step),0,7},6144);if(got[0].bytes!=payload)throw std::runtime_error("current token corruption");events<<"{\"step\":"<<step<<",\"producer\":\"append_complete\",\"append_bytes\":2048,\"selected_bytes\":6144,\"elapsed_ms\":"<<elapsed(t)<<"}\n";}
 double chainms=elapsed(chain);struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
 std::ofstream report(dir/"result.json");report<<std::setprecision(12)<<"{\"status\":\"passed\",\"evidence_mode\":\"D_DIAGNOSTIC\",\"method_id\":\"fixed_request_ids\",\"physical_path\":\"systemfs_buffered\",\"selection_recomputed\":false,\"actual_fpga_executed\":false,\"actual_target_device_io\":false,\"actual_model_feedback\":false,\"synthetic_fixture\":true,\"N\":"<<n<<",\"preparation_ms\":"<<prepms<<",\"fixed_storage_chain_ms\":"<<chainms<<",\"append_bytes\":"<<store.append_write_bytes+writer.append_write_bytes<<",\"selected_read_bytes\":"<<store.selected_read_bytes<<",\"index_allocator_peak_bytes\":"<<store.metadata().peak+writer.metadata().peak<<",\"process_peak_rss_kib\":"<<ru.ru_maxrss<<",\"fpga_kernel_ms\":null,\"physical_nand_bytes\":null,\"formal_speedup_eligible\":false}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
