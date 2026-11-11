// Connect a freshly reproduced external-feedback query to the immutable old C files.
#include "hybrid_eval/gqa_frozen.hpp"
#include <iostream>
using namespace hillserve;using namespace hillserve::replay;
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"QUERY_DIR NEW_SYSTEM_SCORE OLD_SCORE OLD_KV NEW_TRACE_DIR OLD_CAPACITY\n";return 2;}
 const std::filesystem::path query=argv[1],out=argv[5];if(!std::filesystem::create_directory(out))return 2;
 try{json f;std::ifstream(query/"freeze.json")>>f;const std::size_t layer=8,h=32,d=128,valid=2049,n=std::stoul(argv[6]),new_n=f.at("capacity");need(n==2112&&new_n>=valid,"old C frozen dimensions");
  const auto q=load<std::uint16_t>(query/"query-f16.bin",h*d);std::vector<std::uint16_t> old(h*n*d),fresh(h*new_n*d);
  std::ifstream a(argv[3],std::ios::binary),b(argv[2],std::ios::binary);a.seekg(layer*h*n*d*2);a.read(reinterpret_cast<char*>(old.data()),old.size()*2);b.seekg(layer*h*new_n*d*2);b.read(reinterpret_cast<char*>(fresh.data()),fresh.size()*2);need(a&&b,"existing score files short");
  std::size_t errors=0;for(std::size_t head=0;head<h;++head)for(std::size_t t=0;t<valid;++t)for(std::size_t c=0;c<d;++c)errors+=old[(head*n+t)*d+c]!=fresh[(head*new_n+t)*d+c];need(errors==0,"reproduced actual feedback keys differ from old C valid prefix");
  auto scores=host_partial_scores(q,old,h,n,d);auto selected=policy::select_gqa_scores(scaled(scores),h,n,valid,5,.2f);auto ids=load<std::uint64_t>(query/"selected-ids.bin",h*f.at("selected_per_head").get<std::size_t>());need(flatten(selected)==ids,"old file selection differs from actual-feedback frozen IDs");
  std::vector<std::uint16_t> physical(valid*2048);std::ifstream kv(argv[4],std::ios::binary);for(std::size_t t=0;t<valid;++t){std::array<std::uint64_t,4> header{};kv.seekg(4096+(layer*n+t)*4128);kv.read(reinterpret_cast<char*>(header.data()),32);kv.read(reinterpret_cast<char*>(physical.data()+t*2048),4096);need(kv&&header[1]==t&&std::uint32_t(header[3])==layer&&header[0]==checksum(physical.data()+t*2048,4096),"old physical KV integrity/position/layer");}
  save(out/"query-f16.bin",q);save(out/"raw-score-oracle.bin",scores);save(out/"selected-ids.bin",ids);save(out/"kv-physical.bin",physical);std::vector<std::uint64_t> positions(valid);std::iota(positions.begin(),positions.end(),0);save(out/"positions.bin",positions);
  json m={{"status","passed"},{"evidence_class","actual_external_feedback_query_reproduced_against_old_C_files"},{"P",2048},{"layer",layer},{"heads",h},{"capacity",n},{"valid_candidates",valid},{"compact_channels_padded",d},{"normalization_dimension",128},{"alpha",5},{"cap_ratio",.2},{"selected_per_head",selected.k},{"score_file_offset",layer*h*n*d*2},{"kv_record_stride",4128},{"kv_data_offset",4096+layer*n*4128},{"kv_payload_offset",32},{"valid_prefix_score_key_bit_errors",errors},{"original_feedback_selected_ID_errors",0},{"historical_future_slots_exist_but_are_excluded_from_selection",true},{"layout","old_C_full128_masked_query_same_v1"},{"purpose","read_only_timer_anchor_not_compact_performance_comparison"},{"score_file",argv[3]},{"KV_file",argv[4]}};
  std::ofstream(out/"manifest.json")<<m.dump(2)<<'\n';std::cout<<m.dump()<<'\n';return 0;
 }catch(const std::exception&e){std::ofstream(out/"error.txt")<<e.what()<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
