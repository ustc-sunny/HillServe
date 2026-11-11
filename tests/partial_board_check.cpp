#include "smartssd/adapter.hpp"
#include "policy/host_policy.hpp"
#include <fstream>
#include <iostream>
#include <vector>
using namespace hillserve;
template<class T>std::vector<T> read(std::ifstream& f,std::size_t n){std::vector<T>x(n);if(!f.read(reinterpret_cast<char*>(x.data()),n*sizeof(T)))throw std::runtime_error("short real trace");return x;}
int main(int argc,char**argv){try{
 if(argc<5||std::string(argv[3])!="--authorized-new-image")throw std::runtime_error("usage: partial-board-check IMAGE SHA256 --authorized-new-image REAL_TRACE...");
 HardwareContract c;c.abi="HS_PARTIAL_F16_V1";c.image_path=argv[1];c.image_sha256=argv[2];c.image_execute_authorized=true;
 auto device=make_fpga_backend(HardwareMode::fpga_host_staged,c);std::size_t total=0,errors=0;
 for(int file=4;file<argc;++file){std::ifstream f(argv[file],std::ios::binary);auto hdr=read<std::uint32_t>(f,10);if(hdr[0]!=0x48535033||(hdr[1]!=1&&hdr[1]!=2))throw std::runtime_error("trace ABI");auto h=hdr[2],n=hdr[3],d=hdr[4];auto q=read<std::uint16_t>(f,h*d),all=read<std::uint16_t>(f,h*n*d);
  for(std::size_t count:std::vector<std::size_t>{1,127,std::min<std::size_t>(n,256),n}){if(count>n)continue;std::vector<std::uint16_t> keys(h*count*d);for(std::size_t head=0;head<h;++head)std::copy(all.begin()+head*n*d,all.begin()+(head*n+count)*d,keys.begin()+head*count*d);
   auto expected=host_partial_scores(q,keys,h,count,d);auto got=device->score(q,keys,h,count,d);std::size_t mismatch=0;for(std::size_t i=0;i<expected.size();++i)mismatch+=expected[i]!=got.values[i];errors+=mismatch;total+=expected.size();
   std::cout<<"{\"evidence\":\"FPGA_HOST_STAGED_REAL_TRACE_NUMERICAL_VALIDATION\",\"trace\":\""<<argv[file]<<"\",\"heads\":"<<h<<",\"tokens\":"<<count<<",\"channels\":"<<d<<",\"bit_mismatches\":"<<mismatch<<",\"score_count\":"<<expected.size()<<",\"initialization_ms\":"<<got.timings.initialization_ms<<",\"score_path_ms\":"<<got.timings.total_ms<<",\"launch_wait_ms\":"<<got.timings.launch_wait_ms<<",\"actual_storage_read\":false,\"actual_model_feedback\":false,\"simulation\":false}\n"<<std::flush;
   if(mismatch)throw std::runtime_error("strict CPU/FPGA mismatch; performance gate fails");
  }
 }
 std::cout<<"{\"status\":\"passed\",\"total_scores\":"<<total<<",\"total_bit_mismatches\":"<<errors<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
