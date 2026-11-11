#include "smartssd/xrt_backend.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
template<class T>std::vector<T> read_file(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto n=f.tellg();if(n<0||std::size_t(n)%sizeof(T))throw std::runtime_error("input file size");f.seekg(0);std::vector<T> v(std::size_t(n)/sizeof(T));if(!f.read(reinterpret_cast<char*>(v.data()),n))throw std::runtime_error("input read");return v;}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--inspect"){std::cout<<hillserve::inspect_xrt_device(0)<<'\n';return 0;}
 if(argc!=6||std::string(argv[1])!="--explicit-image-execution-granted")throw std::runtime_error("usage: --inspect OR --explicit-image-execution-granted IMAGE Q_I16 K_I16 EXPECTED_I64; caller must verify image SHA and grant");
 auto q=read_file<std::int16_t>(argv[3]),k=read_file<std::int16_t>(argv[4]);auto expected=read_file<std::int64_t>(argv[5]);
 auto r=hillserve::run_xrt_rawsum(argv[2],0,true,q,k,expected.size());std::size_t errors=0;for(std::size_t i=0;i<expected.size();++i)errors+=expected[i]!=r.scores[i];
 std::cout<<std::setprecision(12)<<"{\"evidence\":\"FPGA_HOST_STAGED_RAW_SUM_PRIMITIVE\",\"paper_selector_compatible\":false,\"actual_fpga\":true,\"true_storage_to_fpga\":false,\"mismatches\":"<<errors<<",\"allocation_ms\":"<<r.times.allocation_ms<<",\"h2d_ms\":"<<r.times.h2d_ms<<",\"kernel_launch_wait_ms\":"<<r.times.kernel_wait_ms<<",\"d2h_ms\":"<<r.times.d2h_ms<<",\"wall_excludes_image_load_ms\":"<<r.times.total_ms<<",\"h2d_bytes\":"<<r.times.h2d_bytes<<",\"d2h_bytes\":"<<r.times.d2h_bytes<<"}\n";return errors?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
