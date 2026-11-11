#include "policy/host_policy.hpp"
#include "smartssd/adapter.hpp"
#include <fstream>
#include <iostream>
#include <vector>
extern "C" void hillserve_partial_f16(const std::uint16_t*,const std::uint16_t*,std::uint16_t*,std::uint32_t,std::uint32_t,std::uint32_t);
template<class T>std::vector<T> read(std::ifstream& f,std::size_t n){std::vector<T> out(n);if(!f.read(reinterpret_cast<char*>(out.data()),n*sizeof(T)))throw std::runtime_error("short real trace");return out;}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: partial-kernel-check REAL_TRACE");
 std::ifstream f(argv[1],std::ios::binary);auto hdr=read<std::uint32_t>(f,10);if(hdr[0]!=0x48535033||(hdr[1]!=1&&hdr[1]!=2))throw std::runtime_error("trace ABI");
 auto h=hdr[2],n=hdr[3],d=hdr[4];if(!h||h>32||!n||n>65504||!d||d>128)throw std::runtime_error("shape");
 auto q=read<std::uint16_t>(f,h*d),k=read<std::uint16_t>(f,h*n*d);std::vector<std::uint16_t> result(h*n);
 hillserve_partial_f16(q.data(),k.data(),result.data(),h,n,d);auto host=hillserve::host_partial_scores(q,k,h,n,d);
 std::size_t errors=0;for(std::size_t i=0;i<host.size();++i)errors+=host[i]!=result[i];
 hillserve::HardwareContract c;c.abi="HS_PARTIAL_F16_V1";c.image_path="not-loaded.xclbin";c.image_sha256=std::string(64,'0');bool rejected=false;
 try{hillserve::validate_fpga_contract(hillserve::HardwareMode::fpga_host_staged,c);}catch(const std::runtime_error&){rejected=true;}
 if(!rejected)throw std::runtime_error("missing image action grant accepted");
 c.image_execute_authorized=true;hillserve::validate_fpga_contract(hillserve::HardwareMode::fpga_host_staged,c); // pure contract validation; no device access
 std::cout<<"{\"evidence\":\"CPU_C_SIMULATION_NOT_FPGA\",\"real_input_scores\":"<<h*n<<",\"bit_mismatches\":"<<errors<<",\"image_action_guard_checked\":true}\n";return errors?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
