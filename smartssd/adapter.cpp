#include "smartssd/adapter.hpp"
#include <algorithm>
namespace hillserve {
void validate_fpga_contract(HardwareMode mode,const HardwareContract& c){
    if(mode!=HardwareMode::fpga_host_staged&&mode!=HardwareMode::fpga_storage_streaming)throw std::invalid_argument("only explicit staged or file-to-P2P scoring is implemented");
    if(c.abi!="HS_PARTIAL_F16_V1"&&c.abi!="HS_FULL_GQA_F16_V2"&&c.abi!="HS_FULL_GQA_PIPELINE_F16_V3")throw std::invalid_argument("unsupported score ABI; legacy cross-head raw-sum image cannot be substituted");
    if(c.image_sha256.size()!=64||!std::all_of(c.image_sha256.begin(),c.image_sha256.end(),[](char v){return (v>='0'&&v<='9')||(v>='a'&&v<='f');})||c.image_path.empty())throw std::invalid_argument("exact image path/SHA256 required");
    if(!c.buffer_budget_bytes||c.buffer_budget_bytes>256*1024*1024)throw std::invalid_argument("FPGA BO budget outside bounded pilot");
    if(!c.direct_read_queue_depth||c.direct_read_queue_depth>8)throw std::invalid_argument("direct read queue depth must be1..8");
    if(!c.image_execute_authorized)throw std::runtime_error("specific image load/execute action is not authorized");
}
void require_fpga_backend(HardwareMode mode,const HardwareContract& c){
    validate_fpga_contract(mode,c);
#ifndef HILLSERVE_HAS_XRT
    throw std::runtime_error("this binary was compiled without XRT; build with HILLSERVE_XRT=ON");
#endif
}
std::unique_ptr<FpgaScoreBackend> make_fpga_backend(HardwareMode mode,const HardwareContract& c){
    require_fpga_backend(mode,c);
#ifdef HILLSERVE_HAS_XRT
    return create_xrt_partial_backend(c,mode);
#else
    throw std::logic_error("unreachable: XRT build unavailable");
#endif
}
}
