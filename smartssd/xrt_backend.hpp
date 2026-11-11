#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace hillserve {
struct XrtTimes { double allocation_ms=0,h2d_ms=0,kernel_wait_ms=0,d2h_ms=0,total_ms=0;std::uint64_t h2d_bytes=0,d2h_bytes=0; };
struct XrtRawSumResult { std::vector<std::int64_t> scores;XrtTimes times; };
std::string inspect_xrt_device(unsigned index);
// Only the inspected OPT32x128 raw-sum ABI. It is NOT per-head paper selection.
// Invocation requires an explicit grant for this exact image; no automatic fallback.
XrtRawSumResult run_xrt_rawsum(const std::string& image,unsigned device,bool execution_granted,
 const std::vector<std::int16_t>& query,const std::vector<std::int16_t>& keys,std::size_t tokens);
}
