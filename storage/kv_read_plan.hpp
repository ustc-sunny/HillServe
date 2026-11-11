#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hillserve::storage {
// Describe existing immutable records only. Planning neither reads a device
// nor changes the selection. Extra pages may be read, but never retained as KV.
struct KVReadRecord { std::uint64_t offset; std::size_t output_row; };
struct KVReadWindow {
    std::uint64_t begin, end, required_page_bytes;
    std::size_t first_record, end_record;
};
struct KVReadLimits {
    std::size_t window_bytes=1024*1024, gap_bytes=16*1024;
    unsigned max_amplification_permille=1500;
};
inline void validate_kv_read_limits(const KVReadLimits& limits) {
    if(limits.window_bytes<8192 || limits.window_bytes>1024*1024 || limits.window_bytes%4096 ||
       limits.gap_bytes>64*1024 || limits.gap_bytes%4096 ||
       limits.max_amplification_permille<1000 || limits.max_amplification_permille>2000)
        throw std::runtime_error("invalid bounded complete-KV read limits");
}
// Linear greedy plan with a per-window amplification ceiling relative to the
// union of aligned pages actually containing requested records in that window.
// The ceiling includes gap pages; repeated boundary pages across windows remain
// in the reported syscall byte count. Windows are aligned and <=1 MiB.
inline std::vector<KVReadWindow> plan_kv_reads(
    const std::vector<KVReadRecord>& records, std::uint64_t record_bytes,
    std::uint64_t extent, const KVReadLimits& limits={}) {
    validate_kv_read_limits(limits);
    if(!record_bytes || extent%4096 || record_bytes>limits.window_bytes-4095)
        throw std::runtime_error("invalid complete-KV record geometry");
    std::vector<KVReadWindow> windows; windows.reserve(records.size());
    std::uint64_t previous_end=0, last_page_end=0;
    for(std::size_t i=0;i<records.size();++i){
        const auto offset=records[i].offset;
        if(offset>extent || record_bytes>extent-offset || (i && offset<previous_end))
            throw std::runtime_error("unordered, overlapping or out-of-extent KV record");
        previous_end=offset+record_bytes;
        if(previous_end>std::numeric_limits<std::uint64_t>::max()-4095)
            throw std::runtime_error("complete-KV aligned offset overflow");
        const auto begin=offset/4096*4096, end=(previous_end+4095)/4096*4096;
        if(end>extent)throw std::runtime_error("complete-KV aligned read exceeds extent");
        bool merged=false;
        if(!windows.empty()){
            auto& w=windows.back();
            const auto required=w.required_page_bytes+end-std::max(begin,last_page_end);
            const auto span=end-w.begin, gap=begin>w.end?begin-w.end:0;
            if(span<=limits.window_bytes && gap<=limits.gap_bytes &&
               span*1000<=required*limits.max_amplification_permille){
                w.end=end;w.required_page_bytes=required;w.end_record=i+1;merged=true;
            }
        }
        if(!merged)windows.push_back({begin,end,end-begin,i,i+1});
        last_page_end=end;
    }
    return windows;
}
} // namespace hillserve::storage
