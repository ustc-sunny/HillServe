#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
namespace hillserve {
enum class HardwareMode { cpu_emulation, fpga_dram_resident, fpga_host_staged, fpga_storage_streaming };
struct HardwareContract {
    std::string abi,image_sha256,image_path;
    bool image_execute_authorized=false; // specific image action, NOT source reuse permission
    unsigned device_index=0;
    std::function<void()> on_kernel_submitted; // Optional timing/pipeline barrier, no policy change.
    bool reuse_file_buffers=false; // Explicit replay option; every score_file still reads SSD.
    std::size_t buffer_budget_bytes=256*1024*1024;
    unsigned direct_read_queue_depth=1; // CSS only:1=pread,2..8=native AIO for fragmented chunks.
};
struct ScoreDeviceTimings {double initialization_ms=0,allocation_ms=0,pack_ms=0,h2d_ms=0,launch_wait_ms=0,d2h_ms=0,release_ms=0,total_ms=0;std::uint64_t h2d_bytes=0,d2h_bytes=0,launch_started_ns=0,launch_finished_ns=0;double submission_ms=0,host_wait_ms=0;};
struct DeviceScores {std::vector<std::uint16_t> values;ScoreDeviceTimings timings;};
struct ScoreFileRegion {int fd=-1;std::uint64_t offset=0,bytes=0;bool capture_mapping_evidence=false;};
struct FileDeviceScores : DeviceScores {
    double file_read_ms=0;
    std::uint64_t file_read_bytes=0,key_host_staging_bytes=0,query_h2d_bytes=0;
    std::uint64_t bo_flags=0,bo_device_address=0,bo_mapped_address=0;
    unsigned kernel_key_bank=0,bo_memory_group=0;
    bool p2p_allocation_verified=false,kernel_completed=false,bo_reused=false,resident_keys=false;
    std::string mapping_evidence;
};
// A streamed chunk is head-major [query_head, tokens]. token_offset indexes
// the concatenation of input regions, NOT an implicit original token ID.
// The caller owns that mapping and must wait for complete global coverage
// before final selection. Callback references expire when it returns.
struct FileScoreChunkInfo {
    std::size_t token_offset=0,tokens=0,buffer_slot=0;
    std::uint64_t file_read_bytes=0,score_d2h_bytes=0;
    std::uint64_t file_read_operations=0,aio_submit_calls=0,aio_completions=0;
    unsigned aio_peak_inflight=0;
    std::uint64_t read_started_ns=0,read_finished_ns=0;
    // Host submission and completion-observation timestamps, not independent
    // device cycle counters. Previous-chunk delivery occurs before this wait,
    // so launch_wait_ms can include that host work and delayed observation.
    std::uint64_t kernel_started_ns=0,kernel_wait_started_ns=0,kernel_finished_ns=0;
    std::uint64_t scores_CPU_ready_ns=0,callback_started_ns=0,callback_finished_ns=0;
    double file_read_ms=0,submission_ms=0,host_wait_ms=0,launch_wait_ms=0,d2h_ms=0,callback_ms=0;
};
struct FileScoreChunk { FileScoreChunkInfo info;std::vector<std::uint16_t> values; };
using FileScoreCallback=std::function<void(const FileScoreChunk&)>;
struct FileScoreStreamReport {
    std::vector<FileScoreChunkInfo> chunks;
    double initialization_ms=0,allocation_ms=0,query_h2d_ms=0,total_ms=0;
    std::uint64_t file_read_bytes=0,query_h2d_bytes=0,score_d2h_bytes=0;
    std::size_t device_buffer_bytes=0,host_score_buffer_bytes=0;
    bool buffers_reused=false,p2p_allocation_verified=false;
    unsigned direct_read_queue_depth=1,aio_peak_inflight=0;
    std::uint64_t file_read_operations=0,aio_submit_calls=0,aio_completions=0;
    bool aio_context_reused=false;
    double aio_setup_ms=0;
    // The pipeline uses separate K/S BOs. It overlaps the next direct read
    // and the previous score delivery with an active chunk kernel; actual
    // overlap must be established from timestamps, not inferred from flags.
};
class FpgaScoreBackend {
public:
    // Calls on one backend must be serialized. Input fds must remain open;
    // callbacks must not reenter this backend. Unsupported modes fail explicitly.
    virtual FileScoreStreamReport score_file_regions_raw_gqa_stream(
        const std::vector<std::uint16_t>&,const std::vector<ScoreFileRegion>&,
        std::size_t,std::size_t,std::size_t,std::size_t,const FileScoreCallback&){
        throw std::runtime_error("chunked direct raw GQA scoring unavailable; no fallback");
    }
    // Explicit full-dimension GQA ABI: Q[32,128], K[token,8,128], S[32,token].
    // Never reinterpret the old per-query-head partial-key layout as physical K.
    virtual DeviceScores score_raw_gqa(const std::vector<std::uint16_t>&,
                                      const std::vector<std::uint16_t>&,
                                      std::size_t,std::size_t,std::size_t,std::size_t){
        throw std::invalid_argument("full raw GQA ABI unavailable in this backend");
    }
    virtual FileDeviceScores score_file_raw_gqa(const std::vector<std::uint16_t>&,
                                      const ScoreFileRegion&,
                                      std::size_t,std::size_t,std::size_t,std::size_t){
        throw std::invalid_argument("full raw GQA file ABI unavailable in this backend");
    }
    virtual FileDeviceScores score_resident_file(const std::vector<std::uint16_t>&,std::size_t,std::size_t,std::size_t){
        throw std::runtime_error("explicit preloaded file BO scoring unavailable; no fallback");
    }
    virtual ~FpgaScoreBackend()=default;
    virtual DeviceScores score(const std::vector<std::uint16_t>& q,const std::vector<std::uint16_t>& k,
                               std::size_t heads,std::size_t tokens,std::size_t channels)=0;
    virtual FileDeviceScores score_file(const std::vector<std::uint16_t>&,const ScoreFileRegion&,
                                      std::size_t,std::size_t,std::size_t){
        throw std::runtime_error("file-to-device scoring unavailable in this backend; no staged fallback");
    }
};
void validate_fpga_contract(HardwareMode,const HardwareContract&);
// V3 retains the raw GQA byte layout, with explicit64-token alignment.
inline void validate_raw_gqa_shape(const std::string& abi,std::size_t qh,std::size_t kh,
                                  std::size_t n,std::size_t d){
    const bool pipeline=abi=="HS_FULL_GQA_PIPELINE_F16_V3";
    if((abi!="HS_FULL_GQA_F16_V2"&&!pipeline)||qh!=32||kh!=8||d!=128||!n||n>32768)
        throw std::invalid_argument("raw GQA requires explicit V2/V3 ABI and Q32/KV8/D128,1<=N<=32768");
    if(pipeline&&(n<64||n%64))
        throw std::invalid_argument("HS_FULL_GQA_PIPELINE_F16_V3 requires64<=N<=32768,N%64==0;no padding or fallback");
}
void require_fpga_backend(HardwareMode,const HardwareContract&);
std::unique_ptr<FpgaScoreBackend> make_fpga_backend(HardwareMode,const HardwareContract&);
// Defined only in the real XRT build; never substituted by a CPU implementation.
std::unique_ptr<FpgaScoreBackend> create_xrt_partial_backend(const HardwareContract&,HardwareMode mode=HardwareMode::fpga_host_staged);
}
