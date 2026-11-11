#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace hillserve::runtime {
// Share this fence with the complete-KV writer. SegmentedKVStore appends use
// page RMW; an old record and the next append can occupy the same direct-I/O
// page even though their logical payloads are disjoint. Hold only during file
// reads/writes, never during scoring, CUDA callbacks or future waits.
struct GQAPrefetchIOFence { std::mutex file_io; };

// Immutable once published. One request owns its artifacts; layer+position
// identifies one decode use. No future/current-token payload is included.
// kv_bytes consists of physical_ids.size() rows in physical_ids order. Each
// row is F16 K[8,128] followed by F16 V[8,128], without expanded query heads.
struct GQAPrefetchArtifact {
    static constexpr std::size_t row_bytes=4096, key_bytes=2048;
    std::uint64_t request_id=0;
    int layer=-1;
    std::uint64_t position=0;
    std::vector<std::vector<std::uint64_t>> per_head_ids;
    std::vector<std::uint64_t> physical_ids;
    std::vector<std::uint8_t> kv_bytes;
    std::uint64_t score_ready_ns=0,read_started_ns=0,read_finished_ns=0;
    std::uint64_t hot_bytes=0,cold_bytes=0,cold_syscall_bytes=0,cold_read_calls=0;
    // CPU steady-clock durations, disjoint within this read task. These are
    // host observations, not physical NAND latency or bandwidth estimates.
    std::uint64_t plan_ns=0,hot_copy_ns=0,io_lock_wait_ns=0,extent_check_ns=0;
    std::uint64_t pread_ns=0,validate_copy_ns=0,checksum_ns=0;
    std::uint64_t required_page_bytes=0,gap_read_bytes=0,extent_checks=0;
};
using GQAPrefetchReady=std::function<void(std::shared_ptr<const GQAPrefetchArtifact>)>;
} // namespace hillserve::runtime
