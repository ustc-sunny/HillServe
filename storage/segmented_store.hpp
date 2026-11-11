#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>
namespace hillserve {
// One request, immutable format geometry, rule-based (layer,position) offsets.
// Index memory is O(layers), payload is only in the bounded exclusive regular file.
class SegmentedKVStore {
 int fd_=-1,direct_fd_=-1,direct_write_fd_=-1;std::uint64_t request_,capacity_,record_bytes_,stride_,quota_,file_extent_;bool poisoned_=false;
 std::vector<std::uint64_t> next_;
 std::uint64_t offset(std::size_t layer,std::uint64_t position)const;
 void write_all(const void*,std::size_t,std::uint64_t);
 void read_all(void*,std::size_t,std::uint64_t);
 void append_direct(std::size_t,std::uint64_t,const std::vector<std::uint8_t>&);
public:
 struct Metrics {std::uint64_t logical_append_bytes=0,logical_read_bytes=0,syscall_write_bytes=0,syscall_read_bytes=0,records_appended=0,records_read=0;double append_ms=0,fetch_ms=0,drain_ms=0;std::size_t drain_calls=0;std::uint64_t direct_write_bytes=0,direct_rmw_read_bytes=0,direct_write_calls=0,direct_rmw_read_calls=0;std::size_t max_append_staging_bytes=0,max_fetch_staging_bytes=0;std::uint64_t direct_fetch_read_calls=0;};
 Metrics metrics;
 SegmentedKVStore(const std::filesystem::path&,std::uint64_t request,std::size_t layers,std::uint64_t positions,std::size_t record_bytes,std::uint64_t quota_bytes);
 ~SegmentedKVStore();
 SegmentedKVStore(const SegmentedKVStore&)=delete;
 SegmentedKVStore& operator=(const SegmentedKVStore&)=delete;
 void append(std::size_t layer,std::uint64_t first,const std::vector<std::uint8_t>& records);
 std::vector<std::uint8_t> fetch(std::size_t layer,const std::vector<std::uint64_t>& positions,std::size_t staging_budget);
 void drain();
 void enable_direct_reads(const std::filesystem::path&);
 // Serialized append only, enable before any payload write. No per-append sync.
 void enable_direct_writes(const std::filesystem::path&);
 bool direct_writes_enabled()const{return direct_write_fd_>=0;}
 std::uint64_t extent_bytes()const{return file_extent_;}
 std::size_t index_bytes()const{return next_.capacity()*sizeof(std::uint64_t);}
 std::uint64_t completed(std::size_t layer)const{return next_.at(layer);}
};
}
