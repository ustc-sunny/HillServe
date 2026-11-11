#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace hillserve {
std::uint64_t checksum(const void*, std::size_t);
// Owns one newly-created regular file. Synchronous completion; no implicit device claims.
class FileStore {
    int fd_ = -1;
    std::filesystem::path path_;
    std::size_t limit_, size_ = 0;
    std::uint64_t digest_ = 0;
    bool valid_ = false;
public:
    std::uint64_t write_bytes=0, read_bytes=0;
    FileStore(const std::filesystem::path&,std::size_t limit);
    ~FileStore();
    FileStore(const FileStore&)=delete;
    FileStore& operator=(const FileStore&)=delete;
    void replace(const std::vector<std::uint8_t>&);
    std::vector<std::uint8_t> fetch();
    const std::filesystem::path& path()const{return path_;}
};
struct AllocationMeter { std::size_t current=0,peak=0,limit=8*1024*1024; };
template<class T> struct MeteredAllocator {
    using value_type=T;
    AllocationMeter * meter;
    explicit MeteredAllocator(AllocationMeter* m=nullptr):meter(m){}
    template<class U> MeteredAllocator(const MeteredAllocator<U>& a):meter(a.meter){}
    T* allocate(std::size_t n){
        if(!meter || n>std::numeric_limits<std::size_t>::max()/sizeof(T) || n*sizeof(T)>meter->limit-meter->current)
            throw std::length_error("metadata allocation budget");
        auto p=std::allocator<T>{}.allocate(n);meter->current+=n*sizeof(T);meter->peak=std::max(meter->peak,meter->current);return p;
    }
    void deallocate(T* p,std::size_t n){meter->current-=n*sizeof(T);std::allocator<T>{}.deallocate(p,n);}
    template<class U> bool operator==(const MeteredAllocator<U>& a)const{return meter==a.meter;}
    template<class U> bool operator!=(const MeteredAllocator<U>& a)const{return !(*this==a);}
};
struct KVPayload { std::uint64_t token; std::vector<std::uint8_t> bytes; };
// Append-only selected-payload store; preserves original token IDs and request/layer isolation.
// No hidden full payload cache. Metadata and returned active payload have separate limits.
enum class KVDurability { durable_per_layer, volatile_query_kv };
struct StoreTimings { double pwrite_ms=0,pread_ms=0,checksum_ms=0,index_ms=0,fdatasync_ms=0;std::size_t sync_calls=0; };
class IndexedKVStore {
    using Key=std::tuple<std::uint64_t,std::uint64_t,std::uint64_t>;
    struct Record { std::size_t offset,size; std::uint64_t digest; };
    int fd_=-1;
    std::size_t file_cap_,record_cap_,end_=0;
    bool poisoned_=false;
    bool batch_active_=false;
    KVDurability durability_;
    std::size_t unsynced_bytes_=0,dirty_peak_=0;
    void synchronize(bool final);
    AllocationMeter meter_;
    std::map<Key,Record,std::less<Key>,MeteredAllocator<std::pair<const Key,Record>>> index_;
public:
    StoreTimings timings;
    double final_drain_ms=0;std::size_t final_drain_calls=0;
    std::uint64_t append_write_bytes=0,selected_read_bytes=0;
    IndexedKVStore(const std::filesystem::path&,std::size_t file_cap,std::size_t record_cap,std::size_t metadata_cap=8*1024*1024,KVDurability durability=KVDurability::durable_per_layer);
    ~IndexedKVStore();
    IndexedKVStore(const IndexedKVStore&)=delete;
    IndexedKVStore& operator=(const IndexedKVStore&)=delete;
    void append_kv(std::uint64_t request,std::uint64_t layer,std::uint64_t token,
                   const std::vector<std::uint8_t>& payload);
    std::vector<KVPayload> fetch_selected(std::uint64_t request,std::uint64_t layer,
                                         const std::vector<std::uint64_t>& tokens,std::size_t active_cap);
    void begin_append_batch();
    void end_append_batch();
    void drain();
    std::size_t dirty_upper_bound()const{return unsynced_bytes_;}
    std::size_t dirty_peak_upper_bound()const{return dirty_peak_;}
    std::size_t file_cache_reservation()const{return file_cap_;}
    const AllocationMeter& metadata()const{return meter_;}
    void release_request(std::uint64_t request);
    std::size_t records()const{return index_.size();}
};
}
