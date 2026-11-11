#include "storage/segmented_store.hpp"
#include "storage/file_store.hpp"
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <stdexcept>
#include <array>
#include <algorithm>
#include <memory>
#include <limits>
#include <cstdlib>
#include <sys/stat.h>
namespace hillserve {
using Clock=std::chrono::steady_clock;
static double millis(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct RecordHeader {std::uint64_t digest,position,request;std::uint32_t layer,version;};
static_assert(sizeof(RecordHeader)==32,"fixed on-disk header");
std::uint64_t SegmentedKVStore::offset(std::size_t l,std::uint64_t p)const{
 if(l>=next_.size()||p>=capacity_)throw std::out_of_range("segment identity/capacity");
 return 4096+(l*capacity_+p)*stride_;
}
void SegmentedKVStore::write_all(const void* data,std::size_t size,std::uint64_t off){
 std::size_t done=0;while(done<size){auto n=pwrite(fd_,static_cast<const char*>(data)+done,size-done,off+done);if(n<0&&errno==EINTR)continue;if(n<=0){poisoned_=true;throw std::runtime_error("segmented pwrite failed");}done+=n;metrics.syscall_write_bytes+=n;}
}
void SegmentedKVStore::read_all(void* data,std::size_t size,std::uint64_t off){
 std::size_t done=0;while(done<size){auto n=pread(fd_,static_cast<char*>(data)+done,size-done,off+done);if(n<0&&errno==EINTR)continue;if(n<=0)throw std::runtime_error("segmented short/error read");done+=n;metrics.syscall_read_bytes+=n;}
}
SegmentedKVStore::SegmentedKVStore(const std::filesystem::path& path,std::uint64_t req,std::size_t layers,std::uint64_t positions,std::size_t bytes,std::uint64_t quota)
 :request_(req),capacity_(positions),record_bytes_(bytes),stride_(bytes+32),quota_(quota),file_extent_(0),next_(layers,0){
 if(!layers||layers>128||!positions||positions>66048||!bytes||bytes>65536||quota>9ULL*1024*1024*1024||quota<4096)throw std::invalid_argument("bounded segmented geometry/quota");
 file_extent_=(4096+std::uint64_t(layers)*positions*stride_+4095)/4096*4096;if(file_extent_>quota_)throw std::length_error("segmented extent exceeds explicit quota");
 fd_=open(path.c_str(),O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);if(fd_<0)throw std::runtime_error("exclusive segmented file creation failed");
 try{std::array<std::uint64_t,512> header{};header[0]=0x48534b5653454731ULL;header[1]=1;header[2]=req;header[3]=layers;header[4]=positions;header[5]=bytes;header[6]=stride_;header[7]=file_extent_;write_all(header.data(),4096,0);
  if(ftruncate(fd_,file_extent_))throw std::runtime_error("bounded sparse file extent failed");
 }catch(...){close(fd_);fd_=-1;throw;}
}
SegmentedKVStore::~SegmentedKVStore(){if(direct_write_fd_>=0)close(direct_write_fd_);if(direct_fd_>=0)close(direct_fd_);if(fd_>=0)close(fd_);}
void SegmentedKVStore::enable_direct_reads(const std::filesystem::path& path){
#ifdef O_DIRECT
 if(direct_fd_>=0)throw std::runtime_error("direct reads already enabled");
 int fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);if(fd<0)throw std::runtime_error("O_DIRECT unavailable; no buffered fallback");
 struct stat a{},b{};if(fstat(fd_,&a)||fstat(fd,&b)||a.st_ino!=b.st_ino||a.st_dev!=b.st_dev){close(fd);throw std::runtime_error("direct descriptor identity mismatch");}direct_fd_=fd;
#else
 (void)path;throw std::runtime_error("O_DIRECT unsupported on this platform");
#endif
}
void SegmentedKVStore::enable_direct_writes(const std::filesystem::path& path){
#ifdef O_DIRECT
 if(poisoned_||direct_write_fd_>=0||std::any_of(next_.begin(),next_.end(),[](std::uint64_t n){return n!=0;}))throw std::runtime_error("direct writes require clean store before first payload append");
 int fd=open(path.c_str(),O_RDWR|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);if(fd<0)throw std::runtime_error("O_DIRECT writes unavailable; no buffered fallback");
 struct stat a{},b{};if(fstat(fd_,&a)||fstat(fd,&b)||!S_ISREG(b.st_mode)||a.st_ino!=b.st_ino||a.st_dev!=b.st_dev||std::uint64_t(b.st_size)!=file_extent_){close(fd);throw std::runtime_error("direct writer identity/extent mismatch");}
 direct_write_fd_=fd;
#else
 (void)path;throw std::runtime_error("O_DIRECT writes unsupported on this platform");
#endif
}
void SegmentedKVStore::append_direct(std::size_t layer,std::uint64_t first,const std::vector<std::uint8_t>& data){
 const auto count=data.size()/record_bytes_;const auto logical_begin=offset(layer,first),logical_end=logical_begin+count*stride_;
 const auto begin=logical_begin/4096*4096,end=(logical_end+4095)/4096*4096;
 if(end>file_extent_||end>quota_)throw std::length_error("aligned direct append exceeds file/quota");
 struct stat st{};if(fstat(direct_write_fd_,&st)||std::uint64_t(st.st_size)!=file_extent_)throw std::runtime_error("direct append extent changed externally");
 const std::size_t staging=std::min<std::uint64_t>(65536,end-begin);void* ptr=nullptr;if(posix_memalign(&ptr,4096,staging))throw std::bad_alloc();
 std::unique_ptr<void,decltype(&free)> buffer(ptr,&free);auto* raw=static_cast<char*>(ptr);metrics.max_append_staging_bytes=std::max(metrics.max_append_staging_bytes,staging);
 for(auto at=begin;at<end;at+=staging){const std::size_t size=std::min<std::uint64_t>(staging,end-at);std::memset(raw,0,size);std::size_t done=0;
  // Sparse preallocated holes read as zero. Zero-fill an EOF tail; file extent
  // was checked above, and this executor has no concurrent writers/truncators.
  while(done<size){auto got=pread(direct_write_fd_,raw+done,size-done,at+done);if(got<0&&errno==EINTR)continue;if(got<0)throw std::runtime_error("direct append RMW pread failed; no fallback");++metrics.direct_rmw_read_calls;if(!got)break;done+=got;metrics.direct_rmw_read_bytes+=got;metrics.syscall_read_bytes+=got;if(done<size&&done%4096)throw std::runtime_error("unaligned short direct RMW read");}
  const auto window_end=at+size;
  auto copy_overlap=[&](const void* src,std::uint64_t off,std::size_t length){auto lo=std::max(at,off),hi=std::min(window_end,off+length);if(lo<hi)std::memcpy(raw+(lo-at),static_cast<const char*>(src)+(lo-off),hi-lo);};
  auto index=at>logical_begin?(at-logical_begin)/stride_:0;
  for(;index<count&&logical_begin+index*stride_<window_end;++index){auto off=logical_begin+index*stride_;const auto* payload=data.data()+index*record_bytes_;
   if(off+sizeof(RecordHeader)>at){RecordHeader hdr{checksum(payload,record_bytes_),first+index,request_,std::uint32_t(layer),1};copy_overlap(&hdr,off,sizeof hdr);}
   copy_overlap(payload,off+sizeof(RecordHeader),record_bytes_);
  }
  done=0;while(done<size){auto put=pwrite(direct_write_fd_,raw+done,size-done,at+done);if(put<0&&errno==EINTR)continue;if(put<=0)throw std::runtime_error("direct append pwrite failed; no fallback");++metrics.direct_write_calls;done+=put;metrics.direct_write_bytes+=put;metrics.syscall_write_bytes+=put;if(done<size&&done%4096)throw std::runtime_error("unaligned short direct append write");}
 }
}
void SegmentedKVStore::append(std::size_t layer,std::uint64_t first,const std::vector<std::uint8_t>& data){
 if(poisoned_||layer>=next_.size()||data.empty()||data.size()%record_bytes_||first!=next_[layer])throw std::invalid_argument("append epoch/size/poisoned state");
 const auto count=data.size()/record_bytes_;if(first>capacity_||count>capacity_-first)throw std::length_error("append capacity");auto t=Clock::now();
 try{if(direct_write_fd_>=0)append_direct(layer,first,data);else for(std::size_t i=0;i<count;++i){auto* payload=data.data()+i*record_bytes_;RecordHeader hdr{checksum(payload,record_bytes_),first+i,request_,std::uint32_t(layer),1};auto off=offset(layer,first+i);write_all(&hdr,sizeof hdr,off);write_all(payload,record_bytes_,off+sizeof hdr);}}
 catch(...){poisoned_=true;throw;}
 next_[layer]+=count;metrics.records_appended+=count;metrics.logical_append_bytes+=data.size();metrics.append_ms+=millis(t);
}
std::vector<std::uint8_t> SegmentedKVStore::fetch(std::size_t layer,const std::vector<std::uint64_t>& ids,std::size_t budget){
 if(poisoned_||layer>=next_.size())throw std::invalid_argument("read poisoned/invalid layer");
 if(ids.size()>budget/record_bytes_)throw std::length_error("selected staging budget");
 for(auto id:ids)if(id>=next_[layer])throw std::out_of_range("read before append completion");
 auto t=Clock::now();std::vector<std::uint8_t> data(ids.size()*record_bytes_);
 auto validate=[&](std::size_t i,const RecordHeader& hdr){
  if(hdr.position!=ids[i]||hdr.request!=request_||hdr.layer!=layer||hdr.version!=1)throw std::runtime_error("on-disk identity/version mismatch");
  if(checksum(data.data()+i*record_bytes_,record_bytes_)!=hdr.digest)throw std::runtime_error("selected KV checksum failure");
 };
 if(direct_fd_<0){for(std::size_t i=0;i<ids.size();++i){RecordHeader hdr;auto off=offset(layer,ids[i]);read_all(&hdr,sizeof hdr,off);read_all(data.data()+i*record_bytes_,record_bytes_,off+sizeof hdr);validate(i,hdr);}}
 else if(!ids.empty()){
  // 56KiB aligned data plus a bounded 1000-entry order tile and one header
  // stay within 64KiB extra staging. Sorted callers need no full ID copy.
  constexpr std::size_t window_bytes=56*1024,tile_capacity=1000;
  void* ptr=nullptr;if(posix_memalign(&ptr,4096,window_bytes))throw std::bad_alloc();std::unique_ptr<void,decltype(&free)> buffer(ptr,&free);auto* raw=static_cast<char*>(ptr);
  std::array<std::size_t,tile_capacity> order{};
  metrics.max_fetch_staging_bytes=std::max(metrics.max_fetch_staging_bytes,window_bytes+sizeof order+sizeof(RecordHeader));
  auto read_window=[&](std::uint64_t at,std::size_t bytes){
   if(at%4096||!bytes||bytes%4096||bytes>window_bytes||at>file_extent_||bytes>file_extent_-at)throw std::logic_error("bounded direct fetch window");
   ssize_t got;do{got=pread(direct_fd_,raw,bytes,at);}while(got<0&&errno==EINTR);++metrics.direct_fetch_read_calls;if(got>0)metrics.syscall_read_bytes+=got;
   if(got!=ssize_t(bytes))throw std::runtime_error("coalesced direct KV read failed; no fallback");
  };
  const bool sorted=std::is_sorted(ids.begin(),ids.end());
  for(std::size_t tile=0;tile<ids.size();){const auto count=sorted?ids.size():std::min(tile_capacity,ids.size()-tile);
   if(!sorted){for(std::size_t i=0;i<count;++i)order[i]=tile+i;std::sort(order.begin(),order.begin()+count,[&](std::size_t a,std::size_t b){return ids[a]!=ids[b]?ids[a]<ids[b]:a<b;});}
   auto index=[&](std::size_t i){return sorted?i:order[i];};
   std::size_t cursor=0;
   while(cursor<count){const auto original=index(cursor),off=offset(layer,ids[original]);auto begin=off/4096*4096,end=(off+stride_+4095)/4096*4096;
    if(end-begin>window_bytes){
     // The supported max-size record can exceed one window. Stream its parts
     // into the caller-owned final payload; never allocate a larger read buffer.
     RecordHeader hdr{};auto copy_part=[&](void* dst,std::uint64_t dst_off,std::size_t length,std::uint64_t at,std::size_t bytes){auto lo=std::max(dst_off,at),hi=std::min(dst_off+length,at+bytes);if(lo<hi)std::memcpy(static_cast<char*>(dst)+(lo-dst_off),raw+(lo-at),hi-lo);};
     for(auto at=begin;at<end;at+=window_bytes){auto bytes=std::min<std::uint64_t>(window_bytes,end-at);read_window(at,bytes);copy_part(&hdr,off,sizeof hdr,at,bytes);copy_part(data.data()+original*record_bytes_,off+sizeof hdr,record_bytes_,at,bytes);}
     validate(original,hdr);++cursor;
     while(cursor<count&&ids[index(cursor)]==ids[original]){std::memcpy(data.data()+index(cursor)*record_bytes_,data.data()+original*record_bytes_,record_bytes_);++cursor;}
     continue;
    }
    auto finish=cursor+1;
    while(finish<count){auto next_off=offset(layer,ids[index(finish)]),next_begin=next_off/4096*4096,next_end=(next_off+stride_+4095)/4096*4096;
     if(next_begin>end||next_end-begin>window_bytes)break;
     end=std::max(end,next_end);++finish;
    }
    read_window(begin,end-begin);
    for(auto i=cursor;i<finish;++i){auto dest=index(i),record_off=offset(layer,ids[dest]);RecordHeader hdr;std::memcpy(&hdr,raw+(record_off-begin),sizeof hdr);std::memcpy(data.data()+dest*record_bytes_,raw+(record_off-begin)+sizeof hdr,record_bytes_);validate(dest,hdr);}
    cursor=finish;
   }
   tile+=count;
  }
 }
 metrics.records_read+=ids.size();metrics.logical_read_bytes+=data.size();metrics.fetch_ms+=millis(t);return data;
}
void SegmentedKVStore::drain(){if(poisoned_)throw std::runtime_error("drain poisoned store");auto t=Clock::now();if(fdatasync(fd_)){poisoned_=true;throw std::runtime_error("end-of-request sync failed");}metrics.drain_ms+=millis(t);++metrics.drain_calls;}
}
