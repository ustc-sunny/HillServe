#include "storage/file_store.hpp"
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
namespace hillserve {
std::uint64_t checksum(const void* data,std::size_t n){
    const auto* p=static_cast<const std::uint8_t*>(data);
    std::uint64_t h=14695981039346656037ULL;
    for(std::size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;
}
struct PhaseTimer {
    double& ms;std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    explicit PhaseTimer(double& value):ms(value){}
    ~PhaseTimer(){ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
};
static void io_error(const char* what){throw std::system_error(errno,std::generic_category(),what);}
FileStore::FileStore(const std::filesystem::path& p,std::size_t limit):path_(p),limit_(limit){
    if(!limit||limit>256ULL*1024*1024)throw std::invalid_argument("pilot file cap must be 1..256 MiB");
    fd_=::open(p.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd_<0)io_error("create new store");
    struct stat s{};
    if(fstat(fd_,&s)||!S_ISREG(s.st_mode)){close(fd_);fd_=-1;throw std::runtime_error("not regular file");}
}
FileStore::~FileStore(){if(fd_>=0)close(fd_);}
void FileStore::replace(const std::vector<std::uint8_t>& data){
    if(data.size()>limit_)throw std::length_error("storage cap exceeded");
    valid_=false;
    std::size_t pos=0;
    while(pos<data.size()){
        auto n=pwrite(fd_,data.data()+pos,data.size()-pos,static_cast<off_t>(pos));
        if(n<0&&errno==EINTR)continue;
        if(n<0)io_error("pwrite");
        if(n==0)throw std::runtime_error("zero write");
        pos+=static_cast<std::size_t>(n);write_bytes+=n;
    }
    if(ftruncate(fd_,data.size()))io_error("ftruncate owned file");
    if(fdatasync(fd_))io_error("fdatasync");
    size_=data.size();digest_=checksum(data.data(),data.size());valid_=true;
}
std::vector<std::uint8_t> FileStore::fetch(){
    if(!valid_)throw std::logic_error("no completed write");
    std::vector<std::uint8_t> data(size_);
    std::size_t pos=0;
    while(pos<data.size()){
        auto n=pread(fd_,data.data()+pos,data.size()-pos,static_cast<off_t>(pos));
        if(n<0&&errno==EINTR)continue;
        if(n<0)io_error("pread");
        if(n==0)throw std::runtime_error("short read");
        pos+=static_cast<std::size_t>(n);read_bytes+=n;
    }
    if(checksum(data.data(),data.size())!=digest_)throw std::runtime_error("payload checksum mismatch");
    return data;
}
IndexedKVStore::IndexedKVStore(const std::filesystem::path& path,std::size_t cap,std::size_t records,std::size_t metadata_cap,KVDurability durability)
    :file_cap_(cap),record_cap_(records),durability_(durability),meter_{0,0,metadata_cap},index_(std::less<Key>{},MeteredAllocator<std::pair<const Key,Record>>(&meter_)){
    if(!cap||cap>256ULL*1024*1024||!records||records>1000000)throw std::invalid_argument("invalid bounded store capacity");
    fd_=open(path.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd_<0)io_error("create indexed KV store");
}
IndexedKVStore::~IndexedKVStore(){if(fd_>=0)close(fd_);}
void IndexedKVStore::append_kv(std::uint64_t req,std::uint64_t layer,std::uint64_t token,
                               const std::vector<std::uint8_t>& payload){
    if(poisoned_)throw std::logic_error("previous append failed");
    Key key{req,layer,token};
    std::uint64_t digest;{PhaseTimer timer(timings.checksum_ms);digest=checksum(payload.data(),payload.size());}
    {PhaseTimer timer(timings.index_ms);
    if(index_.count(key))throw std::invalid_argument("duplicate KV identity");
    if(index_.size()>=record_cap_||payload.size()>file_cap_-end_)throw std::length_error("KV backing budget exceeded");
    // Reserve metadata before writing. Roll back visibility on any write/completion failure.
    index_.emplace(key,Record{end_,payload.size(),digest});}
    try{
        std::size_t n=0;
        {PhaseTimer timer(timings.pwrite_ms);while(n<payload.size()){
            auto written=pwrite(fd_,payload.data()+n,payload.size()-n,end_+n);
            if(written<0&&errno==EINTR)continue;
            if(written<0)io_error("append KV");
            if(written==0)throw std::runtime_error("zero KV write");
            n+=written;append_write_bytes+=written;
        }
        }
        unsynced_bytes_+=payload.size();dirty_peak_=std::max(dirty_peak_,unsynced_bytes_);
        if(!batch_active_ && durability_==KVDurability::durable_per_layer)synchronize(false);
        end_+=payload.size();
    }catch(...){index_.erase(key);poisoned_=true;throw;}
}
std::vector<KVPayload> IndexedKVStore::fetch_selected(std::uint64_t req,std::uint64_t layer,
                   const std::vector<std::uint64_t>& tokens,std::size_t active_cap){
    if(poisoned_)throw std::logic_error("previous append failed");
    if(batch_active_)throw std::logic_error("append batch not completed");
    std::size_t total=0;std::set<std::uint64_t> unique;
    {PhaseTimer timer(timings.index_ms);for(auto token:tokens){
        if(!unique.insert(token).second)throw std::invalid_argument("duplicate selected ID");
        auto it=index_.find(Key{req,layer,token});
        if(it==index_.end())throw std::out_of_range("unknown selected KV identity");
        if(it->second.size>active_cap-total)throw std::length_error("active fetch budget exceeded");
        total+=it->second.size;
    }
    }
    std::vector<KVPayload> out;out.reserve(tokens.size());
    for(auto token:tokens){
        Record rec;{PhaseTimer timer(timings.index_ms);rec=index_.at(Key{req,layer,token});}
        KVPayload p{token,std::vector<std::uint8_t>(rec.size)};std::size_t n=0;
        {PhaseTimer timer(timings.pread_ms);while(n<p.bytes.size()){
            auto got=pread(fd_,p.bytes.data()+n,p.bytes.size()-n,rec.offset+n);
            if(got<0&&errno==EINTR)continue;
            if(got<0)io_error("fetch selected KV");
            if(got==0)throw std::runtime_error("short selected KV read");
            n+=got;selected_read_bytes+=got;
        }
        }
        {PhaseTimer timer(timings.checksum_ms);if(checksum(p.bytes.data(),p.bytes.size())!=rec.digest)throw std::runtime_error("selected KV corruption");}
        out.push_back(std::move(p));
    }
    return out;
}
void IndexedKVStore::begin_append_batch(){
    if(poisoned_||batch_active_)throw std::logic_error("invalid append batch state");
    batch_active_=true;
}
void IndexedKVStore::end_append_batch(){
    if(poisoned_||!batch_active_)throw std::logic_error("invalid completion state");
    if(durability_==KVDurability::durable_per_layer)synchronize(false);
    // All pwrite calls have completed successfully; publish batch visibility even without crash durability.
    batch_active_=false;
}
void IndexedKVStore::synchronize(bool final){
    if(final){PhaseTimer timer(final_drain_ms);++final_drain_calls;
        if(fdatasync(fd_)){poisoned_=true;io_error("final KV drain");}}
    else {PhaseTimer timer(timings.fdatasync_ms);++timings.sync_calls;
        if(fdatasync(fd_)){poisoned_=true;io_error("complete KV batch");}}
    unsynced_bytes_=0;
}
void IndexedKVStore::drain(){
    if(poisoned_||batch_active_)throw std::logic_error("drain before successful completion");
    synchronize(true);
}
void IndexedKVStore::release_request(std::uint64_t req){
    if(batch_active_)throw std::logic_error("release before append completion");
    for(auto it=index_.begin();it!=index_.end();)if(std::get<0>(it->first)==req)it=index_.erase(it);else ++it;
}
}
