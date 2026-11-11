#pragma once
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace hillserve::storage {
// Bounded blocking preads on persistent IO workers, not an unbounded async
// queue. The caller holds the page-RMW fence until EVERY read has completed,
// including on failure. Buffers/FDs remain caller-owned. No buffered fallback.
class DirectReadBatch {
public:
    static constexpr std::size_t max_depth=4;
    struct Request { int fd=-1; void* data=nullptr; std::size_t bytes=0; std::uint64_t offset=0; };
private:
    struct Result { ssize_t bytes=-1; int error=0; };
    const std::size_t depth_;
    std::array<std::thread,max_depth-1> workers_;
    std::array<Request,max_depth> requests_{};
    std::array<Result,max_depth> results_{};
    std::mutex dispatch_,mutex_;
    std::condition_variable changed_,done_;
    std::uint64_t generation_=0;
    std::size_t count_=0,remaining_=0;
    bool closing_=false;
    static Result read(const Request& r) noexcept {
        ssize_t n;do{n=pread(r.fd,r.data,r.bytes,static_cast<off_t>(r.offset));}while(n<0&&errno==EINTR);
        return {n,n<0?errno:0};
    }
    void worker(std::size_t slot) {
        std::uint64_t observed=0;
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;){
            changed_.wait(lock,[&]{return closing_||generation_!=observed;});
            if(closing_)return;
            observed=generation_;
            if(slot>=count_)continue;
            const auto request=requests_[slot];lock.unlock();
            const auto result=read(request);lock.lock();
            results_[slot]=result;--remaining_;done_.notify_one();
        }
    }
    void stop() noexcept {
        {std::lock_guard<std::mutex> lock(mutex_);closing_=true;}changed_.notify_all();
        for(auto& worker:workers_)if(worker.joinable())worker.join();
    }
public:
    explicit DirectReadBatch(std::size_t depth):depth_(depth){
        if(depth<1||depth>max_depth)throw std::invalid_argument("direct read depth must be 1..4");
        try{for(std::size_t i=1;i<depth;++i)workers_[i-1]=std::thread([this,i]{worker(i);});}
        catch(...){stop();throw;}
    }
    ~DirectReadBatch(){stop();}
    DirectReadBatch(const DirectReadBatch&)=delete;
    DirectReadBatch& operator=(const DirectReadBatch&)=delete;
    void read_exact(const std::array<Request,max_depth>& requests,std::size_t count){
        if(!count||count>depth_)throw std::invalid_argument("direct read batch exceeds depth");
        for(std::size_t i=0;i<count;++i){const auto& r=requests[i];
            if(r.fd<0||!r.data||reinterpret_cast<std::uintptr_t>(r.data)%4096||
               !r.bytes||r.bytes>1024*1024||r.bytes%4096||r.offset%4096)
                throw std::invalid_argument("unaligned or unbounded direct read request");
        }
        std::lock_guard<std::mutex> dispatch(dispatch_);
        {std::lock_guard<std::mutex> lock(mutex_);requests_=requests;count_=count;remaining_=count-1;++generation_;}
        changed_.notify_all();
        const auto first=read(requests[0]);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock,[&]{return remaining_==0;});results_[0]=first;
        // Drain before throwing: no worker may still write into a released
        // buffer or overlap a subsequent writer's page RMW.
        for(std::size_t i=0;i<count;++i)
            if(results_[i].error||results_[i].bytes!=static_cast<ssize_t>(requests[i].bytes))
                throw std::runtime_error("batched direct KV read failed/short; no fallback");
    }
};
} // namespace hillserve::storage
