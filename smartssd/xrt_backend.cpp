#include "smartssd/xrt_backend.hpp"
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
namespace hillserve {
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
std::string inspect_xrt_device(unsigned index){xrt::device d(index);return d.get_info<xrt::info::device::name>();}
XrtRawSumResult run_xrt_rawsum(const std::string& image,unsigned device,bool granted,
 const std::vector<std::int16_t>& query,const std::vector<std::int16_t>& keys,std::size_t n){
 if(!granted)throw std::runtime_error("image execution not authorized");
 if(!n||n>16384||query.size()!=32*128||keys.size()!=n*32*128)throw std::runtime_error("OPT raw-sum ABI shape mismatch (no padding/head reinterpretation)");
 // Device image loading is deliberately behind the grant and full shape checks.
 xrt::device dev(device);auto uuid=dev.load_xclbin(image);xrt::kernel kernel(dev,uuid,"kv_filter_kernel");
 auto begin=Clock::now(),t=begin;XrtRawSumResult out;out.scores.resize(n);
 xrt::bo qbo(dev,query.size()*2,kernel.group_id(0));
 xrt::bo kbo(dev,keys.size()*2,kernel.group_id(1));
 xrt::bo sbo(dev,n*8,kernel.group_id(2));out.times.allocation_ms=ms(t);
 t=Clock::now();std::memcpy(qbo.map<void*>(),query.data(),query.size()*2);std::memcpy(kbo.map<void*>(),keys.data(),keys.size()*2);
 qbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);kbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);out.times.h2d_ms=ms(t);
 t=Clock::now();auto run=kernel(qbo,kbo,sbo,std::uint32_t(n));auto state=run.wait(60000);
 if(state!=ERT_CMD_STATE_COMPLETED)throw std::runtime_error("XRT command did not complete (timeout/error)");out.times.kernel_wait_ms=ms(t);
 t=Clock::now();sbo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);std::memcpy(out.scores.data(),sbo.map<void*>(),n*8);out.times.d2h_ms=ms(t);
 out.times.total_ms=ms(begin);out.times.h2d_bytes=(query.size()+keys.size())*2;out.times.d2h_bytes=n*8;return out;
}
}

#include "smartssd/adapter.hpp"
#include "smartssd/stream_layout.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sstream>
#include <array>
#include <future>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
namespace hillserve {
namespace {
[[noreturn]] void aio_fail_stop(const char* reason){
    std::fprintf(stderr,"CSS native AIO fatal: %s; in-flight completion unconfirmed; measurement process fail-stop\n",reason);
    std::fflush(stderr);std::_Exit(70);
}
struct NativeAioReadResult {
    std::uint64_t bytes=0,operations=0,submissions=0,completions=0;
    unsigned peak_inflight=0;
};
// One context is reused across CSS calls. Each read() drains all submissions
// before returning or throwing; only one outstanding read() uses this object.
class NativeAioReader {
    aio_context_t context_=0;
public:
    explicit NativeAioReader(unsigned depth){
        if(depth<2||depth>8)throw std::invalid_argument("CSS native AIO depth must be2..8");
        if(syscall(SYS_io_setup,depth,&context_)!=0)
            throw std::runtime_error("CSS io_setup failed errno="+std::to_string(errno));
    }
    ~NativeAioReader(){if(context_)syscall(SYS_io_destroy,context_);}
    NativeAioReader(const NativeAioReader&)=delete;
    NativeAioReader& operator=(const NativeAioReader&)=delete;
    NativeAioReadResult read(const std::vector<DirectChunkRead>& pieces,std::uint8_t* destination,unsigned depth){
        NativeAioReadResult result;
        for(std::size_t next=0;next<pieces.size();){
            const auto count=std::min<std::size_t>(depth,pieces.size()-next);
            std::array<iocb,8> requests{};std::array<iocb*,8> pointers{};
            for(std::size_t i=0;i<count;++i){
                const auto& piece=pieces[next+i];auto& request=requests[i];
                request.aio_data=i+1;request.aio_lio_opcode=IOCB_CMD_PREAD;
                request.aio_fildes=piece.fd;request.aio_buf=reinterpret_cast<std::uintptr_t>(destination+piece.destination_offset);
                request.aio_nbytes=piece.bytes;request.aio_offset=piece.offset;pointers[i]=&request;
            }
            long accepted;const auto submit_deadline=Clock::now()+std::chrono::seconds(60);
            do{accepted=syscall(SYS_io_submit,context_,count,pointers.data());++result.submissions;}
            while(accepted<0&&errno==EINTR&&Clock::now()<submit_deadline);
            // Partial submit is valid. Drain exactly the accepted prefix, then
            // retry the remainder in the next batch. No request is duplicated.
            if(accepted<=0)throw std::runtime_error("CSS io_submit failed/no progress result="+std::to_string(accepted)+" errno="+std::to_string(accepted<0?errno:0)+"; no pread fallback");
            if(accepted>long(count))aio_fail_stop("io_submit returned impossible request count");
            result.peak_inflight=std::max(result.peak_inflight,unsigned(accepted));
            std::array<bool,8> seen{};std::size_t complete=0;std::string first_error;
            const auto deadline=Clock::now()+std::chrono::seconds(60);
            while(complete<std::size_t(accepted)){
                const auto remaining=std::chrono::duration_cast<std::chrono::nanoseconds>(deadline-Clock::now()).count();
                if(remaining<=0)aio_fail_stop("completion deadline exceeded");
                timespec timeout{long(remaining/1000000000),long(remaining%1000000000)};
                std::array<io_event,8> events{};
                const long got=syscall(SYS_io_getevents,context_,1,accepted-long(complete),events.data(),&timeout);
                if(got<0){if(errno==EINTR)continue;const auto error="io_getevents failed errno="+std::to_string(errno);aio_fail_stop(error.c_str());}
                if(got>accepted-long(complete))aio_fail_stop("io_getevents returned impossible count");
                for(long e=0;e<got;++e){
                    const auto& event=events[e];
                    if(event.data<1||event.data>std::uint64_t(accepted))aio_fail_stop("completion request ID mismatch");
                    const auto index=std::size_t(event.data-1);
                    if(seen[index]||event.obj!=reinterpret_cast<std::uintptr_t>(&requests[index]))aio_fail_stop("duplicate/foreign completion");
                    seen[index]=true;++complete;++result.completions;++result.operations;
                    const auto expected=pieces[next+index].bytes;
                    if(event.res!=std::int64_t(expected)||event.res2!=0){
                        if(first_error.empty())first_error="CSS AIO read failed/short: result="+std::to_string(event.res)+" expected="+std::to_string(expected)+" res2="+std::to_string(event.res2);
                    }else result.bytes+=expected;
                }
            }
            // All accepted iocbs have terminated before stack storage or the
            // destination BO can be reused, including negative/short results.
            if(!first_error.empty())throw std::runtime_error(first_error);
            next+=accepted;
        }
        return result;
    }
};
std::string sha256_image(const std::string& path){
    int fd[2];if(pipe(fd))throw std::runtime_error("SHA256 pipe");
    pid_t pid=fork();if(pid<0){close(fd[0]);close(fd[1]);throw std::runtime_error("SHA256 fork");}
    if(pid==0){close(fd[0]);dup2(fd[1],STDOUT_FILENO);close(fd[1]);execl("/usr/bin/sha256sum","sha256sum","--",path.c_str(),static_cast<char*>(nullptr));_exit(127);}
    close(fd[1]);std::string output;char block[1024];ssize_t got;
    while((got=read(fd[0],block,sizeof block))!=0){if(got<0){if(errno==EINTR)continue;break;}if(output.size()<128)output.append(block,std::min<std::size_t>(got,128-output.size()));}
    close(fd[0]);int status=0;while(waitpid(pid,&status,0)<0){if(errno!=EINTR)throw std::runtime_error("SHA256 wait");}
    if(!WIFEXITED(status)||WEXITSTATUS(status)||output.size()<64)throw std::runtime_error("image SHA256 command failed");
    return output.substr(0,64); // argv execution, no shell interpolation; image load follows exact comparison
}

class XrtPartial final:public FpgaScoreBackend {
    HardwareContract contract_;HardwareMode mode_;std::unique_ptr<xrt::device> dev_;std::unique_ptr<xrt::kernel> kernel_;double init_ms_=0;
    bool full_gqa_=false,pipeline_gqa_=false;
    std::unique_ptr<NativeAioReader> native_aio_;
    void require_full_shape(std::size_t qh,std::size_t kh,std::size_t n,std::size_t d)const{
        validate_raw_gqa_shape(contract_.abi,qh,kh,n,d);
    }
public:
    explicit XrtPartial(const HardwareContract& c,HardwareMode mode):contract_(c),mode_(mode){
        if(mode!=HardwareMode::fpga_host_staged&&mode!=HardwareMode::fpga_storage_streaming)throw std::invalid_argument("unsupported XRT transport mode");
        auto start=Clock::now();
        // Also guard the direct factory; source reuse permission is unrelated.
        pipeline_gqa_=c.abi=="HS_FULL_GQA_PIPELINE_F16_V3";
        full_gqa_=c.abi=="HS_FULL_GQA_F16_V2"||pipeline_gqa_;
        if(!c.image_execute_authorized||(c.abi!="HS_PARTIAL_F16_V1"&&!full_gqa_)||!c.buffer_budget_bytes||c.buffer_budget_bytes>256*1024*1024||!c.direct_read_queue_depth||c.direct_read_queue_depth>8)
            throw std::runtime_error("invalid independent-image action/ABI/buffer contract");
        // Hash the actual image BEFORE opening/loading the FPGA.
        if(sha256_image(c.image_path)!=c.image_sha256)throw std::runtime_error("image SHA256 differs from authorized image");
        dev_=std::make_unique<xrt::device>(c.device_index);auto uuid=dev_->load_xclbin(c.image_path);
        const char* name=pipeline_gqa_?"hillserve_full_gqa_pipeline":full_gqa_?"hillserve_full_gqa_f16":"hillserve_partial_f16";
        kernel_=std::make_unique<xrt::kernel>(*dev_,uuid,name);init_ms_=ms(start);
    }
    struct FileBuffers {
        std::size_t h,n,d; xrt::bo q,k,s;
        FileBuffers(xrt::device& dev,xrt::kernel& kernel,std::size_t hh,std::size_t nn,std::size_t dd,std::size_t kh)
          :h(hh),n(nn),d(dd),q(dev,hh*dd*2,kernel.group_id(0)),
           k(dev,kh*nn*dd*2,xrt::bo::flags::p2p,kernel.group_id(1)),s(dev,hh*nn*2,kernel.group_id(2)){}
    };
    std::unique_ptr<FileBuffers> file_buffers_;bool file_keys_loaded_=false;
    std::array<std::unique_ptr<FileBuffers>,2> stream_buffers_;
    void release_stream_buffers(){for(auto& p:stream_buffers_)p.reset();}
    FileScoreStreamReport score_file_regions_raw_gqa_stream(
        const std::vector<std::uint16_t>& query,const std::vector<ScoreFileRegion>& regions,
        std::size_t qh,std::size_t kh,std::size_t d,std::size_t chunk_tokens,
        const FileScoreCallback& callback)override{
        if(mode_!=HardwareMode::fpga_storage_streaming||!pipeline_gqa_)
            throw std::invalid_argument("CSS direct stream requires explicit V3 pipeline image");
        require_full_shape(qh,kh,chunk_tokens,d);
        if(query.size()!=qh*d||!callback)throw std::invalid_argument("CSS missing query or consumer");
        for(auto x:query)if((x&0x7c00)==0x7c00)throw std::invalid_argument("CSS nonfinite query");
        const auto plan=plan_raw_gqa_stream(regions,chunk_tokens);
        for(const auto& region:regions){
            struct stat st{};
            if(fstat(region.fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||
               region.offset+region.bytes>std::uint64_t(st.st_size))
                throw std::invalid_argument("CSS source must be an existing bounded regular file");
            const int flags=fcntl(region.fd,F_GETFL);
            if(flags<0||!(flags&O_DIRECT))throw std::invalid_argument("CSS requires O_DIRECT, no fallback");
        }
        const auto stamp=[](){return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());};
        const std::size_t slots=std::min<std::size_t>(2,plan.size());
        const std::size_t capacity=std::min(chunk_tokens,plan.front().tokens);
        const std::size_t qb=qh*d*2,kb=kh*capacity*d*2,sb=qh*capacity*2;
        if(slots*(qb+kb+sb)>contract_.buffer_budget_bytes)
            throw std::length_error("CSS double-buffer BO reservation exceeds budget");
        auto started=Clock::now(),t=started;FileScoreStreamReport report;
        report.initialization_ms=init_ms_;report.chunks.reserve(plan.size());
        report.direct_read_queue_depth=contract_.direct_read_queue_depth;
        report.device_buffer_bytes=slots*(qb+kb+sb);report.host_score_buffer_bytes=sb;
        // A backend never retains the old whole-file BO alongside streaming BOs.
        file_buffers_.reset();file_keys_loaded_=false;
        report.buffers_reused=bool(stream_buffers_[0]);
        for(std::size_t slot=0;slot<2;++slot){
            auto& b=stream_buffers_[slot];
            if(slot>=slots){b.reset();continue;}
            if(b&&(b->h!=qh||b->n!=capacity||b->d!=d)){b.reset();report.buffers_reused=false;}
            if(!b){b=std::make_unique<FileBuffers>(*dev_,*kernel_,qh,capacity,d,kh);report.buffers_reused=false;}
            if(b->k.get_flags()!=xrt::bo::flags::p2p||b->k.get_memory_group()!=kernel_->group_id(1))
                throw std::runtime_error("CSS P2P BO metadata mismatch");
            auto* mapped=b->k.map<void*>();
            if(!mapped||reinterpret_cast<std::uintptr_t>(mapped)%4096)
                throw std::runtime_error("CSS P2P mapping missing or unaligned");
        }
        report.p2p_allocation_verified=true;report.allocation_ms=ms(t);
        if(contract_.direct_read_queue_depth>1&&std::any_of(plan.begin(),plan.end(),[](const auto& chunk){return chunk.reads.size()>1;})){
            report.aio_context_reused=bool(native_aio_);t=Clock::now();
            if(!native_aio_)native_aio_=std::make_unique<NativeAioReader>(contract_.direct_read_queue_depth);
            report.aio_setup_ms=ms(t);
        }
        t=Clock::now();
        // Query copied only once per live BO set per request, not per chunk.
        for(std::size_t slot=0;slot<slots;++slot){
            auto& b=*stream_buffers_[slot];std::memcpy(b.q.map<void*>(),query.data(),qb);
            b.q.sync(XCL_BO_SYNC_BO_TO_DEVICE);report.query_h2d_bytes+=qb;
        }
        report.query_h2d_ms=ms(t);
        std::vector<FileScoreChunkInfo> events(plan.size());
        for(std::size_t i=0;i<plan.size();++i){events[i].token_offset=plan[i].token_offset;events[i].tokens=plan[i].tokens;events[i].buffer_slot=i%slots;}
        struct ReadResult{std::uint64_t begin=0,end=0,bytes=0,operations=0,submissions=0,completions=0;unsigned peak_inflight=0;};
        // Only this worker touches K for its slot. It never allocates a host K
        // buffer. Read runs can be non-contiguous in the same SmartSSD file.
        const auto read_chunk=[&](std::size_t index){
            const auto& chunk=plan[index];auto& b=*stream_buffers_[index%slots];
            auto* destination=b.k.map<std::uint8_t*>();ReadResult result;result.begin=stamp();
            if(contract_.direct_read_queue_depth>1&&chunk.reads.size()>1){
                const auto read=native_aio_->read(chunk.reads,destination,contract_.direct_read_queue_depth);
                result.bytes=read.bytes;result.operations=read.operations;result.submissions=read.submissions;result.completions=read.completions;result.peak_inflight=read.peak_inflight;
                result.end=stamp();return result;
            }
            for(const auto& piece:chunk.reads){
                ssize_t got;do{got=pread(piece.fd,destination+piece.destination_offset,piece.bytes,piece.offset);}while(got<0&&errno==EINTR);
                const int saved=errno;
                if(got!=ssize_t(piece.bytes)){
                    std::ostringstream error;error<<"CSS direct read failed: chunk="<<index<<" got="<<got<<" expected="<<piece.bytes<<" errno="<<saved;
                    throw std::runtime_error(error.str());
                }
                result.bytes+=got;++result.operations;
            }
            result.end=stamp();return result;
        };
        FileScoreChunk output;output.values.reserve(qh*capacity);
        const auto deliver=[&](std::size_t index){
            auto& info=events[index];auto& b=*stream_buffers_[index%slots];
            const std::size_t bytes=qh*info.tokens*2;
            auto copy_start=Clock::now();
            // The final short chunk uses its actual N-dependent head stride.
            // Reading only its output avoids returning uninitialized BO tails.
            b.s.sync(XCL_BO_SYNC_BO_FROM_DEVICE,bytes,0);
            output.values.resize(qh*info.tokens);std::memcpy(output.values.data(),b.s.map<void*>(),bytes);
            info.d2h_ms=ms(copy_start);info.score_d2h_bytes=bytes;info.scores_CPU_ready_ns=stamp();
            for(auto x:output.values)if((x&0x7c00)==0x7c00)throw std::runtime_error("CSS nonfinite score");
            info.callback_started_ns=stamp();output.info=info;callback(output);
            info.callback_finished_ns=stamp();info.callback_ms=(info.callback_finished_ns-info.callback_started_ns)/1e6;
            report.file_read_bytes+=info.file_read_bytes;report.score_d2h_bytes+=bytes;
            report.file_read_operations+=info.file_read_operations;report.aio_submit_calls+=info.aio_submit_calls;report.aio_completions+=info.aio_completions;
            report.aio_peak_inflight=std::max(report.aio_peak_inflight,info.aio_peak_inflight);report.chunks.push_back(info);
        };
        std::future<ReadResult> pending_read;
        std::unique_ptr<xrt::run> active;
        bool command_started=false;
        try{
            // Prologue has one unavoidable initial read.
            pending_read=std::async(std::launch::async,read_chunk,0);
            for(std::size_t i=0;i<plan.size();++i){
                // Do not hold completed scores behind a slow next read. If K
                // is ready, prefer overlap with its kernel; otherwise consume
                // the previous scores while the outstanding read continues.
                bool previous_delivered=false;
                if(i&&pending_read.wait_for(std::chrono::seconds(0))!=std::future_status::ready){
                    deliver(i-1);previous_delivered=true;
                }
                const auto read=pending_read.get();auto& info=events[i];
                info.read_started_ns=read.begin;info.read_finished_ns=read.end;info.file_read_bytes=read.bytes;info.file_read_ms=(read.end-read.begin)/1e6;
                info.file_read_operations=read.operations;info.aio_submit_calls=read.submissions;info.aio_completions=read.completions;info.aio_peak_inflight=read.peak_inflight;
                auto& b=*stream_buffers_[i%slots];
                // Allocate and own the run handle before submitting work. In
                // particular, a make_unique failure must never leave a kernel
                // launched by operator() without a handle that can drain it.
                active=std::make_unique<xrt::run>(*kernel_);
                active->set_arg(0,b.q);active->set_arg(1,b.k);active->set_arg(2,b.s);
                active->set_arg(3,std::uint32_t(info.tokens));
                const auto launch=Clock::now();info.kernel_started_ns=stamp();
                // start() itself can throw after partial submission. Treat the
                // command as possibly live until a completion is confirmed.
                command_started=true;active->start();
                info.submission_ms=ms(launch);
                if(contract_.on_kernel_submitted)contract_.on_kernel_submitted();
                // K in the other slot may be overwritten now: its kernel was
                // drained last iteration. Its completed S is a different BO.
                if(i+1<plan.size())pending_read=std::async(std::launch::async,read_chunk,i+1);
                // While kernel i computes, deliver i-1 to the host immediately.
                // This is cross-chunk CSS, not splitting scores after full N.
                if(i&&!previous_delivered)deliver(i-1);
                const auto wait=Clock::now();info.kernel_wait_started_ns=stamp();const auto state=active->wait(60000);
                info.host_wait_ms=ms(wait);info.kernel_finished_ns=stamp();info.launch_wait_ms=ms(launch);
                if(state!=ERT_CMD_STATE_COMPLETED)throw std::runtime_error("CSS kernel failed or timed out");
                command_started=false;active.reset();
            }
            deliver(plan.size()-1); // Epilogue: no later kernel to overlap.
        }catch(...){
            // Do not release or overwrite BOs while a command or direct read
            // still owns them. Draining is required even when a callback fails.
            const auto original_error=std::current_exception();
            bool drained=!command_started;
            if(active&&command_started){
                try{drained=active->wait(60000)==ERT_CMD_STATE_COMPLETED;}catch(...){drained=false;}
            }
            if(pending_read.valid()){try{pending_read.get();}catch(...){}}
            if(!drained){
                // No orderly BO/device destruction is safe after an ambiguous
                // XRT drain failure. Fail-stop this measurement process rather
                // than freeing buffers possibly still owned by the device.
                std::fputs("CSS fatal: device completion could not be confirmed; measurement process fail-stop, original error: ",stderr);
                try{std::rethrow_exception(original_error);}catch(const std::exception& error){std::fputs(error.what(),stderr);}catch(...){std::fputs("non-standard exception",stderr);}
                std::fputs("\n",stderr);std::fflush(stderr);std::_Exit(70);
            }
            active.reset();release_stream_buffers();std::rethrow_exception(original_error);
        }
        report.total_ms=ms(started);
        if(!contract_.reuse_file_buffers)release_stream_buffers();
        return report;
    }
    FileDeviceScores score_file(const std::vector<std::uint16_t>& q,const ScoreFileRegion& file,std::size_t h,std::size_t n,std::size_t d)override{
        if(full_gqa_)throw std::invalid_argument("full GQA image requires explicit score_file_raw_gqa");
        return file_compute(q,&file,h,n,d);
    }
    FileDeviceScores score_file_raw_gqa(const std::vector<std::uint16_t>& q,const ScoreFileRegion& file,
                                      std::size_t qh,std::size_t kh,std::size_t n,std::size_t d)override{
        require_full_shape(qh,kh,n,d);return file_compute(q,&file,qh,n,d);
    }
    FileDeviceScores score_resident_file(const std::vector<std::uint16_t>& q,std::size_t h,std::size_t n,std::size_t d)override{
        if(full_gqa_)throw std::invalid_argument("full GQA pilot requires an actual file read on every call");
        if(!contract_.reuse_file_buffers||!file_keys_loaded_||!file_buffers_||file_buffers_->h!=h||file_buffers_->n!=n||file_buffers_->d!=d)
            throw std::runtime_error("resident scoring requires explicitly primed matching persistent BOs");
        return file_compute(q,nullptr,h,n,d);
    }
    FileDeviceScores file_compute(const std::vector<std::uint16_t>& q,const ScoreFileRegion* file,std::size_t h,std::size_t n,std::size_t d){
        release_stream_buffers();
        if(mode_!=HardwareMode::fpga_storage_streaming)throw std::invalid_argument("file scoring requires explicit direct transport mode");
        if(!h||h>32||!n||n>65504||!d||d>128||q.size()!=h*d)throw std::invalid_argument("P2P shape mismatch");
        if(full_gqa_)require_full_shape(h,8,n,d);
        const std::size_t kh=full_gqa_?8:h;
        const std::size_t qb=q.size()*2,kb=kh*n*d*2,sb=h*n*2;
        if(qb+kb+sb>contract_.buffer_budget_bytes)throw std::length_error("P2P BO budget exceeded");
        if(file){
            if(file->fd<0||file->bytes!=kb||kb%4096||file->offset%4096)throw std::invalid_argument("P2P requires exact page-aligned whole block");
            struct stat st{};if(fstat(file->fd,&st)||!S_ISREG(st.st_mode)||file->offset+kb>std::uint64_t(st.st_size))throw std::invalid_argument("P2P source must be an existing bounded regular file");
            int flags=fcntl(file->fd,F_GETFL);if(flags<0||!(flags&O_DIRECT))throw std::invalid_argument("P2P source fd must be O_DIRECT; no fallback");
        }
        for(auto x:q)if((x&0x7c00)==0x7c00)throw std::invalid_argument("nonfinite P2P query");
        auto begin=Clock::now(),t=begin;FileDeviceScores r;r.values.resize(h*n);r.timings.initialization_ms=init_ms_;
        r.bo_reused=bool(file_buffers_);r.resident_keys=!file;
        if(file_buffers_&&(file_buffers_->h!=h||file_buffers_->n!=n||file_buffers_->d!=d)){file_buffers_.reset();file_keys_loaded_=false;r.bo_reused=false;}
        if(!file_buffers_)file_buffers_=std::make_unique<FileBuffers>(*dev_,*kernel_,h,n,d,kh);
        auto& b=*file_buffers_;const auto bank=kernel_->group_id(1);r.kernel_key_bank=bank;
        r.bo_flags=static_cast<std::uint32_t>(b.k.get_flags());r.bo_memory_group=b.k.get_memory_group();r.bo_device_address=b.k.address();
        r.p2p_allocation_verified=b.k.get_flags()==xrt::bo::flags::p2p&&b.k.get_memory_group()==bank;
        if(!r.p2p_allocation_verified)throw std::runtime_error("P2P BO metadata mismatch; no fallback");
        auto* destination=b.k.map<void*>();r.bo_mapped_address=reinterpret_cast<std::uintptr_t>(destination);
        if(!destination||r.bo_mapped_address%4096)throw std::runtime_error("P2P map missing or unaligned");
        if(file&&file->capture_mapping_evidence){std::ifstream maps("/proc/self/smaps");std::string line;bool active=false;std::ostringstream evidence;
            while(std::getline(maps,line)){std::uintptr_t lo=0,hi=0;std::istringstream header(line);char dash=0;if(header>>std::hex>>lo>>dash>>hi&&dash=='-')active=lo<=r.bo_mapped_address&&r.bo_mapped_address<hi;
                if(active&&(line.find('-')!=std::string::npos||line.rfind("VmFlags:",0)==0))evidence<<line<<'\n';}r.mapping_evidence=evidence.str();}
        r.timings.allocation_ms=ms(t);t=Clock::now();std::memcpy(b.q.map<void*>(),q.data(),qb);r.timings.pack_ms=ms(t);
        if(file){t=Clock::now();ssize_t got;do{got=pread(file->fd,destination,kb,file->offset);}while(got<0&&errno==EINTR);int saved_errno=errno;r.file_read_ms=ms(t);
            if(got!=ssize_t(kb)){std::ostringstream e;e<<"P2P O_DIRECT pread failed: got="<<got<<" expected="<<kb<<" errno="<<saved_errno<<" bank="<<bank<<" flags="<<r.bo_flags;throw std::runtime_error(e.str());}
            r.file_read_bytes=got;file_keys_loaded_=true;
        }
        r.query_h2d_bytes=qb;
        // Completed pread into BAR-backed BO; physical NVMe DMA bounce is not observed here.
        t=Clock::now();b.q.sync(XCL_BO_SYNC_BO_TO_DEVICE);r.timings.h2d_ms=ms(t);
        t=Clock::now();auto run=full_gqa_?(*kernel_)(b.q,b.k,b.s,std::uint32_t(n)):(*kernel_)(b.q,b.k,b.s,std::uint32_t(h),std::uint32_t(n),std::uint32_t(d));
        r.timings.submission_ms=ms(t);r.timings.launch_started_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
        if(contract_.on_kernel_submitted)contract_.on_kernel_submitted();
        const auto wait_start=Clock::now();auto state=run.wait(60000);r.timings.host_wait_ms=ms(wait_start);
        r.timings.launch_finished_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
        if(state!=ERT_CMD_STATE_COMPLETED)throw std::runtime_error("P2P score kernel did not complete");r.kernel_completed=true;r.timings.launch_wait_ms=ms(t);
        t=Clock::now();b.s.sync(XCL_BO_SYNC_BO_FROM_DEVICE);std::memcpy(r.values.data(),b.s.map<void*>(),sb);r.timings.d2h_ms=ms(t);
        for(auto x:r.values)if((x&0x7c00)==0x7c00)throw std::runtime_error("P2P score nonfinite/overflow");
        if(!contract_.reuse_file_buffers){t=Clock::now();file_buffers_.reset();file_keys_loaded_=false;r.timings.release_ms=ms(t);}
        r.timings.total_ms=ms(begin);r.timings.h2d_bytes=qb;r.timings.d2h_bytes=sb;return r;
    }
    DeviceScores score(const std::vector<std::uint16_t>& q,const std::vector<std::uint16_t>& k,std::size_t h,std::size_t n,std::size_t d)override{
        if(full_gqa_)throw std::invalid_argument("full GQA image requires explicit score_raw_gqa");
        return staged_compute(q,k,h,h,n,d);
    }
    DeviceScores score_raw_gqa(const std::vector<std::uint16_t>& q,const std::vector<std::uint16_t>& k,
                              std::size_t qh,std::size_t kh,std::size_t n,std::size_t d)override{
        require_full_shape(qh,kh,n,d);return staged_compute(q,k,qh,kh,n,d);
    }
    DeviceScores staged_compute(const std::vector<std::uint16_t>& q,const std::vector<std::uint16_t>& k,
                              std::size_t h,std::size_t kh,std::size_t n,std::size_t d){
        release_stream_buffers();
        if(mode_!=HardwareMode::fpga_host_staged)throw std::invalid_argument("direct transport cannot accept a host K vector; use score_file, no fallback");
        if(!h||h>32||!n||n>65504||!d||d>128||q.size()!=h*d||k.size()!=kh*n*d)throw std::invalid_argument("staged score ABI shape mismatch");
        std::size_t qb=q.size()*2,kb=k.size()*2,sb=h*n*2;if(qb+kb+sb>contract_.buffer_budget_bytes)throw std::length_error("FPGA BO budget exceeded");
        for(auto x:q)if((x&0x7c00)==0x7c00)throw std::invalid_argument("nonfinite score query");
        for(auto x:k)if((x&0x7c00)==0x7c00)throw std::invalid_argument("nonfinite score key");
        auto begin=Clock::now(),t=begin;DeviceScores r;r.values.resize(h*n);r.timings.initialization_ms=init_ms_;
        xrt::bo qbo(*dev_,qb,kernel_->group_id(0)),kbo(*dev_,kb,kernel_->group_id(1)),sbo(*dev_,sb,kernel_->group_id(2));r.timings.allocation_ms=ms(t);
        t=Clock::now();std::memcpy(qbo.map<void*>(),q.data(),qb);std::memcpy(kbo.map<void*>(),k.data(),kb);r.timings.pack_ms=ms(t);
        t=Clock::now();qbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);kbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);r.timings.h2d_ms=ms(t);
        t=Clock::now();auto run=full_gqa_?(*kernel_)(qbo,kbo,sbo,std::uint32_t(n)):(*kernel_)(qbo,kbo,sbo,std::uint32_t(h),std::uint32_t(n),std::uint32_t(d));
        r.timings.submission_ms=ms(t);r.timings.launch_started_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
        if(contract_.on_kernel_submitted)contract_.on_kernel_submitted();
        const auto wait_start=Clock::now();auto state=run.wait(60000);r.timings.host_wait_ms=ms(wait_start);
        r.timings.launch_finished_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
        if(state!=ERT_CMD_STATE_COMPLETED)throw std::runtime_error("per-head FPGA run failed or timed out");r.timings.launch_wait_ms=ms(t);
        t=Clock::now();sbo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);std::memcpy(r.values.data(),sbo.map<void*>(),sb);r.timings.d2h_ms=ms(t);
        for(auto x:r.values)if((x&0x7c00)==0x7c00)throw std::runtime_error("FPGA nonfinite/overflow score");
        r.timings.total_ms=ms(begin);r.timings.h2d_bytes=qb+kb;r.timings.d2h_bytes=sb;return r;
    }
};
}
std::unique_ptr<FpgaScoreBackend> create_xrt_partial_backend(const HardwareContract& c,HardwareMode mode){return std::make_unique<XrtPartial>(c,mode);}
}
