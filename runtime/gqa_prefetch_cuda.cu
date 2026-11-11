#include "runtime/gqa_prefetch_cuda.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
namespace hillserve {
namespace {
using Clock=std::chrono::steady_clock;
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
void check(cudaError_t status,const char* op){if(status!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(status));}
__global__ void gather_gqa(const std::uint16_t* prefetched,const std::uint16_t* misses,
 const std::int32_t* rows,std::uint16_t* k,std::uint16_t* v,std::size_t count){
 const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=count*32*128)return;
 const std::size_t d=i%128,h=(i/128)%32,s=i/(32*128);
 const std::int32_t row=rows[h*count+s];
 std::uint16_t kval=0,vval=0;
 if(row!=-1){const auto* src=row>=0?prefetched+std::size_t(row)*2048:misses+std::size_t(-2-row)*2048;
  kval=src[(h/4)*128+d];vval=src[1024+(h/4)*128+d];}
 k[i]=kval;v[(h*128+d)*count+s]=vval;
}
}
struct GQAPrefetchCuda::Impl {
 int device=0;std::size_t rows=0,bytes=0;
 struct Slot {void* host=nullptr;void* gpu=nullptr;cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,ready=nullptr;std::size_t valid=0;};
 Slot slots[2];void* misses=nullptr;std::int32_t* map=nullptr;cudaEvent_t delivery=nullptr;
 // Only the main/consumer thread changes these. The stream is borrowed from
 // ggml; retain it only while work may still reference our device allocations.
 cudaStream_t consume_stream=nullptr;bool consume_pending=false;
 ~Impl(){
  (void)cudaSetDevice(device);
  // A gather can still read a staging slot if delivery failed before its event
  // was recorded. Drain the actual consumer stream before freeing any slot.
  // Cleanup is non-throwing so it cannot replace the original CUDA failure.
  if(consume_pending&&consume_stream)(void)cudaStreamSynchronize(consume_stream);
  if(delivery)(void)cudaEventSynchronize(delivery);
  for(auto& s:slots)if(s.stream)(void)cudaStreamSynchronize(s.stream);
  for(auto& s:slots){if(s.start)(void)cudaEventDestroy(s.start);if(s.ready)(void)cudaEventDestroy(s.ready);if(s.gpu)(void)cudaFree(s.gpu);if(s.host)(void)cudaFreeHost(s.host);if(s.stream)(void)cudaStreamDestroy(s.stream);}
  if(misses)(void)cudaFree(misses);if(map)(void)cudaFree(map);if(delivery)(void)cudaEventDestroy(delivery);
 }
};
GQAPrefetchCuda::GQAPrefetchCuda(void* pointer,std::size_t max_rows):impl_(new Impl){
 auto& p=*impl_;cudaPointerAttributes attributes{};check(cudaPointerGetAttributes(&attributes,pointer),"prefetch GPU pointer attributes");
 if(attributes.type!=cudaMemoryTypeDevice)throw std::runtime_error("prefetch target must be CUDA device memory");
 p.device=attributes.device;p.rows=max_rows;p.bytes=max_rows*4096;check(cudaSetDevice(p.device),"prefetch set device");
 for(auto& s:p.slots){check(cudaMalloc(&s.gpu,p.bytes),"prefetch slot allocation");check(cudaMallocHost(&s.host,p.bytes),"prefetch pinned slot allocation");check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking),"prefetch private stream");check(cudaEventCreate(&s.start),"prefetch start event");check(cudaEventCreate(&s.ready),"prefetch ready event");}
 check(cudaMalloc(&p.misses,p.bytes),"prefetch miss allocation");check(cudaMalloc(reinterpret_cast<void**>(&p.map),max_rows*32*sizeof(std::int32_t)),"prefetch map allocation");check(cudaEventCreateWithFlags(&p.delivery,cudaEventDisableTiming),"prefetch delivery event");
}
GQAPrefetchCuda::~GQAPrefetchCuda()=default;
std::size_t GQAPrefetchCuda::gpu_bytes()const{return impl_->bytes*3+impl_->rows*32*sizeof(std::int32_t);}
std::size_t GQAPrefetchCuda::host_bytes()const{return impl_->bytes*2;}
GQAPrefetchCuda::Upload GQAPrefetchCuda::stage(unsigned slot,const std::vector<std::uint8_t>& payload){
 auto& p=*impl_;if(slot>=2||payload.size()>p.bytes||payload.size()%4096)throw std::runtime_error("prefetch CUDA slot geometry");auto& s=p.slots[slot];check(cudaSetDevice(p.device),"prefetch worker device");
 Upload out;if(!payload.empty())std::memcpy(s.host,payload.data(),payload.size());out.started_ns=now_ns();out.bytes=payload.size();s.valid=payload.size()/4096;
 try {
 check(cudaEventRecord(s.start,s.stream),"prefetch upload start");if(!payload.empty())check(cudaMemcpyAsync(s.gpu,s.host,payload.size(),cudaMemcpyHostToDevice,s.stream),"prefetch H2D");check(cudaEventRecord(s.ready,s.stream),"prefetch ready record");
 // The policy future owns this callback until all uploads complete. This wait is
 // on the background thread and never touches the active attention workspace.
 check(cudaEventSynchronize(s.ready),"prefetch upload completion");out.ready_observed_ns=now_ns();check(cudaEventElapsedTime(&out.device_ms,s.start,s.ready),"prefetch upload interval");return out;
 }catch(...){
  // Do not let a failed submission release/reuse the pinned source while an
  // earlier successful H2D on this private stream may still be consuming it.
  (void)cudaStreamSynchronize(s.stream);
  throw;
 }
}
GQAPrefetchCuda::Delivery GQAPrefetchCuda::deliver(unsigned slot,const std::vector<std::int32_t>& rows,const std::vector<std::uint8_t>& misses,const float* mask,std::size_t mask_elements,std::size_t count,void* k,void* v,void* gpu_mask,void* raw_stream){
 auto& p=*impl_;if(slot>=2||!raw_stream||count>p.rows||rows.size()!=count*32||misses.size()>p.bytes||misses.size()%4096)throw std::runtime_error("prefetch CUDA delivery geometry/stream");
 auto& s=p.slots[slot];std::size_t valid_pairs=0;for(auto row:rows){if(row>=0&&std::size_t(row)>=s.valid)throw std::runtime_error("prefetch row outside uploaded payload");if(row<=-2&&std::size_t(-2-row)>=misses.size()/4096)throw std::runtime_error("prefetch miss row outside payload");if(row!=-1)++valid_pairs;}
 check(cudaSetDevice(p.device),"prefetch consumer device");const auto stream=static_cast<cudaStream_t>(raw_stream);Delivery out;out.started_ns=now_ns();
 p.consume_stream=stream;p.consume_pending=true;
 try {
 check(cudaStreamWaitEvent(stream,s.ready,0),"attention stream waits for prefetched payload");
 if(!misses.empty())check(cudaMemcpyAsync(p.misses,misses.data(),misses.size(),cudaMemcpyHostToDevice,stream),"prefetch miss H2D");
 check(cudaMemcpyAsync(p.map,rows.data(),rows.size()*sizeof(std::int32_t),cudaMemcpyHostToDevice,stream),"prefetch gather map H2D");
 check(cudaMemcpyAsync(gpu_mask,mask,mask_elements*sizeof(float),cudaMemcpyHostToDevice,stream),"prefetch attention mask H2D");
 gather_gqa<<<(count*32*128+255)/256,256,0,stream>>>(static_cast<const std::uint16_t*>(s.gpu),static_cast<const std::uint16_t*>(p.misses),p.map,static_cast<std::uint16_t*>(k),static_cast<std::uint16_t*>(v),count);check(cudaGetLastError(),"prefetch gather launch");
 check(cudaEventRecord(p.delivery,stream),"prefetch delivery completion record");const auto wait=Clock::now();check(cudaEventSynchronize(p.delivery),"prefetch delivery completion");out.wait_ms=std::chrono::duration<double,std::milli>(Clock::now()-wait).count();out.ready_observed_ns=now_ns();out.h2d_bytes=misses.size()+rows.size()*sizeof(std::int32_t)+mask_elements*sizeof(float);out.d2d_bytes=valid_pairs*128*4;p.consume_pending=false;p.consume_stream=nullptr;return out;
 }catch(...){
  // rows/misses/mask belong to the caller. Their lifetimes must include all
  // already-enqueued copies, including failures before delivery-event record.
  // Ignore cleanup status and rethrow the original submission/completion error.
  if(cudaStreamSynchronize(stream)==cudaSuccess){p.consume_pending=false;p.consume_stream=nullptr;}
  throw;
 }
}
}
