#include "runtime/gqa_delivery.hpp"
#include "storage/file_store.hpp"
namespace hillserve {
using namespace replay;
static void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct GQADeliveryWorkspace::Impl {
 std::size_t count;std::vector<std::uint16_t> kh,vh;std::vector<float> mh;
 ggml_backend_t backend=nullptr;ggml_context*ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_tensor *k=nullptr,*v=nullptr,*mask=nullptr,*q=nullptr,*output=nullptr;ggml_cgraph*graph=nullptr;cudaEvent_t start{},done{};
 explicit Impl(std::size_t c):count(c),kh(c*4096),vh(c*4096),mh(c*64*32){}
 ~Impl(){if(start)cudaEventDestroy(start);if(done)cudaEventDestroy(done);if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);if(backend)ggml_backend_free(backend);}
};
GQADeliveryWorkspace::GQADeliveryWorkspace(std::size_t count,const std::vector<float>& actual):p(new Impl(count)){
 need(count>0&&count<=6592&&count%32==0&&actual.size()==4096,"bounded real GQA delivery shape");
 ggml_backend_load_all();auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(dev,"GPU required");p->backend=ggml_backend_dev_init(dev,nullptr);need(p->backend&&ggml_backend_is_cuda(p->backend),"actual CUDA backend required");p->ctx=ggml_init({2*1024*1024,nullptr,true});need(p->ctx,"graph metadata");
 p->k=ggml_new_tensor_1d(p->ctx,GGML_TYPE_F16,count*4096);p->v=ggml_new_tensor_1d(p->ctx,GGML_TYPE_F16,count*4096);p->mask=ggml_new_tensor_1d(p->ctx,GGML_TYPE_F32,count*64*32);p->q=ggml_new_tensor_4d(p->ctx,GGML_TYPE_F32,128,32,1,1);
 auto k=ggml_view_4d(p->ctx,p->k,128,32,count,1,256,8192,count*8192,0);
 auto v=ggml_view_4d(p->ctx,p->v,count,32,128,1,count*256,count*2,count*8192,0);
 auto mask=ggml_view_4d(p->ctx,p->mask,count,64,32,1,count*4,count*64*4,count*64*32*4,0);
 auto q=ggml_permute(p->ctx,p->q,0,2,1,3);k=ggml_permute(p->ctx,k,0,2,1,3);v=ggml_permute(p->ctx,v,0,2,1,3);
 auto scores=ggml_mul_mat(p->ctx,k,q);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);auto probs=ggml_soft_max_ext(p->ctx,scores,mask,1.f/std::sqrt(128.f),0.f);p->output=ggml_mul_mat(p->ctx,v,probs); // pinned default value precision, no new F32 override
 p->graph=ggml_new_graph(p->ctx);ggml_build_forward_expand(p->graph,p->output);p->buffer=ggml_backend_alloc_ctx_tensors(p->ctx,p->backend);need(p->buffer&&gpu_bytes()<=256ULL*1024*1024,"GPU workspace budget");
 ggml_backend_tensor_set(p->q,actual.data(),0,actual.size()*4);cuda_check(cudaEventCreate(&p->start));cuda_check(cudaEventCreate(&p->done));
}
GQADeliveryWorkspace::~GQADeliveryWorkspace()=default;
std::uint64_t GQADeliveryWorkspace::gpu_bytes()const{return ggml_backend_buffer_get_size(p->buffer);}
GQADeliveryResult GQADeliveryWorkspace::deliver(const std::vector<std::vector<std::uint64_t>>& ids,std::uint64_t position,const std::vector<std::uint64_t>& physical,const std::vector<std::uint8_t>& payload){
 GQADeliveryResult r;auto t=Clock::now();pack_gqa_workspace(ids,{position},physical,payload,p->count,p->kh.data(),p->vh.data(),p->mh.data());r.pack_ms=milliseconds(t);t=Clock::now();
 // b6000 ggml_backend_cuda_buffer_set_tensor uses cudaStreamPerThread and synchronizes it.
 // Record completion on THAT copy stream, not on the unrelated backend compute stream.
 cuda_check(cudaEventRecord(p->start,cudaStreamPerThread));
 ggml_backend_tensor_set(p->k,p->kh.data(),0,p->kh.size()*2);ggml_backend_tensor_set(p->v,p->vh.data(),0,p->vh.size()*2);ggml_backend_tensor_set(p->mask,p->mh.data(),0,p->mh.size()*4);
 cuda_check(cudaEventRecord(p->done,cudaStreamPerThread));cuda_check(cudaEventSynchronize(p->done));
 r.gpu_ready_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();r.upload_wall_ms=milliseconds(t);float interval=0;cuda_check(cudaEventElapsedTime(&interval,p->start,p->done));r.cuda_copy_stream_interval_ms=interval;r.upload_bytes=p->kh.size()*2+p->vh.size()*2+p->mh.size()*4;return r;
}
replay::json GQADeliveryWorkspace::verify_and_consume(){
 std::vector<std::uint8_t> scratch(65536);std::uint64_t checked=0,errors=0;
 auto verify=[&](ggml_tensor*t,const void* expected,std::size_t bytes){const auto* src=static_cast<const std::uint8_t*>(expected);for(std::size_t off=0;off<bytes;off+=scratch.size()){auto n=std::min(scratch.size(),bytes-off);ggml_backend_tensor_get(t,scratch.data(),off,n);for(std::size_t i=0;i<n;++i)errors+=scratch[i]!=src[off+i];checked+=n;}};
 verify(p->k,p->kh.data(),p->kh.size()*2);verify(p->v,p->vh.data(),p->vh.size()*2);verify(p->mask,p->mh.data(),p->mh.size()*4);need(errors==0,"GPU K/V/mask exact readback differs");
 auto t=Clock::now();need(ggml_backend_graph_compute(p->backend,p->graph)==GGML_STATUS_SUCCESS,"actual pinned attention failed");ggml_backend_synchronize(p->backend);double attention_ms=milliseconds(t);std::vector<float> output(4096);ggml_backend_tensor_get(p->output,output.data(),0,output.size()*4);for(auto x:output)need(std::isfinite(x),"actual attention produced nonfinite output");
 return {{"GPU_received_exact_byte_errors",errors},{"GPU_received_bytes_checked_outside_timer",checked},{"K_checksum",std::to_string(checksum(p->kh.data(),p->kh.size()*2))},{"V_checksum",std::to_string(checksum(p->vh.data(),p->vh.size()*2))},{"mask_checksum",std::to_string(checksum(p->mh.data(),p->mh.size()*4))},{"actual_attention_output_checksum",std::to_string(checksum(output.data(),output.size()*4))},{"attention_after_delivery_ms_excluded",attention_ms},{"copy_stream","cudaStreamPerThread from actual pinned ggml set_tensor implementation"},{"completion","cudaEventRecord+cudaEventSynchronize on actual copy stream"},{"GPU_need_KV_time","unavailable"}};
}
}
