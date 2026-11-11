#include "runtime/batch_delivery.hpp"
#include "storage/file_store.hpp"
namespace hillserve {
using namespace replay;
static void check_cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct GQABatchDeliveryWorkspace::Impl {
 std::size_t batch,slots;std::vector<std::uint16_t> kh,vh;std::vector<float> mh;
 std::vector<bool> delivered;std::vector<std::array<std::uint64_t,3>> expected_hashes;
 std::vector<std::uint64_t> ready;ggml_backend_t backend=nullptr;ggml_context*ctx=nullptr;
 ggml_backend_buffer_t buffer=nullptr;ggml_tensor*k=nullptr,*v=nullptr,*mask=nullptr,*q=nullptr,*output=nullptr;
 ggml_cgraph*graph=nullptr;cudaEvent_t start{},done{};
 Impl(std::size_t b,std::size_t n):batch(b),slots(n),kh(n*4096),vh(n*4096),mh(n*64*32),delivered(b,false),expected_hashes(b),ready(b,0){}
 ~Impl(){if(start)cudaEventDestroy(start);if(done)cudaEventDestroy(done);if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);if(backend)ggml_backend_free(backend);}
};
GQABatchDeliveryWorkspace::GQABatchDeliveryWorkspace(std::size_t b,std::size_t n,const std::vector<float>& q,std::uint64_t budget){
 need(b==8&&n>0&&n<=6592&&n%32==0&&q.size()==b*4096,"B8 bounded real sequence-axis shape");
 need(b*n*24576ULL+16*1024*1024<=budget,"B8 GPU byte admission before allocation");
 p=std::make_unique<Impl>(b,n);ggml_backend_load_all();auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(dev,"real GPU required");
 p->backend=ggml_backend_dev_init(dev,nullptr);need(p->backend&&ggml_backend_is_cuda(p->backend),"CUDA required");p->ctx=ggml_init({2*1024*1024,nullptr,true});need(p->ctx,"batch metadata");
 p->k=ggml_new_tensor_1d(p->ctx,GGML_TYPE_F16,b*n*4096);p->v=ggml_new_tensor_1d(p->ctx,GGML_TYPE_F16,b*n*4096);
 p->mask=ggml_new_tensor_1d(p->ctx,GGML_TYPE_F32,b*n*64*32);p->q=ggml_new_tensor_4d(p->ctx,GGML_TYPE_F32,128,32,1,b);
 auto k=ggml_view_4d(p->ctx,p->k,128,32,n,b,256,8192,n*8192,0);
 auto v=ggml_view_4d(p->ctx,p->v,n,32,128,b,n*256,n*2,n*8192,0);
 auto mask=ggml_view_4d(p->ctx,p->mask,n,64,32,b,n*4,n*64*4,n*64*32*4,0);
 auto query=ggml_permute(p->ctx,p->q,0,2,1,3);k=ggml_permute(p->ctx,k,0,2,1,3);v=ggml_permute(p->ctx,v,0,2,1,3);
 auto scores=ggml_mul_mat(p->ctx,k,query);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
 auto probs=ggml_soft_max_ext(p->ctx,scores,mask,1.f/std::sqrt(128.f),0.f);p->output=ggml_mul_mat(p->ctx,v,probs);
 need(p->output->ne[3]==static_cast<std::int64_t>(b),"attention must retain independent batch axis");
 p->graph=ggml_new_graph(p->ctx);ggml_build_forward_expand(p->graph,p->output);p->buffer=ggml_backend_alloc_ctx_tensors(p->ctx,p->backend);
 need(p->buffer&&gpu_bytes()<=budget,"actual B8 GPU allocation exceeds budget");ggml_backend_tensor_set(p->q,q.data(),0,q.size()*4);
 check_cuda(cudaEventCreate(&p->start));check_cuda(cudaEventCreate(&p->done));
}
GQABatchDeliveryWorkspace::~GQABatchDeliveryWorkspace()=default;
std::uint64_t GQABatchDeliveryWorkspace::gpu_bytes()const{return ggml_backend_buffer_get_size(p->buffer);}
std::uint64_t GQABatchDeliveryWorkspace::host_staging_bytes()const{return p->kh.capacity()*2+p->vh.capacity()*2+p->mh.capacity()*4;}
void GQABatchDeliveryWorkspace::begin_round(){std::fill(p->delivered.begin(),p->delivered.end(),false);std::fill(p->ready.begin(),p->ready.end(),0);}
GQADeliveryResult GQABatchDeliveryWorkspace::deliver(std::size_t seq,const std::vector<std::vector<std::uint64_t>>& ids,std::uint64_t pos,const std::vector<std::uint64_t>& physical,const std::vector<std::uint8_t>& payload){
 need(seq<p->batch&&!p->delivered[seq],"each sequence delivered exactly once per round");auto t=Clock::now();GQADeliveryResult r;
 pack_gqa_workspace(ids,{pos},physical,payload,p->slots,p->kh.data(),p->vh.data(),p->mh.data());r.pack_ms=milliseconds(t);
 t=Clock::now();check_cuda(cudaEventRecord(p->start,cudaStreamPerThread));
 ggml_backend_tensor_set(p->k,p->kh.data(),seq*p->kh.size()*2,p->kh.size()*2);
 ggml_backend_tensor_set(p->v,p->vh.data(),seq*p->vh.size()*2,p->vh.size()*2);
 ggml_backend_tensor_set(p->mask,p->mh.data(),seq*p->mh.size()*4,p->mh.size()*4);
 check_cuda(cudaEventRecord(p->done,cudaStreamPerThread));check_cuda(cudaEventSynchronize(p->done));
 r.gpu_ready_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();r.upload_wall_ms=milliseconds(t);float interval=0;check_cuda(cudaEventElapsedTime(&interval,p->start,p->done));r.cuda_copy_stream_interval_ms=interval;r.upload_bytes=p->kh.size()*2+p->vh.size()*2+p->mh.size()*4;
 // Full GPU readback/validation occurs only after all-eight delivery completion.
 p->delivered[seq]=true;p->ready[seq]=r.gpu_ready_ns;return r;
}
std::uint64_t GQABatchDeliveryWorkspace::batch_ready_ns()const{need(std::all_of(p->delivered.begin(),p->delivered.end(),[](bool x){return x;}),"not all batch sequences GPU ready");return *std::max_element(p->ready.begin(),p->ready.end());}
replay::json GQABatchDeliveryWorkspace::verify_sequence(std::size_t seq,const std::vector<std::vector<std::uint64_t>>& ids,std::uint64_t pos,const std::vector<std::uint64_t>& physical,const std::vector<std::uint8_t>& payload){
 (void)batch_ready_ns();need(seq<p->batch,"verify sequence bounds");pack_gqa_workspace(ids,{pos},physical,payload,p->slots,p->kh.data(),p->vh.data(),p->mh.data());
 std::vector<std::uint8_t> scratch(65536);std::uint64_t errors=0,checked=0;
 auto verify=[&](ggml_tensor*t,const void* expected,std::size_t bytes){const auto*src=static_cast<const std::uint8_t*>(expected);for(std::size_t off=0;off<bytes;off+=scratch.size()){auto n=std::min(scratch.size(),bytes-off);ggml_backend_tensor_get(t,scratch.data(),seq*bytes+off,n);for(std::size_t i=0;i<n;++i)errors+=scratch[i]!=src[off+i];checked+=n;}};
 verify(p->k,p->kh.data(),p->kh.size()*2);verify(p->v,p->vh.data(),p->vh.size()*2);verify(p->mask,p->mh.data(),p->mh.size()*4);need(errors==0,"B8 sequence K/V/mask GPU readback differs");
 return {{"sequence",seq},{"GPU_received_exact_byte_errors",errors},{"GPU_received_bytes_checked_outside_timer",checked},{"K_checksum",std::to_string(checksum(p->kh.data(),p->kh.size()*2))},{"V_checksum",std::to_string(checksum(p->vh.data(),p->vh.size()*2))},{"mask_checksum",std::to_string(checksum(p->mh.data(),p->mh.size()*4))}};
}
replay::json GQABatchDeliveryWorkspace::verify_and_consume(){
 (void)batch_ready_ns();auto t=Clock::now();need(ggml_backend_graph_compute(p->backend,p->graph)==GGML_STATUS_SUCCESS,"true B8 attention graph failed");ggml_backend_synchronize(p->backend);
 const double attention_ms=milliseconds(t);std::vector<float> output(p->batch*4096);ggml_backend_tensor_get(p->output,output.data(),0,output.size()*4);for(auto x:output)need(std::isfinite(x),"B8 attention nonfinite output");
 json slices=json::array();for(std::size_t i=0;i<p->batch;++i)slices.push_back(std::to_string(checksum(output.data()+i*4096,4096*4)));
 return {{"per_sequence_attention_output_checksums",slices},{"graph_batch_axis",p->output->ne[3]},{"actual_attention_output_checksum",std::to_string(checksum(output.data(),output.size()*4))},{"attention_ms_after_delivery_excluded",attention_ms},{"full_model_feedback_claim",false},{"GPU_need_KV_time","unavailable"}};
}
}
