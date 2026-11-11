#include "hybrid_eval/method_attention.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace hillserve {
using Clock=std::chrono::steady_clock;static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct MethodAttention::Impl {
 std::size_t h,d,pad,bytes=0;ggml_backend_t be=nullptr;ggml_context*ctx=nullptr;ggml_backend_buffer_t buf=nullptr;
 ggml_tensor *q=nullptr,*k=nullptr,*v=nullptr,*mask=nullptr,*out=nullptr;ggml_cgraph*g=nullptr;
 ~Impl(){if(buf)ggml_backend_buffer_free(buf);if(ctx)ggml_free(ctx);if(be)ggml_backend_free(be);}
};
MethodAttention::MethodAttention(std::size_t h,std::size_t d,std::size_t cap):p(new Impl){
 p->h=h;p->d=d;p->pad=((cap+1+31)/32)*32;
 if(h>128||d!=128||cap>128)throw std::runtime_error("GPU method pilot shape budget");
 ggml_backend_load_all();auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);if(!dev)throw std::runtime_error("real GPU required");
 p->be=ggml_backend_dev_init(dev,nullptr);p->ctx=ggml_init({1024*1024,nullptr,true});if(!p->be||!p->ctx)throw std::runtime_error("GPU initialization");
 p->q=ggml_new_tensor_3d(p->ctx,GGML_TYPE_F32,d,1,h);p->k=ggml_new_tensor_3d(p->ctx,GGML_TYPE_F16,d,p->pad,h);p->v=ggml_new_tensor_3d(p->ctx,GGML_TYPE_F16,p->pad,d,h);p->mask=ggml_new_tensor_2d(p->ctx,GGML_TYPE_F32,p->pad,1);
 auto scores=ggml_mul_mat(p->ctx,p->k,p->q);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
 auto probs=ggml_soft_max_ext(p->ctx,scores,p->mask,1.f,0.f); // Q already includes OPT D^-1/2
 p->out=ggml_mul_mat(p->ctx,p->v,probs); // pinned llama default value precision, no silent F32 control
 p->g=ggml_new_graph(p->ctx);ggml_build_forward_expand(p->g,p->out);p->buf=ggml_backend_alloc_ctx_tensors(p->ctx,p->be);
 if(!p->buf)throw std::runtime_error("GPU allocation");p->bytes=ggml_backend_buffer_get_size(p->buf);if(p->bytes>2*1024*1024)throw std::runtime_error("GPU tensor budget 2MiB exceeded");
}
MethodAttention::~MethodAttention()=default;
std::size_t MethodAttention::gpu_bytes()const{return p->bytes;}
AttentionOutput MethodAttention::run(const std::vector<float>&query,const std::vector<std::uint16_t>&current,const std::vector<KVPayload>&selected,std::size_t count){
 if(query.size()!=p->h*p->d||current.size()!=p->h*p->d*2||selected.size()!=p->h*count||count+1>p->pad)throw std::runtime_error("selected GPU shape");
 auto begin=Clock::now(),t=begin;AttentionOutput result;std::vector<std::uint16_t> k(p->h*p->d*p->pad),v(k.size());std::vector<float> mask(p->pad,-std::numeric_limits<float>::infinity());
 for(std::size_t slot=0;slot<=count;++slot)mask[slot]=0;
 for(std::size_t head=0;head<p->h;++head)for(std::size_t slot=0;slot<=count;++slot){
  const std::uint8_t*src=slot==count?reinterpret_cast<const std::uint8_t*>(current.data()+head*2*p->d):selected[head*count+slot].bytes.data();
  if(slot<count&&selected[head*count+slot].bytes.size()!=p->d*4)throw std::runtime_error("selected payload shape");
  std::memcpy(k.data()+(head*p->pad+slot)*p->d,src,p->d*2);
  for(std::size_t d=0;d<p->d;++d)std::memcpy(&v[(head*p->d+d)*p->pad+slot],src+(p->d+d)*2,2);
 }
 result.pack_ms=ms(t);t=Clock::now();
 ggml_backend_tensor_set(p->q,query.data(),0,query.size()*4);ggml_backend_tensor_set(p->k,k.data(),0,k.size()*2);ggml_backend_tensor_set(p->v,v.data(),0,v.size()*2);ggml_backend_tensor_set(p->mask,mask.data(),0,mask.size()*4);
 ggml_backend_synchronize(p->be);result.h2d_ms=ms(t);t=Clock::now();
 if(ggml_backend_graph_compute(p->be,p->g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU attention failure");ggml_backend_synchronize(p->be);result.compute_wall_ms=ms(t);t=Clock::now();
 result.values.resize(p->h*p->d);ggml_backend_tensor_get(p->out,result.values.data(),0,result.values.size()*4);result.d2h_ms=ms(t);result.total_ms=ms(begin);result.gpu_bytes=p->bytes;result.h2d_bytes=query.size()*4+(k.size()+v.size())*2+mask.size()*4;return result;
}
}
