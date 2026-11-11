#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <sstream>
std::string number(double x){std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();}
std::vector<float> gpu(const std::vector<float>& q,const std::vector<ggml_fp16_t>& data,int n,const std::vector<int>& ids,int pad,bool fp32){
 auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);if(!dev)throw std::runtime_error("GPU unavailable");auto be=ggml_backend_dev_init(dev,nullptr);
 auto ctx=ggml_init({8*1024*1024,nullptr,true});
 auto Q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,1,28);auto K=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,128,pad,4);auto V=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,pad,128,4);auto mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,pad,1);
 auto scores=ggml_mul_mat(ctx,K,Q);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);auto probs=ggml_soft_max_ext(ctx,scores,mask,1/std::sqrt(128.f),0);auto out=ggml_mul_mat(ctx,V,probs);
 if(fp32)ggml_mul_mat_set_prec(out,GGML_PREC_F32);
 auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);auto buf=ggml_backend_alloc_ctx_tensors(ctx,be);
 std::vector<ggml_fp16_t> k(4*128*pad),v(k.size());std::vector<float> m(pad,-INFINITY);
 for(std::size_t slot=0;slot<ids.size();++slot){int t=ids[slot];if(t<0)continue;if(t>=n)throw std::runtime_error("bad id");m[slot]=0;for(int h=0;h<4;++h)for(int d=0;d<128;++d){k[(h*pad+slot)*128+d]=data[t*1024+h*128+d];v[(h*128+d)*pad+slot]=data[t*1024+512+h*128+d];}}
 ggml_backend_tensor_set(Q,q.data(),0,q.size()*4);ggml_backend_tensor_set(K,k.data(),0,k.size()*2);ggml_backend_tensor_set(V,v.data(),0,v.size()*2);ggml_backend_tensor_set(mask,m.data(),0,m.size()*4);
 if(ggml_backend_graph_compute(be,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU compute failure");std::vector<float> result(28*128);ggml_backend_tensor_get(out,result.data(),0,result.size()*4);
 ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(be);return result;
}
std::vector<long double> reference(const std::vector<float>&q,const std::vector<ggml_fp16_t>&kv,const std::vector<int>&ids){
 std::vector<long double> o(28*128);for(int h=0;h<28;++h){std::vector<long double>s;for(int t:ids){long double dot=0;for(int d=0;d<128;++d)dot+=(long double)q[h*128+d]*ggml_fp16_to_fp32(kv[t*1024+(h/7)*128+d]);s.push_back(dot/std::sqrt((long double)128));}
 auto max=*std::max_element(s.begin(),s.end());long double sum=0;for(auto&x:s){x=std::exp(x-max);sum+=x;}for(int d=0;d<128;++d)for(std::size_t i=0;i<ids.size();++i)o[h*128+d]+=s[i]/sum*ggml_fp16_to_fp32(kv[ids[i]*1024+512+(h/7)*128+d]);}return o;
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage frozen_attention FROZEN_BIN");std::ifstream f(argv[1],std::ios::binary);std::uint32_t h[4];f.read((char*)h,16);if(!f||h[0]!=18||h[1]!=128||h[2]!=28||h[3]!=4)throw std::runtime_error("frozen fixture ABI");
 std::vector<float>q(28*128);std::vector<ggml_fp16_t>kv(h[0]*1024);f.read((char*)q.data(),q.size()*4);f.read((char*)kv.data(),kv.size()*2);if(!f)throw std::runtime_error("short fixture");std::vector<float> recorded(28*128);f.read((char*)recorded.data(),recorded.size()*4);bool has_recorded=bool(f);ggml_backend_load_all();
 std::vector<int> selected{17,0,5,2},sorted{0,2,5,17},native(32,-1);for(int t:selected)native[t]=t;
 auto ref=reference(q,kv,selected);auto refsort=reference(q,kv,sorted);std::vector<std::vector<int>> order{native,selected,sorted,selected};std::vector<int>pads{32,32,32,64};std::vector<std::vector<float>>values;
 std::cout<<std::setprecision(17)<<"{\"evidence_mode\":\"D_DIAGNOSTIC\",\"input\":\"frozen_actual_Qwen_step1\",\"independent_reference\":\"CPU_long_double_stable_softmax_GQA7\",\"cases\":[";bool pass=true;
 for(int c=0;c<8;++c){auto a=gpu(q,kv,h[0],order[c%4],pads[c%4],c>=4);double max=0;int violations=0,bits=0;double delta=0,record_delta=0,matched_delta=0;int matched_bits=0;for(int i=0;i<28*128;++i){double e=std::abs((long double)a[i]-ref[i]);max=std::max(max,e);if(e>1e-5+1e-5*std::abs(ref[i]))++violations;if(c){if(std::memcmp(&a[i],&values[0][i],4))++bits;delta=std::max(delta,std::abs(double(a[i])-values[0][i]));}}if(has_recorded)for(int i=0;i<28*128;++i)record_delta=std::max(record_delta,std::abs(double(a[i])-recorded[i]));if(c%4)for(int i=0;i<28*128;++i){const float b=values[c>=4?4:0][i];matched_delta=std::max(matched_delta,std::abs(double(a[i])-b));if(std::memcmp(&a[i],&b,4))++matched_bits;}values.push_back(a);if(c>=4)pass&=!violations;if(c)std::cout<<",";std::cout<<"{\"case\":"<<c<<",\"f32_value_accumulation\":"<<(c>=4?"true":"false")<<",\"pad\":"<<pads[c%4]<<",\"max_error_vs_fp_reference\":"<<max<<",\"tolerance_violations\":"<<violations<<",\"bit_differences_vs_native\":"<<bits<<",\"max_delta_vs_native\":"<<delta<<",\"max_delta_vs_precision_matched_native\":"<<matched_delta<<",\"bits_vs_precision_matched_native\":"<<matched_bits<<",\"max_delta_vs_recorded_attention\":"<<(has_recorded?number(record_delta):"null")<<"}";}
 double perm=0;for(int i=0;i<28*128;++i)perm=std::max(perm,double(std::abs(ref[i]-refsort[i])));std::cout<<"],\"fp_reference_permutation_delta\":"<<perm<<",\"status_scope\":\"F32_control_only_default_failures_retained\",\"status\":\""<<(pass?"passed":"failed")<<"\"}\n";return pass?0:1;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
