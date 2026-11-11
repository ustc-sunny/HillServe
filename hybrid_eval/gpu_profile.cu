#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
__global__ void attention(const float*q,const float*k,const float*v,float*out,int n){
 int h=blockIdx.x,d=threadIdx.x;extern __shared__ float s[];float*score=s+128;s[d]=0;
 for(int t=0;t<n;++t){s[d]=q[h*128+d]*k[(t*4+h/7)*128+d];__syncthreads();for(int z=64;z;z/=2){if(d<z)s[d]+=s[d+z];__syncthreads();}if(d==0)score[t]=s[0]*rsqrtf(128.f);__syncthreads();}
 if(d==0){float mx=score[0];for(int t=1;t<n;++t)mx=fmaxf(mx,score[t]);float den=0;for(int t=0;t<n;++t){score[t]=expf(score[t]-mx);den+=score[t];}for(int t=0;t<n;++t)score[t]/=den;}__syncthreads();float value=0;for(int t=0;t<n;++t)value+=score[t]*v[(t*4+h/7)*128+d];out[h*128+d]=value;
}
int main(int argc,char**argv){try{
 int n=argc==2?std::stoi(argv[1]):257;if(n<32||n>1024)throw std::runtime_error("N cap");
 std::vector<float> q(28*128),k(n*512),v(k.size());for(int i=0;i<int(q.size());++i)q[i]=float(i%19-9)/32;for(int i=0;i<int(k.size());++i){k[i]=float(i%23-11)/32;v[i]=float(i%31-15)/32;}
 float *dq,*dk,*dv,*out,*copy;float*host;std::size_t bytes=2*1024*1024;CK(cudaMalloc(&dq,q.size()*4));CK(cudaMalloc(&dk,k.size()*4));CK(cudaMalloc(&dv,v.size()*4));CK(cudaMalloc(&out,q.size()*4));CK(cudaMalloc(&copy,bytes));CK(cudaMallocHost(&host,bytes));for(std::size_t i=0;i<bytes/4;++i)host[i]=.25f;
 CK(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dk,k.data(),k.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,v.data(),v.size()*4,cudaMemcpyHostToDevice));
 cudaStream_t cs,ts;CK(cudaStreamCreateWithFlags(&cs,cudaStreamNonBlocking));CK(cudaStreamCreateWithFlags(&ts,cudaStreamNonBlocking));cudaEvent_t begin,end,cb,ce;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));CK(cudaEventCreate(&cb));CK(cudaEventCreate(&ce));
 std::cout<<"component,N,iterations,cuda_event_ms,host_wall_ms,logical_copy_bytes\n"<<std::setprecision(12);
 for(int mode=0;mode<3;++mode){auto wall=std::chrono::steady_clock::now();CK(cudaEventRecord(begin,cs));if(mode==2){CK(cudaEventRecord(cb,ts));}
 for(int i=0;i<20;++i){if(mode!=0){attention<<<28,128,(128+n)*4,cs>>>(dq,dk,dv,out,n);CK(cudaGetLastError());}if(mode!=1)CK(cudaMemcpyAsync(copy,host,bytes,cudaMemcpyHostToDevice,mode==2?ts:cs));}
 if(mode==2){CK(cudaEventRecord(ce,ts));CK(cudaStreamWaitEvent(cs,ce,0));}CK(cudaEventRecord(end,cs));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));double w=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();std::cout<<(mode==0?"h2d_pinned":mode==1?"attention_f32_primitive":"attention_plus_independent_h2d")<<','<<n<<",20,"<<ms<<','<<w<<','<<(mode==1?0:bytes*20)<<'\n';}
 std::vector<float> got(q.size());CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));double err=0;int violations=0;
 for(int h=0;h<28;++h){std::vector<double>s(n);for(int t=0;t<n;++t){for(int d=0;d<128;++d)s[t]+=double(q[h*128+d])*k[(t*4+h/7)*128+d];s[t]/=std::sqrt(128.0);}double mx=s[0];for(auto x:s)mx=std::max(mx,x);double den=0;for(auto&x:s){x=std::exp(x-mx);den+=x;}for(int d=0;d<128;++d){double ref=0;for(int t=0;t<n;++t)ref+=s[t]/den*v[(t*4+h/7)*128+d];double delta=std::abs(ref-got[h*128+d]);err=std::max(err,delta);if(delta>1e-5+1e-5*std::abs(ref))++violations;}}
 std::cerr<<std::setprecision(17)<<"{\"status\":\""<<(violations?"failed":"passed")<<"\",\"evidence_mode\":\"M_COMPONENT\",\"fixture\":\"synthetic_not_model\",\"precision\":\"F32\",\"independent_cpu_reference_max_error\":"<<err<<",\"violations\":"<<violations<<",\"gpu_allocator_bytes\":"<<(2*q.size()*4+2*k.size()*4+bytes)<<",\"pinned_host_bytes\":"<<bytes<<",\"not_llama_kernel_profile\":true}\n";
 cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(out);cudaFree(copy);cudaFreeHost(host);cudaStreamDestroy(cs);cudaStreamDestroy(ts);cudaEventDestroy(begin);cudaEventDestroy(end);cudaEventDestroy(cb);cudaEventDestroy(ce);return violations?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
