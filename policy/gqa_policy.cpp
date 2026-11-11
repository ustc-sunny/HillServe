#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "policy/half_order.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <thread>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hillserve::policy {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
void read_exact(int fd,void* data,std::size_t bytes,std::uint64_t offset){
 auto* p=static_cast<char*>(data);std::size_t done=0;
 while(done<bytes){ssize_t r=pread(fd,p+done,bytes-done,off_t(offset+done));if(r<0&&errno==EINTR)continue;need(r>0,"score-key pread failed/short: "+std::string(std::strerror(errno)));done+=std::size_t(r);}
}
void write_exact(int fd,const void* data,std::size_t bytes,std::uint64_t offset){
 const auto* p=static_cast<const char*>(data);std::size_t done=0;
 while(done<bytes){ssize_t r=pwrite(fd,p+done,bytes-done,off_t(offset+done));if(r<0&&errno==EINTR)continue;need(r>0,"score-key pwrite failed: "+std::string(std::strerror(errno)));done+=std::size_t(r);}
}
std::uint32_t u32(std::ifstream& f){unsigned char b[4]{};f.read(reinterpret_cast<char*>(b),4);need(bool(f),"truncated policy artifact header");return std::uint32_t(b[0])|(std::uint32_t(b[1])<<8)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[3])<<24);}
void quota_check(const std::filesystem::path& file,std::uint64_t extra){
 const char* configured = std::getenv("HILLSERVE_SCRATCH_ROOT");
 need(configured && *configured, "HILLSERVE_SCRATCH_ROOT is required for legacy GQA file allocation");
 const auto allowed = std::filesystem::canonical(configured);
 const auto parent=std::filesystem::weakly_canonical(file.parent_path());
 const auto rel=parent.lexically_relative(allowed);need(!rel.empty()&&*rel.begin()!="..","score file outside approved SmartSSD directory");
 std::uint64_t used=0;for(const auto& e:std::filesystem::recursive_directory_iterator(allowed))if(e.is_regular_file())used+=e.file_size();
 need(used<=8ULL*1024*1024*1024&&extra<=8ULL*1024*1024*1024-used,"aggregate approved SmartSSD 8GiB quota exceeded");
}
}
GQASelection select_gqa_scores(const std::vector<std::uint16_t>& score,std::size_t h,std::size_t stride,std::size_t n,float alpha,float ratio,bool fp32_counts){
 need(h>0&&h<=128&&n>0&&n<=stride&&score.size()==h*stride,"GQA selector shape");
 need(std::isfinite(alpha)&&alpha>=0&&std::isfinite(ratio)&&ratio>0&&ratio<=1,"GQA selector numeric parameters");
 // Wide indexing/counting removes the old global-vector shape restriction. The
 // inspected F16 count representation is still preserved, and fails explicitly
 // when it cannot represent a count; no unannounced long-context algorithm.
 float maximum=-std::numeric_limits<float>::infinity();
 for(std::size_t i=0;i<n;++i){float x=half_value(score[i]);need(std::isfinite(x),"nonfinite head0 score");maximum=std::max(maximum,x);}
 float threshold=half_value(half_bits(maximum-alpha));need(std::isfinite(threshold),"F16 threshold overflow");
 GQASelection r;r.ids.resize(h);r.counts.resize(h);float total=0;
 for(std::size_t head=0;head<h;++head){std::uint64_t count=0;for(std::size_t i=0;i<n;++i){float x=half_value(score[head*stride+i]);need(std::isfinite(x),"nonfinite GQA score");count+=x>=threshold;}
  need(count<=std::numeric_limits<std::uint32_t>::max(),"threshold count exceeds u32");r.counts[head]=std::uint32_t(count);
  float rounded=fp32_counts?float(count):half_value(half_bits(float(count)));need(std::isfinite(rounded),"public F16 count overflow; long-count policy extension required");total+=rounded;
 }
 float mean=fp32_counts?total/float(h):half_value(half_bits(total/float(h)));need(std::isfinite(mean),"public F16 mean overflow; long-count policy extension required");
 r.k=std::min({std::size_t(mean),std::size_t(std::floor(double(n)*double(ratio))),n});
 for(std::size_t head=0;head<h;++head){auto& ids=r.ids[head];ids.resize(n);std::iota(ids.begin(),ids.end(),0);
  auto compare=[&](std::uint64_t a,std::uint64_t b){const auto av=finite_half_order_key(score[head*stride+a]),bv=finite_half_order_key(score[head*stride+b]);return av!=bv?av>bv:a<b;};
  std::partial_sort(ids.begin(),ids.begin()+std::min(n,r.k+1),ids.end(),compare);
  if(r.k&&r.k<n&&score[head*stride+ids[r.k-1]]==score[head*stride+ids[r.k]])++r.boundary_tied_heads;
  ids.resize(r.k);
 }
 return r;
}
GQAPolicy::GQAPolicy(const std::string& artifact,int capacity,float alpha,float ratio,const std::string& path):capacity_(capacity),layer_bytes_(heads_*std::size_t(capacity)*dim_*2),alpha_(alpha),ratio_(ratio),path_(path){
 need(capacity>0&&capacity<=65504&&capacity%32==0,"GQA score capacity must be32-aligned and within v1 per-call65504 limit");need(!path.empty(),"explicit new score backing file required; no memory fallback");
 need(std::isfinite(alpha)&&alpha>=0&&std::isfinite(ratio)&&ratio>0&&ratio<=1,"invalid GQA policy parameters");
 std::ifstream f(artifact,std::ios::binary);need(bool(f),"policy artifact missing");char magic[8]{};f.read(magic,8);need(std::string(magic,8)=="HSGLPF01","policy artifact magic");
 need(u32(f)==layers_&&u32(f)==heads_&&u32(f)==kv_heads_&&u32(f)==dim_,"policy artifact Gradient GQA shape");
 basis_.resize(layers_*heads_*dim_*dim_);mask_.resize(layers_*heads_*dim_);f.read(reinterpret_cast<char*>(basis_.data()),basis_.size()*2);f.read(reinterpret_cast<char*>(mask_.data()),mask_.size());need(bool(f)&&f.peek()==std::ifstream::traits_type::eof(),"policy artifact size");
 for(auto b:basis_)need(std::isfinite(half_value(b)),"nonfinite policy basis");
 for(std::size_t h=0;h<layers_*heads_;++h){std::size_t count=0;for(std::size_t c=0;c<dim_;++c){need(mask_[h*dim_+c]<=1,"invalid channel mask");count+=mask_[h*dim_+c];}need(count>0,"empty reference channel mask");}
 active_basis_.resize(heads_*dim_*dim_);stats_.file_extent_bytes=layers_*layer_bytes_;quota_check(path_,stats_.file_extent_bytes);
 fd_=open(path.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);need(fd_>=0,"new exclusive score file required: "+std::string(std::strerror(errno)));
 if(ftruncate(fd_,off_t(stats_.file_extent_bytes))!=0){close(fd_);fd_=-1;throw std::runtime_error("score backing extent creation failed");}
}
GQAPolicy::~GQAPolicy(){if(direct_write_fd_>=0)close(direct_write_fd_);if(direct_fd_>=0)close(direct_fd_);if(fd_>=0)close(fd_);}
void GQAPolicy::configure_backend(const std::string& backend,const HardwareContract& contract,bool direct_reads){
 need(stats_.score_calls==0,"cannot change backend during request");need(backend=="host"||backend=="fpga_staged"||backend=="fpga_direct","unknown GQA score backend");
 if(backend!="host")fpga_=make_fpga_backend(backend=="fpga_direct"?HardwareMode::fpga_storage_streaming:HardwareMode::fpga_host_staged,contract);
 if(direct_reads||backend=="fpga_direct"){
#ifdef O_DIRECT
  direct_fd_=open(path_.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);need(direct_fd_>=0,"direct score read open failed; no buffered fallback");
#else
  throw std::runtime_error("O_DIRECT score backing unavailable on this platform");
#endif
 }
 backend_=backend;
}
void GQAPolicy::enable_direct_writes(){
 need(std::all_of(completed_.begin(),completed_.end(),[](auto n){return n==0;}),"direct writes must be selected before any payload append");
#ifdef O_DIRECT
 direct_write_fd_=open(path_.c_str(),O_RDWR|O_DIRECT|O_CLOEXEC);need(direct_write_fd_>=0,"direct score write open failed; no buffered fallback");
#else
 throw std::runtime_error("direct score writes unavailable on this platform");
#endif
}
void GQAPolicy::write_region(const void* data,std::size_t bytes,std::uint64_t offset){
 if(direct_write_fd_<0){write_exact(fd_,data,bytes,offset);stats_.append_syscall_bytes+=bytes;return;}
 const auto first=offset/4096*4096;const auto end=(offset+bytes+4095)/4096*4096;const auto total=std::size_t(end-first);
 void* block=nullptr;need(posix_memalign(&block,4096,total)==0,"direct score append staging allocation");
 try{if(first!=offset||total!=bytes){read_exact(direct_write_fd_,block,total,first);stats_.append_rmw_read_bytes+=total;}std::memcpy(static_cast<char*>(block)+(offset-first),data,bytes);write_exact(direct_write_fd_,block,total,first);stats_.append_syscall_bytes+=total;}catch(...){free(block);throw;}free(block);
}
void GQAPolicy::configure_cpu_cache(std::size_t budget,unsigned threads){
 need(std::all_of(completed_.begin(),completed_.end(),[](auto n){return n==0;}),"cache placement must be frozen before prefill");need(threads>=1&&threads<=4,"CPU scorer budget is1..4threads");need(!optimized_cpu_,"CPU cache configuration is single use");cpu_threads_=threads;optimized_cpu_=true;cache_budget_=budget;
 // Fixed earliest-layer placement; compact only channels that Q can use.
 // This is a common budgeted placement, not dynamic HPM or lookahead APP.
 for(std::size_t l=2;l<layers_;++l){std::size_t count=0;for(std::size_t h=0;h<heads_;++h){cache_head_offsets_[l][h]=count*capacity_;auto& cols=channels_[l][h];for(std::size_t c=0;c<dim_;++c)if(mask_[(l*heads_+h)*dim_+c])cols.push_back(c);count+=cols.size();}const auto bytes=count*capacity_*2;if(bytes<=budget-cache_allocated_){compressed_cache_[l].resize(bytes/2,0);cache_allocated_+=bytes;}}
}
std::vector<std::uint16_t> GQAPolicy::cpu_scores(int layer,const std::vector<std::uint16_t>& q,const std::vector<std::uint16_t>* keys,std::size_t valid)const{
 std::vector<std::uint16_t> scores(heads_*capacity_,0);std::exception_ptr failure;std::mutex guard;
 auto worker=[&](unsigned worker_id){try{for(std::size_t h=worker_id;h<heads_;h+=cpu_threads_){const auto& columns=channels_[layer][h];std::vector<float> query(columns.size());for(std::size_t c=0;c<columns.size();++c)query[c]=half_value(q[h*dim_+columns[c]]);
  for(std::size_t t=0;t<valid;++t){float sum=0;for(std::size_t c=0;c<columns.size();++c){const auto value=keys?(*keys)[(h*capacity_+t)*dim_+columns[c]]:compressed_cache_[layer][cache_head_offsets_[layer][h]+t*columns.size()+c];sum+=query[c]*half_value(value);}auto result=half_bits(sum);need(std::isfinite(half_value(result)),"host optimized score overflow");scores[h*capacity_+t]=result;}
 }}catch(...){std::lock_guard<std::mutex> lock(guard);if(!failure)failure=std::current_exception();}};
 std::vector<std::thread> threads;try{for(unsigned t=1;t<cpu_threads_;++t)threads.emplace_back(worker,t);worker(0);}catch(...){for(auto& t:threads)t.join();throw;}for(auto& t:threads)t.join();if(failure)std::rethrow_exception(failure);return scores;
}
void GQAPolicy::basis_for(int layer){need(layer>=0&&layer<int(layers_),"GQA policy layer");if(cached_basis_layer_==layer)return;const auto offset=std::size_t(layer)*active_basis_.size();for(std::size_t i=0;i<active_basis_.size();++i)active_basis_[i]=half_value(basis_[offset+i]);cached_basis_layer_=layer;}
std::vector<std::uint16_t> gqa_skew_head_f16(const std::uint16_t* x,const float* basis,const std::uint8_t* mask){
 std::array<float,128> in{},sum{};for(std::size_t i=0;i<128;++i){in[i]=half_value(x[i]);need(std::isfinite(in[i]),"nonfinite current key/rehearsal query");}
 for(std::size_t i=0;i<128;++i)for(std::size_t c=0;c<128;++c)sum[c]+=in[i]*basis[i*128+c];
 std::vector<std::uint16_t> out(128);for(std::size_t c=0;c<128;++c){need(std::isfinite(sum[c]),"skew transform overflow");auto bits=half_bits(sum[c]);need(std::isfinite(half_value(bits)),"F16 skew transform overflow");out[c]=(mask&&!mask[c])?0:bits;}return out;
}
std::vector<std::uint16_t> GQAPolicy::transform(int layer,std::size_t head,const std::uint16_t* x,bool maskquery)const{
 return gqa_skew_head_f16(x,active_basis_.data()+head*dim_*dim_,maskquery?mask_.data()+(std::size_t(layer)*heads_+head)*dim_:nullptr);
}
void GQAPolicy::append(int layer,const std::vector<std::uint64_t>& positions,const std::vector<std::uint16_t>& keys){
 need(layer>=0&&layer<int(layers_)&&!positions.empty()&&keys.size()==positions.size()*kv_heads_*dim_,"GQA append shape");
 need(positions.front()==completed_[layer]&&positions.back()<capacity_,"GQA append position/visibility");for(std::size_t t=1;t<positions.size();++t)need(positions[t]==positions[t-1]+1,"GQA append must be incremental");
 if(layer<2){completed_[layer]+=positions.size();return;}
 auto begin=Clock::now();basis_for(layer);std::vector<std::uint16_t> transformed(heads_*positions.size()*dim_);
 for(std::size_t h=0;h<heads_;++h)for(std::size_t t=0;t<positions.size();++t){auto out=transform(layer,h,keys.data()+(t*kv_heads_+h/4)*dim_,false);std::copy(out.begin(),out.end(),transformed.begin()+(h*positions.size()+t)*dim_);}
 stats_.transform_ms+=elapsed(begin);begin=Clock::now();for(std::size_t h=0;h<heads_;++h){const auto offset=std::uint64_t(layer)*layer_bytes_+(h*capacity_+positions.front())*dim_*2;const auto bytes=positions.size()*dim_*2;write_region(transformed.data()+h*positions.size()*dim_,bytes,offset);stats_.append_bytes+=bytes;}
 stats_.write_ms+=elapsed(begin);
 if(!compressed_cache_[layer].empty())for(std::size_t h=0;h<heads_;++h){const auto& cols=channels_[layer][h];for(std::size_t t=0;t<positions.size();++t)for(std::size_t c=0;c<cols.size();++c)compressed_cache_[layer][cache_head_offsets_[layer][h]+positions[t]*cols.size()+c]=transformed[(h*positions.size()+t)*dim_+cols[c]];}
 completed_[layer]+=positions.size();
}
void GQAPolicy::read_layer(int layer,std::vector<std::uint16_t>& keys){
 auto begin=Clock::now();keys.resize(layer_bytes_/2);const auto offset=std::uint64_t(layer)*layer_bytes_;
 if(direct_fd_>=0){void* aligned=nullptr;need(posix_memalign(&aligned,4096,layer_bytes_)==0,"score read aligned allocation");try{read_exact(direct_fd_,aligned,layer_bytes_,offset);std::memcpy(keys.data(),aligned,layer_bytes_);}catch(...){free(aligned);throw;}free(aligned);}else read_exact(fd_,keys.data(),layer_bytes_,offset);
 stats_.score_read_bytes+=layer_bytes_;stats_.key_host_staging_bytes+=layer_bytes_;stats_.file_read_ms+=elapsed(begin);
}
std::vector<std::vector<std::uint64_t>> GQAPolicy::select(int layer,std::uint64_t position,const std::vector<float>& rehearsal){
 need(layer>=2&&layer<int(layers_)&&rehearsal.size()==heads_*dim_&&completed_[layer]==position+1,"GQA selection needs completed current positions and true rehearsal query");
 const auto before=stats_;Event event;event.layer=layer;event.position=position;event.started_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
 auto begin=Clock::now();basis_for(layer);std::vector<std::uint16_t> q(heads_*dim_);std::array<std::uint16_t,128> halfq{};
 for(std::size_t h=0;h<heads_;++h){for(std::size_t i=0;i<dim_;++i){need(std::isfinite(rehearsal[h*dim_+i]),"nonfinite actual rehearsal");halfq[i]=half_bits(rehearsal[h*dim_+i]);}auto out=transform(layer,h,halfq.data(),true);std::copy(out.begin(),out.end(),q.begin()+h*dim_);}
 stats_.transform_ms+=elapsed(begin);std::vector<std::uint16_t> scores;
 bool cpu_hot=!compressed_cache_[layer].empty();
 if(cpu_hot){begin=Clock::now();scores=cpu_scores(layer,q,nullptr,position+1);stats_.score_ms+=elapsed(begin);++stats_.cache_hits;std::size_t cols=0;for(const auto& c:channels_[layer])cols+=c.size();stats_.cache_hit_valid_bytes+=cols*(position+1)*2;}
 else if(backend_=="fpga_direct"){
  ++stats_.cache_misses;begin=Clock::now();ScoreFileRegion region{direct_fd_,std::uint64_t(layer)*layer_bytes_,layer_bytes_,false};auto device=fpga_->score_file(q,region,heads_,capacity_,dim_);++stats_.actual_fpga_calls;stats_.actual_fpga_rows+=heads_*capacity_;stats_.score_ms+=elapsed(begin);stats_.file_read_ms+=device.file_read_ms;stats_.score_read_bytes+=device.file_read_bytes;stats_.key_host_staging_bytes+=device.key_host_staging_bytes;stats_.query_h2d_bytes+=device.query_h2d_bytes;stats_.score_d2h_bytes+=device.timings.d2h_bytes;stats_.launch_wait_ms+=device.timings.launch_wait_ms;stats_.p2p_api_path_completed=device.p2p_allocation_verified&&device.kernel_completed;scores=std::move(device.values);
 }else{
  ++stats_.cache_misses;std::vector<std::uint16_t> keys;read_layer(layer,keys);begin=Clock::now();
  if(backend_=="host")scores=optimized_cpu_?cpu_scores(layer,q,&keys,position+1):host_partial_scores(q,keys,heads_,capacity_,dim_);
  else{auto device=fpga_->score(q,keys,heads_,capacity_,dim_);++stats_.actual_fpga_calls;stats_.actual_fpga_rows+=heads_*capacity_;stats_.query_h2d_bytes+=q.size()*2;stats_.key_h2d_bytes+=keys.size()*2;stats_.score_d2h_bytes+=device.timings.d2h_bytes;stats_.launch_wait_ms+=device.timings.launch_wait_ms;scores=std::move(device.values);}
  stats_.score_ms+=elapsed(begin);
 }
 if(verify_optimized_){
  const auto verification_start=Clock::now();std::vector<std::uint16_t> original;read_layer(layer,original);auto reference=host_partial_scores(q,original,heads_,capacity_,dim_);
  need(scores.size()==heads_*capacity_,"actual backend score shape mismatch");
  for(auto bits:scores)need(std::isfinite(half_value(bits)),"nonfinite actual backend score, including padded rows");
  std::uint64_t mismatches=0;for(std::size_t h=0;h<heads_;++h)for(std::size_t t=0;t<=position;++t)mismatches+=scores[h*capacity_+t]!=reference[h*capacity_+t];
  stats_.verification_score_count+=heads_*(position+1);stats_.verification_bit_errors+=mismatches;stats_.verification_ms+=elapsed(verification_start);
  need(mismatches==0,"same-input actual backend rawF16 score bit mismatches="+std::to_string(mismatches)+"; no CPU fallback");
 }
 // Preserve the public Llama operation order: HALF dot, then HALF divide.
 for(auto& s:scores)s=half_bits(half_value(s)/std::sqrt(float(dim_)));
 begin=Clock::now();auto choice=select_gqa_scores(scores,heads_,capacity_,position+1,alpha_,ratio_);need(choice.k>0,"public Llama selector produced zero K; no dense/current-token fallback");stats_.select_ms+=elapsed(begin);
 if(const char* capture=std::getenv("HILLSERVE_FREEZE_LAYER8_QUERY")){if(layer==8&&position==2048){
  const std::filesystem::path dest(capture);need(std::filesystem::is_directory(dest),"query freeze directory missing");
  need(!std::filesystem::exists(dest/"query-f16.bin"),"query freeze requires new files");
  std::ofstream fq(dest/"query-f16.bin",std::ios::binary);fq.write(reinterpret_cast<const char*>(q.data()),q.size()*2);need(bool(fq),"query freeze write failed");
  std::ofstream fi(dest/"selected-ids.bin",std::ios::binary);for(const auto& head:choice.ids)fi.write(reinterpret_cast<const char*>(head.data()),head.size()*8);need(bool(fi),"ID freeze write failed");
  std::ofstream fm(dest/"freeze.json");fm<<"{\"source\":\"actual_external_GQA_feedback_query_after_skew\",\"layer\":8,\"position\":2048,\"capacity\":"<<capacity_<<",\"selected_per_head\":"<<choice.k<<",\"alpha\":"<<alpha_<<",\"cap_ratio\":"<<ratio_<<",\"trace_capture_not_formal_timing\":true}\n";
 }}
 stats_.score_calls++;stats_.valid_candidates+=heads_*(position+1);stats_.kernel_rows+=heads_*((cpu_hot||(backend_=="host"&&optimized_cpu_))?position+1:capacity_);stats_.logical_valid_score_key_bytes+=heads_*(position+1)*dim_*2;stats_.selected_pairs+=heads_*choice.k;
 event.ended_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();event.read_bytes=stats_.score_read_bytes-before.score_read_bytes;event.key_host_bytes=stats_.key_host_staging_bytes-before.key_host_staging_bytes;event.key_h2d=stats_.key_h2d_bytes-before.key_h2d_bytes;event.q_h2d=stats_.query_h2d_bytes-before.query_h2d_bytes;event.d2h=stats_.score_d2h_bytes-before.score_d2h_bytes;event.k=choice.k;event.hot=cpu_hot;event.counts=choice.counts;need(events_.size()<30*64,"bounded64-step policy event capacity exceeded");events_.push_back(std::move(event));return choice.ids;
}
void GQAPolicy::drain(){auto begin=Clock::now();
#ifdef __APPLE__
 need(fsync(fd_)==0,"score-key end-of-request drain failed");
#else
 need(fdatasync(fd_)==0,"score-key end-of-request drain failed");
#endif
stats_.drain_ms+=elapsed(begin);++stats_.drain_calls;}
std::size_t GQAPolicy::managed_bytes()const{std::size_t channels=0;for(const auto& l:channels_)for(const auto& h:l)channels+=h.capacity()*sizeof(std::size_t);return basis_.capacity()*2+mask_.capacity()+active_basis_.capacity()*4+2*layer_bytes_+heads_*capacity_*2+heads_*dim_*2+cache_allocated_+channels+4*1024*1024+sizeof(*this);}
void GQAPolicy::write_stats(const std::string& path)const{
 std::ofstream events(path+".events.jsonl");need(bool(events),"cannot save per-layer score events");for(const auto& e:events_){events<<"{\"layer\":"<<e.layer<<",\"position\":"<<e.position<<",\"started_monotonic_ns\":"<<e.started_ns<<",\"ended_monotonic_ns\":"<<e.ended_ns<<",\"execution_backend\":\""<<(e.hot?"cpu_fixed_cache":backend_)<<"\",\"hot\":"<<(e.hot?"true":"false")<<",\"valid_candidates_per_head\":"<<e.position+1<<",\"kernel_capacity\":"<<capacity_<<",\"selected_per_head\":"<<e.k<<",\"retention_ratio\":"<<double(e.k)/double(e.position+1)<<",\"score_file_read_bytes\":"<<e.read_bytes<<",\"score_keys_via_host_bytes\":"<<e.key_host_bytes<<",\"K_H2D_bytes\":"<<e.key_h2d<<",\"Q_H2D_bytes\":"<<e.q_h2d<<",\"score_D2H_bytes\":"<<e.d2h<<",\"threshold_counts\":[";for(std::size_t h=0;h<e.counts.size();++h){if(h)events<<",";events<<e.counts[h];}events<<"]}\n";}

 std::ofstream f(path);need(bool(f),"cannot save GQA policy statistics");f<<std::setprecision(17)<<"{\n\"policy_id\":\"infinigen_public_llama_accuracy_GQA_online_v1\",\"backend\":\""<<backend_<<"\",\"alpha\":"<<alpha_<<",\"cap_ratio\":"<<ratio_<<",\"recent_pin_ratio\":0,\"first_sparse_layer\":2,\"head0_max_broadcast\":true,\"inclusive_threshold\":true,\"current_token_forced\":false,\"full_score_history_resident\":false,\"capacity\":"<<capacity_<<",\"score_channels\":128,\"unfilled_capacity_rows_excluded_from_selection\":true,\"managed_bytes_reservation\":"<<managed_bytes()<<",\"file_extent_bytes\":"<<stats_.file_extent_bytes<<",\"append_bytes\":"<<stats_.append_bytes<<",\"score_read_bytes\":"<<stats_.score_read_bytes<<",\"logical_valid_score_key_bytes\":"<<stats_.logical_valid_score_key_bytes<<",\"key_host_staging_bytes\":"<<stats_.key_host_staging_bytes<<",\"query_h2d_bytes\":"<<stats_.query_h2d_bytes<<",\"key_h2d_bytes\":"<<stats_.key_h2d_bytes<<",\"score_d2h_bytes\":"<<stats_.score_d2h_bytes<<",\"score_calls\":"<<stats_.score_calls<<",\"valid_candidates\":"<<stats_.valid_candidates<<",\"scorer_rows\":"<<stats_.kernel_rows<<",\"actual_fpga_calls\":"<<stats_.actual_fpga_calls<<",\"actual_fpga_rows\":"<<stats_.actual_fpga_rows<<",\"selected_pairs\":"<<stats_.selected_pairs<<",\"transform_ms\":"<<stats_.transform_ms<<",\"write_ms\":"<<stats_.write_ms<<",\"file_read_ms\":"<<stats_.file_read_ms<<",\"score_ms\":"<<stats_.score_ms<<",\"select_ms\":"<<stats_.select_ms<<",\"launch_wait_ms\":"<<stats_.launch_wait_ms<<",\"drain_ms\":"<<stats_.drain_ms<<",\"drain_calls\":"<<stats_.drain_calls<<",\"p2p_api_path_completed\":"<<(stats_.p2p_api_path_completed?"true":"false")<<",\"physical_NAND_bytes\":null,\"score_writes_O_DIRECT\":"<<(direct_write_fd_>=0?"true":"false")<<",\"append_syscall_bytes\":"<<stats_.append_syscall_bytes<<",\"append_rmw_read_bytes\":"<<stats_.append_rmw_read_bytes<<",\"fixed_compressed_cache_budget_bytes\":"<<cache_budget_<<",\"fixed_compressed_cache_allocated_bytes\":"<<cache_allocated_<<",\"cache_hits\":"<<stats_.cache_hits<<",\"cache_misses\":"<<stats_.cache_misses<<",\"cache_hit_valid_bytes\":"<<stats_.cache_hit_valid_bytes<<",\"cpu_threads\":"<<cpu_threads_<<",\"optimization_validation_enabled\":"<<(verify_optimized_?"true":"false")<<",\"verification_score_count\":"<<stats_.verification_score_count<<",\"verification_bit_errors\":"<<stats_.verification_bit_errors<<",\"verification_ms\":"<<stats_.verification_ms<<",\"ordered_masked_channel_scoring\":"<<(optimized_cpu_?"true":"false")<<",\"dynamic_HPM\":false,\"APP\":false,\"HOST_BEST_score_optimizations\":"<<(optimized_cpu_&&backend_=="host"?"true":"false")<<"}\n";
}
}
