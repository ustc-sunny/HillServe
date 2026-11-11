#include "runtime/gqa_executor.hpp"
#include "runtime/gqa_pack.hpp"
#include "storage/file_store.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <map>
#include <iomanip>
#include <cstdlib>
#include <array>
#include <condition_variable>
#include <mutex>
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
#include "runtime/gqa_prefetch_cuda.hpp"
#include "ggml-cuda.h"
#endif
#ifdef HILLSERVE_GQA_CUDA_TIMING
#include <cuda_runtime_api.h>
#endif
namespace hillserve {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void need(bool b,const char* x){if(!b)throw std::runtime_error(x);}

// Match the fixed llama graph when per_query_head=false: K[t,KH,D],
// V[KH,D,slots], mask[paddedQuery,count]. V deliberately uses the allocated
// slots stride, not count; zero padding is uploaded and accounted explicitly.
void pack_compact_dense_prefill(const std::vector<uint64_t>& positions,
 const std::vector<uint8_t>& payload,size_t count,size_t slots,
 uint16_t* k,uint16_t* v,float* mask){
 constexpr size_t key_words=8*128,row_bytes=key_words*4;
 need(!positions.empty()&&positions.size()<=256&&slots&&count<=slots&&
      payload.size()%row_bytes==0,"compact dense prefill geometry");
 const auto tokens=payload.size()/row_bytes;
 need(tokens&&tokens<=count&&positions.back()==tokens-1,"compact prefill must contain complete visible history");
 for(size_t q=0;q<positions.size();++q)
  need(positions[q]<tokens&&(q==0||positions[q]>positions[q-1]),"compact prefill query order");
 std::fill_n(k,count*key_words,uint16_t(0));
 std::fill_n(v,slots*key_words,uint16_t(0));
 for(size_t t=0;t<tokens;++t)
  std::memcpy(k+t*key_words,payload.data()+t*row_bytes,key_words*2);
 // A bounded transpose tile avoids touching 1024 distant V cache lines for
 // every token. Both source and destination retain the original F16 bits.
 for(size_t first=0;first<tokens;first+=32)
  for(size_t channel=0;channel<key_words;channel+=32)
   for(size_t t=first;t<std::min(tokens,first+32);++t)
    for(size_t d=channel;d<channel+32;++d)
     std::memcpy(v+d*slots+t,payload.data()+t*row_bytes+key_words*2+d*2,2);
 const auto qpad=(positions.size()+63)/64*64;
 std::fill_n(mask,count*qpad,-std::numeric_limits<float>::infinity());
 for(size_t q=0;q<positions.size();++q)
  std::fill_n(mask+q*count,positions[q]+1,0.f);
}
}
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
struct GQALayerExecutor::PrefetchState {
 struct Slot {bool busy=false,ready=false;int layer=-1;uint64_t position=0,ticket=0,submitted_ns=0;std::shared_ptr<const runtime::GQAPrefetchArtifact> artifact;GQAPrefetchCuda::Upload upload;};
 std::mutex mutex;std::condition_variable changed;std::array<Slot,2> slots;unsigned active=0;bool closing=false;GQAPrefetchCuda cuda;
 PrefetchState(void* pointer,size_t capacity):cuda(pointer,capacity){}
 static uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
 std::pair<unsigned,uint64_t> reserve(int layer,uint64_t position){std::lock_guard<std::mutex> lock(mutex);need(!closing,"prefetch executor closing");for(unsigned i=0;i<slots.size();++i)if(!slots[i].busy){auto& s=slots[i];s.busy=true;s.ready=false;s.layer=layer;s.position=position;s.submitted_ns=now();++s.ticket;return {i,s.ticket};}throw std::runtime_error("more than two payload prefetches in flight");}
 void stage(unsigned slot,uint64_t ticket,std::shared_ptr<const runtime::GQAPrefetchArtifact> artifact){
  {std::lock_guard<std::mutex> lock(mutex);auto& s=slots.at(slot);need(!closing&&s.busy&&s.ticket==ticket,"stale payload callback");need(artifact&&artifact->layer==s.layer&&artifact->position==s.position,"prefetch artifact identity mismatch");need(artifact->kv_bytes.size()==artifact->physical_ids.size()*runtime::GQAPrefetchArtifact::row_bytes,"prefetch artifact geometry");++active;}
  try {auto upload=cuda.stage(slot,artifact->kv_bytes);std::lock_guard<std::mutex> lock(mutex);auto& s=slots[slot];s.artifact=std::move(artifact);s.upload=upload;s.ready=true;--active;changed.notify_all();}
  catch(...){std::lock_guard<std::mutex> lock(mutex);--active;changed.notify_all();throw;}
 }
 std::pair<unsigned,Slot> get(int layer,uint64_t position){std::lock_guard<std::mutex> lock(mutex);for(unsigned i=0;i<slots.size();++i)if(slots[i].busy&&slots[i].layer==layer&&slots[i].position==position){need(slots[i].ready&&slots[i].artifact,"policy returned before payload callback completed");return {i,slots[i]};}throw std::runtime_error("target layer has no submitted payload prefetch");}
 void release(unsigned i){std::lock_guard<std::mutex> lock(mutex);slots.at(i).artifact.reset();slots[i].busy=false;slots[i].ready=false;}
 void shutdown(){std::unique_lock<std::mutex> lock(mutex);closing=true;changed.wait(lock,[&]{return active==0;});for(auto& s:slots)s.artifact.reset();}
 ~PrefetchState(){shutdown();}
};
#else
struct GQALayerExecutor::PrefetchState {};
#endif
GQALayerExecutor::GQALayerExecutor(const std::string& dir,const std::string& file,int slots,int batch,float ratio,bool memory,bool verify,bool direct,size_t gpu_budget,size_t host_budget,GQAAppend append,GQASelect select,GQARuntimeHooks hooks)
:memory_(memory),verify_(verify),slots_(slots),batch_(batch),cap_ratio_(ratio),rehearsal_by_layer_(L),completed_(L,0),append_policy_(std::move(append)),select_policy_(std::move(select)),hooks_(std::move(hooks)),events_(dir+"/gqa-events.jsonl"),selections_(dir+"/gqa-selections.jsonl"),attention_(dir+"/gqa-attention.jsonl"){
 need(slots>0&&slots<=66048&&slots%32==0&&batch>0&&batch<=256&&ratio>0&&ratio<=1,"GQA workspace geometry unsupported");
 need(!hooks_.full_attention||hooks_.compact_dense_prefill,"full attention requires physical GQA layout");
 need(!hooks_.defer_native_workspace||hooks_.native_dense_prefill,"deferred workspace requires native prefill");
 need(!hooks_.compact_full_workspace||(hooks_.full_attention&&hooks_.compact_dense_prefill&&!hooks_.payload_prefetch),"compact reservation is full-attention only");
 workspace_budget_=gpu_budget;
 const size_t qpad=(batch+63)/64*64;
 // Decode still needs 32 head masks padded to 64 queries, even when dense
 // prefill uses only one mask. Keep the legacy reservation unchanged by default.
 const size_t mask_rows=hooks_.compact_full_workspace?qpad:((hooks_.compact_dense_prefill||hooks_.native_dense_prefill)?std::max(qpad,size_t(64*QH)):qpad*QH);
 const size_t kv_words=size_t(slots)*(hooks_.compact_full_workspace?E:QE);
 const size_t kv=2ULL*kv_words*2,mask=size_t(slots)*mask_rows*4;
 const size_t prefetch_gpu=hooks_.payload_prefetch?size_t(slots)*(3*4096+32*sizeof(int32_t)):0;
 const size_t prefetch_host=hooks_.payload_prefetch?size_t(slots)*(3*4096+34*sizeof(int32_t)):0;
 const size_t staging=kv+mask+size_t(batch)*E*8+size_t(slots)*E*4+16ULL*1024*1024+prefetch_host;
 need(!hooks_.payload_prefetch||(hooks_.early_rehearsal&&hooks_.prepare_history&&hooks_.full_kv_io_fence),"payload prefetch requires early rehearsal, hook, and shared file IO fence");
#ifndef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
 need(!hooks_.payload_prefetch,"payload prefetch requires the CUDA helper build");
#endif
 stats.managed_host_bytes=staging;
 need(kv+mask+prefetch_gpu<=gpu_budget&&staging<=host_budget,"GQA workspace budget admission failed");
 auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(dev,"GQA GPU missing");
 if(hooks_.native_dense_prefill){
  need(!memory&&store_==nullptr&&hooks_.append_full&&(direct||hooks_.external_memory_backing),"native prefill requires explicit incremental backing export");
  const auto* graphs=std::getenv("GGML_CUDA_DISABLE_GRAPHS");
  need(graphs&&std::string(graphs)=="1","native prefill transition requires disabled CUDA graph capture");
  const std::uint64_t minimum_native=std::uint64_t(slots)*L*E*4;
  need(hooks_.native_prefill_kv_budget_bytes>=minimum_native,"explicit temporary native KV budget is too small");
  stats.native_prefill_reserved_bytes=hooks_.native_prefill_kv_budget_bytes;
  size_t free=0,total=0;ggml_backend_dev_memory(dev,&free,&total);
  const auto workspace=hooks_.defer_native_workspace?0:kv+mask+prefetch_gpu;
  need(workspace<=free&&stats.native_prefill_reserved_bytes<=free-workspace,"GPU free memory cannot cover external workspace plus temporary native KV");
 }
 tensors_=ggml_init({16*ggml_tensor_overhead(),nullptr,true});need(tensors_,"GQA tensor metadata allocation failed");
 k_=ggml_new_tensor_1d(tensors_,GGML_TYPE_F16,kv_words);v_=ggml_new_tensor_1d(tensors_,GGML_TYPE_F16,kv_words);mask_=ggml_new_tensor_1d(tensors_,GGML_TYPE_F32,mask/4);
 ggml_set_name(k_,"gqa_external_k");ggml_set_name(v_,"gqa_external_v");ggml_set_name(mask_,"gqa_external_mask");
 stats.prefetch_host_bytes=prefetch_host;
 if(!hooks_.defer_native_workspace)allocate_workspace();
 k_host_.resize(kv_words);v_host_.resize(kv_words);mask_host_.resize(mask/4);
 need(!hooks_.external_memory_backing||(!memory&&hooks_.append_full&&hooks_.fetch),"bounded memory backing requires real append/fetch hooks");
 if(memory||verify){backing_.resize(L);for(auto& layer:backing_)layer.reserve(size_t(slots)*E*4);}
 if(!memory&&!hooks_.external_memory_backing){store_=std::make_unique<SegmentedKVStore>(file,1,L,slots,E*4,hooks_.kv_file_quota_bytes);if(direct)store_->enable_direct_reads(file);if(std::getenv("HILLSERVE_GQA_DIRECT_IO")&&std::string(std::getenv("HILLSERVE_GQA_DIRECT_IO"))=="1")store_->enable_direct_writes(file);}
 events_<<std::setprecision(17);attention_<<std::setprecision(17);
}
void GQALayerExecutor::allocate_workspace(){
 need(!buffer_,"external GPU workspace already allocated");
 const auto started=Clock::now();
 auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);need(dev,"GQA GPU missing");
 buffer_=ggml_backend_alloc_ctx_tensors_from_buft(tensors_,ggml_backend_dev_buffer_type(dev));need(buffer_,"GQA GPU workspace allocation failed");
 stats.workspace_bytes=ggml_backend_buffer_get_size(buffer_);
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
 if(hooks_.payload_prefetch){prefetch_=std::make_shared<PrefetchState>(k_->data,slots_);stats.prefetch_gpu_bytes=prefetch_->cuda.gpu_bytes();stats.workspace_bytes+=stats.prefetch_gpu_bytes;}
#endif
 need(stats.workspace_bytes<=workspace_budget_,"GQA actual workspace budget exceeded");
 events_<<"{\"phase\":\"external_GPU_workspace_allocated\",\"deferred_until_native_release\":"<<(hooks_.defer_native_workspace?"true":"false")<<",\"compact_full_workspace\":"<<(hooks_.compact_full_workspace?"true":"false")<<",\"bytes\":"<<stats.workspace_bytes<<",\"allocation_ms\":"<<ms(started)<<"}\n";
}
GQALayerExecutor::~GQALayerExecutor(){
 if(hooks_.wait_pending)hooks_.wait_pending();
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
 if(prefetch_)prefetch_->shutdown();
 prefetch_.reset();
#endif
 llama_hillserve_configure(nullptr);if(buffer_)ggml_backend_buffer_free(buffer_);if(tensors_)ggml_free(tensors_);}
void GQALayerExecutor::configure(){llama_hillserve_config c{};c.enabled=true;c.native_prefill=hooks_.native_dense_prefill&&!native_prefill_finished_;c.external=!c.native_prefill;c.slots=slots_;c.max_tokens=batch_;c.k=k_;c.v=v_;c.mask=mask_;c.per_query_head=!c.native_prefill&&!hooks_.full_attention&&(decode_||!hooks_.compact_dense_prefill);c.early_rehearsal=hooks_.early_rehearsal;c.h2o_attention_stats=hooks_.h2o_attention_stats;
 if(decode_&&!hooks_.full_attention)c.sparse_slots=std::max(32,((int(std::floor((positions_.back()+1)*cap_ratio_))+31)/32)*32);
 llama_hillserve_configure(&c);
}
void GQALayerExecutor::begin(int step,const std::vector<uint64_t>& p,bool decode){need(!p.empty()&&p.size()<=size_t(batch_)&&p.back()<uint64_t(slots_),"GQA position bound");if(hooks_.native_dense_prefill)need(decode==native_prefill_finished_,"native prefill must finish once before external decode; prefill cannot resume");positions_=p;step_=step;decode_=decode;for(auto& q:rehearsal_by_layer_)q.clear();configure();}
void GQALayerExecutor::prepare(ggml_tensor* t,int layer,int count){
 const auto layer_started=Clock::now();
 const bool compact=hooks_.compact_dense_prefill&&(!decode_||hooks_.full_attention);
 need(t->type==GGML_TYPE_F16&&ggml_is_contiguous(t)&&t->ne[0]==2*D&&t->ne[1]==KH&&t->ne[2]==int(positions_.size()),"GQA current KV ABI mismatch");need(positions_.front()==completed_.at(layer),"GQA incremental position mismatch");
 current_.resize(ggml_nelements(t));ggml_backend_tensor_get(t,current_.data(),0,ggml_nbytes(t));
 for(auto x:current_)need((x&0x7c00)!=0x7c00,"GQA current model KV contains nonfinite F16");
 std::vector<uint8_t> bytes(positions_.size()*E*4);std::vector<uint16_t> keys(positions_.size()*E);
 for(size_t j=0;j<positions_.size();++j)for(int h=0;h<KH;++h){auto* row=current_.data()+(j*KH+h)*2*D;std::memcpy(bytes.data()+j*E*4+h*D*2,row,D*2);std::memcpy(bytes.data()+j*E*4+E*2+h*D*2,row+D,D*2);std::memcpy(keys.data()+j*E+h*D,row,D*2);}
 auto start=Clock::now();if(store_){std::unique_lock<std::mutex> io_lock;if(hooks_.full_kv_io_fence)io_lock=std::unique_lock<std::mutex>(hooks_.full_kv_io_fence->file_io);store_->append(layer,positions_.front(),bytes);}if(memory_||verify_)backing_[layer].insert(backing_[layer].end(),bytes.begin(),bytes.end());stats.append_ms+=ms(start);stats.appends+=positions_.size();completed_[layer]+=positions_.size();
 if(append_policy_)append_policy_(layer,positions_,keys);
 if(hooks_.append_full)hooks_.append_full(layer,positions_,keys,bytes);
 if(hooks_.native_dense_prefill&&!native_prefill_finished_){
  need(!decode_&&(store_||hooks_.external_memory_backing),"native prefill export outside dense backing phase");
  ++stats.native_prefill_layers;stats.native_prefill_export_bytes+=bytes.size();
  if(verify_&&!hooks_.skip_native_export_readback){
   std::unique_lock<std::mutex> lock;if(hooks_.full_kv_io_fence)lock=std::unique_lock<std::mutex>(hooks_.full_kv_io_fence->file_io);
   const auto stored=store_?store_->fetch(layer,positions_,bytes.size()):hooks_.fetch(layer,positions_,[](int,const std::vector<uint64_t>&)->std::vector<uint8_t>{throw std::runtime_error("no SSD fallback for memory backing");});
   need(stored==bytes,"native prefill exported current KV differs from actual file readback");
   stats.native_prefill_export_verified_bytes+=stored.size();
  }
  events_<<"{\"schema\":\"hillserve-native-prefill-v1\",\"phase\":\"native_prefill_export\",\"step\":"<<step_<<",\"layer\":"<<layer<<",\"first_position\":"<<positions_.front()<<",\"current_positions\":"<<positions_.size()<<",\"historical_positions\":"<<completed_[layer]<<",\"current_KV_D2H_bytes\":"<<bytes.size()<<",\"exported_KV_bytes\":"<<bytes.size()<<",\"export_backing\":\""<<(hooks_.external_memory_backing?"bounded_host_memory":"SSD_file")<<"\",\"readback_verified_bytes\":"<<(verify_&&!hooks_.skip_native_export_readback?bytes.size():0)<<",\"file_readback_verified_bytes\":"<<(verify_&&!hooks_.external_memory_backing?bytes.size():0)<<",\"history_fetch_bytes\":0,\"external_KV_mask_H2D_bytes\":0,\"native_mask_transfer_bytes\":null,\"actual_model_feedback\":true,\"native_gpu_history\":true}\n";
  return;
 }
 std::vector<std::vector<uint64_t>> ids(QH);
 std::vector<uint64_t> physical;
 if(decode_&&layer>=2&&!hooks_.full_attention){need(rehearsal_by_layer_.at(layer).size()==QE,"missing true layer-addressed live rehearsal");ids=select_policy_(layer,positions_.back(),rehearsal_by_layer_[layer]);need(ids.size()==QH,"GQA selected head count mismatch");}
 else if(compact){physical.resize(completed_[layer]);std::iota(physical.begin(),physical.end(),0);}
 else for(auto& v:ids){v.resize(completed_[layer]);std::iota(v.begin(),v.end(),0);}
 const auto union_started=Clock::now();
 if(!compact)physical=gqa_physical_union(ids,completed_[layer]);
 const double physical_union_ms=ms(union_started);
 const auto dense_checksum=compact?checksum(physical.data(),physical.size()*sizeof(uint64_t)):0;
 for(int h=0;h<QH;++h){const auto& head=compact?physical:ids[h];need(!head.empty()&&head.size()<=size_t(count),"GQA empty/excess per-head selection");stats.logical_head_selections+=head.size();stats.logical_selected_bytes+=head.size()*D*4;
  selections_<<"{\"step\":"<<step_<<",\"layer\":"<<layer<<",\"q_head\":"<<h<<",\"kv_head\":"<<h/4<<",\"candidate_count\":"<<completed_[layer]<<",\"selected_count\":"<<head.size()<<",\"selected_ids_checksum\":\""<<(compact?dense_checksum:checksum(head.data(),head.size()*sizeof(uint64_t)))<<"\",\"ids_recorded\":"<<(verify_?"true":"false")<<",\"ids\":[";if(verify_)for(size_t i=0;i<head.size();++i){if(i)selections_<<",";selections_<<head[i];}selections_<<"]}\n";
 }
 // Union only deduplicates physical storage reads, never logical attention selections.
 // Physical union was validated once with a bounded direct index above.
 std::vector<uint8_t> payload;
 GQAColdFetch cold_fetch=[&](int l,const std::vector<uint64_t>& selected){
  if(store_)return store_->fetch(l,selected,size_t(slots_)*E*4);
  need(!hooks_.external_memory_backing,"no implicit file/oracle fallback for no-SSD baseline");
  std::vector<uint8_t> result(selected.size()*E*4);for(size_t i=0;i<selected.size();++i)std::memcpy(result.data()+i*E*4,backing_.at(l).data()+selected[i]*E*4,E*4);return result;
 };
 start=Clock::now();payload=hooks_.fetch?hooks_.fetch(layer,physical,cold_fetch):cold_fetch(layer,physical);
 need(payload.size()==physical.size()*E*4,"tiered fetch payload size mismatch");
 const double tiered_fetch_wall_ms=ms(start);stats.fetch_ms+=tiered_fetch_wall_ms;stats.physical_selected_bytes+=payload.size();
 for(size_t i=0;i<physical.size();++i){if(verify_){need(std::memcmp(payload.data()+i*E*4,backing_[layer].data()+physical[i]*E*4,E*4)==0,"GQA exact stored payload mismatch");stats.exact_bytes+=E*4;}}
 const size_t k_words=size_t(count)*(compact?E:QE),v_words=compact?size_t(slots_)*E:k_words;
 int qpad=(positions_.size()+63)/64*64;const size_t mask_count=size_t(count)*qpad*(compact?1:QH);
 const size_t k_bytes=k_words*2,v_bytes=v_words*2,final_gpu_bytes=k_bytes+v_bytes+mask_count*4;
 need(k_words<=k_host_.size()&&v_words<=v_host_.size()&&mask_count<=mask_host_.size(),"GQA active workspace exceeds reservation");
 const auto pack_started=Clock::now();
 if(compact){pack_compact_dense_prefill(positions_,payload,count,slots_,k_host_.data(),v_host_.data(),mask_host_.data());++stats.compact_prefill_layers;}
 else if(hooks_.payload_prefetch&&decode_&&layer>=2&&!verify_){
  std::fill_n(mask_host_.data(),mask_count,-std::numeric_limits<float>::infinity());
  for(size_t h=0;h<ids.size();++h)for(size_t s=0;s<ids[h].size();++s)for(size_t q=0;q<positions_.size();++q)if(ids[h][s]<=positions_[q])mask_host_[(h*qpad+q)*count+s]=0;
 }else pack_gqa_workspace(ids,positions_,physical,payload,count,k_host_.data(),v_host_.data(),mask_host_.data());
 const double CPU_KV_mask_pack_ms=ms(pack_started);
 uint64_t gpu_upload_started_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
 uint64_t actual_gpu_ready_ns=0;
 start=Clock::now();uint64_t layer_h2d_bytes=0,layer_d2d_bytes=0;
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
 if(hooks_.payload_prefetch&&decode_&&layer>=2){
  need(attention_stream_,"actual attention CUDA backend stream has not been observed");auto staged=prefetch_->get(layer,positions_.back());const auto slot=staged.first;const auto& ready=staged.second;const auto& artifact=*ready.artifact;
  std::vector<int32_t> prefetched_rows(slots_,-1),miss_rows(slots_,-1);for(size_t i=0;i<artifact.physical_ids.size();++i){auto id=artifact.physical_ids[i];need(id<uint64_t(slots_)&&prefetched_rows[id]==-1,"duplicate/out-of-range prefetched physical ID");prefetched_rows[id]=int32_t(i);}
  std::vector<uint8_t> misses;misses.reserve(payload.size());size_t useful=0;
  for(size_t i=0;i<physical.size();++i){const auto found=prefetched_rows[physical[i]];if(found>=0){if(verify_)need(std::memcmp(payload.data()+i*E*4,artifact.kv_bytes.data()+size_t(found)*E*4,E*4)==0,"prefetched/final payload differs");++useful;}else{miss_rows[physical[i]]=int32_t(misses.size()/(E*4));misses.insert(misses.end(),payload.begin()+i*E*4,payload.begin()+(i+1)*E*4);}}
  std::vector<int32_t> row_map(size_t(count)*QH,-1);for(size_t h=0;h<ids.size();++h)for(size_t s=0;s<ids[h].size();++s){const auto id=ids[h][s];auto found=prefetched_rows[id];need(found>=0||miss_rows[id]>=0,"selected KV absent from prefetch and miss payload");row_map[h*count+s]=found>=0?found:-2-miss_rows[id];}
  auto delivered=prefetch_->cuda.deliver(slot,row_map,misses,mask_host_.data(),mask_count,count,k_->data,v_->data,mask_->data,attention_stream_);
  gpu_upload_started_ns=delivered.started_ns;actual_gpu_ready_ns=delivered.ready_observed_ns;
  layer_h2d_bytes=ready.upload.bytes+delivered.h2d_bytes;layer_d2d_bytes=delivered.d2d_bytes;++stats.prefetch_layers;stats.prefetch_h2d_bytes+=ready.upload.bytes;stats.prefetch_useful_bytes+=useful*E*4;stats.prefetch_wasted_bytes+=ready.upload.bytes-useful*E*4;stats.prefetch_miss_h2d_bytes+=misses.size();stats.gpu_d2d_bytes+=delivered.d2d_bytes;stats.prefetch_consume_wait_ms+=delivered.wait_ms;stats.prefetch_gpu_upload_ms+=ready.upload.device_ms;
  events_<<"{\"schema\":\"hillserve-payload-prefetch-v1\",\"phase\":\"payload_prefetch_consumed\",\"payload_prefetch_enabled\":true,\"step\":"<<step_<<",\"layer\":"<<layer<<",\"position\":"<<positions_.back()<<",\"prefetch_submitted_ns\":"<<ready.submitted_ns<<",\"prefetch_score_ready_ns\":"<<artifact.score_ready_ns<<",\"prefetch_read_started_ns\":"<<artifact.read_started_ns<<",\"prefetch_read_finished_ns\":"<<artifact.read_finished_ns<<",\"prefetch_cpu_ready_ns\":"<<artifact.read_finished_ns<<",\"prefetch_h2d_started_ns\":"<<ready.upload.started_ns<<",\"prefetch_gpu_ready_observed_ns\":"<<ready.upload.ready_observed_ns<<",\"target_append_started_ns\":"<<std::chrono::duration_cast<std::chrono::nanoseconds>(layer_started.time_since_epoch()).count()<<",\"consume_wait_scope\":\"final_copy_and_gather_completion_wait\",\"gpu_demand_wait\":\"unavailable\",\"consume_wait_ms\":"<<delivered.wait_ms<<",\"prefetch_h2d_bytes\":"<<ready.upload.bytes<<",\"miss_h2d_bytes\":"<<misses.size()<<",\"map_h2d_bytes\":"<<row_map.size()*sizeof(int32_t)<<",\"mask_h2d_bytes\":"<<mask_count*4<<",\"d2d_bytes\":"<<delivered.d2d_bytes<<",\"useful_prefetch_bytes\":"<<useful*E*4<<",\"wasted_prefetch_bytes\":"<<ready.upload.bytes-useful*E*4<<",\"matched_physical_positions\":"<<useful<<",\"missing_physical_positions\":"<<misses.size()/(E*4)<<",\"max_inflight_slots\":2,\"ready_time_kind\":\"host_observation_after_cuda_event\"}\n";
  prefetch_->release(slot);
 }else
#endif
 {
  ggml_backend_tensor_set(k_,k_host_.data(),0,k_bytes);ggml_backend_tensor_set(v_,v_host_.data(),0,v_bytes);ggml_backend_tensor_set(mask_,mask_host_.data(),0,mask_count*4);
#ifdef HILLSERVE_GQA_CUDA_TIMING
  cudaEvent_t done{};need(cudaEventCreateWithFlags(&done,cudaEventDisableTiming)==cudaSuccess,"GPU delivery event create failed");
  auto recorded=cudaEventRecord(done,cudaStreamPerThread);auto synced=recorded==cudaSuccess?cudaEventSynchronize(done):recorded;cudaEventDestroy(done);need(recorded==cudaSuccess&&synced==cudaSuccess,"GPU KV delivery completion failed");
#endif
  actual_gpu_ready_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
  layer_h2d_bytes=final_gpu_bytes;
 }
 const auto GPU_ready_ns=actual_gpu_ready_ns;
 double h2d=double(GPU_ready_ns-gpu_upload_started_ns)/1e6;stats.h2d_ms+=h2d;stats.gpu_h2d_bytes+=layer_h2d_bytes;
 if(verify_){
  // Debug only: bounded readback of the actual GPU tensors, after the recorded
  // delivery endpoint. No second full workspace or hidden host KV copy.
  std::vector<uint8_t> check(256*1024);
  auto equal=[&](ggml_tensor* dst,const void* src,size_t bytes){
   const auto* expected=static_cast<const uint8_t*>(src);
   for(size_t offset=0;offset<bytes;offset+=check.size()){
    const auto n=std::min(check.size(),bytes-offset);ggml_backend_tensor_get(dst,check.data(),offset,n);
    need(!std::memcmp(check.data(),expected+offset,n),"GQA actual GPU K/V/mask byte mismatch");stats.exact_gpu_bytes+=n;
   }
  };
  equal(k_,k_host_.data(),k_bytes);equal(v_,v_host_.data(),v_bytes);equal(mask_,mask_host_.data(),mask_count*4);
 }
 events_<<"{\"schema\":\"hillserve-gqa-delivery-v2\",\"request\":1,\"phase\":\"selected_KV_ready\",\"started_ns\":"<<std::chrono::duration_cast<std::chrono::nanoseconds>(layer_started.time_since_epoch()).count()<<",\"ended_ns\":"<<GPU_ready_ns<<",\"GPU_readback_verified\":"<<(verify_?"true":"false")<<",\"step\":"<<step_<<",\"decode\":"<<(decode_?"true":"false")<<",\"layer\":"<<layer<<",\"current_positions\":"<<positions_.size()<<",\"historical_positions\":"<<completed_[layer]<<",\"attention_slots\":"<<count<<",\"physical_unique_positions\":"<<physical.size()<<",\"physical_payload_bytes\":"<<payload.size()<<",\"gpu_h2d_bytes\":"<<layer_h2d_bytes<<",\"gpu_d2d_bytes\":"<<layer_d2d_bytes<<",\"gpu_final_KV_mask_bytes\":"<<final_gpu_bytes<<",\"gpu_readback_bytes\":"<<(verify_?final_gpu_bytes:0)<<",\"gpu_upload_started_ns\":"<<gpu_upload_started_ns<<",\"gpu_ready_observed_ns\":"<<GPU_ready_ns<<",\"payload_prefetch_enabled\":"<<(hooks_.payload_prefetch?"true":"false")<<",\"packed_K_checksum\":\""<<(verify_?std::to_string(checksum(k_host_.data(),k_bytes)):"")<<"\",\"packed_V_checksum\":\""<<(verify_?std::to_string(checksum(v_host_.data(),v_bytes)):"")<<"\",\"packed_mask_checksum\":\""<<(verify_?std::to_string(checksum(mask_host_.data(),mask_count*4)):"")<<"\",\"physical_payload_checksum\":\""<<(verify_?std::to_string(checksum(payload.data(),payload.size())):"")<<"\",\"h2d_ms\":"<<h2d<<",\"physical_union_ms\":"<<physical_union_ms<<",\"tiered_fetch_wall_ms\":"<<tiered_fetch_wall_ms<<",\"CPU_KV_mask_pack_ms\":"<<CPU_KV_mask_pack_ms<<",\"actual_model_feedback\":true,\"per_query_head_selection\":"<<(!compact?"true":"false")<<",\"compact_dense_prefill\":"<<(compact?"true":"false")<<",\"GPU_KV_heads\":"<<(compact?KH:QH)<<",\"GPU_mask_heads\":"<<(compact?1:QH)<<",\"GPU_V_stride_tokens\":"<<(compact?slots_:count)<<",\"gpu_K_bytes\":"<<k_bytes<<",\"gpu_V_bytes\":"<<v_bytes<<",\"gpu_mask_bytes\":"<<mask_count*4<<",\"native_gpu_history\":false}\n";
}
void GQALayerExecutor::inspect(ggml_tensor* t){auto* b=t->buffer?t->buffer:t->view_src?t->view_src->buffer:nullptr;need(b,"GQA attention buffer missing");auto dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b));need(!ggml_backend_buffer_is_host(b)&&dev&&ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU,"GQA attention must use actual GPU device memory, not pinned host memory");++stats.attention_calls;
 events_<<"{\"request\":1,\"phase\":\"attention_complete\",\"step\":"<<step_<<",\"tensor\":\""<<t->name<<"\",\"completed_ns\":"<<std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()<<"}\n";
 if(verify_){std::vector<float> data(ggml_nelements(t));need(t->type==GGML_TYPE_F32,"GQA attention output dtype mismatch");ggml_backend_tensor_get(t,data.data(),0,ggml_nbytes(t));for(float x:data)need(std::isfinite(x),"GQA attention nonfinite");attention_<<"{\"step\":"<<step_<<",\"tensor\":\""<<t->name<<"\",\"elements\":"<<data.size()<<",\"checksum\":\""<<checksum(data.data(),data.size()*4)<<"\"}\n";}}
void GQALayerExecutor::accounting(llama_context* c){
 stats.native_history_bytes=llama_hillserve_native_kv_bytes(c);
 const auto graph=llama_hillserve_graph_bytes(c,true);stats.graph_gpu_bytes=std::max<uint64_t>(stats.graph_gpu_bytes,graph);
 if(hooks_.native_dense_prefill&&!native_prefill_finished_){
  const auto native_gpu=llama_hillserve_native_gpu_kv_bytes(c);
  stats.native_prefill_context_tokens=llama_n_ctx(c);
  const auto expected=stats.native_prefill_context_tokens*L*E*4;
  need(!decode_&&native_gpu==expected&&stats.native_history_bytes==native_gpu,"native prefill cache must be actual all32-layer F16 GQA GPU payload");
  need(native_gpu<=stats.native_prefill_reserved_bytes,"actual padded native KV allocation exceeds explicit budget");
  stats.native_prefill_peak_bytes=std::max<uint64_t>(stats.native_prefill_peak_bytes,native_gpu);
 }else need(!stats.native_history_bytes,"GQA full native KV unexpectedly present during external execution");
 stats.phase_gpu_managed_peak_bytes=std::max(stats.phase_gpu_managed_peak_bytes,stats.workspace_bytes+stats.native_history_bytes+graph);
}
void GQALayerExecutor::finish_prefill(llama_context* c,std::uint64_t expected_prompt_tokens){
 need(hooks_.native_dense_prefill&&!native_prefill_finished_&&!decode_&&failure.empty(),"invalid/repeated native prefill transition");
 need(expected_prompt_tokens>0&&expected_prompt_tokens<=std::uint64_t(slots_)&&!positions_.empty()&&positions_.back()+1==expected_prompt_tokens,"native prefill does not match complete requested prompt");
 for(auto tokens:completed_)need(tokens==expected_prompt_tokens,"native prefill has not exported every position in all32 layers");
 need(stats.native_prefill_export_bytes==expected_prompt_tokens*L*E*4&&stats.native_prefill_layers==stats.attention_calls,"native prefill export/attention coverage incomplete");
 accounting(c);const auto before=stats.native_history_bytes;const auto started=Clock::now();
 const auto freed=llama_hillserve_finish_native_prefill(c,std::uint32_t(expected_prompt_tokens));
 stats.native_prefill_release_ms=ms(started);need(freed==before&&freed>0,"native prefill release accounting mismatch");
 stats.native_prefill_released_bytes=freed;native_prefill_finished_=true;
 if(hooks_.defer_native_workspace)allocate_workspace();
 configure();accounting(c);
 need(stats.native_history_bytes==0,"native KV payload survived external-decode transition");
 events_<<"{\"schema\":\"hillserve-native-prefill-v1\",\"phase\":\"native_prefill_released\",\"step\":"<<step_<<",\"prompt_tokens\":"<<expected_prompt_tokens<<",\"native_context_tokens\":"<<stats.native_prefill_context_tokens<<",\"released_bytes\":"<<freed<<",\"remaining_native_KV_bytes\":0,\"release_ms\":"<<stats.native_prefill_release_ms<<",\"all32_layers_exported\":true,\"native_prefill_peak_bytes\":"<<stats.native_prefill_peak_bytes<<",\"phase_gpu_managed_peak_bytes\":"<<stats.phase_gpu_managed_peak_bytes<<"}\n";
}
void GQALayerExecutor::drain(){if(store_){std::unique_lock<std::mutex> io_lock;if(hooks_.full_kv_io_fence)io_lock=std::unique_lock<std::mutex>(hooks_.full_kv_io_fence->file_io);store_->drain();}}
void GQALayerExecutor::observe_backend(ggml_backend_t backend) noexcept {
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
 try{if((hooks_.payload_prefetch||hooks_.h2o_attention_stats)&&backend&&ggml_backend_is_cuda(backend)&&ggml_backend_get_device(backend)==ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)){auto stream=ggml_backend_cuda_hillserve_stream(backend);need(stream,"actual CUDA attention stream unavailable");attention_stream_=stream;}}
 catch(const std::exception& e){failure=std::string("observe attention backend: ")+e.what();}
#else
 (void)backend;
#endif
}
bool GQALayerExecutor::callback(ggml_tensor* t,bool ask,void* u){auto& r=*static_cast<GQALayerExecutor*>(u);std::string n=t->name;bool a=n.rfind("hs_append_",0)==0,q=n.rfind("hs_rehearsal_",0)==0,at=n.rfind("kqv_out-",0)==0,h2o=n.rfind("hs_h2o_prob_",0)==0;if(ask)return a||q||at||h2o;if(!r.failure.empty())return false;try{if(q){need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t)&&ggml_nelements(t)==QE,"GQA rehearsal ABI");int target=-1;need(std::sscanf(n.c_str(),"hs_rehearsal_%d",&target)==1&&target>=2&&target<L,"GQA rehearsal target identity");auto& query=r.rehearsal_by_layer_.at(target);query.resize(QE);ggml_backend_tensor_get(t,query.data(),0,QE*4);++r.stats.rehearsal_calls;
 r.events_<<"{\"phase\":\"query_ready\",\"step\":"<<r.step_<<",\"layer\":"<<target<<",\"position\":"<<r.positions_.back()<<",\"ready_ns\":"<<std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()<<",\"F32_checksum\":\""<<checksum(query.data(),query.size()*4)<<"\",\"early_rehearsal\":"<<(r.hooks_.early_rehearsal?"true":"false")<<"}\n";
 if(r.hooks_.early_rehearsal&&r.hooks_.prepare_history){need(r.decode_&&r.positions_.size()==1,"early rehearsal requires single-token decode");runtime::GQAPrefetchReady ready;
#ifdef HILLSERVE_GQA_PAYLOAD_PREFETCH_CUDA
 if(r.hooks_.payload_prefetch){auto reserved=r.prefetch_->reserve(target,r.positions_.back());std::weak_ptr<PrefetchState> weak=r.prefetch_;ready=[weak,reserved](std::shared_ptr<const runtime::GQAPrefetchArtifact> artifact){auto state=weak.lock();need(bool(state),"payload executor destroyed before callback");state->stage(reserved.first,reserved.second,std::move(artifact));};}
#endif
 r.hooks_.prepare_history(target,r.positions_.back(),query,std::move(ready));}}else if(a){int l,c;need(std::sscanf(n.c_str(),"hs_append_%d_%d",&l,&c)==2,"GQA barrier name");r.prepare(t,l,c);}else if(at)r.inspect(t);else if(h2o){int layer=-1;need(std::sscanf(n.c_str(),"hs_h2o_prob_%d",&layer)==1&&layer>=0&&layer<L&&bool(r.hooks_.observe_attention),"H2O attention hook identity");r.hooks_.observe_attention(t,layer,r.positions_.front(),r.positions_.size(),r.completed_.at(layer),r.attention_stream_);}return true;}catch(const std::exception& e){r.failure=n+": "+e.what();return false;}}
}
