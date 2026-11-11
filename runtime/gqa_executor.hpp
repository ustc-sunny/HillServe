#pragma once
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "storage/segmented_store.hpp"
#include "runtime/gqa_prefetch.hpp"
#include <functional>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
namespace hillserve {
using GQAAppend=std::function<void(int,const std::vector<uint64_t>&,const std::vector<uint16_t>&)>;
using GQASelect=std::function<std::vector<std::vector<uint64_t>>(int,uint64_t,const std::vector<float>&)>;
using GQAColdFetch=std::function<std::vector<uint8_t>(int,const std::vector<uint64_t>&)>;
struct GQARuntimeHooks {
 std::function<void(int,const std::vector<uint64_t>&,const std::vector<uint16_t>&,const std::vector<uint8_t>&)> append_full;
 std::function<std::vector<uint8_t>(int,const std::vector<uint64_t>&,const GQAColdFetch&)> fetch;
 std::function<void(int,uint64_t,const std::vector<float>&,runtime::GQAPrefetchReady)> prepare_history;
 std::function<void()> wait_pending;
 bool payload_prefetch=false;
 std::shared_ptr<runtime::GQAPrefetchIOFence> full_kv_io_fence;
 bool early_rehearsal=false;
 // Dense prefill keeps the physical eight GQA KV heads and one causal mask.
 // Decode retains the existing per-query-head selection and delivery layout.
 bool compact_dense_prefill=false;
 // One-way native dense prefill -> external decode; never keeps native history
 // during decoding. The temporary native allocation is separately budgeted.
 bool native_dense_prefill=false;
 // Opt-in capacity support: native prefill never reads external tensors.
 // Allocate their GPU backing only after releasing native history.
 bool defer_native_workspace=false;
 // Full-attention always uses physical eight-head GQA layout. Default keeps
 // the historical larger reservation; opt-in avoids unused query-head slots.
 bool compact_full_workspace=false;
 // Explicit budgeted in-memory backing supplied by hooks; never the debug oracle.
 bool external_memory_backing=false;
 // Fixed no-SSD residency intentionally discards cold exports. Debug verification
 // then checks full fetch and actual GPU bytes during decode, not export rereads.
 bool skip_native_export_readback=false;
 bool full_attention=false;
 bool h2o_attention_stats=false;
 std::function<void(ggml_tensor*,int,std::uint64_t,std::size_t,std::size_t,void*)> observe_attention;
 std::uint64_t native_prefill_kv_budget_bytes=0;
 std::uint64_t kv_file_quota_bytes=8ULL*1024*1024*1024;
};
struct GQAStats {uint64_t appends=0,attention_calls=0,rehearsal_calls=0,logical_selected_bytes=0,physical_selected_bytes=0,exact_bytes=0,exact_gpu_bytes=0,logical_head_selections=0,gpu_h2d_bytes=0,native_history_bytes=0,graph_gpu_bytes=0,workspace_bytes=0,managed_host_bytes=0;uint64_t compact_prefill_layers=0,prefetch_layers=0,prefetch_gpu_bytes=0,prefetch_host_bytes=0,prefetch_h2d_bytes=0,prefetch_useful_bytes=0,prefetch_wasted_bytes=0,prefetch_miss_h2d_bytes=0,gpu_d2d_bytes=0;double prefetch_consume_wait_ms=0,prefetch_gpu_upload_ms=0;double append_ms=0,fetch_ms=0,h2d_ms=0,attention_max_abs=0;
 uint64_t native_prefill_reserved_bytes=0,native_prefill_peak_bytes=0,native_prefill_context_tokens=0;
 uint64_t native_prefill_layers=0,native_prefill_export_bytes=0,native_prefill_export_verified_bytes=0,native_prefill_released_bytes=0;
 uint64_t phase_gpu_managed_peak_bytes=0;
 double native_prefill_release_ms=0;
};
class GQALayerExecutor {
 static constexpr int D=128,QH=32,KH=8,L=32,E=D*KH,QE=D*QH;
 bool memory_,verify_,decode_=false;int slots_,batch_,step_=0;float cap_ratio_;
 bool native_prefill_finished_=false;
 size_t workspace_budget_=0;
 std::vector<uint64_t> positions_;std::vector<float> mask_host_;std::vector<std::vector<float>> rehearsal_by_layer_;
 std::vector<ggml_fp16_t> current_,k_host_,v_host_;
 std::unique_ptr<SegmentedKVStore> store_;
 std::vector<std::vector<uint8_t>> backing_;
 std::vector<uint64_t> completed_;
 ggml_context* tensors_=nullptr;ggml_backend_buffer_t buffer_=nullptr;
 ggml_tensor *k_=nullptr,*v_=nullptr,*mask_=nullptr;
 GQAAppend append_policy_;GQASelect select_policy_;GQARuntimeHooks hooks_;
 std::ofstream events_,selections_,attention_;
 struct PrefetchState;std::shared_ptr<PrefetchState> prefetch_;
 void* attention_stream_=nullptr;
 void prepare(ggml_tensor*,int,int);void inspect(ggml_tensor*);
 void allocate_workspace();
public:
 GQAStats stats;std::string failure;
 GQALayerExecutor(const std::string& dir,const std::string& backing_file,int slots,int batch,float cap_ratio,bool memory,bool verify,bool direct,size_t gpu_budget,size_t host_budget,GQAAppend,GQASelect,GQARuntimeHooks hooks={});
 ~GQALayerExecutor();
 void begin(int step,const std::vector<uint64_t>& positions,bool decode);
 void configure();void accounting(llama_context*);void drain();
 // Must be called after the complete live prompt and before begin(...,true).
 void finish_prefill(llama_context*,std::uint64_t expected_prompt_tokens);
 // Called on the scheduler/main thread from the actual backend profile hook.
 void observe_backend(ggml_backend_t) noexcept;
 const SegmentedKVStore* store()const{return store_.get();}
 static bool callback(ggml_tensor*,bool,void*);
};
}
