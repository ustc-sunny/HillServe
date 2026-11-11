#pragma once
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "storage/file_store.hpp"
#include "runtime/fetch_plan.hpp"
#include "runtime/graph_profile.hpp"
#include <map>
#include <fstream>
#include <memory>
#include <string>
namespace hillserve {
using Plans=std::map<std::pair<int,int>,std::vector<std::uint64_t>>;
struct RuntimeStats {
    std::size_t native_history_bytes=0,current_prefill=0,current_decode=0,metadata_prefill=0,metadata_decode=0;
    std::size_t appends=0,gathers=0,attention_checks=0,kv_identity_checks=0;
    std::size_t current_peak=0,host_payload_peak=0,gpu_workspace_bytes=0,mask_bytes=0,kv_bytes=0;
    std::size_t append_prefill=0,append_decode=0,fetch_bytes=0,graph_prefill_gpu=0,graph_decode_gpu=0,graph_cpu=0;
    std::size_t storage_exact_checks=0,own_shadow_bytes=0,kv_drift_words=0,attention_violations=0;
    double kv_drift_max=0,pack_ms=0;
    double final_drain_ms=0;std::size_t dirty_peak_bytes=0,file_cache_reservation_bytes=0;
    double append_ms=0,fetch_ms=0,h2d_ms=0,max_attention_error=0;
};
class LayerExecutor {
    std::unique_ptr<IndexedKVStore> store_;
    ggml_context* tensor_ctx_=nullptr;
    ggml_backend_buffer_t buffer_=nullptr;
    ggml_tensor * k_=nullptr,*v_=nullptr,*mask_=nullptr;
    std::vector<std::uint64_t> positions_;
    std::vector<ggml_fp16_t> current_,host_k_,host_v_;
    std::vector<float> host_mask_;
    const Plans* plans_;
    bool external_,observe_,compact_oracle_;
    bool compact() const { return external_ || compact_oracle_; }
    int slots_,max_tokens_,step_=-1;
    std::uint64_t request_=1;
    std::size_t host_budget_;
    std::ofstream events_,selections_;
    KVDurability durability_=KVDurability::durable_per_layer;
    std::string directory_;
    std::unique_ptr<GraphProfiler> profiler_;
    std::vector<float> frozen_q_;
    bool first_bit_=false, first_kv_=false, numerical_=false;
    std::map<std::string,std::vector<ggml_fp16_t>> own_kv_;
    std::ofstream numerical_log_;
    std::map<std::string,std::vector<float>>* oracle_;
    std::map<std::string,std::vector<ggml_fp16_t>>* kv_oracle_;
    std::map<int,std::uint64_t> next_;
    void prepare(ggml_tensor*,int layer,int count);
    void inspect(ggml_tensor*);
public:
    RuntimeStats stats;
    std::string failure;
    LayerExecutor(const std::string& directory,bool external,int slots,int max_tokens,
                  std::size_t gpu_budget,std::size_t host_budget,const Plans*,
                  std::map<std::string,std::vector<float>>* oracle,
                  std::map<std::string,std::vector<ggml_fp16_t>>* kv_oracle,bool observe,bool compact_oracle=false,bool numerical=false);
    ~LayerExecutor();
    void begin(int step,const std::vector<std::uint64_t>& positions);
    void configure();
    void attach_profile(llama_context* ctx);
    void accounting(llama_context*,bool prefill);
    std::size_t metadata_peak()const{return store_?store_->metadata().peak:0;}
    std::size_t backing_bytes()const{return store_?store_->append_write_bytes:0;}
    void release();
    static bool callback(ggml_tensor*,bool ask,void*);
};
}
