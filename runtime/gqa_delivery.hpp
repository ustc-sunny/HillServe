#pragma once
#include "hybrid_eval/gqa_frozen.hpp"
#include "runtime/gqa_pack.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <memory>
namespace hillserve {
struct GQADeliveryResult { double pack_ms=0,upload_wall_ms=0,cuda_copy_stream_interval_ms=0;std::uint64_t gpu_ready_ns=0,upload_bytes=0; };
class GQADeliveryWorkspace {
 struct Impl;std::unique_ptr<Impl> p;
public:
 GQADeliveryWorkspace(std::size_t count,const std::vector<float>& actual_attention_query);
 ~GQADeliveryWorkspace();
 GQADeliveryResult deliver(const std::vector<std::vector<std::uint64_t>>& ids,std::uint64_t query_position,const std::vector<std::uint64_t>& physical,const std::vector<std::uint8_t>& payload);
 replay::json verify_and_consume();
 std::uint64_t gpu_bytes()const;
};
}
