#pragma once
#include "runtime/gqa_delivery.hpp"
namespace hillserve {
// One GGML attention graph with a real independent sequence axis (ne[3]).
// Preparation may serialize on the single FPGA, but this never labels eight B1
// model invocations as a batched model decode. This is a batched layer replay.
class GQABatchDeliveryWorkspace {
 struct Impl;std::unique_ptr<Impl> p;
public:
 GQABatchDeliveryWorkspace(std::size_t batch,std::size_t slots,const std::vector<float>& queries,
                           std::uint64_t gpu_budget_bytes);
 ~GQABatchDeliveryWorkspace();
 void begin_round();
 GQADeliveryResult deliver(std::size_t sequence,const std::vector<std::vector<std::uint64_t>>& ids,
   std::uint64_t position,const std::vector<std::uint64_t>& physical,const std::vector<std::uint8_t>& payload);
 std::uint64_t batch_ready_ns()const;
 replay::json verify_sequence(std::size_t sequence,const std::vector<std::vector<std::uint64_t>>& ids,
   std::uint64_t position,const std::vector<std::uint64_t>& physical,const std::vector<std::uint8_t>& payload);
 replay::json verify_and_consume();
 std::uint64_t gpu_bytes()const;
 std::uint64_t host_staging_bytes()const;
};
}
