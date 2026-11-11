#pragma once
#include "storage/file_store.hpp"
#include <vector>
#include <cstdint>
#include <memory>
namespace hillserve {
struct AttentionOutput {std::vector<float> values;double pack_ms=0,h2d_ms=0,compute_wall_ms=0,d2h_ms=0,total_ms=0;std::size_t h2d_bytes=0,gpu_bytes=0;};
class MethodAttention {
 struct Impl;std::unique_ptr<Impl> p;
public:
 MethodAttention(std::size_t heads,std::size_t dim,std::size_t cap);
 ~MethodAttention();
 AttentionOutput run(const std::vector<float>& scaled_query,const std::vector<std::uint16_t>& current_kv,const std::vector<KVPayload>& selected,std::size_t k);
 std::size_t gpu_bytes()const;
};
}
