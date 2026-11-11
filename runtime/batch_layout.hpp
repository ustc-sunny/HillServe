#pragma once
// Capacity/addressing contract only. This does not execute eight requests and
// does not turn eight serial B=1 runs into a B=8 model measurement.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hillserve::batch {
using Bytes = std::uint64_t;
inline Bytes checked_add(Bytes a, Bytes b) {
    if (b > std::numeric_limits<Bytes>::max() - a)
        throw std::overflow_error("batch byte addition overflow");
    return a + b;
}
inline Bytes checked_mul(Bytes a, Bytes b) {
    if (a && b > std::numeric_limits<Bytes>::max() / a)
        throw std::overflow_error("batch byte multiplication overflow");
    return a * b;
}
inline Bytes checked_align(Bytes n, Bytes alignment) {
    if (!alignment) throw std::invalid_argument("zero alignment");
    const auto rem = n % alignment;
    return rem ? checked_add(n, alignment - rem) : n;
}
inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

struct TokenRef { Bytes sequence, position; };
enum class KVPart { key, value };
enum class SSDBacking { complete_KV_copy, cold_KV_only };
struct Spec {
    Bytes sequences = 8;
    Bytes prompt_tokens = 32768;
    Bytes cold_prompt_tokens = 16384;
    Bytes decode_capacity = 1;
    // Actual per-layer compact channel counts must be supplied by the producer.
    // Zero is deliberately invalid: no guessed uniform D across model layers.
    std::array<Bytes, 32> compact_channels{};
};
struct StorageSizes {
    Bytes prompt_full_KV = 0;
    Bytes capacity_full_KV = 0;
    Bytes cold_KV = 0;
    Bytes host_hot_KV_capacity = 0;
    Bytes cold_partial_K = 0;
    Bytes host_hot_partial_K_capacity = 0;
};
struct Admission {
    Bytes already_used = 0;
    Bytes new_bytes = 0;
    Bytes peak_bytes = 0;
    Bytes quota_bytes = 0;
    Bytes shortfall_bytes = 0;
    bool admitted = false;
};
inline Admission admit(Bytes new_bytes, Bytes logical, Bytes allocated, Bytes quota) {
    Admission a;
    a.already_used = std::max(logical, allocated);
    a.new_bytes = new_bytes;
    a.peak_bytes = checked_add(a.already_used, new_bytes);
    a.quota_bytes = quota;
    a.admitted = a.peak_bytes <= quota;
    a.shortfall_bytes = a.admitted ? 0 : a.peak_bytes - quota;
    return a;
}
struct WorkspacePlan {
    Bytes simultaneous_sequence_buffers = 0;
    Bytes packed_slots_per_sequence = 0;
    Bytes GPU_KV_mask_bytes = 0; // Existing expanded-Q-head packed ABI.
    Bytes explicit_GPU_other_reservation = 0; // Caller supplies graph/workspace.
    Bytes GPU_planned_bytes = 0;
    Bytes host_hot_KV_one_layer = 0;
    Bytes host_hot_partial_K_one_layer = 0;
    Bytes host_work_buffers = 0;
    Bytes host_one_layer_planned_bytes = 0;
    Bytes host_full_model_planned_bytes = 0;
    Bytes FPGA_BO_one_sequence_bytes = 0;
    // Library/driver RSS is not covered by this application buffer accounting.
};

class BatchLayout {
public:
    static constexpr Bytes layers = 32, query_heads = 32, KV_heads = 8;
    static constexpr Bytes dimension = 128, scalar_bytes = 2;
    static constexpr Bytes physical_token_bytes = 4096;
    explicit BatchLayout(Spec spec) : spec_(validated(spec)) {}
    const Spec& spec() const { return spec_; }
    Bytes capacity_per_sequence() const {
        return checked_add(spec_.prompt_tokens, spec_.decode_capacity);
    }
    Bytes valid_per_sequence(Bytes completed_decode_tokens) const {
        require(completed_decode_tokens <= spec_.decode_capacity, "decode capacity exceeded");
        return checked_add(spec_.prompt_tokens, completed_decode_tokens);
    }
    Bytes cold_per_sequence() const { return spec_.cold_prompt_tokens; }
    Bytes hot_per_sequence(Bytes completed_decode_tokens) const {
        return valid_per_sequence(completed_decode_tokens) - cold_per_sequence();
    }
    double cold_position_fraction(Bytes completed_decode_tokens) const {
        return double(cold_per_sequence()) / double(valid_per_sequence(completed_decode_tokens));
    }
    bool is_cold(TokenRef token, Bytes completed_decode_tokens) const {
        validate_token(token, completed_decode_tokens);
        return token.position < cold_per_sequence();
    }
    // Complete backing layout: [layer, sequence, position, K_or_V, KVhead, dim].
    Bytes full_KV_offset(Bytes layer, TokenRef token, KVPart part,
                         Bytes physical_head, Bytes channel, Bytes completed_decode_tokens) const {
        validate_coordinate(layer, token, physical_head, channel, completed_decode_tokens);
        return checked_add(checked_mul(record(layer, token.sequence, capacity_per_sequence(), token.position),
                                       physical_token_bytes), component_offset(part, physical_head, channel));
    }
    // Cold and hot stores retain the sequence namespace; position IDs never
    // become globally sorted IDs from a different sequence.
    Bytes cold_KV_offset(Bytes layer, TokenRef token, KVPart part,
                         Bytes physical_head, Bytes channel, Bytes completed_decode_tokens) const {
        validate_coordinate(layer, token, physical_head, channel, completed_decode_tokens);
        require(token.position < cold_per_sequence(), "hot position requested from cold store");
        return checked_add(checked_mul(record(layer, token.sequence, cold_per_sequence(), token.position),
                                       physical_token_bytes), component_offset(part, physical_head, channel));
    }
    Bytes hot_KV_offset(Bytes layer, TokenRef token, KVPart part,
                        Bytes physical_head, Bytes channel, Bytes completed_decode_tokens) const {
        validate_coordinate(layer, token, physical_head, channel, completed_decode_tokens);
        require(token.position >= cold_per_sequence(), "cold position requested from hot cache");
        return checked_add(checked_mul(record(layer, token.sequence,
                                  capacity_per_sequence() - cold_per_sequence(),
                                  token.position - cold_per_sequence()), physical_token_bytes),
                           component_offset(part, physical_head, channel));
    }
    // Variable-D layer file layout: [layer][sequence,Qhead,cold_position,D_l].
    Bytes cold_score_offset(Bytes layer, TokenRef token, Bytes query_head,
                            Bytes compact_channel, Bytes completed_decode_tokens) const {
        validate_token(token, completed_decode_tokens);
        require(layer < layers && query_head < query_heads, "score layer/head bounds");
        require(token.position < cold_per_sequence(), "cold score position bounds");
        require(compact_channel < spec_.compact_channels[layer], "compact channel bounds");
        Bytes base = 0;
        for (Bytes l = 0; l < layer; ++l) base = checked_add(base, score_layer_bytes(l, cold_per_sequence()));
        auto idx = checked_add(checked_mul(token.sequence, query_heads), query_head);
        idx = checked_add(checked_mul(idx, cold_per_sequence()), token.position);
        idx = checked_add(checked_mul(idx, spec_.compact_channels[layer]), compact_channel);
        return checked_add(base, checked_mul(idx, scalar_bytes));
    }
    StorageSizes storage_sizes() const {
        StorageSizes s;
        const auto factor = checked_mul(checked_mul(layers, spec_.sequences), physical_token_bytes);
        s.prompt_full_KV = checked_mul(factor, spec_.prompt_tokens);
        s.capacity_full_KV = checked_mul(factor, capacity_per_sequence());
        s.cold_KV = checked_mul(factor, cold_per_sequence());
        s.host_hot_KV_capacity = s.capacity_full_KV - s.cold_KV;
        for (Bytes l = 0; l < layers; ++l) {
            s.cold_partial_K = checked_add(s.cold_partial_K, score_layer_bytes(l, cold_per_sequence()));
            s.host_hot_partial_K_capacity = checked_add(s.host_hot_partial_K_capacity,
                score_layer_bytes(l, capacity_per_sequence() - cold_per_sequence()));
        }
        return s;
    }
    Bytes SSD_bytes(SSDBacking mode) const {
        require(mode == SSDBacking::complete_KV_copy || mode == SSDBacking::cold_KV_only,
                "unknown SSD backing mode");
        const auto s = storage_sizes();
        return checked_add(mode == SSDBacking::complete_KV_copy ? s.capacity_full_KV : s.cold_KV,
                           s.cold_partial_K);
    }
    Admission admit_SSD(SSDBacking mode, Bytes logical, Bytes allocated, Bytes quota) const {
        return admit(SSD_bytes(mode), logical, allocated, quota);
    }
    Bytes layer_KV_bytes(Bytes positions_per_sequence) const {
        require(positions_per_sequence <= capacity_per_sequence(), "layer positions exceed capacity");
        return checked_mul(checked_mul(spec_.sequences, positions_per_sequence), physical_token_bytes);
    }
    Bytes score_layer_bytes(Bytes layer, Bytes positions_per_sequence) const {
        require(layer < layers && positions_per_sequence <= capacity_per_sequence(), "score layer extent bounds");
        return checked_mul(checked_mul(checked_mul(checked_mul(spec_.sequences, query_heads),
                  positions_per_sequence), spec_.compact_channels[layer]), scalar_bytes);
    }
    WorkspacePlan workspace(Bytes layer, Bytes selected_cap_per_head,
                            Bytes simultaneous_sequence_buffers, Bytes GPU_other_bytes,
                            Bytes fixed_host_overhead = 16ULL * 1024 * 1024) const {
        require(layer < layers && selected_cap_per_head > 0 &&
                selected_cap_per_head <= capacity_per_sequence(), "workspace selection bounds");
        require(simultaneous_sequence_buffers > 0 && simultaneous_sequence_buffers <= spec_.sequences,
                "sequence workspace concurrency bounds");
        WorkspacePlan w;
        w.simultaneous_sequence_buffers = simultaneous_sequence_buffers;
        w.packed_slots_per_sequence = checked_align(selected_cap_per_head, 32);
        // K and V F16 expanded to32 Qheads; mask F32 [32,64,slots].
        const auto packed_KV = checked_mul(w.packed_slots_per_sequence, query_heads * dimension * 2 * scalar_bytes);
        const auto mask = checked_mul(w.packed_slots_per_sequence, query_heads * 64 * 4);
        const auto one_upload = checked_add(packed_KV, mask);
        w.GPU_KV_mask_bytes = checked_mul(one_upload, simultaneous_sequence_buffers);
        w.explicit_GPU_other_reservation = GPU_other_bytes;
        w.GPU_planned_bytes = checked_add(w.GPU_KV_mask_bytes, GPU_other_bytes);
        w.host_hot_KV_one_layer = layer_KV_bytes(capacity_per_sequence() - cold_per_sequence());
        w.host_hot_partial_K_one_layer = score_layer_bytes(layer, capacity_per_sequence() - cold_per_sequence());
        const auto cold_keys = checked_mul(checked_mul(checked_mul(query_heads, cold_per_sequence()),
                                                    spec_.compact_channels[layer]), scalar_bytes);
        const auto scores_one = checked_mul(checked_mul(query_heads, checked_align(capacity_per_sequence(), 64)), scalar_bytes);
        const auto selected_pairs = checked_mul(query_heads, selected_cap_per_head);
        const auto union_tokens = std::min(capacity_per_sequence(), selected_pairs);
        const auto gathered_KV_and_ID = checked_mul(union_tokens, physical_token_bytes + sizeof(Bytes));
        const auto selected_IDs = checked_mul(selected_pairs, sizeof(Bytes));
        // Four score vectors bound raw/normalized/split output/oracle. The
        // complete original KV oracle remains file-backed, not resident here.
        auto one_host = checked_add(one_upload, checked_align(cold_keys, 4096));
        one_host = checked_add(one_host, checked_mul(scores_one, 4));
        one_host = checked_add(one_host, gathered_KV_and_ID);
        one_host = checked_add(one_host, selected_IDs);
        one_host = checked_add(one_host, 64 * 1024); // finite gather/readback buffer
        w.host_work_buffers = checked_add(checked_mul(one_host, simultaneous_sequence_buffers), fixed_host_overhead);
        w.host_one_layer_planned_bytes = checked_add(checked_add(w.host_hot_KV_one_layer,
                                  w.host_hot_partial_K_one_layer), w.host_work_buffers);
        const auto s = storage_sizes();
        w.host_full_model_planned_bytes = checked_add(checked_add(s.host_hot_KV_capacity,
                                  s.host_hot_partial_K_capacity), w.host_work_buffers);
        const auto query = checked_mul(query_heads * scalar_bytes, spec_.compact_channels[layer]);
        const auto cold_scores = checked_mul(query_heads * scalar_bytes, cold_per_sequence());
        w.FPGA_BO_one_sequence_bytes = checked_add(checked_add(cold_keys, query), cold_scores);
        return w;
    }
private:
    const Spec spec_;
    static Spec validated(Spec spec) {
        require(spec.sequences > 0 && spec.sequences <= 8, "bounded pilot sequence count 1..8");
        require(spec.prompt_tokens > 0 && spec.prompt_tokens <= 32768, "bounded pilot prompt 1..32768");
        require(spec.decode_capacity <= 64, "bounded pilot decode capacity 0..64");
        require(spec.cold_prompt_tokens <= spec.prompt_tokens, "cold prompt exceeds prompt");
        for (auto d : spec.compact_channels) require(d > 0 && d <= dimension, "actual per-layer channels required");
        // Verify the maximum physical address and score extent before use.
        auto cap = checked_add(spec.prompt_tokens, spec.decode_capacity);
        checked_mul(checked_mul(checked_mul(layers, spec.sequences), cap), physical_token_bytes);
        return spec;
    }
    void validate_token(TokenRef token, Bytes completed_decode_tokens) const {
        require(token.sequence < spec_.sequences && token.position < valid_per_sequence(completed_decode_tokens),
                "sequence/position is not visible in this batch");
    }
    void validate_coordinate(Bytes layer, TokenRef token, Bytes head, Bytes channel, Bytes done) const {
        validate_token(token, done);
        require(layer < layers && head < KV_heads && channel < dimension, "physical KV coordinate bounds");
    }
    Bytes record(Bytes layer, Bytes sequence, Bytes stride, Bytes position) const {
        return checked_add(checked_mul(checked_add(checked_mul(layer, spec_.sequences), sequence), stride), position);
    }
    static Bytes component_offset(KVPart part, Bytes head, Bytes channel) {
        require(part == KVPart::key || part == KVPart::value, "unknown KV component");
        return checked_mul(checked_add(checked_add(part == KVPart::value ? KV_heads * dimension : 0,
                                                  checked_mul(head, dimension)), channel), scalar_bytes);
    }
};
} // namespace hillserve::batch
