#include "runtime/batch_layout.hpp"
#include <iostream>
#include <set>
#include <string>
using namespace hillserve::batch;
static void check(bool x, const char* message) { if (!x) throw std::runtime_error(message); }
template<class F> static void rejects(F f, const char* message) {
    bool threw = false; try { f(); } catch (const std::exception&) { threw = true; }
    check(threw, message);
}
static Spec example() { Spec s; s.compact_channels.fill(49); return s; }
int main() { try {
    const auto spec = example(); const BatchLayout layout(spec); const auto sizes = layout.storage_sizes();
    check(sizes.prompt_full_KV == 34359738368ULL, "B8/32K physical KV must be32GiB");
    check(sizes.cold_KV == 17179869184ULL, "cold prompt half16GiB");
    check(sizes.host_hot_KV_capacity == 17180917760ULL, "hot half plus eight new decode tokens/all layers");
    check(layout.cold_position_fraction(0) == .5 && layout.cold_position_fraction(1) < .5,
          "post-decode ratio must not claim exact50percent");
    check(layout.hot_per_sequence(0) == 16384 && layout.hot_per_sequence(1) == 16385, "decode appends to hot");
    std::set<Bytes> offsets;
    for (Bytes layer : {0ULL, 8ULL, 31ULL}) for (Bytes seq = 0; seq < 8; ++seq)
        for (Bytes pos : {0ULL, 16383ULL, 16384ULL, 32767ULL, 32768ULL})
            for (auto part : {KVPart::key, KVPart::value}) for (Bytes head = 0; head < 8; ++head) {
                const auto off = layout.full_KV_offset(layer, {seq, pos}, part, head, 127, 1);
                check(off + 2 <= sizes.capacity_full_KV, "physical address within extent");
                check(offsets.insert(off).second, "batch sequence/layer/head alias");
            }
    check(layout.full_KV_offset(0,{1,0},KVPart::key,0,0,1) == 32769ULL*4096, "sequence stride");
    check(layout.full_KV_offset(0,{0,1},KVPart::key,0,0,1) == 4096, "token stride");
    check(layout.full_KV_offset(0,{0,0},KVPart::value,0,0,1) == 2048, "V physical offset");
    check(layout.cold_KV_offset(0,{1,0},KVPart::key,0,0,1) == 16384ULL*4096, "cold sequence stride");
    check(layout.hot_KV_offset(0,{1,16384},KVPart::key,0,0,1) == 16385ULL*4096, "hot sequence stride");
    check(layout.cold_score_offset(0,{1,0},0,0,1) == 32ULL*16384*49*2, "score sequence stride");
    Spec variable = spec; variable.compact_channels[0] = 25; variable.compact_channels[1] = 50;
    const BatchLayout varying(variable);
    check(varying.cold_score_offset(1,{0,0},0,0,1) == 8ULL*32*16384*25*2, "variableD layer offset");
    check(varying.cold_score_offset(2,{0,0},0,0,1) == 8ULL*32*16384*75*2, "variableD second prefix");
    rejects([&]{layout.full_KV_offset(0,{8,0},KVPart::key,0,0,1);}, "out-of-batch sequence accepted");
    rejects([&]{layout.full_KV_offset(0,{0,32768},KVPart::key,0,0,0);}, "future decode position visible");
    rejects([&]{layout.full_KV_offset(0,{0,0},KVPart::key,8,0,1);}, "Qhead mistaken for physical head");
    rejects([&]{layout.cold_KV_offset(0,{0,16384},KVPart::key,0,0,1);}, "hot served from cold index");
    rejects([&]{layout.hot_KV_offset(0,{0,16383},KVPart::key,0,0,1);}, "cold served from hot index");
    rejects([&]{layout.valid_per_sequence(2);}, "decode capacity overflow accepted");
    rejects([&]{Spec bad=spec;bad.sequences=9;BatchLayout x(bad);}, "batch bound ignored");
    rejects([&]{Spec bad=spec;bad.prompt_tokens=65504;BatchLayout x(bad);}, "prompt bound ignored");
    rejects([&]{Spec bad=spec;bad.compact_channels[31]=0;BatchLayout x(bad);}, "unspecified layerD guessed");
    rejects([&]{checked_add(~Bytes(0),1);}, "addition overflow accepted");
    rejects([&]{checked_mul(~Bytes(0),2);}, "multiplication overflow accepted");
    rejects([&]{checked_align(~Bytes(0),4096);}, "alignment overflow accepted");
    rejects([&]{checked_align(1,0);}, "zero alignment accepted");
    const auto full = layout.SSD_bytes(SSDBacking::complete_KV_copy), cold = layout.SSD_bytes(SSDBacking::cold_KV_only);
    check(full-cold == sizes.host_hot_KV_capacity, "full backing and cold-only cannot be conflated");
    rejects([&]{layout.SSD_bytes(static_cast<SSDBacking>(9));}, "unknown SSD mode accepted");
    const auto rejected=layout.admit_SSD(SSDBacking::cold_KV_only,8507297792ULL,8352468992ULL,8589934592ULL);
    check(!rejected.admitted && rejected.shortfall_bytes == cold-82636800, "old8GiB quota rejects actual new files");
    check(admit(100,0,20,120).admitted && !admit(101,0,20,120).admitted, "quota exact boundary");
    const auto one=layout.workspace(8,6553,1,1048576), all=layout.workspace(8,6553,8,1048576);
    check(one.packed_slots_per_sequence==6560, "padded selected count");
    check(all.GPU_KV_mask_bytes==8*one.GPU_KV_mask_bytes, "true eight-buffer batch memory");
    check(one.FPGA_BO_one_sequence_bytes==52431936, "FPGA Q/K/score bytes");
    check(one.host_hot_KV_one_layer==536903680, "hot layer all8seq accounted");
    check(one.host_full_model_planned_bytes > 17179869184ULL, "all-layer CPUhot not hidden by perseq reuse");
    rejects([&]{layout.workspace(8,6553,9,0);}, "workspace batch count exceeds batch");
    rejects([&]{layout.workspace(8,0,1,0);}, "zero selection cap accepted");
    Spec old=spec;old.sequences=1;old.prompt_tokens=16384;old.cold_prompt_tokens=10240;
    const BatchLayout legacy(old);
    check(legacy.layer_KV_bytes(16385)==67112960, "old16K physical extent changed");
    check(legacy.score_layer_bytes(8,10240)==32112640, "old16K cold score extent changed");
    check(legacy.hot_per_sequence(1)==6145, "old16K hot count changed");
    std::cout << "passed: B8 sequence isolation, physical GQA layout, 50:50 prompt placement, hot decode, "
                 "variable-D score extents, budget/overflow rejection, bounded workspaces and old16K sizes\n";
    std::cout << "contract_only=true actual_B8_model_execution=false cold_KV_bytes=" << sizes.cold_KV
              << " full_KV_bytes_with_decode_capacity=" << sizes.capacity_full_KV
              << " one_sequence_GPU_pack_bytes=" << one.GPU_KV_mask_bytes << '\n';
    return 0;
} catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;} }
