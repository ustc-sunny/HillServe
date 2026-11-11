#include "runtime/full_gqa_contract.hpp"
#include "policy/full_gqa_runtime.hpp"
#include "policy/full_gqa_cpu.hpp"
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "storage/segmented_store.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace hillserve;
using namespace hillserve::policy;
using namespace hillserve::runtime;
namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
FullGqaRuntimeSpec spec() {
    FullGqaRuntimeSpec s;
    s.prompt_tokens = 128;
    s.generated_capacity = 3;
    s.capacity = 192;
    s.beta = 1;
    s.hot_kv_budget_bytes = 32 * 67 * 4096ULL;
    s.additional_workspace_bytes = 8ULL << 20;
    s.total_host_workspace_budget_bytes = s.hot_kv_budget_bytes + s.additional_workspace_bytes;
    s.chunk_tokens = 64;
    return s;
}
void capacity_contracts() {
    auto s = spec();
    auto p = plan_full_gqa_runtime(s);
    check(p.prompt_hot_tokens == 64 && p.prompt_cold_tokens == 64,
          "fixed beta prompt partition");
    check(p.prompt_cold_tokens % 64 == 0 && p.recent_pin_tokens == 64,
          "cold64 alignment and bounded recent window");
    check(p.reserved_all_layers_hot_bytes == 32 * 67 * 4096ULL,
          "must reserve generated tail in ALL32 layers");
    check(p.managed_total == s.total_host_workspace_budget_bytes,
          "hot KV and explicit additional workspace share total budget");
    check(p.raw_K_extent_bytes == 32 * 192 * 2048ULL &&
          p.raw_K_layer_stride_bytes % 4096 == 0,
          "raw score file covers aligned complete layer extents");
    check(p.segmented_KV_extent_bytes ==
          full_gqa_align_page(4096 + 32 * 192 * 4128ULL),
          "existing segmented format record headers included");
    check(p.storage_total_bytes == p.raw_K_extent_bytes + p.segmented_KV_extent_bytes,
          "both complete backing and rawK file accounted");
    check(p.bo_reserved_bytes == 8192 + 64 * (2048 + 64),
          "one cold chunk BO accounting");
    for (std::size_t t = 0; t < 192; ++t) {
        const auto expected = t >= 64 && t < 131 ? std::int64_t(t - 64) : -1;
        check(p.hot_slot_by_position[t] == expected,
              "prompt hot map, reserved tail, and padding distinct");
    }
    p.require_visible(31, 127, 128);
    rejects([&] { p.require_visible(31, 128, 128); }, "future reserved tail is not readable");
    rejects([&] { p.require_visible(32, 0, 128); }, "layer bounds");
    rejects([&] { p.require_visible(0, 0, 132); }, "completed outside decode allowance");
    auto bad = s; bad.hot_kv_budget_bytes = p.reserved_all_layers_hot_bytes - 1;
    rejects([&] { plan_full_gqa_runtime(bad); }, "all-layer hot budget one-byte overflow");
    bad = s; bad.hot_kv_budget_bytes = 67 * 4096;
    rejects([&] { plan_full_gqa_runtime(bad); }, "one-layer budget cannot be reused32 times");
    bad = s; --bad.total_host_workspace_budget_bytes;
    rejects([&] { plan_full_gqa_runtime(bad); }, "workspace admission one-byte overflow");
    bad = s; bad.additional_workspace_bytes = std::numeric_limits<std::uint64_t>::max();
    rejects([&] { plan_full_gqa_runtime(bad); }, "checked total overflow");
    bad = s; bad.storage_quota_bytes = p.storage_total_bytes - 1;
    rejects([&] { plan_full_gqa_runtime(bad); }, "two-file storage cap");
    bad = s; bad.bo_budget_bytes = p.bo_reserved_bytes - 1;
    rejects([&] { plan_full_gqa_runtime(bad); }, "actual BO reservation bounded");
    bad = s; bad.bo_budget_bytes = full_gqa_max_bo_bytes + 1;
    rejects([&] { plan_full_gqa_runtime(bad); }, "cannot increase authorized BO maximum");
    bad = s; bad.layers = 1;
    rejects([&] { plan_full_gqa_runtime(bad); }, "complete32-layer accounting mandatory");
    bad = s; bad.capacity = 131;
    rejects([&] { plan_full_gqa_runtime(bad); }, "odd layer stride must not reach O_DIRECT");
    bad = s; bad.capacity = 128;
    rejects([&] { plan_full_gqa_runtime(bad); }, "all generated tokens must fit");
    bad = s; bad.generated_capacity = 513;
    rejects([&] { plan_full_gqa_runtime(bad); }, "bounded decode capacity");
    bad = s; bad.prompt_tokens = 129;
    rejects([&] { plan_full_gqa_runtime(bad); }, "prompt tail cannot silently become cold padding");
    // Exact natural prompt length: incomplete last block is explicitly hot,
    // cold extents remain O_DIRECT/FPGA aligned and generated IDs are not shifted.
    for (std::uint64_t n : {129ULL, 191ULL, 32767ULL}) {
        auto partial = s; partial.prompt_tokens=n; partial.generated_capacity=512;
        partial.capacity=((n+512+63)/64)*64; partial.allow_partial_prompt=true;
        partial.hot_kv_budget_bytes=8ULL<<30; partial.total_host_workspace_budget_bytes=9ULL<<30;
        const auto tail=plan_full_gqa_runtime(partial);
        check(tail.prompt_tail_hot_tokens==n%64 && tail.prompt_cold_tokens%64==0,
              "partial prompt cold blocks and pinned tail");
        check(tail.prompt_hot_tokens+tail.prompt_cold_tokens==n,
              "natural prompt partition has no lost or artificial tokens");
        for(auto t=n-n%64;t<n;++t) check(tail.hot_slot_by_position[t]>=0,"tail must stay hot");
        for(auto t=n;t<n+512;++t) check(tail.hot_slot_by_position[t]==std::int64_t(t-n+tail.prompt_hot_tokens),"generated slots follow real prompt");
        check(tail.reserved_all_layers_hot_bytes==32*(tail.prompt_hot_tokens+512)*4096ULL,"all32 layers include partial tail and long continuation");
        rejects([&]{tail.require_visible(0,n,n);},"tail reservation not prematurely readable");
        --partial.hot_kv_budget_bytes; partial.hot_kv_budget_bytes=tail.reserved_all_layers_hot_bytes-1;
        rejects([&]{plan_full_gqa_runtime(partial);},"tail counts in hot budget");
    }
    auto aligned=s;aligned.allow_partial_prompt=true;
    check(plan_full_gqa_runtime(aligned).hot_slot_by_position==p.hot_slot_by_position,
          "opt-in natural length preserves aligned historical partition");
    bad = s; bad.chunk_tokens = 65;
    auto wide=s;wide.prompt_tokens=65536;wide.generated_capacity=32;wide.capacity=65600;
    wide.hot_kv_budget_bytes=5ULL<<30;wide.total_host_workspace_budget_bytes=12ULL<<30;
    rejects([&]{plan_full_gqa_runtime(wide);},"64K must explicitly extend count representation");
    wide.fp32_selection_counts=true;
    const auto wp=plan_full_gqa_runtime(wide);
    check(wp.capacity==65600&&wp.storage_total_bytes==12964663296ULL&&wp.segmented_KV_extent_bytes==8665501696ULL,"64K+decode file headers and padding accounted");
    rejects([&] { plan_full_gqa_runtime(bad); }, "cold chunk alignment");
    bad = s; bad.chunk_tokens = 65536;
    rejects([&] { plan_full_gqa_runtime(bad); }, "V3 chunk capacity");
    rejects([&] { full_gqa_hot_blocks(128, -1); }, "negative beta rejected");
    rejects([&] { full_gqa_hot_blocks(128, std::numeric_limits<double>::quiet_NaN()); },
            "NaN beta rejected");
    check(full_gqa_hot_blocks(32768, 1) == 256 &&
          full_gqa_hot_blocks(32768, 3) == 384 &&
          full_gqa_hot_blocks(32768, 15) == 480,
          "same offline beta maps deterministically to prompt block capacity");
    bad = s; bad.beta = 0;
    auto cold = plan_full_gqa_runtime(bad);
    check(cold.prompt_hot_tokens == 0 && cold.reserved_all_layers_hot_bytes == 32 * 3 * 4096ULL &&
          cold.bo_reserved_bytes == 2 * (8192 + 64 * (2048 + 64)),
          "all-cold prompt still reserves generated tail and double BOs");
    bad = s; bad.beta = std::numeric_limits<double>::infinity();
    bad.hot_kv_budget_bytes = 32 * 131 * 4096ULL;
    bad.total_host_workspace_budget_bytes = bad.hot_kv_budget_bytes + bad.additional_workspace_bytes;
    auto hot = plan_full_gqa_runtime(bad);
    check(hot.prompt_cold_tokens == 0 && hot.bo_reserved_bytes == 0,
          "explicit all-hot control naturally bypasses FPGA buffer reservation");
}

// Deterministic, finite, layer/position/head-specific software test tensors.
// These are NOT model traces and these timings are never performance results.
std::uint16_t key_word(std::size_t layer, std::uint64_t position,
                       std::size_t head, std::size_t d) {
    const float t = float(int(position % 97) - 48);
    return half_bits(t * 0.001f + float(layer) * 0.0001f +
                     float(head) * 0.0002f + float(d) * 0.000001f);
}
std::vector<std::uint8_t> records(std::size_t layer, std::uint64_t first, std::size_t count) {
    std::vector<std::uint16_t> words(count * 2048);
    for (std::size_t t = 0; t < count; ++t) {
        for (std::size_t kh = 0; kh < 8; ++kh) {
            for (std::size_t d = 0; d < 128; ++d) {
                words[t * 2048 + kh * 128 + d] = key_word(layer, first + t, kh, d);
                words[t * 2048 + 1024 + kh * 128 + d] =
                    half_bits(float(layer + 1) * 0.01f + float(first + t) * 0.0001f +
                              float(kh * 128 + d) * 0.000001f);
            }
        }
    }
    std::vector<std::uint8_t> bytes(words.size() * 2);
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}
std::vector<std::uint16_t> keys_from(const std::vector<std::uint8_t>& kv) {
    std::vector<std::uint16_t> keys(kv.size() / 4);
    for (std::size_t t = 0; t < kv.size() / 4096; ++t)
        std::memcpy(keys.data() + t * 1024, kv.data() + t * 4096, 2048);
    return keys;
}
std::vector<float> query(std::size_t layer, std::size_t step) {
    std::vector<float> q(32 * 128);
    for (std::size_t h = 0; h < 32; ++h)
        for (std::size_t d = 0; d < 128; ++d)
            q[h * 128 + d] = (h % 2 ? -1.f : 1.f) *
                (0.01f + float(layer + step) * 0.0001f + float(d) * 0.00001f);
    return q;
}
std::vector<std::vector<std::uint64_t>> expected_ids(
        std::size_t layer, std::size_t valid, const std::vector<float>& q) {
    std::vector<std::uint16_t> scaled(32 * valid);
    for (std::size_t h = 0; h < 32; ++h) {
        for (std::size_t t = 0; t < valid; ++t) {
            volatile float sum = 0;
            for (std::size_t d = 0; d < 128; ++d) {
                volatile float mul = half_value(half_bits(q[h * 128 + d])) *
                                     half_value(key_word(layer, t, h / 4, d));
                sum = sum + mul;
            }
            const auto raw = half_bits(sum);
            scaled[h * valid + t] = half_bits(half_value(raw) / std::sqrt(128.f));
        }
    }
    return select_gqa_scores(scaled, 32, valid, valid, 5, 0.2f).ids;
}
std::vector<std::uint64_t> physical_union(const std::vector<std::vector<std::uint64_t>>& per_head) {
    std::vector<std::uint64_t> ids;
    for (const auto& h : per_head) ids.insert(ids.end(), h.begin(), h.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}
// CPU-only contract fixture, not model input or measured performance. Compare
// the shared old replay scorer against an independent channel-order oracle
// and the existing automatic implementation, including non-tile-aligned tails.
std::size_t legacy_cpu_contracts() {
    std::size_t compared=0;
    const std::uint16_t edges[]={0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
        0x3c00,0xbc00,0x3555,0xb555};
    for(unsigned threads:{1u,3u,8u}){
        FullGqaCpuScorer automatic(threads);
        for(std::size_t n:{1u,7u,65u})for(std::size_t stride:{1024u,2048u}){
            std::vector<std::uint16_t> q(32*128),keys(n*stride,0x7e00);
            for(std::size_t i=0;i<q.size();++i)q[i]=edges[(i*7+i/128)%12];
            for(std::size_t t=0;t<n;++t)for(std::size_t d=0;d<1024;++d)
                keys[t*stride+d]=(t+d)%11==0?edges[(t*3+d)%12]:half_bits(float(int((t*7+d*13)%53)-26)*0.003f);
            std::vector<std::uint16_t> expected(32*n),optimized;
            for(std::size_t h=0;h<32;++h)for(std::size_t t=0;t<n;++t){
                volatile float sum=0;
                for(std::size_t d=0;d<128;++d){
                    volatile float product=half_value(q[h*128+d])*half_value(keys[t*stride+(h/4)*128+d]);
                    sum=sum+product;
                }
                expected[h*n+t]=half_bits(sum);
            }
            const auto old=score_full_gqa_legacy_cpu(q,keys.data(),n,stride,threads);
            automatic.score(q,{{keys.data(),n,stride}},optimized);
            check(old==expected,"shared legacy score must match ordered raw F16 oracle bitwise");
            check(old==optimized,"legacy/automatic CPU scores must match bitwise");
            compared+=expected.size()*2;
        }
    }
    std::vector<std::uint16_t> q(32*128),key(1024);
    rejects([&]{score_full_gqa_legacy_cpu(q,key.data(),1,1024,0);},"legacy zero threads rejected");
    rejects([&]{score_full_gqa_legacy_cpu(q,key.data(),1,1024,33);},"legacy threads beyond original32 rejected");
    rejects([&]{score_full_gqa_legacy_cpu(q,key.data(),1,17,1);},"legacy unsupported stride rejected");
    rejects([&]{score_full_gqa_legacy_cpu(q,key.data(),0,1024,1);},"legacy empty input rejected");
    return compared;
}
}

int main(int argc, char** argv) {
    std::filesystem::path dir;
    try {
        capacity_contracts();
        if (argc == 2 && std::string(argv[1]) == "--contract-only") {
            std::cout << "{\"status\":\"passed\",\"evidence\":\"CPU_capacity_contract_only\",\"layers\":32}\n";
            return 0;
        }
        const auto legacy_compared=legacy_cpu_contracts();
        if(argc==2&&std::string(argv[1])=="--cpu-scorer-only"){
            std::cout<<"{\"status\":\"passed\",\"evidence\":\"CPU_legacy_scorer_contract_fixture_only\",\"legacy_implementation\":\""<<full_gqa_legacy_cpu_name<<"\",\"bitwise_score_comparisons\":"<<legacy_compared<<",\"GPU_executed\":false,\"FPGA_executed\":false}\n";
            return 0;
        }
        const bool legacy=argc==2&&std::string(argv[1])=="--legacy";
        check(argc == 1 || legacy, "unknown runtime test arguments");
        dir = std::filesystem::temp_directory_path() /
              ("hillserve-full-gqa-runtime-" + std::to_string(getpid()));
        check(std::filesystem::create_directory(dir), "exclusive software test directory");
        const auto s = spec(); const auto p = plan_full_gqa_runtime(s);
        {
            SegmentedKVStore backing(dir / "complete-kv", 991, 32, s.capacity, 4096,
                                      p.segmented_KV_extent_bytes);
            check(backing.extent_bytes() == p.segmented_KV_extent_bytes,
                  "planned full KV extent must equal actual existing store format");
            FullGqaRuntimePolicy::Config cfg;
            cfg.layers = 32; cfg.capacity = s.capacity; cfg.prompt_tokens = s.prompt_tokens;
            cfg.generated_capacity = s.generated_capacity; cfg.beta = s.beta;
            cfg.hot_kv_budget_bytes = p.reserved_all_layers_hot_bytes;
            cfg.total_host_workspace_budget_bytes = 128ULL << 20;
            cfg.chunk_tokens = 64; cfg.hot_threads = 2; cfg.cold_threads = 1;
            cfg.mode = "HOST_MATCHED"; cfg.score_file = (dir / "raw-k").string();
            cfg.verify_scores = true;cfg.cpu_implementation=legacy?"legacy":"auto";
            auto bad = cfg; bad.score_file = (dir / "reject-budget").string();
            --bad.hot_kv_budget_bytes;
            rejects([&] { FullGqaRuntimePolicy invalid(bad); }, "policy must enforce complete hot budget");
            check(!std::filesystem::exists(bad.score_file), "budget rejection before file creation");
            bad=cfg;bad.cpu_implementation="unknown";bad.score_file=(dir/"reject-cpu-impl").string();
            rejects([&] { FullGqaRuntimePolicy invalid(bad); }, "unknown CPU implementation rejected");
            check(!std::filesystem::exists(bad.score_file), "CPU implementation rejection before file creation");
            FullGqaRuntimePolicy policy(cfg);
            check(policy.stats().cpu_implementation_requested==cfg.cpu_implementation,"CPU implementation request recorded");
            check(legacy?policy.stats().cpu_implementation_actual==full_gqa_legacy_cpu_name:
                  (policy.stats().cpu_implementation_actual=="scalar_ordered_fp32"||policy.stats().cpu_implementation_actual=="avx2_f16c_ordered_fp32"),"actual CPU implementation recorded");
            check((policy.stats().legacy_cpu_workspace_bytes>0)==legacy,"legacy packed K/transient scores included in memory reservation");
            check(policy.hot_prompt_tokens() == 64 && policy.cold_prompt_tokens() == 64,
                  "policy uses frozen prompt partition");
            check(policy.stats().hot_kv_cache_bytes == p.reserved_all_layers_hot_bytes,
                  "runtime hot allocation reserves all layers and decode tail");
            check(policy.managed_bytes() <= cfg.total_host_workspace_budget_bytes,
                  "runtime complete managed budget");
            check(policy.stats().score_file_extent_bytes == p.raw_K_extent_bytes,
                  "runtime rawK extent agrees with contract");
            rejects([&] { policy.select(0, 0, query(0, 0)); }, "select before any append");
            auto append = [&](std::size_t layer, std::uint64_t first, std::size_t count) {
                auto kv = records(layer, first, count); auto keys = keys_from(kv);
                std::vector<std::uint64_t> positions(count);
                std::iota(positions.begin(), positions.end(), first);
                backing.append(layer, first, kv);
                policy.append(int(layer), positions, keys, kv);
            };
            for (std::size_t l = 0; l < 32; ++l) {
                append(l, 0, 64);
                append(l, 64, 64);
                check(policy.stats().layers[l].completed_tokens == 128,
                      "per-layer prefill completion");
                check(records(l, 0, 1) != records((l + 1) % 32, 0, 1),
                      "different layers must not share identical fake payload");
            }
            std::uint64_t cold_callbacks = 0, verified_bytes = 0;
            auto fetcher = [&](int layer, const std::vector<std::uint64_t>& ids) {
                check(!ids.empty() && ids.size() <= 64, "bounded cold fetch batch");
                for (auto id : ids) {
                    check(!policy.is_hot(id), "hot KV must not use cold storage callback");
                    check(id < backing.completed(layer), "callback must never see future IDs");
                }
                ++cold_callbacks;
                return backing.fetch(layer, ids, 64 * 4096);
            };
            auto verify_fetch = [&](std::size_t layer, const std::vector<std::uint64_t>& ids) {
                auto got = policy.fetch(int(layer), ids, fetcher);
                check(got.size() == ids.size() * 4096, "fetch exact payload size");
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    const auto expected = records(layer, ids[i], 1);
                    check(std::equal(expected.begin(), expected.end(), got.begin() + i * 4096),
                          "hot/cold fetch preserves layer, position, original bytes and caller order");
                }
                verified_bytes += got.size();
            };
            for (std::size_t step = 0; step < 3; ++step) {
                if (step == 0) {
                    // Two distinct layer histories may be outstanding. They
                    // share the bounded history engine, while each layer's
                    // current token is completed independently after append.
                    policy.prepare_history(0, 128, query(0, step));
                    rejects([&] { policy.prepare_history(0, 128, query(0, step)); },
                            "same-layer duplicate history preparation rejected");
                    policy.prepare_history(1, 128, query(1, step));
                    rejects([&] { policy.prepare_history(2, 128, query(2, step)); },
                            "third outstanding history preparation rejected");
                    check(policy.managed_bytes() <= cfg.total_host_workspace_budget_bytes,
                          "two outstanding histories remain within admitted host reservation");
                }
                for (std::size_t l = 0; l < 32; ++l) {
                    const auto position = 128 + step; auto q = query(l, step);
                    // Legal earlier-Q contract only: history exists before the
                    // current KV; this test does not claim model/GPU overlap.
                    if (step != 0 || l >= 2) policy.prepare_history(int(l), position, q);
                    append(l, position, 1);
                    auto ids = policy.select(int(l), position, q);
                    check(ids == expected_ids(l, position + 1, q),
                          "merged hot/cold and incremental-tail scores preserve ordered selected IDs");
                    for (const auto& h : ids) for (auto id : h)
                        check(id <= position, "selector returned future or padded position");
                    verify_fetch(l, physical_union(ids));
                    verify_fetch(l, {position, 0, 127, 63, 64});
                    rejects([&] { policy.fetch(int(l), {0, 0}, fetcher); },
                            "physical union contract rejects duplicate fetch IDs");
                    check(policy.hot_prompt_tokens() == 64 && policy.cold_prompt_tokens() == 64,
                          "offline beta cannot change across layers or steps");
                    rejects([&] { policy.fetch(int(l), {position + 1}, fetcher); },
                            "future pre-reserved hot slot cannot be fetched");
                    rejects([&] { policy.select(int(l), position + 1, q); },
                            "selection requires current layer append completion");
                }
            }
            check(cold_callbacks > 0, "test must actually use ordinary-file cold fetches");
            for (std::size_t l = 0; l < 32; ++l) {
                const auto& st = policy.stats().layers[l];
                check(st.completed_tokens == 131 && st.score_calls == 3 && st.fpga_calls == 0,
                      "32 layers,3 steps,CPU path only");
                check(st.hot_fetch_bytes > 0 && st.cold_fetch_bytes > 0,
                      "each layer exercises both hot and cold KV");
                check(st.verification_scores > 0, "independent score check enabled per layer");
                check(st.hot_cpu_calls==3&&st.cold_cpu_calls==3,"both hot/cold CPU scorers executed every step");
                check(legacy?st.hot_cpu_packed_key_bytes==(64+65+66)*2048ULL:st.hot_cpu_packed_key_bytes==0,
                      "legacy packs history hot K once per score call; automatic stays on original views");
                check(st.history_prefetch_calls == 3 && st.current_token_score_calls == 3,
                      "every prepared history has an independently scored current-token completion");
            }
            policy.drain(); backing.drain();
            check(policy.stats().drain_calls == 1 && backing.metrics.drain_calls == 1,
                  "explicit end-of-request drain only");
            std::cout << "{\"status\":\"passed\",\"evidence\":\"CPU_software_regular_file_contract_not_model_or_FPGA\","
                      << "\"layers\":32,\"prompt_tokens\":128,\"decode_steps\":3,"
                      << "\"frozen_beta\":1,\"cold_prompt_tokens\":64,\"hot_prompt_tokens\":64,"
                      << "\"verified_KV_bytes\":" << verified_bytes
                      << ",\"cold_fetch_callbacks\":" << cold_callbacks
                      << ",\"hot_KV_all_layers_bytes\":" << policy.stats().hot_kv_cache_bytes
                      << ",\"managed_reserved_bytes\":" << policy.managed_bytes()
                      << ",\"managed_budget_bytes\":" << cfg.total_host_workspace_budget_bytes
                      << ",\"CPU_scorer\":\"" << policy.stats().cpu_implementation_actual << "\""
                      << ",\"legacy_scorer_bitwise_comparisons\":" << legacy_compared
                      << ",\"hot_score_threads\":" << cfg.hot_threads
                      << ",\"cold_score_threads\":" << cfg.cold_threads
                      << ",\"two_outstanding_histories_verified\":true"
                      << ",\"duplicate_and_third_history_rejected\":true"
                      << ",\"GPU_executed\":false,\"FPGA_executed\":false}\n";
        }
        std::filesystem::remove_all(dir);
        return 0;
    } catch (const std::exception& e) {
        if (!dir.empty()) std::filesystem::remove_all(dir);
        std::cerr << "full GQA runtime contract failed: " << e.what() << '\n';
        return 1;
    }
}
