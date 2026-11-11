#include "runtime/prompt_recompute.hpp"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <future>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace hillserve::runtime {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t layers = 32, kv_heads = 8, dimension = 128, row_bytes = 512;
void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::size_t add(std::size_t a, std::size_t b) {
    need(b <= std::numeric_limits<std::size_t>::max() - a, "prompt recompute size addition overflow");
    return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
    need(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a, "prompt recompute size multiplication overflow");
    return a * b;
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct ThreadConfiguration {
    ThreadConfiguration() {
        need(!llama_hillserve_configuration().enabled, "recompute worker inherited an active HillServe context");
    }
    ~ThreadConfiguration() { llama_hillserve_configure(nullptr); }
};
struct Batch {
    llama_batch value;
    explicit Batch(int size) : value(llama_batch_init(size, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
};
struct PhaseTime {
    double& milliseconds;
    Clock::time_point started = Clock::now();
    ~PhaseTime() { milliseconds += elapsed(started); }
};
struct ContextDeleter {
    PromptRecompute::Call* event;
    void operator()(llama_context* context) const {
        const auto started = Clock::now();
        llama_free(context);
        event->context_destroy_ms += elapsed(started);
    }
};
struct EventFile {
    int fd = -1;
    ~EventFile() { if (fd >= 0) (void)::close(fd); }
    void create(const std::string& path) {
        if (path.empty()) return;
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error(std::string("cannot exclusively create recompute event log: ") + std::strerror(errno));
    }
    void append(const std::string& row) {
        if (fd < 0) return;
        std::size_t done = 0;
        while (done < row.size()) {
            const auto bytes = ::write(fd, row.data() + done, row.size() - done);
            if (bytes < 0 && errno == EINTR) continue;
            if (bytes <= 0) throw std::runtime_error(std::string("cannot flush recompute event log: ") + (bytes < 0 ? std::strerror(errno) : "zero-byte write"));
            done += std::size_t(bytes);
        }
        // No userspace buffering: all bytes have reached the file descriptor.
        // Deliberately not fsync: these are receipts, not durable KV writes.
    }
};
void json_error(std::ostream& out, const char* error) {
    if (!error) { out << "null"; return; }
    out << '"';
    constexpr char hex[] = "0123456789abcdef";
    // Bound exception text, including non-ASCII bytes, to keep both the journal
    // allocation and JSON syntax valid even for an arbitrary backend error.
    for (std::size_t i = 0; i < 1024 && error[i]; ++i) {
        const auto c = static_cast<unsigned char>(error[i]);
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32 || c >= 127) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << char(c);
    }
    out << '"';
}
using HeadPositions = std::array<const std::vector<std::uint64_t>*, kv_heads>;
using HeadOutputs = std::vector<std::vector<std::uint8_t>>;
using LayerOutputs = std::vector<HeadOutputs>;
struct Requested { std::uint64_t position; std::size_t head, destination; };
// Contains only recovery payload for this generation step. A move to the
// caller does not duplicate bytes; the caller's output is covered by the same
// group bound until its fetch finishes. No next group overlaps that lifetime.
struct DecodeCache {
    bool active = false;
    std::uint64_t position = 0, last_position = 0;
    bool have_previous = false;
    std::size_t next_layer = 0, first_layer = 0, live_bytes = 0, peak_bytes = 0;
    std::vector<std::vector<std::uint64_t>> expected;
    LayerOutputs payload;
    void begin(std::uint64_t p) {
        need(!active && live_bytes == 0 && payload.empty(), "recovery cache crossed a decode boundary");
        need(!have_previous || p == last_position + 1, "recovery decode positions must be consecutive");
        active = true; position = p; next_layer = 0; expected.clear();
    }
    void request(int layer, const std::vector<std::vector<std::uint64_t>>& ids,
                 std::size_t prompt_tokens, std::size_t max_rows) {
        need(active && layer >= 0 && std::size_t(layer) == next_layer && next_layer < layers,
             "grouped recovery layers must be consumed once in order within a decode");
        need(ids.size() == kv_heads && !ids[0].empty() && ids[0].size() <= max_rows,
             "grouped recovery requires bounded nonempty eight-head rows");
        for (const auto& head : ids) {
            need(head == ids[0], "grouped full-attention recovery head sets differ");
            for (auto p : head) need(p < prompt_tokens, "grouped recovery requested generated/future KV");
        }
        if (expected.empty()) expected = ids;
        else need(expected == ids, "grouped recovery missing positions changed between layers");
    }
    void install(int first, LayerOutputs bytes, std::size_t budget) {
        need(active && std::size_t(first) == next_layer && payload.empty() && live_bytes == 0,
             "recovery group overlaps unconsumed payload");
        need(!bytes.empty() && bytes.size() <= layers - next_layer, "invalid recovery group range");
        std::size_t total = 0;
        for (const auto& layer : bytes) {
            need(layer.size() == kv_heads, "recovery group head count");
            for (std::size_t h = 0; h < kv_heads; ++h) {
                need(layer[h].size() == expected[h].size() * row_bytes, "recovery group row count");
                total = add(total, layer[h].size());
            }
        }
        need(total <= budget, "recovery group exceeds explicit payload reservation");
        first_layer = first; payload = std::move(bytes); live_bytes = total;
        peak_bytes = std::max(peak_bytes, total);
    }
    HeadOutputs take(int layer) {
        need(active && std::size_t(layer) == next_layer && next_layer >= first_layer &&
             next_layer - first_layer < payload.size(), "no matching recovered layer");
        auto result = std::move(payload[next_layer - first_layer]);
        for (const auto& head : result) live_bytes -= head.size();
        ++next_layer;
        if (next_layer == first_layer + payload.size()) {
            need(live_bytes == 0, "recovery payload byte accounting mismatch"); payload.clear();
        }
        return result;
    }
    void end() {
        need(active && next_layer == layers && payload.empty() && live_bytes == 0,
             "decode ended before every recovery layer was consumed");
        active = false; expected.clear(); last_position = position; have_previous = true;
    }
};
struct Capture {
    int target_layer;
    std::size_t first = 0, count = 0, cursor = 0;
    std::size_t target_count = 1, active_target = 0, rows_per_layer = 0;
    std::array<std::size_t, layers> cursors{};
    std::vector<Requested> requested;
    std::vector<std::uint8_t> captured;
    std::vector<std::uint16_t> current;
    HeadOutputs& output;
    LayerOutputs* group;
    PromptRecompute::Call& observation;
    std::array<std::uint64_t, layers> append_calls{}, attention_calls{};
    std::exception_ptr failure;

    Capture(int layer, const HeadPositions& positions,
            std::size_t chunk, HeadOutputs& bytes, PromptRecompute::Call& event,
            LayerOutputs* grouped = nullptr)
        : target_layer(layer), captured(event.requested_rows, 0),
          current(chunk * kv_heads * dimension * 2), output(bytes), group(grouped), observation(event) {
        target_count = grouped ? grouped->size() : 1;
        need(target_count > 0 && target_count <= layers - std::size_t(layer) &&
             event.requested_rows % target_count == 0, "recompute target-layer group geometry");
        rows_per_layer = event.requested_rows / target_count;
        requested.resize(rows_per_layer);
        std::size_t next = 0;
        for (std::size_t head = 0; head < kv_heads; ++head) {
            if (!positions[head]) continue;
            for (std::size_t i = 0; i < positions[head]->size(); ++i)
                requested[next++] = {(*positions[head])[i], head, i};
        }
        need(next == requested.size(), "recompute head-row request accounting mismatch");
        std::sort(requested.begin(), requested.end(), [](const auto& a, const auto& b) {
            if (a.position != b.position) return a.position < b.position;
            if (a.head != b.head) return a.head < b.head;
            return a.destination < b.destination;
        });
    }
    void gather_current_rows() {
        while (cursor < requested.size() && requested[cursor].position < first + count) {
            const auto index = cursor++;
            const auto request = requested[index];
            need(request.position >= first, "recompute capture skipped a requested position");
            const auto relative = std::size_t(request.position - first);
            const auto* row = current.data() + (relative * kv_heads + request.head) * dimension * 2;
            for (std::size_t i = 0; i < dimension * 2; ++i)
                need((row[i] & 0x7c00) != 0x7c00, "recomputed model KV contains nonfinite F16");
            const auto flag = active_target * rows_per_layer + index;
            need(!captured[flag], "recompute captured a requested output twice");
            auto& target = group ? (*group)[active_target] : output;
            std::memcpy(target[request.head].data() + request.destination * row_bytes, row, row_bytes);
            captured[flag] = 1;
            ++observation.captured_rows;
        }
        cursors[active_target] = cursor;
    }
    static bool callback(ggml_tensor* tensor, bool ask, void* opaque) {
        auto& self = *static_cast<Capture*>(opaque);
        const bool append = std::strncmp(tensor->name, "hs_append_", 10) == 0;
        const bool attention = std::strncmp(tensor->name, "kqv_out-", 8) == 0;
        if (ask) return append || attention;
        if (self.failure) return false;
        try {
            int layer = -1;
            if (attention) {
                need(std::sscanf(tensor->name, "kqv_out-%d", &layer) == 1 && layer >= 0 && layer < int(layers),
                     "recompute attention layer identity mismatch");
                auto* buffer = tensor->buffer ? tensor->buffer : tensor->view_src ? tensor->view_src->buffer : nullptr;
                need(buffer != nullptr, "recompute attention has no actual backend buffer");
                auto* device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
                need(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU,
                     "recompute attention did not execute on GPU");
                ++self.attention_calls[layer]; ++self.observation.attention_steps;
            } else if (append) {
                unsigned count = 0;
                need(std::sscanf(tensor->name, "hs_append_%d_%u", &layer, &count) == 2 &&
                     layer >= 0 && layer < int(layers), "recompute KV export layer identity mismatch");
                need(tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor) &&
                     tensor->ne[0] == 2 * int64_t(dimension) && tensor->ne[1] == int64_t(kv_heads) &&
                     tensor->ne[2] == int64_t(self.count) && tensor->ne[3] == 1,
                     "recompute live current KV tensor ABI mismatch");
                ++self.append_calls[layer]; ++self.observation.layer_steps;
                if (layer < self.target_layer || std::size_t(layer - self.target_layer) >= self.target_count) return true;
                self.active_target = std::size_t(layer - self.target_layer);
                self.cursor = self.cursors[self.active_target];
                if (self.cursor < self.requested.size() &&
                    self.requested[self.cursor].position < self.first + self.count) {
                    need(self.requested[self.cursor].position >= self.first, "recompute capture skipped a requested position");
                    const auto bytes = ggml_nbytes(tensor);
                    need(bytes <= self.current.size() * sizeof(std::uint16_t), "recompute capture staging exceeded");
                    ggml_backend_tensor_get(tensor, self.current.data(), 0, bytes);
                    self.observation.capture_D2H_bytes += bytes;
                    self.gather_current_rows();
                }
            }
            return true;
        } catch (...) {
            self.failure = std::current_exception(); return false;
        }
    }
};
} // namespace

struct PromptRecompute::Impl {
    llama_model* model;
    Config config;
    std::vector<llama_token> prompt;
    std::vector<Call> events;
    Stats statistics;
    std::size_t context_tokens = 0, max_rows = 0;
    DecodeCache decode_cache;
    std::uint64_t step_replays_before = 0;
    std::mutex serial;
    EventFile event_file;

    Impl(llama_model* m, const std::vector<llama_token>& tokens, const Config& c)
        : model(m), config(c), prompt(tokens) {
        need(model && !prompt.empty() && prompt.size() <= 65536, "recompute needs a live model and bounded original prompt1..65536");
        need(c.cpu_threads >= 1 && c.cpu_threads <= 32 && c.chunk_tokens >= 1 && c.chunk_tokens <= 256,
             "recompute thread/chunk geometry unsupported");
        need(c.max_calls > 0 && c.max_calls <= 65536, "recompute max_calls must be1..65536");
        need(c.max_batch_heads >= 1 && c.max_batch_heads <= kv_heads, "recompute batch head bound must be1..8");
        need(c.max_target_layers >= 1 && c.max_target_layers <= layers &&
             (c.max_target_layers == 1 || c.max_batch_heads == kv_heads),
             "layer grouping requires1..32 layers and all eight KV heads");
        need(std::isfinite(c.rope_freq_base) && c.rope_freq_base > 0 &&
             std::isfinite(c.rope_freq_scale) && c.rope_freq_scale > 0, "invalid recompute RoPE configuration");
        need(llama_model_n_layer(model) == int(layers) && llama_model_n_head(model) == 32 &&
             llama_model_n_head_kv(model) == int(kv_heads) && llama_model_n_embd(model) == 4096,
             "recompute requires verified Gradient8B GQA geometry");
        need(prompt.size() <= std::size_t(llama_model_n_ctx_train(model)), "recompute exceeds actual model context");
        const auto vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (auto token : prompt) need(token >= 0 && token < vocab_size, "recompute prompt token outside model vocabulary");
        max_rows = c.max_requested_rows ? c.max_requested_rows : prompt.size();
        need(max_rows > 0 && max_rows <= prompt.size(), "recompute output row bound exceeds original prompt");
        context_tokens = (prompt.size() + 255) / 256 * 256;
        statistics.native_GPU_planned_bytes = mul(mul(context_tokens, layers), kv_heads * dimension * 4);
        need(add(statistics.native_GPU_planned_bytes, c.gpu_graph_budget_bytes) <= c.gpu_budget_bytes,
             "recompute native KV plus graph allowance exceeds incremental GPU budget");
        events.reserve(c.max_calls);
        std::size_t host = sizeof(Impl) + sizeof(PromptRecompute) + sizeof(Capture) + sizeof(Batch);
        host = add(host, mul(prompt.capacity(), sizeof(llama_token)));
        host = add(host, mul(events.capacity(), sizeof(Call)));
        host = add(host, mul(mul(max_rows, c.max_batch_heads), row_bytes + sizeof(Requested) + sizeof(std::uint8_t)));
        host = add(host, sizeof(HeadPositions) + sizeof(HeadOutputs) + kv_heads * sizeof(std::vector<std::uint8_t>));
        host = add(host, mul(c.chunk_tokens, kv_heads * dimension * 4));
        host = add(host, mul(c.chunk_tokens, sizeof(llama_token) + sizeof(llama_pos) +
                            sizeof(int32_t) + sizeof(llama_seq_id*) + sizeof(llama_seq_id) + sizeof(int8_t)));
        host = add(host, 4 * ggml_tensor_overhead() + 4096); // private dummy-mask metadata and runtime bookkeeping
        host = add(host, c.host_context_reserve_bytes);
        host = add(host, add(config.event_log_path.capacity() + 1, 16 * 1024)); // bounded JSON receipt scratch
        if (c.max_target_layers > 1) {
            // Share one request list and transfer slab across the target layers.
            // Only output payload and initialization flags scale with the group.
            host = add(host, mul(mul(max_rows, kv_heads), sizeof(std::uint64_t)));
            host = add(host, kv_heads * sizeof(std::vector<std::uint64_t>) +
                             c.max_target_layers * sizeof(HeadOutputs));
            need(host <= c.host_budget_bytes, "grouped recovery metadata exceeds host reserve");
            const auto extra_layer = add(mul(mul(max_rows, kv_heads), row_bytes + 1),
                                         kv_heads * sizeof(std::vector<std::uint8_t>));
            statistics.target_layers_per_replay = std::min(c.max_target_layers,
                1 + (c.host_budget_bytes - host) / extra_layer);
            host = add(host, mul(statistics.target_layers_per_replay - 1, extra_layer));
            statistics.recovery_cache_reserved_bytes = mul(
                mul(max_rows, kv_heads * row_bytes), statistics.target_layers_per_replay);
        }
        need(host <= c.host_budget_bytes, "recompute prompt/output/capture/context metadata exceeds host reserve");
        statistics.host_managed_reserved_bytes = host;
        event_file.create(c.event_log_path);
    }

    void execute(int layer, const HeadPositions& positions, HeadOutputs& result, Call& event,
                 LayerOutputs* grouped = nullptr) {
        ThreadConfiguration thread_configuration;
        auto* device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        need(device, "recompute requires the existing real GPU backend");
        std::size_t free_before = 0, total = 0;
        ggml_backend_dev_memory(device, &free_before, &total);
        event.GPU_free_before_bytes = free_before;
        need(free_before <= total, "recompute GPU memory query returned free greater than total");
        ++event.GPU_memory_samples;
        event.global_GPU_used_peak_bytes = total - free_before;
        need(!config.total_gpu_budget_bytes || event.global_GPU_used_peak_bytes <= config.total_gpu_budget_bytes,
             "recompute device-wide GPU use already exceeds total budget");
        need(free_before >= config.gpu_free_reserve_bytes &&
             config.gpu_budget_bytes <= free_before - config.gpu_free_reserve_bytes,
             "recompute GPU admission cannot preserve main context/free reserve");

        using Meta = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
        Meta metadata(ggml_init({4 * ggml_tensor_overhead() + 4096, nullptr, true}), ggml_free);
        need(bool(metadata), "recompute dummy mask metadata allocation failed");
        auto* mask = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_F32, 1);
        need(mask, "recompute dummy native mask tensor failed");
        llama_hillserve_config hs{};
        hs.enabled = true; hs.external = false; hs.native_prefill = true;
        hs.slots = static_cast<std::uint32_t>(context_tokens);
        hs.max_tokens = static_cast<std::uint32_t>(config.chunk_tokens); hs.mask = mask;
        // Native attention uses its own causal mask. hs.mask is only the graph
        // contract sentinel and is never read as external attention data.
        llama_hillserve_configure(&hs);
        Capture capture(layer, positions, config.chunk_tokens, result, event, grouped);
        auto cp = llama_context_default_params();
        cp.n_ctx = static_cast<std::uint32_t>(context_tokens);
        cp.n_batch = cp.n_ubatch = static_cast<std::uint32_t>(config.chunk_tokens); cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = static_cast<int>(config.cpu_threads);
        cp.type_k = cp.type_v = GGML_TYPE_F16; cp.flash_attn = false;
        cp.offload_kqv = true; cp.op_offload = true;
        cp.rope_freq_base = config.rope_freq_base; cp.rope_freq_scale = config.rope_freq_scale;
        cp.cb_eval = Capture::callback; cp.cb_eval_user_data = &capture;
        using Context = std::unique_ptr<llama_context, ContextDeleter>;
        Context ctx(nullptr, ContextDeleter{&event});
        { PhaseTime init{event.context_init_ms}; ctx.reset(llama_init_from_model(model, cp)); }
        need(bool(ctx), "private recompute context allocation failed");
        auto check_memory = [&] {
            const auto native = llama_hillserve_native_gpu_kv_bytes(ctx.get());
            const auto graph_gpu = llama_hillserve_graph_bytes(ctx.get(), true);
            const auto graph_host = llama_hillserve_graph_bytes(ctx.get(), false);
            event.native_GPU_peak_bytes = std::max(event.native_GPU_peak_bytes, native);
            event.graph_GPU_peak_bytes = std::max(event.graph_GPU_peak_bytes, graph_gpu);
            event.graph_host_peak_bytes = std::max(event.graph_host_peak_bytes, graph_host);
            std::size_t free_now = 0, current_total = 0;
            ggml_backend_dev_memory(device, &free_now, &current_total);
            need(free_now <= current_total, "recompute GPU memory query returned free greater than total");
            ++event.GPU_memory_samples;
            event.global_GPU_used_peak_bytes = std::max(event.global_GPU_used_peak_bytes, current_total - free_now);
            const auto increment = free_before > free_now ? free_before - free_now : 0;
            event.observed_GPU_increment_peak_bytes = std::max(event.observed_GPU_increment_peak_bytes, increment);
            need(native == std::size_t(llama_n_ctx(ctx.get())) * layers * kv_heads * dimension * 4 &&
                 llama_hillserve_native_kv_bytes(ctx.get()) == native,
                 "recompute KV must be actual all-layer F16 GPU history");
            need(native <= statistics.native_GPU_planned_bytes && graph_gpu <= config.gpu_graph_budget_bytes &&
                 add(native, graph_gpu) <= config.gpu_budget_bytes, "recompute actual managed GPU budget exceeded");
            need(graph_host <= config.host_context_reserve_bytes, "recompute CPU graph exceeds context host reserve");
            need(!config.total_gpu_budget_bytes || event.global_GPU_used_peak_bytes <= config.total_gpu_budget_bytes,
                 "recompute device-wide GPU use exceeds total budget");
            need(free_now >= config.gpu_free_reserve_bytes && increment <= config.gpu_budget_bytes,
                 "recompute observed GPU allocation exceeds incremental/free reserve");
        };
        check_memory();
        Batch batch(static_cast<int>(config.chunk_tokens));
        {
        PhaseTime replay{event.replay_ms};
        for (std::size_t first = 0; first < prompt.size(); first += config.chunk_tokens) {
            const auto count = std::min(config.chunk_tokens, prompt.size() - first);
            capture.first = first; capture.count = count;
            batch.value.n_tokens = static_cast<int>(count);
            for (std::size_t i = 0; i < count; ++i) {
                batch.value.token[i] = prompt[first + i]; batch.value.pos[i] = static_cast<llama_pos>(first + i);
                batch.value.n_seq_id[i] = 1; batch.value.seq_id[i][0] = 0;
                batch.value.logits[i] = i + 1 == count;
            }
            const int status = llama_decode(ctx.get(), batch.value);
            llama_synchronize(ctx.get());
            check_memory();
            if (capture.failure) std::rethrow_exception(capture.failure);
            need(status == 0, "actual prompt recomputation llama_decode failed");
            ++event.decode_calls; event.replayed_tokens += count;
        }
        }
        need(event.replayed_tokens == prompt.size() &&
             event.captured_rows == event.requested_rows, "recompute did not produce every requested prompt KV row");
        for (std::size_t l = 0; l < capture.target_count; ++l)
            need(capture.cursors[l] == capture.rows_per_layer, "recompute left a target layer incomplete");
        for (auto present : capture.captured) need(present == 1, "recompute left a requested row uninitialized");
        for (std::size_t l = 0; l < layers; ++l)
            need(capture.append_calls[l] == event.decode_calls && capture.attention_calls[l] == event.decode_calls,
                 "recompute did not execute/export every actual model layer");
        ctx.reset();
        // All context-owned KV has been freed before the worker returns; only
        // the bounded requested output rows survive in the caller's workspace.
        if (grouped) {
            for (const auto& target : *grouped) for (const auto& head : target) event.returned_bytes += head.size();
        } else for (const auto& head : result) event.returned_bytes += head.size();
        event.passed = true;
    }

    void record(const Call& event, const char* error = nullptr) {
        // Preserve the bounded in-memory event array even if a caller repeatedly
        // violates max_calls. Every such failure still gets a journal receipt.
        if (events.size() < config.max_calls) events.push_back(event);
        ++statistics.calls;
        if (event.passed) ++statistics.successful_calls; else ++statistics.failed_calls;
        statistics.replayed_tokens += event.replayed_tokens;
        statistics.layer_steps += event.layer_steps; statistics.attention_steps += event.attention_steps;
        statistics.returned_rows += event.passed ? event.requested_rows : 0;
        statistics.returned_bytes += event.returned_bytes; statistics.capture_D2H_bytes += event.capture_D2H_bytes;
        statistics.native_GPU_peak_bytes = std::max(statistics.native_GPU_peak_bytes, event.native_GPU_peak_bytes);
        statistics.graph_GPU_peak_bytes = std::max(statistics.graph_GPU_peak_bytes, event.graph_GPU_peak_bytes);
        statistics.graph_host_peak_bytes = std::max(statistics.graph_host_peak_bytes, event.graph_host_peak_bytes);
        statistics.observed_GPU_increment_peak_bytes = std::max(statistics.observed_GPU_increment_peak_bytes, event.observed_GPU_increment_peak_bytes);
        statistics.global_GPU_used_peak_bytes = std::max(statistics.global_GPU_used_peak_bytes, event.global_GPU_used_peak_bytes);
        statistics.wall_ms += event.wall_ms;
        if (event_file.fd < 0) return;
        std::ostringstream row;
        row.imbue(std::locale::classic());
        row << std::setprecision(17)
            << "{\"schema\":\"hillserve.prompt_recompute.v3\",\"implementation\":\""
            << (config.max_target_layers > 1 ? "full_prompt_replay_layer_groups" : "full_prompt_replay_batched_heads") << "\""
            << ",\"call_index\":" << statistics.calls << ",\"layer\":" << event.layer << ",\"kv_head\":" << event.kv_head
            << ",\"target_layers\":" << event.target_layers << ",\"decode_position\":" << event.decode_position
            << ",\"requested_heads\":" << event.requested_heads
            << ",\"replay_scope\":\"all_32_layers_one_replay_per_target_group\""
            << ",\"capture_scope\":\"requested_layer_range_only\""
            << ",\"passed\":" << (event.passed ? "true" : "false")
            << ",\"requested_rows\":" << event.requested_rows << ",\"returned_bytes\":" << event.returned_bytes
            << ",\"replayed_tokens\":" << event.replayed_tokens << ",\"decode_calls\":" << event.decode_calls
            << ",\"layer_steps\":" << event.layer_steps << ",\"attention_steps\":" << event.attention_steps
            << ",\"captured_rows\":" << event.captured_rows << ",\"capture_D2H_bytes\":" << event.capture_D2H_bytes
            << ",\"native_GPU_peak_bytes\":" << event.native_GPU_peak_bytes << ",\"graph_GPU_peak_bytes\":" << event.graph_GPU_peak_bytes
            << ",\"graph_host_peak_bytes\":" << event.graph_host_peak_bytes
            << ",\"observed_GPU_increment_peak_bytes\":" << event.observed_GPU_increment_peak_bytes
            << ",\"GPU_free_before_bytes\":" << event.GPU_free_before_bytes
            << ",\"global_GPU_used_peak_bytes\":" << event.global_GPU_used_peak_bytes
            << ",\"GPU_memory_samples\":" << event.GPU_memory_samples
            << ",\"global_GPU_peak_scope\":\"device_total_minus_free_sampled_before_context_after_context_and_each_decode\""
            << ",\"incremental_GPU_budget_bytes\":" << config.gpu_budget_bytes
            << ",\"host_managed_reserved_bytes\":" << statistics.host_managed_reserved_bytes
            << ",\"max_batch_heads\":" << config.max_batch_heads
            << ",\"total_GPU_budget_bytes\":" << config.total_gpu_budget_bytes
            << ",\"context_init_ms\":" << event.context_init_ms << ",\"replay_ms\":" << event.replay_ms
            << ",\"context_destroy_ms\":" << event.context_destroy_ms << ",\"wall_ms\":" << event.wall_ms
            << ",\"error\":";
        json_error(row, error);
        row << ",\"error_truncated\":" << (error && ::strnlen(error, 1025) > 1024 ? "true" : "false") << "}\n";
        event_file.append(row.str());
    }

    HeadOutputs grouped_run(int layer, const std::vector<std::vector<std::uint64_t>>& positions) {
        std::lock_guard<std::mutex> lock(serial);
        decode_cache.request(layer, positions, prompt.size(), max_rows);
        const bool reused = !decode_cache.payload.empty();
        if (!reused) {
            Call event; event.layer = layer; event.kv_head = -1;
            event.decode_position = decode_cache.position;
            event.target_layers = std::min(statistics.target_layers_per_replay, layers - std::size_t(layer));
            event.requested_heads = kv_heads;
            event.requested_rows = mul(mul(positions[0].size(), kv_heads), event.target_layers);
            const auto started = Clock::now();
            try {
                need(statistics.calls < config.max_calls, "grouped recovery exceeded replay call budget");
                LayerOutputs output(event.target_layers, HeadOutputs(kv_heads));
                for (auto& target : output) for (std::size_t h = 0; h < kv_heads; ++h)
                    target[h].resize(mul(positions[h].size(), row_bytes));
                HeadPositions requests{};
                for (std::size_t h = 0; h < kv_heads; ++h) requests[h] = &positions[h];
                std::packaged_task<void()> work([&] { execute(layer, requests, output[0], event, &output); });
                auto completion = work.get_future();
                std::thread worker(std::move(work)); worker.join(); completion.get();
                decode_cache.install(layer, std::move(output), statistics.recovery_cache_reserved_bytes);
            } catch (const std::exception& error) {
                event.passed = false; event.wall_ms = elapsed(started); record(event, error.what()); throw;
            } catch (...) {
                event.passed = false; event.wall_ms = elapsed(started); record(event, "grouped recovery exception"); throw;
            }
            event.wall_ms = elapsed(started); record(event);
        }
        auto output = decode_cache.take(layer);
        ++statistics.layer_requests;
        statistics.reused_layer_requests += reused;
        statistics.recovery_cache_live_bytes = decode_cache.live_bytes;
        statistics.recovery_cache_peak_bytes = decode_cache.peak_bytes;
        return output;
    }

    void begin_decode(std::uint64_t position) {
        if (config.max_target_layers == 1) return;
        std::lock_guard<std::mutex> lock(serial);
        need(position >= prompt.size(), "recovery decode must follow the original prompt");
        decode_cache.begin(position); step_replays_before = statistics.calls;
    }
    void end_decode() {
        if (config.max_target_layers == 1) return;
        std::lock_guard<std::mutex> lock(serial);
        decode_cache.end(); ++statistics.completed_decode_steps;
        need(statistics.recovery_cache_live_bytes == 0, "recovery payload retained across decode steps");
        std::ostringstream row;
        row << "{\"schema\":\"hillserve.prompt_recompute.decode_end.v1\",\"decode_position\":"
            << decode_cache.position << ",\"consumed_layers\":32,\"model_replays\":"
            << statistics.calls - step_replays_before << ",\"recovery_cache_live_bytes\":0,\"passed\":true}\n";
        event_file.append(row.str());
    }

    HeadOutputs run(int layer, int head, const HeadPositions& positions,
                    bool batch, std::size_t supplied_heads) {
        std::lock_guard<std::mutex> lock(serial);
        Call event; event.layer = layer; event.kv_head = head;
        HeadOutputs result;
        const auto started = Clock::now();
        try {
            need(layer >= 0 && layer < int(layers), "recompute target layer out of range");
            need(batch ? head == -1 : head >= 0 && head < int(kv_heads), "recompute target KV head out of range");
            need(supplied_heads == kv_heads, "batch recompute requires exactly eight per-head request vectors");
            for (const auto* requested : positions) {
                if (!requested) continue;
                event.requested_rows = add(event.requested_rows, requested->size());
                event.requested_heads += !requested->empty();
                need(requested->size() <= max_rows, "recompute requested output exceeds reserved per-head row bound");
                for (auto position : *requested)
                    need(position < prompt.size(), "recompute cannot recover generated/future KV; it must remain resident");
            }
            need(event.requested_heads <= config.max_batch_heads, "recompute active heads exceed explicit reserved batch bound");
            need(statistics.calls < config.max_calls, "recompute exceeded explicit bounded call/event budget");
            result.resize(kv_heads);
            if (!event.requested_rows) event.passed = true;
            else {
                for (std::size_t h = 0; h < kv_heads; ++h)
                    if (positions[h]) result[h].resize(mul(positions[h]->size(), row_bytes));
                std::packaged_task<void()> work([&] { execute(layer, positions, result, event); });
                auto completion = work.get_future();
                std::thread worker(std::move(work)); worker.join();
                completion.get(); // Exceptions propagate only after private context cleanup.
            }
        } catch (const std::exception& error) {
            event.wall_ms = elapsed(started); record(event, error.what()); throw;
        } catch (...) {
            event.wall_ms = elapsed(started); record(event, "non-standard recomputation exception"); throw;
        }
        event.wall_ms = elapsed(started); record(event);
        return result;
    }
};

PromptRecompute::PromptRecompute(llama_model* model, const std::vector<llama_token>& prompt, const Config& config)
    : impl_(std::make_unique<Impl>(model, prompt, config)) {}
PromptRecompute::~PromptRecompute() = default;
std::vector<std::uint8_t> PromptRecompute::call(int layer, int head,
    const std::vector<std::uint64_t>& positions) {
    HeadPositions requests{};
    if (head >= 0 && head < int(kv_heads)) requests[std::size_t(head)] = &positions;
    auto result = impl_->run(layer, head, requests, false, kv_heads);
    return std::move(result[std::size_t(head)]);
}
std::vector<std::vector<std::uint8_t>> PromptRecompute::batch_call(int layer,
    const std::vector<std::vector<std::uint64_t>>& positions_by_head) {
    if (impl_->config.max_target_layers > 1) return impl_->grouped_run(layer, positions_by_head);
    HeadPositions requests{};
    for (std::size_t head = 0; head < std::min(kv_heads, positions_by_head.size()); ++head)
        requests[head] = &positions_by_head[head];
    return impl_->run(layer, -1, requests, true, positions_by_head.size());
}
const PromptRecompute::Stats& PromptRecompute::stats() const { return impl_->statistics; }
void PromptRecompute::begin_decode(std::uint64_t position) { impl_->begin_decode(position); }
void PromptRecompute::end_decode() { impl_->end_decode(); }
const std::vector<PromptRecompute::Call>& PromptRecompute::events() const { return impl_->events; }

} // namespace hillserve::runtime
