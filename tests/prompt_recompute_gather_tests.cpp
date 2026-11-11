// Standalone CPU test of the actual internal row gather; deliberately includes
// the implementation so no test-only API is added to the runtime. Link unused
// worker functions away with -ffunction-sections -fdata-sections and
// -Wl,-dead_strip (macOS) or -Wl,--gc-sections (Linux). It does not link llama,
// execute a model, or constitute GPU/recomputation correctness evidence.
#include "runtime/prompt_recompute.cpp"
#include <iostream>

namespace {
using namespace hillserve::runtime;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
std::uint16_t fixture(std::size_t position, std::size_t head, std::size_t word) {
    // Arbitrary finite F16 bit patterns distinguish token/head/K/V/dimension.
    return static_cast<std::uint16_t>((position * 101 + head * 37 + word) % 0x7800);
}
void fill(Capture& capture, std::size_t first, std::size_t count) {
    capture.first = first; capture.count = count;
    for (std::size_t t = 0; t < count; ++t)
        for (std::size_t h = 0; h < kv_heads; ++h)
            for (std::size_t d = 0; d < dimension * 2; ++d)
                capture.current[(t * kv_heads + h) * dimension * 2 + d] = fixture(first + t, h, d);
}
void batch_order_and_duplicates() {
    std::vector<std::vector<std::uint64_t>> positions{
        {4,0,2,0}, {3,1}, {2}, {}, {4,3,0}, {1,4}, {0,1,2,3,4}, {4,4,1}};
    HeadPositions pointers{}; HeadOutputs outputs(kv_heads);
    PromptRecompute::Call event;
    for (std::size_t h = 0; h < kv_heads; ++h) {
        pointers[h] = &positions[h];
        outputs[h].resize(positions[h].size() * row_bytes);
        event.requested_rows += positions[h].size();
    }
    Capture capture(8, pointers, 2, outputs, event);
    for (std::size_t first = 0; first < 5; first += 2) {
        fill(capture, first, std::min<std::size_t>(2, 5 - first));
        capture.gather_current_rows();
    }
    check(capture.cursor == event.requested_rows && event.captured_rows == event.requested_rows,
          "batched gather did not produce every requested head-row");
    for (auto flag : capture.captured) check(flag == 1, "uninitialized batched output");
    for (std::size_t h = 0; h < kv_heads; ++h)
        for (std::size_t r = 0; r < positions[h].size(); ++r)
            for (std::size_t d = 0; d < dimension * 2; ++d) {
                std::uint16_t actual = 0;
                std::memcpy(&actual, outputs[h].data() + r * row_bytes + d * 2, 2);
                check(actual == fixture(positions[h][r], h, d), "head/order/KV/dimension mapping changed");
            }
    const auto count = event.captured_rows;
    capture.gather_current_rows();
    check(event.captured_rows == count, "completed batch copied twice");
    check(outputs[3].empty(), "empty head acquired output storage");
}
void single_head_and_failures() {
    std::vector<std::uint64_t> positions{0};
    HeadPositions pointers{}; pointers[7] = &positions;
    HeadOutputs outputs(kv_heads); outputs[7].resize(row_bytes);
    PromptRecompute::Call event; event.requested_rows = 1;
    Capture capture(31, pointers, 1, outputs, event);
    fill(capture, 0, 1); capture.gather_current_rows();
    check(event.captured_rows == 1, "single-head gather failed");
    for (std::size_t h = 0; h < 7; ++h) check(outputs[h].empty(), "single-head gather wrote another head");

    PromptRecompute::Call bad_event; bad_event.requested_rows = 1;
    Capture nonfinite(31, pointers, 1, outputs, bad_event);
    fill(nonfinite, 0, 1); nonfinite.current[7 * dimension * 2] = 0x7c00;
    rejects([&] { nonfinite.gather_current_rows(); }, "nonfinite recomputed KV accepted");
    check(bad_event.captured_rows == 0, "nonfinite row counted as returned");

    PromptRecompute::Call skipped_event; skipped_event.requested_rows = 1;
    Capture skipped(31, pointers, 1, outputs, skipped_event);
    fill(skipped, 1, 1);
    rejects([&] { skipped.gather_current_rows(); }, "skipped requested position accepted");

    HeadPositions empty{}; HeadOutputs no_output(kv_heads);
    PromptRecompute::Call empty_event;
    Capture nothing(0, empty, 1, no_output, empty_event);
    fill(nothing, 0, 1); nothing.gather_current_rows();
    check(empty_event.captured_rows == 0, "empty capture produced rows");
}
void grouped_capture() {
    std::vector<std::vector<std::uint64_t>> ids(kv_heads, {4,0,2,0});
    HeadPositions pointers{};
    for (std::size_t h=0;h<kv_heads;++h)pointers[h]=&ids[h];
    LayerOutputs output(3,HeadOutputs(kv_heads));
    for(auto& layer:output)for(auto& head:layer)head.resize(4*row_bytes);
    PromptRecompute::Call event;event.target_layers=3;event.requested_rows=3*kv_heads*4;
    Capture c(5,pointers,2,output[0],event,&output);
    for(std::size_t first=0;first<5;first+=2)for(std::size_t l=0;l<3;++l){
        fill(c,first,std::min<std::size_t>(2,5-first));
        for(auto& word:c.current)word=static_cast<std::uint16_t>((word+l*1024)%0x7800);
        c.active_target=l;c.cursor=c.cursors[l];c.gather_current_rows();
    }
    check(event.captured_rows==event.requested_rows,"grouped capture omitted head/layer rows");
    for(auto flag:c.captured)check(flag==1,"grouped capture left uninitialized row");
    for(std::size_t l=0;l<3;++l)for(std::size_t h=0;h<kv_heads;++h)
        for(std::size_t r=0;r<4;++r)for(std::size_t d=0;d<256;++d){
            std::uint16_t got;std::memcpy(&got,output[l][h].data()+r*row_bytes+d*2,2);
            check(got==(fixture(ids[h][r],h,d)+l*1024)%0x7800,"grouped capture mixed layer/head/K/V/order");
        }
}
void bounded_decode_cache() {
    DecodeCache c;
    std::vector<std::vector<std::uint64_t>> ids(kv_heads,{0,2});
    const std::size_t per_layer=kv_heads*2*row_bytes;
    rejects([&]{c.request(0,ids,5,2);},"cache request outside decode accepted");
    c.begin(5);
    rejects([&]{c.begin(6);},"cache crossed a generation boundary");
    rejects([&]{c.request(1,ids,5,2);},"out-of-order recovery accepted");
    rejects([&]{c.request(0,ids,5,1);},"oversized recovery accepted");
    auto bad=ids;bad[7]={0,3};
    rejects([&]{c.request(0,bad,5,2);},"different physical head sets accepted");
    c.request(0,ids,5,2);
    LayerOutputs too_big(2,HeadOutputs(kv_heads,std::vector<std::uint8_t>(2*row_bytes,7)));
    rejects([&]{c.install(0,std::move(too_big),2*per_layer-1);},"over-budget recovery cache accepted");
    check(c.live_bytes==0&&c.payload.empty(),"failed cache admission retained payload");
    for(int l=0;l<32;++l){
        c.request(l,ids,5,2);
        if(c.payload.empty()){
            const auto n=std::min(3,32-l);
            LayerOutputs bytes(n,HeadOutputs(kv_heads,std::vector<std::uint8_t>(2*row_bytes)));
            for(int j=0;j<n;++j)for(auto& head:bytes[j])std::fill(head.begin(),head.end(),std::uint8_t(l+j));
            c.install(l,std::move(bytes),3*per_layer);
        }
        auto bytes=c.take(l);
        for(const auto& head:bytes)for(auto b:head)check(b==l,"cached recovery layer mismatch");
        if(l<31)rejects([&]{c.end();},"incomplete decode cache consumption accepted");
        if(l==0){auto changed=ids;for(auto& h:changed)h={0,3};
            rejects([&]{c.request(1,changed,5,2);},"changed missing positions accepted");}
    }
    c.end();check(c.live_bytes==0&&c.payload.empty()&&!c.active,"cache retained KV after decode");
    rejects([&]{c.begin(5);},"stale decode epoch accepted");
    c.begin(6);check(c.expected.empty()&&c.payload.empty(),"previous step payload reused");
}
} // namespace

int main() {
    try {
        batch_order_and_duplicates(); single_head_and_failures(); grouped_capture(); bounded_decode_cache();
        std::cout << "passed: KV gather order/duplicates/nonfinite, multi-layer capture, budget rejection, layer/epoch ordering and zero payload across decode steps (CPU fixture only)\n";
    } catch (const std::exception& error) {
        std::cerr << "failed: " << error.what() << '\n'; return 1;
    }
}
