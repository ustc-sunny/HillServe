#include "storage/bounded_memory_kv.hpp"
#include <algorithm>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using hillserve::BoundedMemoryKVStore;
namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, message);
}
BoundedMemoryKVStore::Config config(std::size_t layers, std::uint64_t capacity,
                                   std::size_t bytes, std::size_t resident) {
    BoundedMemoryKVStore::Config c{layers, capacity, bytes, resident, 0};
    c.resident_budget_bytes = BoundedMemoryKVStore::estimate_bytes(c);
    return c;
}
std::vector<std::uint8_t> rows(std::size_t lane, std::size_t first,
                              std::size_t count, std::size_t bytes) {
    std::vector<std::uint8_t> out(count * bytes);
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < bytes; ++j)
            out[i * bytes + j] = static_cast<std::uint8_t>((lane * 131 + (first + i) * 17 + j * 3) % 251);
    return out;
}
void check_rows(const std::vector<std::uint8_t>& actual, std::size_t lane,
                const std::vector<std::uint64_t>& positions, std::size_t bytes) {
    check(actual.size() == positions.size() * bytes, "fetched size differs");
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto expected = rows(lane, positions[i], 1, bytes);
        check(std::equal(expected.begin(), expected.end(), actual.begin() + i * bytes), "stored KV bytes or requested order changed");
    }
}
}

int main() {
    try {
        auto c = config(2, 6, 16, 3);
        auto too_small = c; --too_small.resident_budget_bytes;
        rejects([&] { BoundedMemoryKVStore s(too_small); }, "resident allocation must reject one-byte budget deficit");
        BoundedMemoryKVStore store(c);
        const auto allocation = store.metrics();
        check(allocation.reserved_payload_bytes == 2 * 3 * 16, "payload slab is not exactly accounted");
        check(allocation.reserved_index_bytes == 2 * 6 * 8 + 2 * 3 * 16 + 2 * (8 + sizeof(std::size_t)), "future-position/reverse/free indexes not fully accounted");
        check(store.reserved_bytes() == allocation.reserved_payload_bytes + allocation.reserved_index_bytes + sizeof(store), "object or metadata missing from reservation");
        store.append(0, 0, rows(0, 0, 3, 16));
        store.append(1, 0, rows(1, 0, 2, 16));
        check_rows(store.fetch(0, {2, 0, 2}, 48), 0, {2, 0, 2}, 16);
        check_rows(store.fetch(1, {1, 0}, 32), 1, {1, 0}, 16);
        rejects([&] { store.fetch(0, {0, 1}, 31); }, "fetch must enforce caller staging budget");
        rejects([&] { store.append(0, 3, rows(0, 3, 1, 16)); }, "full resident store must not evict implicitly");
        check(store.completed(0) == 3 && store.resident_records(0) == 3, "rejected append changed progress");
        check(!store.contains(0, 3), "future row must not be resident");
        rejects([&] { store.fetch(0, {3}, 16); }, "future row cannot be returned");
        rejects([&] { store.evict(0, {0, 3}); }, "eviction containing future ID must reject atomically");
        check(store.contains(0, 0), "bad eviction removed a valid earlier ID");
        store.evict(0, {1, 1});
        check(store.metrics().records_evicted == 1 && store.resident_records(0) == 2, "duplicate eviction changed accounting");
        check(store.missing_positions(0, {2, 1, 0}) == std::vector<std::uint64_t>{1}, "missing lookup incorrect");
        rejects([&] { store.fetch(0, {1}, 16); }, "missing row must not return stale or zero bytes");
        store.restore(0, 1, rows(0, 1, 1, 16));
        check_rows(store.fetch(0, {1, 2, 0}, 48), 0, {1, 2, 0}, 16);
        rejects([&] { store.restore(0, 1, rows(0, 1, 1, 16)); }, "resident KV cannot silently be overwritten");
        store.evict(0, {1});
        store.append(0, 3, rows(0, 3, 1, 16));
        check_rows(store.fetch(0, {3, 0, 2}, 48), 0, {3, 0, 2}, 16);
        check(!store.contains(0, 1), "reused slot leaked an old position");
        rejects([&] { store.restore(0, 1, rows(0, 1, 1, 16)); }, "restore must not overrun resident slots");
        store.evict(0, {2}); store.restore(0, 1, rows(0, 1, 1, 16));
        check(store.completed(0) == 4, "restore changed sequential append position");
        check_rows(store.fetch(0, {1, 3}, 32), 0, {1, 3}, 16);
        check_rows(store.fetch(1, {0, 1}, 32), 1, {0, 1}, 16);

        // Real GQA lane geometry: all history plus generated reserve fits, so
        // nothing is evicted merely to manufacture a recomputation workload.
        auto full = config(32 * 8, 8, 512, 8);
        BoundedMemoryKVStore resident(full);
        for (std::size_t lane = 0; lane < full.layers; ++lane) {
            resident.append(lane, 0, rows(lane, 0, 6, 512));
            resident.append(lane, 6, rows(lane, 6, 2, 512));
            check(resident.missing_positions(lane, {0, 5, 6, 7}).empty(), "fitting history was evicted");
            check_rows(resident.fetch(lane, {7, 0, 6}, 3 * 512), lane, {7, 0, 6}, 512);
        }
        check(resident.metrics().records_evicted == 0 && resident.metrics().records_restored == 0, "all-fit workload manufactured eviction/restoration");

        auto concurrent_config = config(4, 32, 64, 32);
        BoundedMemoryKVStore concurrent(concurrent_config);
        std::vector<std::future<void>> workers;
        for (std::size_t lane = 0; lane < 4; ++lane) workers.push_back(std::async(std::launch::async, [&, lane] {
            for (std::size_t t = 0; t < 32; ++t) {
                concurrent.append(lane, t, rows(lane, t, 1, 64));
                check_rows(concurrent.fetch(lane, {t, 0}, 128), lane, {t, 0}, 64);
            }
        }));
        for (auto& worker : workers) worker.get();
        check(concurrent.metrics().resident_records == 128, "concurrent lanes lost resident accounting");

        auto huge = c; huge.layers = std::numeric_limits<std::size_t>::max();
        rejects([&] { BoundedMemoryKVStore::estimate_bytes(huge); }, "capacity arithmetic must reject overflow");
        auto malformed = c; malformed.resident_rows_per_layer = 7;
        rejects([&] { BoundedMemoryKVStore::estimate_bytes(malformed); }, "resident slots exceed logical capacity");

        const auto old_cwd = std::filesystem::current_path();
        const auto directory = std::filesystem::temp_directory_path() / ("hillserve-memory-store-test-" + std::to_string(getpid()));
        check(std::filesystem::create_directory(directory), "exclusive empty test directory required");
        try {
            std::filesystem::current_path(directory);
            BoundedMemoryKVStore no_file(config(1, 2, 512, 2));
            no_file.append(0, 0, rows(0, 0, 2, 512));
            check_rows(no_file.fetch(0, {1, 0}, 1024), 0, {1, 0}, 512);
            check(std::filesystem::is_empty(directory), "memory storage created a backing file");
            std::filesystem::current_path(old_cwd); std::filesystem::remove(directory);
        } catch (...) {
            std::filesystem::current_path(old_cwd); std::filesystem::remove(directory); throw;
        }
        std::cout << "bounded memory KV contracts passed; no hardware/synthetic latency used\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
