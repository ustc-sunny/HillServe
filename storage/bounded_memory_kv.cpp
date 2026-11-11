#include "storage/bounded_memory_kv.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace hillserve {
namespace {
constexpr std::uint64_t absent = std::numeric_limits<std::uint64_t>::max();
void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::size_t add(std::size_t a, std::size_t b) {
    need(b <= std::numeric_limits<std::size_t>::max() - a, "memory KV size addition overflow");
    return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
    need(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a, "memory KV size multiplication overflow");
    return a * b;
}
struct Plan { std::size_t payload, index, total, positions, slots; };
Plan plan(const BoundedMemoryKVStore::Config& c) {
    need(c.layers > 0 && c.position_capacity > 0 && c.row_bytes > 0,
         "memory KV requires nonzero layers, position capacity and row bytes");
    need(c.position_capacity <= std::numeric_limits<std::size_t>::max(), "memory KV capacity exceeds address range");
    need(c.resident_rows_per_layer > 0 && c.resident_rows_per_layer <= c.position_capacity,
         "memory KV resident rows must be within position capacity");
    Plan p{};
    p.positions = mul(c.layers, static_cast<std::size_t>(c.position_capacity));
    p.slots = mul(c.layers, c.resident_rows_per_layer);
    p.payload = mul(p.slots, c.row_bytes);
    p.index = mul(p.positions, sizeof(std::uint64_t));
    p.index = add(p.index, mul(p.slots, 2 * sizeof(std::uint64_t))); // reverse map and free slots
    p.index = add(p.index, mul(c.layers, sizeof(std::uint64_t) + sizeof(std::size_t)));
    p.total = add(add(p.payload, p.index), sizeof(BoundedMemoryKVStore));
    return p;
}
} // namespace

std::size_t BoundedMemoryKVStore::estimate_bytes(const Config& c) { return plan(c).total; }

BoundedMemoryKVStore::BoundedMemoryKVStore(const Config& c) : config_(c) {
    const auto p = plan(c);
    need(p.total <= c.resident_budget_bytes, "memory KV resident budget exceeded before allocation");
    reserved_bytes_ = p.total;
    // Fixed-size slabs avoid unreported vector growth and retain no hidden
    // complete-history copy. Uninitialized payload is never readable until its
    // position has been installed; every successful fetch copies actual bytes.
    payload_.reset(new std::uint8_t[p.payload]);
    position_to_slot_.reset(new std::uint64_t[p.positions]);
    slot_to_position_.reset(new std::uint64_t[p.slots]);
    free_slots_.reset(new std::uint64_t[p.slots]);
    next_position_.reset(new std::uint64_t[c.layers]);
    free_count_.reset(new std::size_t[c.layers]);
    std::fill_n(position_to_slot_.get(), p.positions, absent);
    std::fill_n(slot_to_position_.get(), p.slots, absent);
    std::fill_n(next_position_.get(), c.layers, 0);
    std::fill_n(free_count_.get(), c.layers, c.resident_rows_per_layer);
    for (std::size_t layer = 0; layer < c.layers; ++layer)
        for (std::size_t i = 0; i < c.resident_rows_per_layer; ++i)
            free_slots_[slot_index(layer, i)] = c.resident_rows_per_layer - 1 - i;
    metrics_.reserved_payload_bytes = p.payload;
    metrics_.reserved_index_bytes = p.index;
    metrics_.reserved_object_bytes = sizeof(*this);
    metrics_.reserved_bytes = p.total;
}

void BoundedMemoryKVStore::check_layer(std::size_t layer) const {
    need(layer < config_.layers, "memory KV layer out of range");
}
std::size_t BoundedMemoryKVStore::position_index(std::size_t layer, std::uint64_t position) const {
    return layer * static_cast<std::size_t>(config_.position_capacity) + static_cast<std::size_t>(position);
}
std::size_t BoundedMemoryKVStore::slot_index(std::size_t layer, std::uint64_t slot) const {
    return layer * config_.resident_rows_per_layer + static_cast<std::size_t>(slot);
}
void BoundedMemoryKVStore::check_existing_positions(
    std::size_t layer, const std::vector<std::uint64_t>& positions) const {
    check_layer(layer);
    need(positions.size() <= config_.position_capacity, "memory KV query IDs exceed declared capacity");
    for (auto position : positions)
        need(position < next_position_[layer], "memory KV query references a future/out-of-range position");
}
void BoundedMemoryKVStore::install(std::size_t layer, std::uint64_t position, const std::uint8_t* record) {
    const auto slot = free_slots_[slot_index(layer, --free_count_[layer])];
    const auto index = slot_index(layer, slot);
    std::memcpy(payload_.get() + index * config_.row_bytes, record, config_.row_bytes);
    slot_to_position_[index] = position;
    position_to_slot_[position_index(layer, position)] = slot;
    ++metrics_.resident_records;
    metrics_.peak_resident_records = std::max(metrics_.peak_resident_records, metrics_.resident_records);
}

void BoundedMemoryKVStore::append(std::size_t layer, std::uint64_t first,
                                 const std::vector<std::uint8_t>& records) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_layer(layer);
    need(!records.empty() && records.size() % config_.row_bytes == 0, "memory KV append record geometry mismatch");
    const auto count = records.size() / config_.row_bytes;
    need(first == next_position_[layer] && first <= config_.position_capacity,
         "memory KV append must be consecutive");
    need(count <= config_.position_capacity - first, "memory KV append exceeds position capacity");
    need(count <= free_count_[layer], "memory KV append needs explicit eviction or a larger resident budget");
    for (std::size_t i = 0; i < count; ++i)
        install(layer, first + i, records.data() + i * config_.row_bytes);
    next_position_[layer] += count;
    metrics_.records_appended += count;
    metrics_.logical_append_bytes += records.size();
}

bool BoundedMemoryKVStore::contains(std::size_t layer, std::uint64_t position) const {
    std::lock_guard<std::mutex> lock(mutex_);
    check_layer(layer);
    need(position < config_.position_capacity, "memory KV contains position out of range");
    return position < next_position_[layer] && position_to_slot_[position_index(layer, position)] != absent;
}

std::vector<std::uint64_t> BoundedMemoryKVStore::missing_positions(
    std::size_t layer, const std::vector<std::uint64_t>& positions) const {
    std::lock_guard<std::mutex> lock(mutex_);
    check_existing_positions(layer, positions);
    std::size_t count = 0;
    for (auto position : positions) count += position_to_slot_[position_index(layer, position)] == absent;
    std::vector<std::uint64_t> missing(count);
    std::size_t i = 0;
    for (auto position : positions)
        if (position_to_slot_[position_index(layer, position)] == absent) missing[i++] = position;
    metrics_.max_missing_output_bytes = std::max(metrics_.max_missing_output_bytes,
                                                missing.capacity() * sizeof(std::uint64_t));
    return missing;
}

std::vector<std::uint8_t> BoundedMemoryKVStore::fetch(std::size_t layer,
    const std::vector<std::uint64_t>& positions, std::size_t staging_budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_existing_positions(layer, positions);
    const auto bytes = mul(positions.size(), config_.row_bytes);
    need(bytes <= staging_budget, "memory KV fetch staging budget exceeded");
    for (auto position : positions) {
        if (position_to_slot_[position_index(layer, position)] == absent) {
            ++metrics_.failed_fetches;
            throw std::runtime_error("memory KV fetch missing row; actual recomputation required");
        }
    }
    std::vector<std::uint8_t> result(bytes);
    need(result.capacity() <= staging_budget, "memory KV actual fetch allocation exceeds staging budget");
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto slot = position_to_slot_[position_index(layer, positions[i])];
        const auto index = slot_index(layer, slot);
        need(slot_to_position_[index] == positions[i], "memory KV internal resident index mismatch");
        std::memcpy(result.data() + i * config_.row_bytes,
                    payload_.get() + index * config_.row_bytes, config_.row_bytes);
    }
    ++metrics_.fetch_calls;
    metrics_.logical_read_bytes += bytes;
    metrics_.max_fetch_output_bytes = std::max(metrics_.max_fetch_output_bytes, result.capacity());
    return result;
}

void BoundedMemoryKVStore::evict(std::size_t layer, const std::vector<std::uint64_t>& positions) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_existing_positions(layer, positions); // All IDs checked before mutation.
    for (auto position : positions) {
        auto& slot = position_to_slot_[position_index(layer, position)];
        if (slot == absent) continue;
        const auto index = slot_index(layer, slot);
        need(slot_to_position_[index] == position && free_count_[layer] < config_.resident_rows_per_layer,
             "memory KV internal eviction index mismatch");
        slot_to_position_[index] = absent;
        free_slots_[slot_index(layer, free_count_[layer]++)] = slot;
        slot = absent;
        --metrics_.resident_records;
        ++metrics_.records_evicted;
    }
}

void BoundedMemoryKVStore::restore(std::size_t layer, std::uint64_t position,
                                  const std::vector<std::uint8_t>& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_layer(layer);
    need(position < next_position_[layer], "memory KV restore requires a previously produced position");
    need(record.size() == config_.row_bytes, "memory KV restore requires exactly one row");
    need(position_to_slot_[position_index(layer, position)] == absent, "memory KV restore cannot overwrite a resident row");
    need(free_count_[layer] > 0, "memory KV restore needs explicit eviction or a larger resident budget");
    install(layer, position, record.data());
    ++metrics_.records_restored;
    metrics_.restored_bytes += record.size();
}

std::uint64_t BoundedMemoryKVStore::completed(std::size_t layer) const {
    std::lock_guard<std::mutex> lock(mutex_); check_layer(layer); return next_position_[layer];
}
std::size_t BoundedMemoryKVStore::resident_records(std::size_t layer) const {
    std::lock_guard<std::mutex> lock(mutex_); check_layer(layer);
    return config_.resident_rows_per_layer - free_count_[layer];
}
BoundedMemoryKVStore::Metrics BoundedMemoryKVStore::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_); return metrics_;
}

} // namespace hillserve
