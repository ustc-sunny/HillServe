#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace hillserve {

// Request-local, file-free KV storage. A "layer" may also be a (layer, KV-head)
// lane. This class neither chooses victims nor reconstructs missing values.
// The caller must supply bytes produced by the actual model to append/restore.
class BoundedMemoryKVStore {
public:
    struct Config {
        std::size_t layers = 0;
        std::uint64_t position_capacity = 0;
        std::size_t row_bytes = 0;
        std::size_t resident_rows_per_layer = 0;
        std::size_t resident_budget_bytes = 0;
    };
    struct Metrics {
        std::size_t reserved_payload_bytes = 0;
        std::size_t reserved_index_bytes = 0;
        std::size_t reserved_object_bytes = 0;
        std::size_t reserved_bytes = 0;
        std::uint64_t resident_records = 0, peak_resident_records = 0;
        std::uint64_t records_appended = 0, records_restored = 0, records_evicted = 0;
        std::uint64_t logical_append_bytes = 0, logical_read_bytes = 0, restored_bytes = 0;
        std::uint64_t fetch_calls = 0, failed_fetches = 0;
        // Returned vectors belong to the caller's workspace, not this store.
        std::size_t max_fetch_output_bytes = 0, max_missing_output_bytes = 0;
    };

    // Includes every requested persistent allocation and sizeof(this object).
    // Allocator bookkeeping, page rounding and caller-owned query/output
    // vectors are not process-RSS estimates and must be accounted separately.
    static std::size_t estimate_bytes(const Config&);
    explicit BoundedMemoryKVStore(const Config&);
    ~BoundedMemoryKVStore() = default;
    BoundedMemoryKVStore(const BoundedMemoryKVStore&) = delete;
    BoundedMemoryKVStore& operator=(const BoundedMemoryKVStore&) = delete;

    // Consecutive new positions only. Admission is atomic: insufficient free
    // resident slots rejects the entire append, without eviction or progress.
    void append(std::size_t layer, std::uint64_t first_position,
                const std::vector<std::uint8_t>& records);
    // A future position within the declared capacity is not resident.
    bool contains(std::size_t layer, std::uint64_t position) const;
    // Both queries reject future/out-of-range positions. Order (and duplicate
    // IDs) is preserved. Input/output ID metadata is caller-owned workspace.
    std::vector<std::uint64_t> missing_positions(
        std::size_t layer, const std::vector<std::uint64_t>& positions) const;
    std::vector<std::uint8_t> fetch(std::size_t layer,
        const std::vector<std::uint64_t>& positions, std::size_t staging_budget);
    // Idempotent for already absent positions and duplicate IDs; future IDs
    // reject the entire operation. Victim choice always belongs to the caller.
    void evict(std::size_t layer, const std::vector<std::uint64_t>& positions);
    // One previously produced, currently missing row. Never changes the next
    // append position; an existing resident row cannot be silently overwritten.
    void restore(std::size_t layer, std::uint64_t position,
                 const std::vector<std::uint8_t>& record);

    std::uint64_t completed(std::size_t layer) const;
    std::size_t resident_records(std::size_t layer) const;
    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }
    Metrics metrics() const;

private:
    Config config_;
    std::size_t reserved_bytes_ = 0;
    std::unique_ptr<std::uint8_t[]> payload_;
    std::unique_ptr<std::uint64_t[]> position_to_slot_, slot_to_position_, free_slots_, next_position_;
    std::unique_ptr<std::size_t[]> free_count_;
    mutable std::mutex mutex_;
    mutable Metrics metrics_;
    std::size_t position_index(std::size_t layer, std::uint64_t position) const;
    std::size_t slot_index(std::size_t layer, std::uint64_t slot) const;
    void check_layer(std::size_t layer) const;
    void check_existing_positions(std::size_t layer,
                                  const std::vector<std::uint64_t>& positions) const;
    void install(std::size_t layer, std::uint64_t position, const std::uint8_t* record);
};

} // namespace hillserve
