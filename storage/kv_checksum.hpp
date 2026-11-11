#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace hillserve::storage {
// Same FNV-1a digest, byte order and constants as checksum(). Interleave four
// independent records to remove the single-record multiply dependency from
// the throughput path. This never substitutes a weaker integrity check.
inline std::array<std::uint64_t,4> checksum_four(
    const std::array<const unsigned char*,4>& rows,std::size_t bytes,std::size_t count){
    if(!count||count>4)throw std::invalid_argument("checksum batch count");
    constexpr std::uint64_t seed=14695981039346656037ULL,prime=1099511628211ULL;
    std::array<std::uint64_t,4> h{seed,seed,seed,seed};
    if(count==4){
        for(std::size_t i=0;i<bytes;++i){
            h[0]=(h[0]^rows[0][i])*prime;h[1]=(h[1]^rows[1][i])*prime;
            h[2]=(h[2]^rows[2][i])*prime;h[3]=(h[3]^rows[3][i])*prime;
        }
    }else for(std::size_t j=0;j<count;++j)
        for(std::size_t i=0;i<bytes;++i)h[j]=(h[j]^rows[j][i])*prime;
    return h;
}
} // namespace hillserve::storage
