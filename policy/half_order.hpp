#pragma once
#include <cstdint>
namespace hillserve::policy {
// Numeric ordering of already-validated finite IEEE F16 values. Canonicalize
// signed zero because the original FP32 comparator treats -0 == +0; ties are
// still broken by original token ID. Scores themselves are never rewritten.
inline std::uint16_t finite_half_order_key(std::uint16_t bits) noexcept {
    if((bits&0x7fff)==0)bits=0;
    return (bits&0x8000)?std::uint16_t(~bits):std::uint16_t(bits^0x8000);
}
}
