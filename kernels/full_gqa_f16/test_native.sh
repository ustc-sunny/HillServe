#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(mktemp -d "${TMPDIR:-/tmp}/hillserve-full-gqa-test.XXXXXX")
trap 'rm -rf "$OUT"' EXIT
"${CXX:-c++}" -std=c++14 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-label \
  -I"$ROOT" "$ROOT/kernels/full_gqa_f16/kernel.cpp" \
  "$ROOT/policy/host_policy.cpp" "$ROOT/tests/full_gqa_kernel_check.cpp" \
  -o "$OUT/full-gqa-kernel-check"
"$OUT/full-gqa-kernel-check" "$@"
