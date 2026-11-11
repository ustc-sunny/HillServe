#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:---cpu}
[[ $# -le 1 && ( "$mode" == --cpu || "$mode" == --gpu ) ]] || {
  echo 'usage: bash scripts/build.sh [--cpu | --gpu]' >&2; exit 2;
}
bash scripts/prepare_upstream.sh
jobs=${HILLSERVE_BUILD_JOBS:-4}
if [[ "$mode" == --cpu ]]; then
  cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release \
    -DHILLSERVE_JSON_INCLUDE="$PWD/third_party/llama.cpp/vendor"
  cmake --build build/cpu -j "$jobs"
  ctest --test-dir build/cpu --output-on-failure
  exit
fi
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
arch=${HILLSERVE_CUDA_ARCHITECTURES:-89}
xrt=${HILLSERVE_ENABLE_XRT:-ON}
if [[ "$xrt" == ON ]]; then
  setup=${HILLSERVE_XRT_SETUP:-/opt/xilinx/xrt/setup.sh}
  [[ -f "$setup" ]] || { echo "XRT setup not found: $setup" >&2; exit 1; }
  set +u
  source "$setup"
  set -u
fi
cmake -S third_party/llama-round2 -B build/llama-cuda \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$arch" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build/llama-cuda --target llama -j "$jobs"
extra=()
if [[ -n "${HILLSERVE_V3_IMAGE_SHA256:-}" ]]; then
  extra+=("-DHILLSERVE_V3_IMAGE_SHA256=$HILLSERVE_V3_IMAGE_SHA256")
fi
cmake -S . -B build/gpu -DCMAKE_BUILD_TYPE=Release \
  -DHILLSERVE_LAYER_RUNTIME=ON -DHILLSERVE_XRT="$xrt" \
  -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$arch" \
  -DLLAMA_SOURCE="$PWD/third_party/llama-round2" \
  -DLLAMA_BUILD="$PWD/build/llama-cuda" \
  -DNATIVE_LLAMA_SOURCE="$PWD/third_party/llama.cpp" "${extra[@]}"
cmake --build build/gpu -j "$jobs"
ctest --test-dir build/gpu --output-on-failure
