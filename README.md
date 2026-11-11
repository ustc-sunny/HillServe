# HillServe

HillServe is a research implementation of heterogeneous KV-cache management and
near-storage scoring for LLM inference. This repository contains the current
llama.cpp implementation, independently authored FPGA kernels, tests, and build
tools.

The GPU runs the model and attention over selected KV. The CPU scores hot keys,
merges scores, selects candidates, and coordinates delivery. The SmartSSD FPGA
scores cold keys. Layer-wise prefetching uses an InfiniGen-style rehearsal query
from the current layer's normalized activations and the next layer's query
projection and RoPE.

## Source layout

| Directory | Contents |
|---|---|
| `core/` | Selection interfaces, block-level KV management, offline APP partitioning |
| `policy/` | Host scoring, GQA selection, and the incremental hot/cold runtime |
| `runtime/` | Layer execution, KV packing, CUDA delivery, prefetch, recomputation |
| `storage/` | Indexed and segmented KV stores and bounded direct reads |
| `smartssd/` | XRT backends, image checks, buffer management, and score streaming |
| `kernels/` | Independent HLS kernels, ABI definitions, and build scripts |
| `adapters/llama_cpp/` | Model, quality, performance, and tensor-capture entry points |
| `hybrid_eval/` | Component replay, calibration, and analysis code |
| `patches/` | Patches against the fixed llama.cpp revision and its MIT notice |
| `tests/` | Host, kernel C-simulation, storage, and scheduling checks |

No model weights, input datasets, captured tensors, bitstreams, experiment
outputs, or machine access credentials are distributed here.

## Dependencies and fixed upstream

The implementation targets Linux x86-64, C++17, CMake 3.18 or newer, Git, and
Python 3. CUDA builds require the CUDA toolkit; hardware scoring additionally
requires XRT and a compatible SmartSSD platform. The FPGA build scripts target
Vitis/Vitis HLS 2021.2. Default CUDA architecture is RTX 4090 (`89`).

llama.cpp is pinned to **b6000**, commit
`4762ad7316dcdec20016ab5985fb46a27902204d`. It is downloaded separately;
`scripts/prepare_upstream.sh` verifies the commit and applies the ordered patch
series to a separate checkout. Unknown source edits are rejected. The upstream
checkout supplies the nlohmann JSON header used by the host build.

## Build and host tests

```sh
git clone https://github.com/ustc-sunny/HillServe.git
cd HillServe
bash scripts/build.sh --cpu
```

This runs CPU contract tests and native kernel checks. It does not open an FPGA,
load an image, or run a model. The Python timing-model tests are separate:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

Build the model runners and XRT backend:

```sh
bash scripts/build.sh --gpu
# Host-only scoring with CUDA model execution, without the XRT dependency:
# HILLSERVE_ENABLE_XRT=OFF bash scripts/build.sh --gpu
```

`CUDACXX`, `HILLSERVE_CUDA_ARCHITECTURES`, `HILLSERVE_BUILD_JOBS`, and
`HILLSERVE_XRT_SETUP` override local tool locations and build settings.

## Model execution

The current GQA runner targets Gradient-Llama-3-8B with BF16 GGUF weights,
32 layers, 32 query heads, 8 KV heads, and dimension 128. Supply the model under
its own license and tokenize the same input for all compared methods:

```sh
mkdir -p inputs outputs
build/gpu/hillserve-quality-runtime --tokenize \
  models/gradient-8b-instruct-bf16.gguf inputs/prompt.txt inputs/prompt.tokens.bin
```

The token file contains little-endian int32 IDs. The ordinary runner accepts
128--32,768 prompt tokens; prepare an input within that range using the actual
model tokenizer. It does not apply a chat template automatically.

Copy an example JSON to a local configuration, set the model and input paths,
and provide **new** output and scratch filenames. For SSD experiments, place
the scratch directory on an already mounted SmartSSD filesystem and check its
available capacity before execution. These scripts do not mount devices or
modify partitions, drivers, or P2P settings.

```sh
# Edit paths in a local copy of examples/host.json first.
build/gpu/hillserve-quality-runtime path/to/host-config.json
# After building and validating a compatible FPGA image:
build/gpu/hillserve-quality-runtime path/to/hillserve-config.json
```

`HOST_MATCHED` is the SSD-backed CPU-scoring path; `HILLSERVE` performs cold
scoring on the FPGA. Both use the same hot scoring and selector. The example
configurations share the scoring representation, budgets, and partition.
The offline `beta` setting is fixed throughout a run; recalibrate it when the
device or execution configuration changes. `selector_alpha` is the score
margin, distinct from the fixed 20% retention cap. The first two layers use
dense attention and the remaining layers use selected KV. HPM is disabled in
the examples; its implementation and tests are included.

The runner also provides `FULL_NATIVE`, `FULL_RECOMPUTE`, and memory-only
prefetch/recomputation modes. Recompute modes recover missing KV and use their
explicit residency and workspace budgets; they are extensions to llama.cpp.
For quality evaluation, `task: "qa"` generates answers and
`task: "teacher_forced"` consumes `target_tokens_file` for continuation NLL.
For timing, `task: "performance"` generates a fixed number of tokens and writes
absolute per-step measurements. Hardware replay, model execution, and timing
predictions are separate entry points; a timing prediction is not a hardware
measurement.

## FPGA kernel build and validation

The current model path uses `HS_FULL_GQA_PIPELINE_F16_V3`. It consumes F16
`Q[32,128]` and `K[N,8,128]`, accumulates in ordered FP32, and returns F16
`scores[32,N]`. Query head `h` maps to physical KV head `h/4`. Each invocation
requires `64 <= N <= 32768` and `N % 64 == 0`; unsupported shapes are rejected.

```sh
HILLSERVE_XILINX_ROOT=/path/to/Xilinx \
  bash kernels/full_gqa_pipeline/build_xclbin.sh --synth-only
# Use a new output directory when linking:
HILLSERVE_XILINX_ROOT=/path/to/Xilinx \
HILLSERVE_PIPELINE_BUILD_DIR=build/v3-image \
  bash kernels/full_gqa_pipeline/build_xclbin.sh --link
```

`HILLSERVE_PLATFORM` supplies the installed platform `.xpfm`.
`HILLSERVE_VITIS_HLS` and `HILLSERVE_VPP` may select tool executables directly.
Build scripts never load an image. Validate the resulting kernel numerically
and check its ABI before authorizing board execution.

The model runners require a build-time approved image hash. After validating
your own V3 image, rebuild with its actual SHA256:

```sh
HILLSERVE_V3_IMAGE_SHA256=YOUR_VALIDATED_64_CHARACTER_SHA256 \
  bash scripts/build.sh --gpu
```

Set the same image path and hash in the local `fpga` configuration and explicitly
enable `execute_authorized` to run it. The backend checks the file's actual hash
before loading. Examples leave execution disabled. The direct-file backend
uses ordinary-file `O_DIRECT` reads into an XRT P2P buffer; support depends on
the host platform and does not by itself prove the absence of physical host-RAM
bounce traffic. Failed direct reads are not replaced with CPU scoring.

The legacy partial-key GQA allocator additionally requires
`HILLSERVE_SCRATCH_ROOT` and retains an 8-GiB directory quota. The legacy device
replay accepts explicit `--scratch-root` and `--ssd-device` arguments and checks
that the directory belongs to that block device.

## Third-party notices

The llama.cpp patch context is covered by the upstream MIT license, retained in
[`patches/llama-MIT-LICENSE`](patches/llama-MIT-LICENSE). Upstream dependencies
retain their respective notices. InfiniGen is the reference for the inherited
rehearsal-query mechanism; its implementation is not vendored. No restricted
framework source or prebuilt FPGA image is included.
