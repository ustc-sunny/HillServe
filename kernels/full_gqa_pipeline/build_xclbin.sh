#!/usr/bin/env bash
# Build only. This script never opens XRT devices or loads/executes an image.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODE=${1:---synth-only}
[[ $# -le 1 && ( "$MODE" == --synth-only || "$MODE" == --link ) ]] || {
  echo 'usage: build_xclbin.sh [--synth-only (default) | --link]' >&2; exit 2;
}
TOOL_ROOT=${HILLSERVE_XILINX_ROOT:-/opt/Xilinx}
HLS=${HILLSERVE_VITIS_HLS:-$TOOL_ROOT/Vitis_HLS/2021.2/bin/vitis_hls}
VPP=${HILLSERVE_VPP:-$TOOL_ROOT/Vitis/2021.2/bin/v++}
PLATFORM=${HILLSERVE_PLATFORM:-/opt/xilinx/platforms/xilinx_u2_gen3x4_xdma_gc_2_202110_1/xilinx_u2_gen3x4_xdma_gc_2_202110_1.xpfm}
OUT=${HILLSERVE_PIPELINE_BUILD_DIR:-build/full-gqa-pipeline-hw}
[[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
[[ -x "$HLS" && -x "$VPP" && -f "$PLATFORM" ]]
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export HILLSERVE_PIPELINE_HLS_PROJECT="$OUT/hls"
export HILLSERVE_PIPELINE_EXPORT_XO=0
if [[ "$MODE" == --link ]]; then export HILLSERVE_PIPELINE_EXPORT_XO=1; fi
if ! "$HLS" -f kernels/full_gqa_pipeline/hls.tcl 2>&1 | tee "$OUT/hls.log"; then
  IP="$OUT/hls/solution1/impl/ip"
  # Only the known2022+ core_revision packaging failure is repairable here.
  # Preserve hls.log and the helper's original-Tcl backup/revision-fix report.
  [[ "$MODE" == --link && -f "$IP/run_ippack.tcl" && -f "$IP/pack.sh" ]] || exit 1
  grep -Eq 'core_revision|Revision.*(invalid|bad lexical cast|out of range)' "$OUT/hls.log" || exit 1
  python3 scripts/fix_generated_ip_revision.py --generated-root "$OUT/hls" \
    "$IP/run_ippack.tcl" 2>&1 | tee "$OUT/revision-fix.log"
  (cd "$IP" && sh pack.sh) 2>&1 | tee "$OUT/repack.log"
  # export_design with an explicit -output makes pack.sh write that exact XO,
  # unlike the historical v++ compile flow's ../export.xo default.
  if [[ ! -f "$OUT/hls/hillserve_full_gqa_pipeline.xo" ]]; then
    [[ -f "$IP/../export.xo" ]]
    cp "$IP/../export.xo" "$OUT/hls/hillserve_full_gqa_pipeline.xo"
  fi
fi
if [[ "$MODE" == --synth-only ]]; then
  echo "Synthesis only: reports under $OUT/hls/solution1/syn/report; no XO export/link/image load."
  exit 0
fi
"$VPP" -l -t hw --platform "$PLATFORM" --kernel_frequency 250 \
  --temp_dir "$OUT/link_tmp" --log_dir "$OUT/link_logs" --report_dir "$OUT/link_reports" \
  -o "$OUT/hillserve_full_gqa_pipeline.xclbin" "$OUT/hls/hillserve_full_gqa_pipeline.xo" \
  2>&1 | tee "$OUT/link.log"
sha256sum "$OUT/hillserve_full_gqa_pipeline.xclbin" | tee "$OUT/IMAGE_SHA256.txt"
echo 'Build only: record actual image SHA and pass numerical/permission gates before image load.'
