#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
TOOL_ROOT=${HILLSERVE_XILINX_ROOT:-/opt/Xilinx}
VPP="$TOOL_ROOT/Vitis/2021.2/bin/v++"
PLATFORM=/opt/xilinx/platforms/xilinx_u2_gen3x4_xdma_gc_2_202110_1/xilinx_u2_gen3x4_xdma_gc_2_202110_1.xpfm
OUT=build/partial-f16-hw
mkdir -p "$OUT"
[[ ! -e "$OUT/hillserve_partial_f16.xclbin" ]]
if ! "$VPP" -c -t hw --platform "$PLATFORM" -k hillserve_partial_f16 \
 --hls.pre_tcl kernels/partial_f16/compile_flags.tcl \
 --temp_dir "$OUT/compile_tmp" --log_dir "$OUT/compile_logs" --report_dir "$OUT/compile_reports" \
 -o "$OUT/hillserve_partial_f16.xo" kernels/partial_f16/kernel.cpp; then
 # Vitis2021.2 date-based core_revision overflows signed int32 after 2021.
 # Repair ONLY this generated project's IP metadata. Never patch installed tools.
 IP="$OUT/compile_tmp/hillserve_partial_f16/hillserve_partial_f16/hillserve_partial_f16/solution/impl/ip"
 python3 scripts/fix_generated_ip_revision.py "$IP/run_ippack.tcl"
 (cd "$IP" && sh pack.sh)
 cp "$IP/../export.xo" "$OUT/hillserve_partial_f16.xo"
fi
"$VPP" -l -t hw --platform "$PLATFORM" \
 --temp_dir "$OUT/link_tmp" --log_dir "$OUT/link_logs" --report_dir "$OUT/link_reports" \
 -o "$OUT/hillserve_partial_f16.xclbin" "$OUT/hillserve_partial_f16.xo"
sha256sum "$OUT/hillserve_partial_f16.xclbin"
