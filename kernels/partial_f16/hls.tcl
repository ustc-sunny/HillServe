# Invoke from repository root with vitis_hls -f kernels/partial_f16/hls.tcl
set root [file normalize [file join [file dirname [info script]] ../..]]
cd [file join $root build]
open_project hls_partial_f16
set_top hillserve_partial_f16
add_files [file join $root kernels/partial_f16/kernel.cpp] -cflags "-std=c++14 -ffp-contract=off"
open_solution -flow_target vitis solution1
set_part {xcku15p-ffva1156-2LV-e}
create_clock -period 3.333333 -name default
# Keep dependency-ordered arithmetic; no unsafe reduction reassociation.
# unsafe_math_optimizations is an enable-only switch in2021.2; leave disabled (default).
csynth_design
exit
