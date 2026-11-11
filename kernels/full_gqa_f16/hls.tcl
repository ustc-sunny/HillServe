# Run from any directory; no FPGA device is opened by this script.
set root [file normalize [file join [file dirname [info script]] ../..]]
set project [file join $root build full-gqa-f16-hls]
if {[info exists ::env(HILLSERVE_FULL_GQA_HLS_PROJECT)]} {set project [file normalize $::env(HILLSERVE_FULL_GQA_HLS_PROJECT)]}
if {[file exists $project]} {error "Refusing to overwrite existing HLS project: $project"}
file mkdir [file dirname $project]
cd [file dirname $project]
open_project [file tail $project]
set_top hillserve_full_gqa_f16
add_files [file join $root kernels/full_gqa_f16/kernel.cpp] -cflags "-std=c++14 -ffp-contract=off"
open_solution -flow_target vitis solution1
set_part xcku15p-ffva1156-2LV-e
create_clock -period 4 -name default
# Default unsafe_math_optimizations remains disabled; no reassociation directive.
csynth_design
if {[info exists ::env(HILLSERVE_FULL_GQA_EXPORT_XO)] && $::env(HILLSERVE_FULL_GQA_EXPORT_XO)=="1"} {
    export_design -format xo -output [file join $project hillserve_full_gqa_f16.xo]
}
exit
