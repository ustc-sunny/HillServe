set root [file normalize [file join [file dirname [info script]] ../..]]
set project [file join $root build full-gqa-f16-csim]
if {[info exists ::env(HILLSERVE_FULL_GQA_CSIM_PROJECT)]} {set project [file normalize $::env(HILLSERVE_FULL_GQA_CSIM_PROJECT)]}
if {[file exists $project]} {error "Refusing to overwrite existing C-sim project: $project"}
file mkdir [file dirname $project]
cd [file dirname $project]
open_project [file tail $project]
set_top hillserve_full_gqa_f16
add_files [file join $root kernels/full_gqa_f16/kernel.cpp] -cflags "-std=c++14 -ffp-contract=off"
add_files -tb [file join $root tests/full_gqa_kernel_check.cpp] -cflags "-std=c++14 -I$root -ffp-contract=off"
add_files -tb [file join $root policy/host_policy.cpp] -cflags "-std=c++14 -I$root -ffp-contract=off"
open_solution -flow_target vitis solution1
set_part xcku15p-ffva1156-2LV-e
create_clock -period 4 -name default
csim_design -clean
exit
