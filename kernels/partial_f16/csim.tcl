set root [file normalize [file join [file dirname [info script]] ../..]]
if {![info exists ::env(HILLSERVE_REAL_TRACE)]} { error "HILLSERVE_REAL_TRACE required" }
set trace [file normalize $::env(HILLSERVE_REAL_TRACE)]
cd [file join $root build]
open_project partial_f16_csim
set_top hillserve_partial_f16
add_files [file join $root kernels/partial_f16/kernel.cpp] -cflags "-std=c++14 -ffp-contract=off"
add_files -tb [file join $root tests/partial_kernel_check.cpp] -cflags "-std=c++14 -I$root -ffp-contract=off"
add_files -tb [file join $root policy/host_policy.cpp] -cflags "-std=c++14 -I$root -ffp-contract=off"
add_files -tb [file join $root smartssd/adapter.cpp] -cflags "-std=c++14 -I$root"
open_solution solution -flow_target vitis
set_part xcku15p-ffva1156-2LV-e
create_clock -period 4
csim_design -clean -argv $trace
exit
