# Verify an already synthesized project's RTL. No board or XRT device access.
set root [file normalize [file join [file dirname [info script]] ../..]]
if {![info exists ::env(HILLSERVE_FULL_GQA_HLS_PROJECT)]} {
    error "HILLSERVE_FULL_GQA_HLS_PROJECT must identify the existing synthesized project"
}
set project [file normalize $::env(HILLSERVE_FULL_GQA_HLS_PROJECT)]
set solution [file join $project solution1]
if {![file isdirectory $project] || ![file exists [file join $solution syn report hillserve_full_gqa_f16_csynth.xml]]} {
    error "Existing full-GQA synthesis report required; no implicit synthesis or new project"
}
if {[file exists [file join $solution sim report hillserve_full_gqa_f16_cosim.rpt]]} {
    error "Refusing to overwrite an existing RTL co-simulation report"
}
cd [file dirname $project]
open_project [file tail $project]
open_solution solution1
add_files -tb [file join $root tests/full_gqa_kernel_check.cpp] -cflags "-std=c++14 -I$root -ffp-contract=off"
add_files -tb [file join $root policy/host_policy.cpp] -cflags "-std=c++14 -I$root -ffp-contract=off"
csim_design -clean -argv "--rtl-smoke"
cosim_design -rtl verilog -tool xsim -argv "--rtl-smoke"
exit
