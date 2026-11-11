# Finite3/4/3-tile transactions verify actual bounded DATAFLOW RTL behavior.
set root [file normalize [file join [file dirname [info script]] ../..]]
if {![info exists ::env(HILLSERVE_PIPELINE_HLS_PROJECT)]} {error "existing HILLSERVE_PIPELINE_HLS_PROJECT required"}
set project [file normalize $::env(HILLSERVE_PIPELINE_HLS_PROJECT)]
set report [file join $project solution1 syn report hillserve_full_gqa_pipeline_csynth.xml]
if {![file exists $report]} {error "existing pipeline synthesis report required"}
if {[file exists [file join $project solution1 sim report hillserve_full_gqa_pipeline_cosim.rpt]]} {error "refusing old cosim report"}
cd [file dirname $project]
open_project [file tail $project]
open_solution solution1
cosim_design -rtl verilog -tool xsim -argv "--rtl-smoke"
exit
