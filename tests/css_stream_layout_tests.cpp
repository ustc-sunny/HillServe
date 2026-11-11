#include "smartssd/stream_layout.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool condition,const char* reason){if(!condition)throw std::runtime_error(reason);}
template<class F> void rejects(F fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}check(rejected,"invalid stream layout accepted");}
}
int main(){
    using namespace hillserve;
    constexpr std::size_t block=64*8*128*2;
    try{
        // Three separated64-token blocks become a128-token chunk plus a64
        // token tail. Their original file offsets remain exact; no gap is read.
        auto plan=plan_raw_gqa_stream({{3,0,block,false},{3,3*block,block,false},{3,7*block,block,false}},128);
        check(plan.size()==2&&plan[0].tokens==128&&plan[1].tokens==64,"tail shape");
        check(plan[1].token_offset==128&&plan[0].reads.size()==2,"cold index map");
        check(plan[0].reads[1].offset==3*block&&plan[0].reads[1].destination_offset==block,"noncontiguous direct placement");
        check(plan[1].reads[0].offset==7*block&&plan[1].reads[0].destination_offset==0,"tail original identity");
        // Adjacent runs coalesce, while a run crossing a chunk boundary splits
        // with no duplicate or omitted bytes.
        plan=plan_raw_gqa_stream({{3,block,block,false},{3,2*block,4*block,false}},128);
        check(plan.size()==3&&plan[0].reads.size()==1&&plan[0].reads[0].bytes==2*block,"coalescing");
        check(plan[1].reads[0].offset==3*block&&plan[2].reads[0].offset==5*block,"split coverage");
        std::size_t read_bytes=0;
        for(const auto& chunk:plan){
            std::size_t dest=0;
            for(const auto& read:chunk.reads){check(read.destination_offset==dest&&read.offset%4096==0&&read.bytes%4096==0,"direct alignment/coverage");dest+=read.bytes;}
            check(dest==chunk.tokens*8*128*2,"chunk exact coverage");read_bytes+=dest;
        }
        check(read_bytes==5*block,"total byte coverage");
        plan=plan_raw_gqa_stream({{3,0,block,false},{4,block,block,false}},128);
        check(plan[0].reads.size()==2,"different fd must not coalesce");
        rejects([&]{plan_raw_gqa_stream({},64);});
        rejects([&]{plan_raw_gqa_stream({{3,0,block,false}},0);});
        rejects([&]{plan_raw_gqa_stream({{3,0,block,false}},65);});
        rejects([&]{plan_raw_gqa_stream({{3,0,block,false}},32832);});
        rejects([&]{plan_raw_gqa_stream({{-1,0,block,false}},64);});
        rejects([&]{plan_raw_gqa_stream({{3,1,block,false}},64);});
        rejects([&]{plan_raw_gqa_stream({{3,0,block-4096,false}},64);});
        rejects([&]{plan_raw_gqa_stream({{3,0,0,false}},64);});
        rejects([&]{plan_raw_gqa_stream({{3,0,513*block,false}},64);});
        rejects([&]{plan_raw_gqa_stream({{3,std::numeric_limits<std::uint64_t>::max()-4095,block,false}},64);});
        std::cout<<"CSS stream layout contracts passed; no FPGA emulation or hardware result\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
