#include "storage/kv_read_plan.hpp"
#include <iostream>
#include <random>
#include <string>
using namespace hillserve::storage;
namespace {
void check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"invalid read plan accepted");}
void verify(const std::vector<KVReadRecord>& r,const std::vector<KVReadWindow>& w,const KVReadLimits& limits){
    std::size_t next=0;
    for(const auto& span:w){
        check(span.first_record==next&&span.end_record>next,"every record covered exactly once");
        check(span.begin%4096==0&&span.end%4096==0&&span.end-span.begin<=limits.window_bytes,"aligned bounded read");
        std::uint64_t required=0,previous_end=0;
        for(auto j=span.first_record;j<span.end_record;++j){
            check(r[j].offset>=span.begin&&r[j].offset+4128<=span.end,"complete record covered");
            const auto begin=r[j].offset/4096*4096,end=(r[j].offset+4128+4095)/4096*4096;
            if(j>span.first_record)check(begin<=previous_end+limits.gap_bytes,"gap ceiling");
            required+=end-std::max(begin,previous_end);previous_end=end;
        }
        check(required==span.required_page_bytes,"required page union accounting");
        check((span.end-span.begin)*1000<=required*limits.max_amplification_permille,"read amplification ceiling");
        next=span.end_record;
    }
    check(next==r.size(),"missing read records");
}
}
int main(){try{
    constexpr std::uint64_t extent=4096ULL*100000;
    KVReadLimits merged,old{56*1024,0,1000};
    check(plan_kv_reads({},4128,extent).empty(),"empty read plan");
    std::vector<KVReadRecord> sparse;
    for(std::uint64_t id=0;id<16384;++id)if(id%7<4)sparse.push_back({4096+id*4128,sparse.size()});
    const auto a=plan_kv_reads(sparse,4128,extent,old),b=plan_kv_reads(sparse,4128,extent,merged);
    verify(sparse,a,old);verify(sparse,b,merged);
    check(b.size()*4<a.size(),"bounded gap merging must reduce constructed fixture call count");
    std::mt19937 random(173);
    for(unsigned trial=0;trial<300;++trial){
        std::vector<KVReadRecord> records;
        for(std::uint64_t id=0;id<1000;++id)if(random()%100<trial%101)records.push_back({4096+id*4128,records.size()});
        KVReadLimits lim{std::size_t(3+random()%254)*4096,std::size_t(random()%17)*4096,unsigned(1000+random()%1001)};
        verify(records,plan_kv_reads(records,4128,extent,lim),lim);
    }
    rejects([&]{plan_kv_reads({{4096,0},{4096,1}},4128,extent);});
    rejects([&]{plan_kv_reads({{8192,0},{4096,1}},4128,extent);});
    rejects([&]{plan_kv_reads({{extent-4096,0}},4128,extent);});
    rejects([&]{plan_kv_reads({{4096,0}},4128,extent,{2*1024*1024,0,1500});});
    rejects([&]{plan_kv_reads({{4096,0}},4128,extent,{1024*1024,1,1500});});
    rejects([&]{plan_kv_reads({{4096,0}},4128,extent,{1024*1024,0,999});});
    rejects([&]{plan_kv_reads({{UINT64_MAX-4095,0}},4128,UINT64_MAX-4095);});
    std::cout<<"{\"status\":\"passed\",\"evidence\":\"CPU_planning_fixture_no_IO_or_performance_claim\",\"randomized_patterns\":300,\"fixture_old_windows\":"<<a.size()<<",\"fixture_new_windows\":"<<b.size()<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
