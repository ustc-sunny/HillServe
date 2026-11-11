#include "core/core.hpp"
#include "runtime/fetch_plan.hpp"
#include "storage/file_store.hpp"
#include "smartssd/adapter.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <unistd.h>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
using namespace hillserve;
static void check(bool b){if(!b)throw std::runtime_error("assertion failed");}
template<class F>static void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught);}
int main(){
    int passed=0,failed=0;
    auto test=[&](const char* name,std::function<void()> fn){try{fn();++passed;std::cout<<"PASS "<<name<<'\n';}catch(const std::exception& e){++failed;std::cerr<<"FAIL "<<name<<": "<<e.what()<<'\n';}};
    Scope s{1,2,3,4,5};
    const Token large=(1ULL<<55)+3;
    auto merger=[&]{return ScoreMerge(s,"diagnostic-f64",{{"host",{0,large}},{"cold",{2}}},{0,large,2});};
    auto block=[&](std::string p,std::vector<Score> scores){return ScoreBlock{s,"diagnostic-f64",p,scores};};
    test("partition overlap rejected",[&]{rejects([&]{ScoreMerge m(s,"x",{{"a",{1}},{"b",{1}}},{1});});});
    test("partition omissions and extra IDs rejected",[&]{rejects([&]{ScoreMerge m(s,"x",{{"a",{1}}},{1,2});});rejects([&]{ScoreMerge m(s,"x",{{"a",{2}}},{1});});});
    test("incomplete selection barrier",[&]{auto m=merger();m.accept(block("host",{{0,1}}));rejects([&]{m.finish();});});
    test("out of order and long IDs",[&]{auto m=merger();m.accept(block("cold",{{2,4}}));m.accept(block("host",{{large,3},{0,2}}));auto r=m.finish();check(r.size()==3&&r.back().token==large);});
    test("epoch request layer step head isolation",[&]{for(int i=0;i<5;++i){auto m=merger();auto b=block("host",{{0,1}});auto& t=b.scope;switch(i){case 0:++t.request;break;case 1:++t.layer;break;case 2:++t.step;break;case 3:++t.head;break;default:++t.epoch;}rejects([&]{m.accept(b);});}});
    test("scale mismatch rejected",[&]{auto m=merger();auto b=block("host",{{0,1}});b.representation="other";rejects([&]{m.accept(b);});});
    test("duplicate block rejected",[&]{auto m=merger();m.accept(block("host",{{0,1}}));rejects([&]{m.accept(block("host",{{0,1}}));});});
    test("invalid block atomicity",[&]{auto m=merger();rejects([&]{m.accept(block("host",{{0,1},{2,2}}));});m.accept(block("host",{{0,1},{large,3}}));m.accept(block("cold",{{2,2}}));check(m.complete());});
    test("nonfinite scores rejected",[&]{auto m=merger();rejects([&]{m.accept(block("host",{{0,std::numeric_limits<double>::quiet_NaN()}}));});});
    test("empty universe",[&]{ScoreMerge m(s,"x",{},{});check(m.complete()&&m.finish().empty());});
    test("diagnostic ties stable and no hot auto retention",[&]{auto r=diagnostic_top_k({{large,2},{0,2},{2,1}},1);check(r==std::vector<Token>{0});});
    test("diagnostic invalid selection budget",[&]{rejects([&]{diagnostic_top_k({},1);});});
    test("diagnostic dot analytical oracle",[&]{auto r=diagnostic_full_dot_product({1,2,-3},{4,5,6,-1,0,2});check(r==std::vector<double>{-4,-7});rejects([&]{diagnostic_full_dot_product({1,2},{1});});});
    test("placement partition invariance diagnostic only",[&]{for(int n=0;n<=4;++n){std::map<std::string,std::vector<Token>> p;for(Token i=0;i<4;++i)p[i<unsigned(n)?"host":"cold"].push_back(i);ScoreMerge m(s,"diagnostic-f64",p,{0,1,2,3});for(int i=3;i>=0;--i)m.accept(block(i<n?"host":"cold",{{Token(i),double(i)}}));check(diagnostic_top_k(m.finish(),2)==std::vector<Token>({3,2}));}});
    test("bounded producer consumer ownership and tails",[&]{ScoreRing r(2,2*sizeof(Score));std::atomic<int> count{0};std::thread p([&]{for(int i=0;i<17;++i)r.push(block("host",{{Token(i),double(i)}}));r.close();});ScoreBlock b;while(r.pop(b)){check(b.scores[0].token==Token(count.load()));++count;}p.join();check(count==17);});
    test("ring failures propagate",[&]{ScoreRing r(1,16);r.fail("device failed");ScoreBlock b;rejects([&]{r.pop(b);});rejects([&]{r.push(block("host",{}));});});
    test("ring oversized and closed reject",[&]{ScoreRing r(1,16);rejects([&]{r.push(block("host",{{0,1},{1,2}}));});r.close();rejects([&]{r.push(block("host",{}));});});
    test("hot pool exact budget and duplicate replacement",[&]{HotPool h(4);h.apply(0,{1},{},{{1,{1,2,3,4}}});h.apply(1,{1},{},{{1,{5,6,7,8}}});check(h.used()==4&&h.members()==std::set<Token>{1});});
    test("hot pool over budget atomic rollback",[&]{HotPool h(4);h.apply(0,{1},{},{{1,{1,2,3,4}}});rejects([&]{h.apply(1,{1,2},{},{{2,{5}}});});check(h.used()==4&&h.epoch()==1);});
    test("demotion backing and epoch guard",[&]{HotPool h(4);h.apply(0,{1},{},{{1,{1}}});rejects([&]{h.apply(0,{1},{1},{});});rejects([&]{h.apply(1,{},{1},{});});h.apply(1,{1},{1},{});check(h.used()==0);});
    test("all host buffers counted overflow safe",[&]{Budget{1,2,3,4,5,15}.validate();rejects([&]{Budget{1,2,3,4,5,14}.validate();});rejects([&]{Budget{UINT64_MAX,1,0,0,0,UINT64_MAX}.validate();});});
    test("paper beta equation balance",[&]{Profile p{20,4,10,2};double r=.2,b=paper_beta_estimate(p,r);check(std::abs(b-5)<1e-12);check(std::abs(b*(1./20+r/10)-(1./4+r/2))<1e-12);check(std::abs(hot_fraction(3)-.75)<1e-12);});
    test("APP rejects unavailable or unconfirmed future query",[&]{rejects([&]{validate_prefetch(true,false,true);});rejects([&]{validate_prefetch(true,true,false);});validate_prefetch(false,false,false);});
    test("invalid profile values",[&]{rejects([&]{paper_beta_estimate({1,0,1,1},.2);});rejects([&]{hot_fraction(-1);});});
    test("hardware cannot silently fall back",[&]{for(auto m:{HardwareMode::fpga_dram_resident,HardwareMode::fpga_host_staged,HardwareMode::fpga_storage_streaming})rejects([&]{require_fpga_backend(m,{});});});
    test("full GQA ABI retains exact-image execution gate",[&]{
        HardwareContract c;c.abi="HS_FULL_GQA_F16_V2";c.image_path="software-contract-fixture.xclbin";c.image_sha256=std::string(64,'a');
        rejects([&]{validate_fpga_contract(HardwareMode::fpga_storage_streaming,c);});
        c.image_execute_authorized=true;validate_fpga_contract(HardwareMode::fpga_storage_streaming,c);
        c.image_sha256="";rejects([&]{validate_fpga_contract(HardwareMode::fpga_storage_streaming,c);});
    });
    test("pipeline GQA explicitly rejects unaligned shapes without fallback",[&]{
        const std::string abi="HS_FULL_GQA_PIPELINE_F16_V3";
        for(std::size_t n:{64u,192u,16384u,32768u})validate_raw_gqa_shape(abi,32,8,n,128);
        for(std::size_t n:{0u,1u,32u,63u,65u,16385u,32769u})rejects([&]{validate_raw_gqa_shape(abi,32,8,n,128);});
        rejects([&]{validate_raw_gqa_shape(abi,32,32,16384,128);});
        rejects([&]{validate_raw_gqa_shape(abi,32,8,16384,49);});
        validate_raw_gqa_shape("HS_FULL_GQA_F16_V2",32,8,17,128);
        HardwareContract c;c.abi=abi;c.image_path="software-contract-fixture.xclbin";c.image_sha256=std::string(64,'a');
        rejects([&]{validate_fpga_contract(HardwareMode::fpga_storage_streaming,c);});
        c.image_execute_authorized=true;validate_fpga_contract(HardwareMode::fpga_storage_streaming,c);
    });
    auto dir=std::filesystem::temp_directory_path()/("hillserve-test-"+std::to_string(getpid()));
    if(!std::filesystem::create_directory(dir))return 2;
    test("file completion, tail, empty, reuse",[&]{FileStore f(dir/"owned.bin",4096);rejects([&]{f.fetch();});for(int n:{0,1,255,4096,3}){std::vector<std::uint8_t> v(n);for(int i=0;i<n;++i)v[i]=i%251;f.replace(v);check(f.fetch()==v);}check(f.write_bytes==4355&&f.read_bytes==4355);});
    test("never overwrite existing files",[&]{rejects([&]{FileStore f(dir/"owned.bin",4096);});});
    test("file cap",[&]{FileStore f(dir/"cap.bin",2);rejects([&]{f.replace({1,2,3});});f.replace({1,2});check(f.fetch().size()==2);});
    test("corruption detected",[&]{FileStore f(dir/"bad.bin",20);f.replace({1,2,3});{std::fstream g(dir/"bad.bin",std::ios::in|std::ios::out|std::ios::binary);g.put(9);}rejects([&]{f.fetch();});});
    test("short read detected",[&]{FileStore f(dir/"short.bin",20);f.replace({1,2,3});std::filesystem::resize_file(dir/"short.bin",1);rejects([&]{f.fetch();});});
    test("symlink rejected",[&]{std::filesystem::create_symlink(dir/"owned.bin",dir/"link");rejects([&]{FileStore f(dir/"link",4096);});});
    test("selected KV IDs, order and exact bytes",[&]{IndexedKVStore f(dir/"indexed.bin",100,10);f.append_kv(1,2,large,{9,8,7});f.append_kv(1,2,5,{4,3});auto r=f.fetch_selected(1,2,{5,large},5);check(r.size()==2&&r[0].token==5&&r[0].bytes==std::vector<std::uint8_t>({4,3})&&r[1].bytes==std::vector<std::uint8_t>({9,8,7}));check(f.append_write_bytes==5&&f.selected_read_bytes==5);});
    test("selected KV request and layer isolation",[&]{IndexedKVStore f(dir/"isolated.bin",100,10);f.append_kv(1,0,7,{1});f.append_kv(2,0,7,{2});f.append_kv(1,1,7,{3});check(f.fetch_selected(1,1,{7},1)[0].bytes[0]==3);f.release_request(1);rejects([&]{f.fetch_selected(1,0,{7},1);});check(f.fetch_selected(2,0,{7},1)[0].bytes[0]==2&&f.records()==1);});
    test("selected KV duplicate and active budgets",[&]{IndexedKVStore f(dir/"selected-budget.bin",100,10);f.append_kv(1,0,0,{1,2});rejects([&]{f.append_kv(1,0,0,{3});});rejects([&]{f.fetch_selected(1,0,{0,0},4);});rejects([&]{f.fetch_selected(1,0,{0},1);});check(f.selected_read_bytes==0);check(f.fetch_selected(1,0,{},0).empty());});
    test("append KV file and metadata caps",[&]{IndexedKVStore f(dir/"indexed-cap.bin",2,1);f.append_kv(1,0,0,{1,2});rejects([&]{f.append_kv(2,0,0,{3});});f.release_request(1);rejects([&]{f.append_kv(2,0,0,{3});});});
    test("FetchPlan sparse reordered tail/current absolute positions",[&]{FetchPlan{1,2,3,3,{17,0,5,2}}.validate(17,32,{17});FetchPlan{1,2,0,0,{2,0,1}}.validate(2,32,{0,1,2});});
    test("FetchPlan duplicate invalid empty and overflow rejected",[&]{for(auto ids:std::vector<std::vector<std::uint64_t>>{{0,0},{18},{},{0,1,2}})rejects([&]{FetchPlan{1,0,0,0,ids}.validate(17,2,{17});});});
    test("FetchPlan future-only all masked prefill rejected",[&]{rejects([&]{FetchPlan{1,0,0,0,{2}}.validate(2,32,{0,1,2});});});
    test("GQA per-head differences fail without union",[&]{require_shared_heads({{2,0},{2,0}});rejects([&]{require_shared_heads({{2,0},{0,2}});});rejects([&]{require_shared_heads({{2},{0}});});});
    test("append completion barrier and metadata accounting",[&]{IndexedKVStore f(dir/"batch.bin",100,10,4096);f.begin_append_batch();f.append_kv(1,0,0,{1});f.append_kv(1,0,1,{2});rejects([&]{f.fetch_selected(1,0,{0},1);});rejects([&]{f.release_request(1);});rejects([&]{f.begin_append_batch();});f.end_append_batch();check(f.fetch_selected(1,0,{1,0},2)[0].bytes[0]==2);check(f.metadata().current>0&&f.metadata().peak==f.metadata().current);f.release_request(1);check(f.metadata().current==0&&f.metadata().peak>0);rejects([&]{f.end_append_batch();});});
    test("metadata allocation cap precedes payload write",[&]{IndexedKVStore f(dir/"metadata.bin",100,10,1);rejects([&]{f.append_kv(1,0,0,{1});});check(f.append_write_bytes==0&&f.records()==0&&f.metadata().current==0);});

    test("volatile KV visibility requires completed writes and explicit final drain",[&]{
        IndexedKVStore f(dir/"volatile.bin",4096,8,4096,KVDurability::volatile_query_kv);
        f.begin_append_batch();f.append_kv(1,0,0,{1,2});f.append_kv(1,0,1,{3,4});
        rejects([&]{f.fetch_selected(1,0,{0},2);});rejects([&]{f.drain();});
        f.end_append_batch();check(f.timings.sync_calls==0&&f.dirty_upper_bound()==4);
        auto got=f.fetch_selected(1,0,{1,0},4);check(got[0].bytes==std::vector<std::uint8_t>({3,4})&&got[1].token==0);
        check(f.selected_read_bytes==4);f.drain();check(f.final_drain_calls==1&&f.dirty_upper_bound()==0&&f.dirty_peak_upper_bound()==4);f.release_request(1);
    });
    test("volatile KV corruption and short-read failures remain visible",[&]{
        IndexedKVStore f(dir/"volatile-bad.bin",4096,8,4096,KVDurability::volatile_query_kv);
        f.append_kv(1,0,0,{1,2,3});{std::fstream g(dir/"volatile-bad.bin",std::ios::in|std::ios::out|std::ios::binary);g.put(9);}
        rejects([&]{f.fetch_selected(1,0,{0},3);});std::filesystem::resize_file(dir/"volatile-bad.bin",1);
        rejects([&]{f.fetch_selected(1,0,{0},3);});
    });


    test("volatile partial write error poisons visibility",[&]{
        const pid_t child=fork();check(child>=0);
        if(child==0){int result=0;try{
            std::signal(SIGXFSZ,SIG_IGN);struct rlimit cap{};check(getrlimit(RLIMIT_FSIZE,&cap)==0);cap.rlim_cur=32;check(setrlimit(RLIMIT_FSIZE,&cap)==0);
            IndexedKVStore f(dir/"volatile-writefail.bin",4096,8,4096,KVDurability::volatile_query_kv);
            f.begin_append_batch();rejects([&]{f.append_kv(1,0,0,std::vector<std::uint8_t>(128,7));});
            check(f.append_write_bytes==32&&f.records()==0);rejects([&]{f.end_append_batch();});rejects([&]{f.fetch_selected(1,0,{0},128);});rejects([&]{f.drain();});
        }catch(...){result=1;}_exit(result);}
        int status=0;check(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0);
    });

    // Only files in this test's exclusive directory are removed.
    for(auto name:{"owned.bin","cap.bin","bad.bin","short.bin","link","indexed.bin","isolated.bin","selected-budget.bin","indexed-cap.bin","batch.bin","metadata.bin","volatile.bin","volatile-bad.bin","volatile-writefail.bin"})std::filesystem::remove(dir/name);
    std::filesystem::remove(dir);
    std::cout<<"{\"passed\":"<<passed<<",\"failed\":"<<failed<<",\"hardware\":\"cpu_emulation\"}\n";
    return failed?1:0;
}
