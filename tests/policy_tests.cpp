#include "policy/host_policy.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace hillserve;
static void check(bool condition){if(!condition)throw std::runtime_error("policy contract failed");}
int main(){try{
 for(unsigned x=0;x<65536;++x)if(!std::isnan(half_value(x)))check(half_bits(half_value(x))==x);
 check(half_bits(1.00048828125f)==half_bits(1.0f));
 check(half_bits(1.00146484375f)==half_bits(1.001953125f));
 std::vector<std::uint16_t> scores;for(float x:{4,3,0,0,3,4})scores.push_back(half_bits(x));
 auto strict=select_partial_scores(scores,2,3,1,3),inclusive=select_partial_scores(scores,2,3,1,3,true);
 check(strict.k==1&&strict.ids==std::vector<std::uint64_t>({0,2}));check(inclusive.k==2);
 check(select_partial_scores(scores,2,3,0,3).k==0);
 check(select_partial_scores(scores,2,3,4,1).k==1);
 auto tie=select_partial_scores({half_bits(1),half_bits(1)},1,2,1,1);check(tie.boundary_tied_heads==1);
 auto dot=host_partial_scores({half_bits(2),half_bits(-1)},{half_bits(3),half_bits(4),half_bits(-2),half_bits(1)},1,2,2);
 check(dot==std::vector<std::uint16_t>({half_bits(2),half_bits(-5)}));
 bool rejected=false;try{host_partial_scores({half_bits(1)},{half_bits(1)},2,1,1);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
 std::cout<<"binary16 exhaustive finite roundtrip, RN-even, per-head strict/inclusive/cap/zero/tie and dot contracts passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
