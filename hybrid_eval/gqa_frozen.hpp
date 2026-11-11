#pragma once
#include "policy/gqa_policy.hpp"
#include "policy/host_policy.hpp"
#include "storage/file_store.hpp"
#include "nlohmann/json.hpp"
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace hillserve::replay {
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
inline double milliseconds(Clock::time_point x){return std::chrono::duration<double,std::milli>(Clock::now()-x).count();}
inline void need(bool x,const std::string& s){if(!x)throw std::runtime_error(s);}
template<class T> std::vector<T> load(const std::filesystem::path& p,std::size_t n){need(std::filesystem::file_size(p)==n*sizeof(T),"frozen payload extent mismatch: "+p.string());std::ifstream f(p,std::ios::binary);std::vector<T> v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));need(bool(f),"frozen payload read failed");return v;}
template<class T> void save(const std::filesystem::path& p,const std::vector<T>& v){need(!std::filesystem::exists(p),"new frozen payload required");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));need(bool(f),"frozen payload write failed");}
struct LayerArtifact {
 std::vector<float> basis;std::vector<std::uint8_t> mask;std::array<std::vector<std::size_t>,32> columns;std::size_t channels=0;
 LayerArtifact(const std::string& path,int layer){
  need(layer>=2&&layer<32,"supported sparse layer");std::ifstream f(path,std::ios::binary);char magic[8]{};std::array<std::uint32_t,4> shape{};f.read(magic,8);f.read(reinterpret_cast<char*>(shape.data()),16);
  need(std::string(magic,8)=="HSGLPF01"&&shape==std::array<std::uint32_t,4>{32,32,8,128},"artifact shape mismatch");
  f.seekg(24+std::size_t(layer)*32*128*128*2);std::vector<std::uint16_t> b(32*128*128);f.read(reinterpret_cast<char*>(b.data()),b.size()*2);need(bool(f),"artifact basis short");basis.resize(b.size());for(std::size_t i=0;i<b.size();++i)basis[i]=half_value(b[i]);
  f.seekg(24+32ULL*32*128*128*2+std::size_t(layer)*32*128);mask.resize(32*128);f.read(reinterpret_cast<char*>(mask.data()),mask.size());need(bool(f),"artifact mask short");
  for(std::size_t h=0;h<32;++h){for(std::size_t c=0;c<128;++c)if(mask[h*128+c])columns[h].push_back(c);need(!columns[h].empty(),"empty channel mask");channels=std::max(channels,columns[h].size());}
 }
 std::vector<std::uint16_t> query(const std::vector<float>& q){need(q.size()==4096,"rehearsal shape");std::vector<std::uint16_t> out(4096);std::array<std::uint16_t,128> in{};for(std::size_t h=0;h<32;++h){for(std::size_t c=0;c<128;++c)in[c]=half_bits(q[h*128+c]);auto v=policy::gqa_skew_head_f16(in.data(),basis.data()+h*128*128,mask.data()+h*128);std::copy(v.begin(),v.end(),out.begin()+h*128);}return out;}
};
inline std::vector<std::uint16_t> scaled(const std::vector<std::uint16_t>& raw){auto out=raw;for(auto& b:out)b=half_bits(half_value(b)/std::sqrt(128.f));return out;}
inline std::vector<std::uint64_t> flatten(const policy::GQASelection& s){std::vector<std::uint64_t> out;for(const auto& h:s.ids)out.insert(out.end(),h.begin(),h.end());return out;}
}
