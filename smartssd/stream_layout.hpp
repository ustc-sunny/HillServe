#pragma once
#include "smartssd/adapter.hpp"
#include <algorithm>
#include <limits>

namespace hillserve {
struct DirectChunkRead {
    int fd=-1;
    std::uint64_t offset=0;
    std::size_t bytes=0,destination_offset=0;
};
struct DirectChunkPlan {
    std::size_t token_offset=0,tokens=0;
    std::vector<DirectChunkRead> reads;
};
// Pure layout validation, shared by the real transport and CPU contract tests.
// Every piece is page aligned; no host bounce buffer or padded token exists.
inline std::vector<DirectChunkPlan> plan_raw_gqa_stream(
    const std::vector<ScoreFileRegion>& regions,std::size_t chunk_tokens) {
    constexpr std::size_t key_bytes=8*128*2,block_bytes=64*key_bytes;
    if(regions.empty()||!chunk_tokens||chunk_tokens>32768||chunk_tokens%64)
        throw std::invalid_argument("CSS requires nonempty64-token regions and64<=chunk<=32768");
    std::size_t total_tokens=0;
    for(const auto& r:regions){
        if(r.fd<0||!r.bytes||r.offset%4096||r.bytes%block_bytes||
           r.offset>std::numeric_limits<std::uint64_t>::max()-r.bytes)
            throw std::invalid_argument("CSS region must be aligned exact64-token raw GQA blocks");
        if(r.bytes/key_bytes>32768-total_tokens)
            throw std::length_error("CSS pilot exceeds32768 cold tokens");
        total_tokens+=r.bytes/key_bytes;
    }
    std::vector<DirectChunkPlan> plan;
    std::size_t ri=0;std::uint64_t consumed=0;
    for(std::size_t base=0;base<total_tokens;base+=chunk_tokens){
        DirectChunkPlan chunk;chunk.token_offset=base;chunk.tokens=std::min(chunk_tokens,total_tokens-base);
        std::size_t needed=chunk.tokens*key_bytes,destination=0;
        while(needed){
            const auto& region=regions[ri];
            const auto bytes=std::min<std::uint64_t>(needed,region.bytes-consumed);
            DirectChunkRead read{region.fd,region.offset+consumed,std::size_t(bytes),destination};
            if(!chunk.reads.empty()&&chunk.reads.back().fd==read.fd&&
               chunk.reads.back().offset+chunk.reads.back().bytes==read.offset)
                chunk.reads.back().bytes+=read.bytes;
            else chunk.reads.push_back(read);
            needed-=bytes;destination+=bytes;consumed+=bytes;
            if(consumed==region.bytes){++ri;consumed=0;}
        }
        plan.push_back(std::move(chunk));
    }
    return plan;
}
}
