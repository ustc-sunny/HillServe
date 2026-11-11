#pragma once
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
namespace hillserve {
// Brackets actual scheduled graph segments on the backend's own CUDA stream.
// Device interval includes stream idle time during host enqueue; not a sum of kernel times.
class GraphProfiler {
    using Clock=std::chrono::steady_clock;
    cudaEvent_t start_{},end_{};
    Clock::time_point wall_,enqueued_;
    std::ofstream out_;
    bool gpu_=false;
    static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
public:
    int step=-1;
    explicit GraphProfiler(const std::string& path):out_(path){check(cudaEventCreate(&start_));check(cudaEventCreate(&end_));out_<<std::setprecision(17);}
    ~GraphProfiler(){cudaEventDestroy(start_);cudaEventDestroy(end_);}
    static void callback(ggml_backend_t backend,ggml_tensor* last,int phase,int nodes,void* data){
        auto& p=*static_cast<GraphProfiler*>(data);
        if(phase==0){p.wall_=Clock::now();p.gpu_=ggml_backend_is_cuda(backend);if(p.gpu_)check(cudaEventRecord(p.start_,static_cast<cudaStream_t>(ggml_backend_cuda_hillserve_stream(backend))));}
        else if(phase==1){if(p.gpu_)check(cudaEventRecord(p.end_,static_cast<cudaStream_t>(ggml_backend_cuda_hillserve_stream(backend))));p.enqueued_=Clock::now();}
        else {const auto done=Clock::now();float device_ms=0;if(p.gpu_)check(cudaEventElapsedTime(&device_ms,p.start_,p.end_));
            auto dur=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
            p.out_<<"{\"step\":"<<p.step<<",\"last_tensor\":\""<<last->name<<"\",\"backend\":\""<<ggml_backend_name(backend)<<"\",\"nodes\":"<<nodes<<",\"shape\":["<<last->ne[0]<<","<<last->ne[1]<<","<<last->ne[2]<<","<<last->ne[3]<<"],\"output_type\":\""<<ggml_type_name(last->type)<<"\",\"host_enqueue_ms\":"<<dur(p.wall_,p.enqueued_)<<",\"host_wait_ms\":"<<dur(p.enqueued_,done)<<",\"host_segment_ms\":"<<dur(p.wall_,done)<<",\"cuda_stream_interval_ms\":";
            if(p.gpu_)p.out_<<device_ms;else p.out_<<"null";
            p.out_<<",\"timer_kind\":\"steady_clock_and_cuda_events\",\"units\":\"one_actual_graph_segment_ms\",\"repetitions_per_call\":1,\"evidence_class\":\"M_COMPONENT\",\"policy_id\":null}\n";
        }
    }
};
}
