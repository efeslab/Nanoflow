#pragma once
#include <span>

// #include "config.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <set>
#include <string>
// #include <spdlog/spdlog.h>
#include "config.h"
struct OperatorWrapper {
    cudaStream_t stream;
    cudaEvent_t start_event, end_event;
    std::string name;
    bool streamInited = false;
    
    // std::set<cudaEvent_t> wait_event;

    OperatorWrapper(bool create_start = false, bool create_end = true) {
        if (create_start) {
            cudaEventCreate(&start_event);
        }
        else{
            start_event = nullptr;
        }

        if (create_end) {
            cudaEventCreate(&end_event);
        }
        else{
            end_event = nullptr;
        }

        stream = nullptr;
    }

    void updateEventExistance(bool create_start, bool create_end){
        if (create_start) {
            cudaEventCreate(&start_event);
        }
        else{
            if (start_event) {
                cudaEventDestroy(start_event);
            }
            start_event = nullptr;
        }

        if (create_end) {
            cudaEventCreate(&end_event);
        }
        else{
            if (end_event) {
                cudaEventDestroy(end_event);
            }
            end_event = nullptr;
        }
    }

    virtual ~OperatorWrapper() {
        if (start_event) {
            cudaEventDestroy(start_event);
        }

        if (end_event) {
            cudaEventDestroy(end_event);
        }
    }

    OperatorWrapper& setStream(cudaStream_t stream) {
        this->stream = stream;
        streamInited = true;
        return *this;
    }

    OperatorWrapper& setName(std::string name) {
        this->name = name + " ";
        return *this;
    }

    virtual void work() = 0;

    void recordStartEvent() {
        if (start_event) {
            cudaEventRecord(start_event, stream);
        }
    }

    void recordEndEvent() {
        if (end_event) {
            cudaEventRecord(end_event, stream);
        }
    }
};