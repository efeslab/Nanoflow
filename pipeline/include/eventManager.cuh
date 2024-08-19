#pragma once
#include <cuda.h>
#include "helper.h"
#include <vector>

class EventManager {
public:
	enum EVENT_NAME
	{
		GEMM_TIMING_START = 0,
		GEMV_TIMING_START,
		NET_TIMING_START,
		GEMV_TIMING_END,
		GEMM_TIMING_END,
		GEMM_TIMING_JOIN,
		O1_FINISH,
		AG_O1_START,
		AG_O1_FINISH,
		O2_FINISH,
		AG_O2_FINISH,
		AG_O2_START,
		UG1_FINISH,
		D1_FINISH,
		UG2_FINISH,
		D2_FINISH,
		AG_D1_FINISH,
		KQV1_FINISH,
		KQV1_ROPE_START,
		KQV2_FINISH,
		KQV2_ROPE_START,
		KQV3_FINISH,
		KQV3_ROPE_START,
		KQV4_FINISH,
		KQV4_ROPE_START,
		GEMV1_FINISH,
		GEMV2_FINISH,
		GEMV3_FINISH,
		GEMV4_FINISH,
		AR1_FINISH,
		AR2_FINISH,
		AG1_GEMV_FINISH,
		AG2_GEMV_FINISH,
		CAPTURE_GEMM_START,
		CAPTURE_GEMV_END,
		CAPTURE_NET_END,
		LN_MODEL1_FINISH,
		LN_MODEL2_FINISH,
		LOGITS1_FINISH,
		LOGITS2_FINISH,
		AG_LOGITS1_FINISH,
		AG_LOGITS2_FINISH,
		NUM
	};
	constexpr static int NUM_EVENTS = NUM + 1;
	std::vector<cudaEvent_t> events;

	EventManager()
		: events(NUM_EVENTS) {
		for(int i = 0; i < NUM_EVENTS; i++) {
			CUDA_CHECK(cudaEventCreate(&events[i]));
		}
	}

	~EventManager() {
		for(int i = 0; i < NUM_EVENTS; i++) {
			CUDA_CHECK(cudaEventDestroy(events[i]));
		}
	}
};