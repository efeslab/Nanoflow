#pragma once

#include "comm.h"
#include "config.h"
#include "vortexData.cuh"
#include <vector>

class SimpleThreadSync {
	// implement simple thread synchronization methods for c++ std::thread
	std::mutex mtx;
	std::condition_variable cv;
	int count;
	int total;

public:
	SimpleThreadSync(int total)
		: total(total)
		, count(0) { }
	void barrier() {
		std::unique_lock<std::mutex> lck(mtx);
		count++;
		if(count == total) {
			count = 0;
			cv.notify_all();
		} else {
			cv.wait(lck);
		}
	}

	std::mutex& getMutex() {
		return mtx;
	}
};

enum class WorkerOp
{
	STOP,
	RUN,
	UPDATE,
	CONFIG,
};

struct SharedState {
	mscclpp::UniqueId uniqueId;
	WorkerOp op;
	std::vector<vortexUpdateData> * updates_ptr;
	std::vector<vortexConfigData> * config_ptr;
};

#ifdef ENABLE_MPI
class NetworkManager
{
public:
    int nranks;
    int rank;
    bool initialized;
    void init(int argc, char** argv);
    void finalize();

};

extern std::shared_ptr<NetworkManager> netmgr;
#else
extern SharedState shared_state;
// sync between nranks workers
extern std::unique_ptr<SimpleThreadSync> worker_sync;
// sync between nranks workers and the management/main thread
extern std::unique_ptr<SimpleThreadSync> global_sync;
#endif

