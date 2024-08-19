#ifndef FLASHINFER_SMALL_BLK_UTILS
#define FLASHINFER_SMALL_BLK_UTILS

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace flashinfer
{
/*!
     * \brief The type of launch. Whether use small blk to run the kernel or not.
     */
enum class LaunchType
{
	// Use all blk to launch, default flashinfer
	AllBlk = 0,
	// Use constrained sm to run. Need specify how many sm.
	SmallBlk = 1,
};

inline std::string LaunchTypeToString(const LaunchType& lType) {
  switch (lType) {
    case LaunchType::AllBlk:
      return "All";
    case LaunchType::SmallBlk:
      return "Small";
    default:
      return "Unknown";
  }
}

} // namespace flashinfer

#define DISPATCH_LAUNCH(ltype, LTYPE, ...)                    \
	switch(ltype) {                                           \
	case LaunchType::AllBlk: {                                \
		constexpr LaunchType LTYPE = LaunchType::AllBlk;      \
		__VA_ARGS__                                           \
		break;                                                \
	}                                                         \
	case LaunchType::SmallBlk: {                              \
		constexpr LaunchType LTYPE = LaunchType::SmallBlk;    \
		__VA_ARGS__                                           \
		break;                                                \
	}                                                         \
	default: {                                                \
		std::ostringstream err_msg;                           \
		err_msg << "Unsupported launch type: " << int(ltype); \
		throw std::invalid_argument(err_msg.str());           \
	}                                                         \
	}

#endif // FLASHINFER_SMALL_BLK_UTILS