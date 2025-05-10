// This source file checks that:
// 1) Header <${header_str}> compiles without error.
// 2) Common macro collisions with platform/system headers are avoided.

// Turn off failures for certain configurations:
#ifndef NVBench_IGNORE_MACRO_CHECKS

// Define NVBench_MACRO_CHECK(macro, header), which emits a diagnostic indicating
// a potential macro collision and halts.
//
// Hacky way to build a string, but it works on all tested platforms.
#define NVBench_MACRO_CHECK(MACRO, HEADER)                                      \
  NVBench_MACRO_CHECK_IMPL(Identifier MACRO should not be used from NVBench      \
                           headers due to conflicts with HEADER macros.)

// Use raw platform checks instead of the NVBench_HOST_COMPILER macros since we
// don't want to #include any headers other than the one being tested.
//
// This is only implemented for MSVC/GCC/Clang.
#if defined(_MSC_VER) // MSVC

// Fake up an error for MSVC
#define NVBench_MACRO_CHECK_IMPL(msg)                                           \
  /* Print message that looks like an error: */                                \
  __pragma(message(__FILE__ ":" NVBench_MACRO_CHECK_IMPL0(__LINE__)             \
                   ": error: " #msg))                                          \
  /* abort compilation due to static_assert or syntax error: */                \
  static_assert(false, #msg);
#define NVBench_MACRO_CHECK_IMPL0(x) NVBench_MACRO_CHECK_IMPL1(x)
#define NVBench_MACRO_CHECK_IMPL1(x) #x

#elif defined(__clang__) || defined(__GNUC__)

// GCC/clang are easy:
#define NVBench_MACRO_CHECK_IMPL(msg) NVBench_MACRO_CHECK_IMPL0(GCC error #msg)
#define NVBench_MACRO_CHECK_IMPL0(expr) _Pragma(#expr)

#endif

// complex.h conflicts
#define I NVBench_MACRO_CHECK('I', complex.h)

// windows.h conflicts
#define small NVBench_MACRO_CHECK('small', windows.h)
// We can't enable these checks without breaking some builds -- some standard
// library implementations unconditionally `#undef` these macros, which then
// causes random failures later.
// Leaving these commented out as a warning: Here be dragons.
//#define min(...) NVBench_MACRO_CHECK('min', windows.h)
//#define max(...) NVBench_MACRO_CHECK('max', windows.h)

// termios.h conflicts (NVIDIA/thrust#1547)
#define B0 NVBench_MACRO_CHECK("B0", termios.h)

#endif // NVBench_IGNORE_MACRO_CHECKS

#include <${header_str}>
