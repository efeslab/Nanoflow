#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

if [ $# -ge 2 ] ; then
    GPU_TARGETS=$2
else
    GPU_TARGETS="gfx908;gfx90a;gfx940"
fi

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_CXX_FLAGS="-Xclang -mllvm -Xclang -enable-post-misched=0 -std=c++17 -O3 -ftemplate-backtrace-limit=0  -fPIE  -Wno-gnu-line-marker"     \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D BUILD_DEV=ON                                                                                   \
-D GPU_TARGETS=$GPU_TARGETS                                                                       \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                 \
-D USE_BITINT_EXTENSION_INT4=OFF                                                                  \
${MY_PROJECT_SOURCE}
