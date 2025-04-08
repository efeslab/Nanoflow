#include <rocprofiler-sdk-roctx/roctx.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(bind_marker, m) {
    m.def("roctxRangePush", [](const char *name) {
        roctxRangePush(name);
    }, "Start a marker with the given name");

    m.def("roctxRangePop", []() {
        roctxRangePop();
    }, "Stop the most recent marker");

    m.def("roctxRangeStart", [](const char *name) -> uint64_t {
        return roctxRangeStart(name);
    }, "Start a marker with the given name");
    

    m.def("roctxRangeStop", [](int id) {
        roctxRangeStop(id);
    }, "Stop the most recent marker");

    m.def("roctxMark", [](const char *name) {
        roctxMark(name);
    }, "Mark a point in the current range");
}