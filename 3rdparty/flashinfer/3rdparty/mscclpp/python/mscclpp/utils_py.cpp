// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_utils(nb::module_& m) {
  nb::class_<Timer>(m, "Timer")
      .def(nb::init<int>(), nb::arg("timeout") = -1)
      .def("elapsed", &Timer::elapsed)
      .def("set", &Timer::set, nb::arg("timeout"))
      .def("reset", &Timer::reset)
      .def("print", &Timer::print, nb::arg("name"));

  nb::class_<ScopedTimer, Timer>(m, "ScopedTimer").def(nb::init<std::string>(), nb::arg("name"));

  m.def("get_host_name", &getHostName, nb::arg("maxlen"), nb::arg("delim"));
  m.def("is_nvls_supported", &isNvlsSupported);
}
