// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/core.hpp>
#include <mscclpp/nvls.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_nvls(nb::module_& m) {
  nb::class_<NvlsConnection::DeviceMulticastPointer>(m, "DeviceMulticastPointer")
      .def("get_device_ptr",
           [](NvlsConnection::DeviceMulticastPointer* self) { return (uintptr_t)self->getDevicePtr(); })
      .def("device_handle", &NvlsConnection::DeviceMulticastPointer::deviceHandle);

  nb::class_<NvlsConnection::DeviceMulticastPointer::DeviceHandle>(m, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("devicePtr", &NvlsConnection::DeviceMulticastPointer::DeviceHandle::devicePtr)
      .def_rw("mcPtr", &NvlsConnection::DeviceMulticastPointer::DeviceHandle::mcPtr)
      .def_rw("size", &NvlsConnection::DeviceMulticastPointer::DeviceHandle::bufferSize)
      .def_prop_ro("raw", [](const NvlsConnection::DeviceMulticastPointer::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<NvlsConnection>(m, "NvlsConnection")
      .def("allocate_bind_memory", &NvlsConnection::allocateAndBindCuda)
      .def("get_multicast_min_granularity", &NvlsConnection::getMultiCastMinGranularity);

  m.def("connect_nvls_collective", &connectNvlsCollective, nb::arg("communicator"), nb::arg("allRanks"),
        nb::arg("bufferSize") = NvlsConnection::DefaultNvlsBufferSize);
}
