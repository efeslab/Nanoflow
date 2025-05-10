// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <mscclpp/semaphore.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_semaphore(nb::module_& m) {
  nb::class_<Host2DeviceSemaphore> host2DeviceSemaphore(m, "Host2DeviceSemaphore");
  host2DeviceSemaphore
      .def(nb::init<Communicator&, std::shared_ptr<Connection>>(), nb::arg("communicator"), nb::arg("connection"))
      .def("connection", &Host2DeviceSemaphore::connection)
      .def("signal", &Host2DeviceSemaphore::signal)
      .def("device_handle", &Host2DeviceSemaphore::deviceHandle);

  nb::class_<Host2DeviceSemaphore::DeviceHandle>(host2DeviceSemaphore, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("inbound_semaphore_id", &Host2DeviceSemaphore::DeviceHandle::inboundSemaphoreId)
      .def_rw("expected_inbound_semaphore_id", &Host2DeviceSemaphore::DeviceHandle::expectedInboundSemaphoreId)
      .def_prop_ro("raw", [](const Host2DeviceSemaphore::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<Host2HostSemaphore>(m, "Host2HostSemaphore")
      .def(nb::init<Communicator&, std::shared_ptr<Connection>>(), nb::arg("communicator"), nb::arg("connection"))
      .def("connection", &Host2HostSemaphore::connection)
      .def("signal", &Host2HostSemaphore::signal)
      .def("poll", &Host2HostSemaphore::poll)
      .def("wait", &Host2HostSemaphore::wait, nb::call_guard<nb::gil_scoped_release>(),
           nb::arg("max_spin_count") = 10000000);

  nb::class_<SmDevice2DeviceSemaphore> smDevice2DeviceSemaphore(m, "SmDevice2DeviceSemaphore");
  smDevice2DeviceSemaphore
      .def(nb::init<Communicator&, std::shared_ptr<Connection>>(), nb::arg("communicator"), nb::arg("connection"))
      .def("device_handle", &SmDevice2DeviceSemaphore::deviceHandle);

  nb::class_<SmDevice2DeviceSemaphore::DeviceHandle>(smDevice2DeviceSemaphore, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("inboundSemaphoreId", &SmDevice2DeviceSemaphore::DeviceHandle::inboundSemaphoreId)
      .def_rw("outboundSemaphoreId", &SmDevice2DeviceSemaphore::DeviceHandle::outboundSemaphoreId)
      .def_rw("remoteInboundSemaphoreId", &SmDevice2DeviceSemaphore::DeviceHandle::remoteInboundSemaphoreId)
      .def_rw("expectedInboundSemaphoreId", &SmDevice2DeviceSemaphore::DeviceHandle::expectedInboundSemaphoreId)
      .def_prop_ro("raw", [](const SmDevice2DeviceSemaphore::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });
}
