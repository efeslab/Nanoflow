// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/proxy_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_proxy_channel(nb::module_& m) {
  nb::class_<BaseProxyService>(m, "BaseProxyService")
      .def("start_proxy", &BaseProxyService::startProxy)
      .def("stop_proxy", &BaseProxyService::stopProxy);

  nb::class_<ProxyService, BaseProxyService>(m, "ProxyService")
      .def(nb::init<size_t>(), nb::arg("fifoSize") = DEFAULT_FIFO_SIZE)
      .def("start_proxy", &ProxyService::startProxy)
      .def("stop_proxy", &ProxyService::stopProxy)
      .def("build_and_add_semaphore", &ProxyService::buildAndAddSemaphore, nb::arg("comm"), nb::arg("connection"))
      .def("add_semaphore", &ProxyService::addSemaphore, nb::arg("semaphore"))
      .def("add_memory", &ProxyService::addMemory, nb::arg("memory"))
      .def("semaphore", &ProxyService::semaphore, nb::arg("id"))
      .def("proxy_channel", &ProxyService::proxyChannel, nb::arg("id"));

  nb::class_<ProxyChannel>(m, "ProxyChannel")
      .def(nb::init<SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Proxy>>(),
           nb::arg("semaphoreId"), nb::arg("semaphore"), nb::arg("proxy"))
      .def("device_handle", &ProxyChannel::deviceHandle);

  nb::class_<ProxyChannel::DeviceHandle>(m, "ProxyChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("semaphoreId_", &ProxyChannel::DeviceHandle::semaphoreId_)
      .def_rw("semaphore_", &ProxyChannel::DeviceHandle::semaphore_)
      .def_rw("fifo_", &ProxyChannel::DeviceHandle::fifo_)
      .def_prop_ro("raw", [](const ProxyChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<SimpleProxyChannel>(m, "SimpleProxyChannel")
      .def(nb::init<ProxyChannel, MemoryId, MemoryId>(), nb::arg("proxyChan"), nb::arg("dst"), nb::arg("src"))
      .def(nb::init<SimpleProxyChannel>(), nb::arg("proxyChan"))
      .def("device_handle", &SimpleProxyChannel::deviceHandle);

  nb::class_<SimpleProxyChannel::DeviceHandle>(m, "SimpleProxyChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("proxyChan_", &SimpleProxyChannel::DeviceHandle::proxyChan_)
      .def_rw("src_", &SimpleProxyChannel::DeviceHandle::src_)
      .def_rw("dst_", &SimpleProxyChannel::DeviceHandle::dst_)
      .def_prop_ro("raw", [](const SimpleProxyChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });
};
