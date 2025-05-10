// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>

#include <mscclpp/fifo.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_fifo(nb::module_& m) {
  nb::class_<ProxyTrigger>(m, "ProxyTrigger").def_rw("fst", &ProxyTrigger::fst).def_rw("snd", &ProxyTrigger::snd);

  nb::class_<FifoDeviceHandle>(m, "FifoDeviceHandle")
      .def_rw("triggers", &FifoDeviceHandle::triggers)
      .def_rw("tail_replica", &FifoDeviceHandle::tailReplica)
      .def_rw("head", &FifoDeviceHandle::head)
      .def_rw("size", &FifoDeviceHandle::size)
      .def_prop_ro("raw", [](const FifoDeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<Fifo>(m, "Fifo")
      .def(nb::init<int>(), nb::arg("size") = DEFAULT_FIFO_SIZE)
      .def("poll", &Fifo::poll)
      .def("pop", &Fifo::pop)
      .def("flush_tail", &Fifo::flushTail, nb::arg("sync") = false)
      .def("size", &Fifo::size)
      .def("device_handle", &Fifo::deviceHandle);
}
