// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/errors.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_error(nb::module_& m) {
  nb::enum_<ErrorCode>(m, "ErrorCode")
      .value("SystemError", ErrorCode::SystemError)
      .value("InternalError", ErrorCode::InternalError)
      .value("RemoteError", ErrorCode::RemoteError)
      .value("InvalidUsage", ErrorCode::InvalidUsage)
      .value("Timeout", ErrorCode::Timeout)
      .value("Aborted", ErrorCode::Aborted)
      .value("ExecutorError", ErrorCode::ExecutorError);

  nb::class_<BaseError>(m, "BaseError")
      .def(nb::init<std::string&, int>(), nb::arg("message"), nb::arg("errorCode"))
      .def("get_error_code", &BaseError::getErrorCode)
      .def("what", &BaseError::what);

  nb::class_<Error, BaseError>(m, "Error")
      .def(nb::init<const std::string&, ErrorCode>(), nb::arg("message"), nb::arg("errorCode"))
      .def("get_error_code", &Error::getErrorCode);

  nb::class_<SysError, BaseError>(m, "SysError")
      .def(nb::init<const std::string&, int>(), nb::arg("message"), nb::arg("errorCode"));

  nb::class_<CudaError, BaseError>(m, "CudaError")
      .def(nb::init<const std::string&, int>(), nb::arg("message"), nb::arg("errorCode"));

  nb::class_<CuError, BaseError>(m, "CuError")
      .def(nb::init<const std::string&, int>(), nb::arg("message"), nb::arg("errorCode"));

  nb::class_<IbError, BaseError>(m, "IbError")
      .def(nb::init<const std::string&, int>(), nb::arg("message"), nb::arg("errorCode"));
}
