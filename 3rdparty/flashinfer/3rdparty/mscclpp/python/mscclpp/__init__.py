# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os as _os

from ._mscclpp import (
    Communicator,
    Connection,
    EndpointConfig,
    Fifo,
    Host2DeviceSemaphore,
    Host2HostSemaphore,
    numa,
    ProxyService,
    RegisteredMemory,
    SimpleProxyChannel,
    SmChannel,
    SmDevice2DeviceSemaphore,
    TcpBootstrap,
    Transport,
    TransportFlags,
    DataType,
    Executor,
    ExecutionPlan,
    PacketType,
    version,
    is_nvls_supported,
)

__version__ = version()

if _os.environ.get("MSCCLPP_HOME", None) is None:
    _os.environ["MSCCLPP_HOME"] = _os.path.abspath(_os.path.dirname(__file__))


def get_include():
    """Return the directory that contains the MSCCL++ headers."""
    return _os.path.join(_os.path.dirname(__file__), "include")


def get_lib():
    """Return the directory that contains the MSCCL++ headers."""
    return _os.path.join(_os.path.dirname(__file__), "lib")
