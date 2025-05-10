# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from concurrent.futures import ThreadPoolExecutor
import os
import time
import threading

import cupy as cp
import numpy as np
import netifaces as ni
import pytest

from mscclpp import (
    DataType,
    EndpointConfig,
    ExecutionPlan,
    Executor,
    Fifo,
    Host2DeviceSemaphore,
    Host2HostSemaphore,
    ProxyService,
    SmDevice2DeviceSemaphore,
    TcpBootstrap,
    Transport,
    is_nvls_supported,
)
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder, pack
from ._cpp import _ext
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group

ethernet_interface_name = "eth0"


def all_ranks_on_the_same_node(mpi_group: MpiGroup):
    if (ethernet_interface_name in ni.interfaces()) is False:
        pytest.skip(f"{ethernet_interface_name} is not an interface to use on this node")
    my_ip = ni.ifaddresses(ethernet_interface_name)[ni.AF_INET][0]["addr"]
    root_ip = mpi_group.comm.bcast(my_ip, 0)
    last_rank_ip = mpi_group.comm.bcast(my_ip, mpi_group.comm.size - 1)
    return last_rank_ip == root_ip


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("ifIpPortTrio", ["eth0:localhost:50000", ethernet_interface_name, ""])
def test_group_with_ip(mpi_group: MpiGroup, ifIpPortTrio: str):
    if (ethernet_interface_name in ni.interfaces()) is False:
        pytest.skip(f"{ethernet_interface_name} is not an interface to use on this node")
    my_ip = ni.ifaddresses(ethernet_interface_name)[ni.AF_INET][0]["addr"]
    root_ip = mpi_group.comm.bcast(my_ip, 0)
    if ifIpPortTrio == ethernet_interface_name:
        ifIpPortTrio += ":" + root_ip + ":50000"  # some random port

    if all_ranks_on_the_same_node(mpi_group) is False and "localhost" in ifIpPortTrio:
        # ranks are on different nodes
        pytest.skip("this case is not supported as localhost will be different for different nodes")

    group = mscclpp_comm.CommGroup(mpi_group.comm, ifIpPortTrio)

    nelem = 1024
    memory = np.zeros(nelem, dtype=np.int32)
    nelemPerRank = nelem // group.nranks
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = group.my_rank + 1
    memory_expected = np.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1

    for rank in range(group.nranks):
        if rank == group.my_rank:
            continue
        group.send(memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))], rank, 0)
    for rank in range(group.nranks):
        if rank == group.my_rank:
            continue
        group.recv(memory[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))], rank, 0)

    assert np.array_equal(memory, memory_expected)


@parametrize_mpi_groups(2, 4, 8, 16)
def test_bootstrap_init_gil_release(mpi_group: MpiGroup):
    bootstrap = TcpBootstrap.create(mpi_group.comm.rank, mpi_group.comm.size)
    uniq_id = None
    if mpi_group.comm.rank == 0:
        # similar to NCCL's unique id
        uniq_id = bootstrap.create_unique_id()
    uniq_id_global = mpi_group.comm.bcast(uniq_id, 0)

    if mpi_group.comm.rank == 0:
        # rank 0 never initializes the bootstrap, making other ranks block
        pass
    else:
        check_list = []

        def check_target():
            check_list.append("this thread could run.")

        def init_target():
            try:
                # expected to raise a timeout after 3 seconds
                bootstrap.initialize(uniq_id_global, 3)
            except:
                pass

        init_thread = threading.Thread(target=init_target)
        check_thread = threading.Thread(target=check_target)
        init_thread.start()

        time.sleep(0.1)

        # check that the check thread is not blocked
        s = time.time()
        check_thread.start()
        check_thread.join()
        e = time.time()
        assert e - s < 0.1
        assert len(check_list) == 1

        init_thread.join()

    mpi_group.comm.barrier()


def create_connection(group: mscclpp_comm.CommGroup, transport: str):
    if transport == "NVLS":
        all_ranks = list(range(group.nranks))
        tran = Transport.Nvls
        connection = group.make_connection(all_ranks, tran)
        return connection

    remote_nghrs = list(range(group.nranks))
    remote_nghrs.remove(group.my_rank)
    if transport == "NVLink":
        tran = Transport.CudaIpc
    elif transport == "IB":
        tran = group.my_ib_device(group.my_rank % 8)
    else:
        assert False
    connections = group.make_connection(remote_nghrs, tran)
    return connections


def create_group_and_connection(mpi_group: MpiGroup, transport: str):
    if (transport == "NVLink" or transport == "NVLS") and all_ranks_on_the_same_node(mpi_group) is False:
        pytest.skip("cannot use nvlink/nvls for cross node")
    group = mscclpp_comm.CommGroup(mpi_group.comm)
    connection = create_connection(group, transport)
    return group, connection


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
def test_group_with_connections(mpi_group: MpiGroup, transport: str):
    create_group_and_connection(mpi_group, transport)


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
def test_connection_write(mpi_group: MpiGroup, transport: Transport, nelem: int):
    group, connections = create_group_and_connection(mpi_group, transport)
    memory = cp.zeros(nelem, dtype=cp.int32)
    nelemPerRank = nelem // group.nranks
    sizePerRank = nelemPerRank * memory.itemsize
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = group.my_rank + 1
    memory_expected = cp.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1
    group.barrier()
    all_reg_memories = group.register_tensor_with_connections(memory, connections)
    for rank in connections:
        connections[rank].write(
            all_reg_memories[rank],
            sizePerRank * group.my_rank,
            all_reg_memories[group.my_rank],
            sizePerRank * group.my_rank,
            sizePerRank,
        )
    poll_for = 100
    for i in range(poll_for):
        all_correct = cp.array_equal(memory, memory_expected)
        if all_correct:
            break
        time.sleep(0.1)
    for conn in connections:
        connections[conn].flush()
    cp.cuda.runtime.deviceSynchronize()
    group.barrier()
    assert all_correct


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20, 27]])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_connection_write_and_signal(mpi_group: MpiGroup, transport: Transport, nelem: int, device: str):
    # this test starts with a random tensor on rank 0 and rotates it all the way through all ranks
    # and finally, comes back to rank 0 to make sure it matches all the original values

    if device == "cpu" and transport == "NVLink":
        pytest.skip("nvlink doesn't work with host allocated memory")
    group, connections = create_group_and_connection(mpi_group, transport)
    xp = cp if device == "cuda" else np
    if group.my_rank == 0:
        memory = xp.random.randn(nelem)
        memory = memory.astype(xp.float32)
        memory_expected = memory.copy()
    else:
        memory = xp.zeros(nelem, dtype=xp.float32)
    if device == "cuda":
        cp.cuda.runtime.deviceSynchronize()

    signal_memory = xp.zeros(1, dtype=xp.int64)
    all_reg_memories = group.register_tensor_with_connections(memory, connections)
    all_signal_memories = group.register_tensor_with_connections(signal_memory, connections)

    next_rank = (group.my_rank + 1) % group.nranks
    bufferSize = nelem * memory.itemsize
    dummy_memory_on_cpu = np.zeros(1, dtype=np.int64)

    signal_val = 123
    if group.my_rank != 0:
        while signal_memory[0] != signal_val:
            time.sleep(0.1)
    connections[next_rank].write(all_reg_memories[next_rank], 0, all_reg_memories[group.my_rank], 0, bufferSize)
    connections[next_rank].flush()
    if group.my_rank == 0:
        memory[:] = 0
        if device == "cuda":
            cp.cuda.runtime.deviceSynchronize()
    connections[next_rank].update_and_sync(
        all_signal_memories[next_rank], 0, dummy_memory_on_cpu.ctypes.data, signal_val
    )
    all_correct = False
    if group.my_rank == 0:
        while signal_memory[0] != signal_val:
            time.sleep(0.1)
        all_correct = cp.array_equal(memory, memory_expected)
    group.barrier()
    all_correct = mpi_group.comm.bcast(all_correct, 0)
    assert all_correct


@parametrize_mpi_groups(2, 4, 8, 16)
def test_h2h_semaphores(mpi_group: MpiGroup):
    group, connections = create_group_and_connection(mpi_group, "IB")

    semaphores = group.make_semaphore(connections, Host2HostSemaphore)
    for rank in connections:
        semaphores[rank].signal()

    for rank in connections:
        semaphores[rank].wait()
    group.barrier()


@parametrize_mpi_groups(2, 4, 8, 16)
def test_h2h_semaphores_gil_release(mpi_group: MpiGroup):
    group, connections = create_group_and_connection(mpi_group, "IB")

    semaphores = group.make_semaphore(connections, Host2HostSemaphore)

    def target_wait(sems, conns):
        for rank in conns:
            sems[rank].wait(-1)

    def target_signal(sems, conns):
        # sleep 1 sec to let target_wait() starts a bit earlier
        time.sleep(1)
        # if wait() doesn't release GIL, this will block forever
        for rank in conns:
            sems[rank].signal()

    wait_thread = threading.Thread(target=target_wait, args=(semaphores, connections))
    signal_thread = threading.Thread(target=target_signal, args=(semaphores, connections))
    wait_thread.start()
    signal_thread.start()
    signal_thread.join()
    wait_thread.join()

    group.barrier()


@parametrize_mpi_groups(8)
@pytest.mark.skipif(is_nvls_supported() is False, reason="NVLS is not supported")
def test_nvls_connection(mpi_group: MpiGroup):
    if all_ranks_on_the_same_node(mpi_group) is False:
        pytest.skip("cannot use nvls for cross node")
    group = mscclpp_comm.CommGroup(mpi_group.comm)
    all_ranks = list(range(group.nranks))
    endpoint = EndpointConfig(Transport.Nvls, 2**22)
    nvls_connection = group.make_connection(all_ranks, endpoint)
    mem_handle1 = nvls_connection.allocate_bind_memory(2**21)
    mem_handle2 = nvls_connection.allocate_bind_memory(2**21)
    with pytest.raises(Exception):
        mem_handle3 = nvls_connection.allocate_bind_memory(2**21)
    # the memory is freed on the destructor of mem_handle2
    mem_handle2 = None
    mem_handle3 = nvls_connection.allocate_bind_memory(2**21)


class MscclppKernel:
    def __init__(
        self,
        test_name,
        my_rank=None,
        nranks=None,
        semaphore_or_channels=None,
        tensor=None,
        use_packet=False,
        scratch=None,
        fifo=None,
        nvls_mem_handle=None,
        nvls_buffer_size=None,
    ):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        if test_name == "h2d_semaphore":
            self._kernel = KernelBuilder(
                file="h2d_semaphore_test.cu", kernel_name="h2d_semaphore", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = 1
            self.nthreads = nranks
        elif test_name == "d2d_semaphore":
            self._kernel = KernelBuilder(
                file="d2d_semaphore_test.cu", kernel_name="d2d_semaphore", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = 1
            self.nthreads = nranks
        elif test_name == "sm_channel":
            self._kernel = KernelBuilder(
                file="sm_channel_test.cu", kernel_name="sm_channel", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = nranks
            self.nthreads = 1024
        elif test_name == "fifo":
            self._kernel = KernelBuilder(
                file="fifo_test.cu", kernel_name="fifo", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = 1
            self.nthreads = 1
        elif test_name == "proxy":
            self._kernel = KernelBuilder(
                file="proxy_test.cu", kernel_name="proxy", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = 1
            self.nthreads = nranks
        elif test_name == "simple_proxy_channel":
            self._kernel = KernelBuilder(
                file="simple_proxy_channel_test.cu", kernel_name="simple_proxy_channel", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = 1
            self.nthreads = 1024
        elif test_name == "nvls":
            self._kernel = KernelBuilder(
                file="nvls_test.cu", kernel_name="nvls_test", file_dir=file_dir
            ).get_compiled_kernel()
            self.nblocks = 64
            self.nthreads = 1024
        else:
            assert False

        self.params = b""
        if semaphore_or_channels != None:
            first_arg = next(iter(semaphore_or_channels.values()))
            size_of_semaphore_or_channels = len(first_arg.device_handle().raw)
            device_handles = []
            for rank in range(nranks):
                if rank == my_rank:
                    device_handles.append(
                        bytes(size_of_semaphore_or_channels)
                    )  # just zeros for semaphores that do not exist
                else:
                    device_handles.append(semaphore_or_channels[rank].device_handle().raw)
            # keep a reference to the device handles so that they don't get garbage collected
            self._d_semaphore_or_channels = cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)

        if test_name in ["h2d_semaphore", "d2d_semaphore", "sm_channel", "simple_proxy_channel"]:
            self.params += pack(self._d_semaphore_or_channels, my_rank, nranks)
            if test_name == "sm_channel":
                self.params += pack(tensor.size, use_packet)
            if test_name == "simple_proxy_channel":
                self.params += pack(tensor, scratch, tensor.size, use_packet)
        elif test_name == "fifo":
            self.params = fifo.device_handle().raw
        elif test_name == "proxy":
            self.params = pack(my_rank, nranks) + fifo.raw + pack(self._d_semaphore_or_channels)
        elif test_name == "nvls":
            self.params = (
                nvls_mem_handle.device_handle().raw
                + pack(self._d_semaphore_or_channels)
                + pack(my_rank, nranks, nvls_buffer_size)
            )

    def __call__(self):
        return self._kernel.launch_kernel(self.params, self.nblocks, self.nthreads, 0, None)


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("transport", ["NVLink", "IB"])
def test_h2d_semaphores(mpi_group: MpiGroup, transport: str):
    def signal(semaphores):
        for rank in semaphores:
            semaphores[rank].signal()

    group, connections = create_group_and_connection(mpi_group, transport)

    semaphores = group.make_semaphore(connections, Host2DeviceSemaphore)
    kernel = MscclppKernel("h2d_semaphore", group.my_rank, group.nranks, semaphores)
    kernel()

    # workaround: use a separate thread to to let cudaMemcpyAsync run concurrently with the kernel
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(signal, semaphores)

    cp.cuda.runtime.deviceSynchronize()
    group.barrier()


@parametrize_mpi_groups(2, 4, 8, 16)
def test_d2d_semaphores(mpi_group: MpiGroup):
    group, connections = create_group_and_connection(mpi_group, "NVLink")

    semaphores = group.make_semaphore(connections, SmDevice2DeviceSemaphore)
    group.barrier()
    kernel = MscclppKernel("d2d_semaphore", group.my_rank, group.nranks, semaphores)
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    group.barrier()


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("use_packet", [False, True])
def test_sm_channels(mpi_group: MpiGroup, nelem: int, use_packet: bool):
    group, connections = create_group_and_connection(mpi_group, "NVLink")

    memory = cp.zeros(nelem, dtype=cp.int32)
    if use_packet:
        scratch = cp.zeros(nelem * 2, dtype=cp.int32)
    else:
        scratch = None
    nelemPerRank = nelem // group.nranks
    nelemPerRank * memory.itemsize
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = group.my_rank + 1
    memory_expected = cp.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1

    if use_packet:
        channels = group.make_sm_channels_with_scratch(memory, scratch, connections)
    else:
        channels = group.make_sm_channels(memory, connections)
    kernel = MscclppKernel("sm_channel", group.my_rank, group.nranks, channels, memory, use_packet, scratch)

    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    group.barrier()
    assert cp.array_equal(memory, memory_expected)


@parametrize_mpi_groups(2, 4, 8, 16)
def test_fifo(
    mpi_group: MpiGroup,
):
    fifo = Fifo()
    kernel = MscclppKernel("fifo", fifo=fifo)

    kernel()
    poll_for = 100
    for _ in range(poll_for):
        trigger = fifo.poll()
        if trigger.fst == 123:
            return
        time.sleep(0.1)
    assert False


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("transport", ["IB", "NVLink"])
def test_proxy(mpi_group: MpiGroup, nelem: int, transport: str):
    group, connections = create_group_and_connection(mpi_group, transport)

    memory = cp.zeros(nelem, dtype=cp.int32)
    nelemPerRank = nelem // group.nranks
    nelemPerRank * memory.itemsize
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = group.my_rank + 1
    memory_expected = cp.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1
    group.barrier()
    all_reg_memories = group.register_tensor_with_connections(memory, connections)

    semaphores = group.make_semaphore(connections, Host2DeviceSemaphore)

    list_conn = []
    list_sem = []
    list_reg_mem = []
    first_conn = next(iter(connections.values()))
    first_sem = next(iter(semaphores.values()))
    for rank in range(group.nranks):
        if rank in connections:
            list_conn.append(connections[rank])
            list_sem.append(semaphores[rank])
        else:
            list_conn.append(first_conn)  # just for simplicity of indexing
            list_sem.append(first_sem)

        list_reg_mem.append(all_reg_memories[rank])

    proxy = _ext.MyProxyService(group.my_rank, group.nranks, nelem * memory.itemsize, list_conn, list_reg_mem, list_sem)

    fifo_device_handle = proxy.fifo_device_handle()

    kernel = MscclppKernel(
        "proxy", my_rank=group.my_rank, nranks=group.nranks, semaphore_or_channels=semaphores, fifo=fifo_device_handle
    )
    proxy.start()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy.stop()
    group.barrier()
    assert cp.array_equal(memory, memory_expected)


@parametrize_mpi_groups(2, 4, 8, 16)
@pytest.mark.parametrize("nelem", [2**i for i in [10, 15, 20]])
@pytest.mark.parametrize("transport", ["NVLink", "IB"])
@pytest.mark.parametrize("use_packet", [False, True])
def test_simple_proxy_channel(mpi_group: MpiGroup, nelem: int, transport: str, use_packet: bool):
    group, connections = create_group_and_connection(mpi_group, transport)

    memory = cp.zeros(nelem, dtype=cp.int32)
    if use_packet:
        scratch = cp.zeros(nelem * 2, dtype=cp.int32)
    else:
        scratch = cp.zeros(1, dtype=cp.int32)  # just so that we can pass a valid ptr
    nelemPerRank = nelem // group.nranks
    nelemPerRank * memory.itemsize
    memory[(nelemPerRank * group.my_rank) : (nelemPerRank * (group.my_rank + 1))] = group.my_rank + 1
    memory_expected = cp.zeros_like(memory)
    for rank in range(group.nranks):
        memory_expected[(nelemPerRank * rank) : (nelemPerRank * (rank + 1))] = rank + 1
    group.barrier()

    proxy_service = ProxyService()
    if use_packet:
        memory_to_register = scratch
    else:
        memory_to_register = memory
    simple_channels = group.make_proxy_channels(proxy_service, memory_to_register, connections)

    kernel = MscclppKernel(
        "simple_proxy_channel",
        my_rank=group.my_rank,
        nranks=group.nranks,
        semaphore_or_channels=simple_channels,
        tensor=memory,
        use_packet=use_packet,
        scratch=scratch,
    )
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    assert cp.array_equal(memory, memory_expected)


@parametrize_mpi_groups(4, 8)
@pytest.mark.skipif(is_nvls_supported() is False, reason="NVLS is not supported")
def test_nvls(mpi_group: MpiGroup):
    group, nvls_connection = create_group_and_connection(mpi_group, "NVLS")
    nbytes = 2**21
    mem_handle = nvls_connection.allocate_bind_memory(nbytes)

    nvlinks_connections = create_connection(group, "NVLink")
    semaphores = group.make_semaphore(nvlinks_connections, SmDevice2DeviceSemaphore)

    kernel = MscclppKernel(
        "nvls",
        my_rank=group.my_rank,
        nranks=group.nranks,
        nvls_mem_handle=mem_handle,
        nvls_buffer_size=nbytes,
        semaphore_or_channels=semaphores,
    )

    kernel()
    cp.cuda.runtime.deviceSynchronize()
    group.barrier()


@parametrize_mpi_groups(2)
@pytest.mark.parametrize("filename", ["allreduce.json", "allreduce_packet.json"])
def test_executor(mpi_group: MpiGroup, filename: str):
    if all_ranks_on_the_same_node(mpi_group) is False:
        pytest.skip("algo not support cross node")
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mscclpp_group = mscclpp_comm.CommGroup(mpi_group.comm)
    executor = Executor(mscclpp_group.communicator)
    execution_plan = ExecutionPlan("allreduce_pairs", os.path.join(project_dir, "test", "execution-files", filename))

    nelems = 1024 * 1024
    cp.random.seed(42)
    buffer = cp.random.random(nelems).astype(cp.float16)
    sub_arrays = cp.split(buffer, mpi_group.comm.size)
    sendbuf = sub_arrays[mpi_group.comm.rank]
    expected = cp.zeros_like(sendbuf)
    for i in range(mpi_group.comm.size):
        expected += sub_arrays[i]
    mscclpp_group.barrier()

    stream = cp.cuda.Stream(non_blocking=True)
    executor.execute(
        mpi_group.comm.rank,
        sendbuf.data.ptr,
        sendbuf.data.ptr,
        sendbuf.nbytes,
        sendbuf.nbytes,
        DataType.float16,
        512,
        execution_plan,
        stream.ptr,
    )
    stream.synchronize()
    assert cp.allclose(sendbuf, expected, atol=1e-3 * mpi_group.comm.size)
