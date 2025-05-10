# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import cupy as cp
from mscclpp_op import (
    MscclppAllReduce1,
    MscclppAllReduce2,
    MscclppAllReduce3,
    MscclppAllReduce4,
    MscclppAllReduce5,
    MscclppAllReduce6,
)
from nccl_op import NcclAllReduce
from mpi4py import MPI
import cupy.cuda.nccl as nccl
import mscclpp.comm as mscclpp_comm
from mscclpp import ProxyService, is_nvls_supported
from prettytable import PrettyTable
import netifaces as ni

data_type = cp.float32

if data_type == cp.float16:
    dtype_str = "fp16"
elif data_type == cp.float32:
    dtype_str = "fp32"
elif data_type == cp.int32:
    dtype_str = "int32"
else:
    raise RuntimeError("Unknown data type")


def plot_graph(sizes, mscclpp_algbw, nccl_algbw, speed_ups):
    import matplotlib.pyplot as plt

    human_readable_sizes = [human_readable_size(size) for size in sizes]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting AlgBW for MSCCLPP and NCCL on the primary y-axis
    (line1,) = ax1.plot(sizes, mscclpp_algbw, marker="o", color="blue", label="MSCCLPP AlgBW")
    (line2,) = ax1.plot(sizes, nccl_algbw, marker="x", color="red", label="NCCL AlgBW")
    ax1.set_ylabel("AlgBW (GB/s)")
    ax1.set_xlabel("Data Size")

    # Logarithmic x-axis
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(human_readable_sizes, rotation=45)

    # Adding secondary y-axis for Speed Up
    ax2 = ax1.twinx()
    (line3,) = ax2.plot(sizes, speed_ups, marker="^", color="green", label="Speed Up")
    ax2.set_ylabel("Speed Up (NCCL Time / MSCCLPP Time)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Set the lower bound of the secondary y-axis to 0
    ax2.set_ylim(bottom=0)

    # Creating legends
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Setting title and grid
    num_nodes = MPI.COMM_WORLD.size // N_GPUS_PER_NODE
    ax1.set_title(f"MSCCLPP vs NCCL -- {num_nodes} Nodes")
    ax2.grid(True, which="both", ls="--")

    # Saving the plot
    plt.savefig(f"mscclpp_vs_nccl_comparison_num_nodes_{num_nodes}.jpeg", format="jpeg")


def human_readable_size(size, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def check_correctness(memory, func, niter=100):
    ac = True
    for p in range(niter):
        memory[:] = cp.ones(memory.shape).astype(data_type) * (p * MPI.COMM_WORLD.size + MPI.COMM_WORLD.rank)
        cp.cuda.runtime.deviceSynchronize()
        output_memory = func(None)
        cp.cuda.runtime.deviceSynchronize()
        expected = cp.zeros_like(memory)
        for i in range(MPI.COMM_WORLD.size):
            expected += cp.ones(memory.shape).astype(data_type) * (p * MPI.COMM_WORLD.size + i)

        is_close = cp.isclose(output_memory, expected, rtol=1.0e-2, atol=2)
        icf = is_close == 0
        all_close = cp.all(is_close)
        ac = ac and all_close
        if not all_close:
            print(
                f"not close: p={p}, rank={MPI.COMM_WORLD.rank}, output={output_memory[icf][0]}, expected={expected[icf][0]}",
                flush=True,
            )

    ac = MPI.COMM_WORLD.allreduce(ac, op=MPI.SUM)
    return ac


def bench_time(niter: int, func):
    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            func(stream)
        graph = stream.end_capture()

    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niter * 1000.0


def find_best_algo(mscclpp_algos, niter):
    assert len(mscclpp_algos) > 0
    best_time = 10000000.0
    best_algo = None
    for algo in mscclpp_algos:
        config, cur_time = find_best_config(algo, niter)
        if cur_time < best_time:
            best_time = cur_time
            best_algo = algo
            algo.set_params(*config)
    if MPI.COMM_WORLD.rank == 0:
        print(best_algo, end="", flush=True)
    return best_algo


def find_best_config(mscclpp_call, niter):
    best_time = 10000000.0
    for config in mscclpp_call.auto_tune():
        cur_time = bench_time(niter, mscclpp_call)
        if cur_time < best_time:
            best_time = cur_time
            best_config = config
        if MPI.COMM_WORLD.rank == 0:
            print("t", end="", flush=True)
    best_config = MPI.COMM_WORLD.bcast(best_config, root=0)
    if MPI.COMM_WORLD.rank == 0:
        print(best_config, end="", flush=True)
    return best_config, best_time


def run_benchmark(
    mscclpp_group: mscclpp_comm.CommGroup, nccl_op: nccl.NcclCommunicator, table: PrettyTable, niter: int, nelem: int
):
    memory = cp.zeros(nelem, dtype=data_type)
    memory_out = cp.zeros(nelem, dtype=data_type)
    cp.cuda.runtime.deviceSynchronize()

    proxy_service = ProxyService()
    if MPI.COMM_WORLD.size // N_GPUS_PER_NODE == 1:
        if memory.nbytes < 2**20:
            mscclpp_algos = [MscclppAllReduce2(mscclpp_group, memory, memory_out)]
        else:
            mscclpp_algos = [
                MscclppAllReduce1(mscclpp_group, memory),
                MscclppAllReduce3(mscclpp_group, memory, proxy_service),
            ]
            if is_nvls_supported():
                mscclpp_algos.append(MscclppAllReduce6(mscclpp_group, nelem, data_type))
    else:
        if memory.nbytes < 2**22:
            mscclpp_algos = [MscclppAllReduce5(mscclpp_group, memory, memory_out, N_GPUS_PER_NODE, proxy_service)]
        else:
            mscclpp_algos = [MscclppAllReduce4(mscclpp_group, memory, N_GPUS_PER_NODE, proxy_service)]

    proxy_service.start_proxy()
    MPI.COMM_WORLD.barrier()
    mscclpp_call = find_best_algo(mscclpp_algos, 20)
    if isinstance(mscclpp_call, MscclppAllReduce6):
        memory = mscclpp_call.get_memory()

    nccl_call = NcclAllReduce(nccl_op, memory)

    memory_nbytes = memory.nbytes
    mscclpp_time = bench_time(niter, mscclpp_call)
    mscclpp_algBw = memory_nbytes / mscclpp_time / 1e3
    mscclpp_check = "PASS" if check_correctness(memory, mscclpp_call) else "FAIL"

    nccl_time = bench_time(niter, nccl_call)
    nccl_algBw = memory_nbytes / nccl_time / 1e3
    nccl_check = "PASS" if check_correctness(memory, nccl_call) else "FAIL"

    MPI.COMM_WORLD.barrier()
    proxy_service.stop_proxy()

    speed_up = nccl_time / mscclpp_time
    if MPI.COMM_WORLD.rank == 0:
        table.add_row(
            [
                human_readable_size(memory_nbytes),
                "{:.2f}".format(mscclpp_time),
                "{:.2f}".format(mscclpp_algBw),
                mscclpp_check,
                "{:.2f}".format(nccl_time),
                "{:.2f}".format(nccl_algBw),
                nccl_check,
                "{:.2f}".format(speed_up),
            ]
        )
    if MPI.COMM_WORLD.rank == 0:
        print(".", end="", flush=True)

    return memory.nbytes, mscclpp_algBw, nccl_algBw, speed_up


if __name__ == "__main__":
    shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    N_GPUS_PER_NODE = shm_comm.size
    shm_comm.Free()
    cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()

    # create a MscclppGroup
    network_interface = "eth0"
    my_ip = ni.ifaddresses(network_interface)[ni.AF_INET][0]["addr"]
    root_ip = MPI.COMM_WORLD.bcast(my_ip, root=0)
    ifIpPortTrio = network_interface + ":" + root_ip + ":50000"  # some random port
    mscclpp_group = mscclpp_comm.CommGroup(
        interfaceIpPortTrio=ifIpPortTrio, rank=MPI.COMM_WORLD.rank, size=MPI.COMM_WORLD.size
    )

    # create a NcclComm
    if MPI.COMM_WORLD.rank == 0:
        uid = nccl.get_unique_id()
    else:
        uid = None
    uid = MPI.COMM_WORLD.bcast(uid, root=0)
    nccl_comm = nccl.NcclCommunicator(MPI.COMM_WORLD.size, uid, MPI.COMM_WORLD.rank)

    table = None
    if MPI.COMM_WORLD.rank == 0:
        # Set table headers
        table = PrettyTable()
        table.field_names = [
            f"Size ({dtype_str})",
            "Time (us)",
            "AlgBW (GB/s)",
            "Correctness",
            "NCCL Time (us)",
            "NCCL AlgBW (GB/s)",
            "NCCL Correctness",
            "Speed Up",
        ]

    sizes = []
    mscclpp_algbw = []
    nccl_algbw = []
    speed_ups = []
    end_range = 28 if is_nvls_supported() else 29
    for i in range(10, end_range):
        if MPI.COMM_WORLD.size // N_GPUS_PER_NODE == 1:
            nelems = 2**i
        elif MPI.COMM_WORLD.size // N_GPUS_PER_NODE == 2:
            nelems = 3 * 2**i
        else:
            raise RuntimeError("Only support one node/two nodes communication")

        if nelems * data_type().itemsize > 2**32:
            break  # due to trigger bit width limitation, we can only support up to 2**32

        size, mscclpp_algBw, nccl_algBw, speed_up = run_benchmark(mscclpp_group, nccl_comm, table, 100, nelems)
        sizes.append(size)
        mscclpp_algbw.append(mscclpp_algBw)
        nccl_algbw.append(nccl_algBw)
        speed_ups.append(speed_up)

    if MPI.COMM_WORLD.rank == 0:
        print()
        print(table)

        plot_graph(sizes, mscclpp_algbw, nccl_algbw, speed_ups)

    mscclpp_group = None
    nccl_comm = None
