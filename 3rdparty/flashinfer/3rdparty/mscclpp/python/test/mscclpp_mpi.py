# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
import logging

import cupy as cp
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from mpi4py import MPI
import pytest

N_GPUS_PER_NODE = 8

logging.basicConfig(level=logging.INFO)


def init_mpi():
    if not MPI.Is_initialized():
        MPI.Init()
        shm_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
        N_GPUS_PER_NODE = shm_comm.size
        shm_comm.Free()
        cp.cuda.Device(MPI.COMM_WORLD.rank % N_GPUS_PER_NODE).use()


# Define a function to finalize MPI
def finalize_mpi():
    if MPI.Is_initialized():
        MPI.Finalize()


# Register the function to be called on exit
atexit.register(finalize_mpi)


class MpiGroup:
    def __init__(self, ranks: list = []):
        world_group = MPI.COMM_WORLD.group
        if len(ranks) == 0:
            self.comm = MPI.COMM_WORLD
        else:
            group = world_group.Incl(ranks)
            self.comm = MPI.COMM_WORLD.Create(group)


@pytest.fixture
def mpi_group(request: pytest.FixtureRequest):
    MPI.COMM_WORLD.barrier()
    if request.param is None:
        pytest.skip(f"Skip for rank {MPI.COMM_WORLD.rank}")
    yield request.param


def parametrize_mpi_groups(*tuples: tuple):
    def decorator(func):
        mpi_groups = []
        for group_size in list(tuples):
            if MPI.COMM_WORLD.size < group_size:
                logging.warning(f"MPI.COMM_WORLD.size < {group_size}, skip")
                continue
            mpi_group = MpiGroup(list(range(group_size)))
            if mpi_group.comm == MPI.COMM_NULL:
                mpi_groups.append(None)
            else:
                mpi_groups.append(mpi_group)
        return pytest.mark.parametrize("mpi_group", mpi_groups, indirect=True)(func)

    return decorator


init_mpi()
