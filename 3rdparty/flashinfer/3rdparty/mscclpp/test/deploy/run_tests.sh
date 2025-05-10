set -e
HOSTFILE=/root/mscclpp/test/deploy/hostfile_mpi

function run_mscclpp_test()
{
  echo "=================Run allgather_test_perf on 2 nodes========================="
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  # For kernel 2, the message size must can be divided by 3
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 3K -e 3G -f 2 -k 2 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allgather_test_perf -b 1K -e 1G -f 2 -k 3 -o /root/mscclpp/output.jsonl

  echo "==================Run allreduce_test_perf on 2 nodes========================="
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1G -f 2 -k 1 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 1K -e 1M -f 2 -k 2 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 3K -e 3G -f 2 -k 3 -o /root/mscclpp/output.jsonl

  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/allreduce_test_perf -b 3K -e 3G -f 2 -k 4 -o /root/mscclpp/output.jsonl

  echo "==================Run alltoall_test_perf on 2 nodes========================="
  /usr/local/mpi/bin/mpirun --allow-run-as-root -np 16 --bind-to numa -hostfile ${HOSTFILE} \
    -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
    -npernode 8 /root/mscclpp/build/test/mscclpp-test/alltoall_test_perf -b 1K -e 1G -f 2 -k 0 -o /root/mscclpp/output.jsonl

  echo "========================Run performance check==============================="
  python3 /root/mscclpp/test/mscclpp-test/check_perf_result.py --perf-file /root/mscclpp/output.jsonl \
    --baseline-file /root/mscclpp/test/deploy/perf_ndmv4.jsonl
}

function run_mp_ut()
{
  echo "============Run multi-process unit tests on 2 nodes (np=2, npernode=1)========================="
  /usr/local/mpi/bin/mpirun -allow-run-as-root -tag-output -np 2 --bind-to numa \
  -hostfile ${HOSTFILE} -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 1 /root/mscclpp/build/test/mp_unit_tests -ip_port mscclit-000000:20003

  echo "============Run multi-process unit tests on 2 nodes (np=16, npernode=8)========================="
  /usr/local/mpi/bin/mpirun -allow-run-as-root -tag-output -np 16 --bind-to numa \
  -hostfile ${HOSTFILE} -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -npernode 8 /root/mscclpp/build/test/mp_unit_tests -ip_port mscclit-000000:20003
}

function run_pytests()
{
  echo "==================Run python tests================================"
  /usr/local/mpi/bin/mpirun -allow-run-as-root -tag-output -np 16 --bind-to numa \
  -hostfile ${HOSTFILE} -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -x MSCCLPP_HOME=/root/mscclpp -npernode 8 bash /root/mscclpp/test/deploy/pytest.sh
}

function run_py_benchmark()
{
  echo "==================Run python benchmark================================"
  /usr/local/mpi/bin/mpirun -allow-run-as-root -np 16 --bind-to numa \
  -hostfile ${HOSTFILE} -x MSCCLPP_DEBUG=WARN -x LD_LIBRARY_PATH=/root/mscclpp/build:$LD_LIBRARY_PATH \
  -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 \
  -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml \
  -x NCCL_NET_PLUGIN=none -x NCCL_IB_DISABLE=0 -x NCCL_MIN_NCHANNELS=32 -x NCCL_DEBUG=WARN -x NCCL_P2P_DISABLE=0 -x NCCL_SHM_DISABLE=0 \
  -x MSCCLPP_HOME=/root/mscclpp -np 16 -npernode 8 python3 /root/mscclpp/python/mscclpp_benchmark/allreduce_bench.py
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <mscclpp-test/mp-ut/run_pytests/run_py_benchmark>"
    exit 1
fi
test_name=$1
case $test_name in
  mscclpp-test)
    echo "==================Run mscclpp-test on 2 nodes========================="
    run_mscclpp_test
    ;;
  mp-ut)
    echo "==================Run mp-ut on 2 nodes================================"
    run_mp_ut
    ;;
  pytests)
    echo "==================Run python tests===================================="
    run_pytests
    ;;
  py-benchmark)
    echo "==================Run python benchmark================================"
    run_py_benchmark
    ;;
  *)
    echo "Unknown test name: $test_name"
    exit 1
    ;;
esac
