set -e

MSCCLPP_SRC_DIR="/mnt/mscclpp"
NPKIT_RUN_DIR="/mnt/npkit_run"
MPI_HOME="/usr/local/mpi"
HOSTFILE="hostfile"
LEADER_IP_PORT="10.6.0.4:50000"

cd ${MSCCLPP_SRC_DIR}
make clean
MPI_HOME=${MPI_HOME} make -j NPKIT=1

parallel-ssh -h ${HOSTFILE} "rm -rf ${NPKIT_RUN_DIR}"
parallel-ssh -h ${HOSTFILE} "mkdir -p ${NPKIT_RUN_DIR}"
parallel-scp -r -h ${HOSTFILE} ${MSCCLPP_SRC_DIR} ${NPKIT_RUN_DIR}
parallel-ssh -h ${HOSTFILE} "mkdir -p ${NPKIT_RUN_DIR}/npkit_dump"
parallel-ssh -h ${HOSTFILE} "mkdir -p ${NPKIT_RUN_DIR}/npkit_trace"

# --bind-to numa is required because hardware timer from different cores (or core groups) can be non-synchronized.
mpirun --allow-run-as-root -hostfile ${HOSTFILE} -map-by ppr:8:node --bind-to numa -x LD_PRELOAD=${NPKIT_RUN_DIR}/mscclpp/build/lib/libmscclpp.so -x MSCCLPP_DEBUG=WARN -x NPKIT_DUMP_DIR=${NPKIT_RUN_DIR}/npkit_dump ${NPKIT_RUN_DIR}/mscclpp/build/bin/tests/allgather_test -ip_port ${LEADER_IP_PORT} -kernel 0

parallel-ssh -h ${HOSTFILE} "cd ${NPKIT_RUN_DIR}/mscclpp/tools/npkit && python npkit_trace_generator.py --npkit_dump_dir ${NPKIT_RUN_DIR}/npkit_dump --npkit_event_header_path ${NPKIT_RUN_DIR}/mscclpp/src/include/npkit/npkit_event.h --output_dir ${NPKIT_RUN_DIR}/npkit_trace"
