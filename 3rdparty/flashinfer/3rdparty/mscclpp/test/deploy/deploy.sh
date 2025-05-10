set -e

KeyFilePath=${SSHKEYFILE_SECUREFILEPATH}
ROOT_DIR="${SYSTEM_DEFAULTWORKINGDIRECTORY}/"
DST_DIR="/tmp/mscclpp"
HOSTFILE="${SYSTEM_DEFAULTWORKINGDIRECTORY}/test/deploy/hostfile"
SSH_OPTION="StrictHostKeyChecking=no"

chmod 400 ${KeyFilePath}
ssh-keygen -t rsa -f sshkey -P ""

while true; do
  set +e
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "hostname"
  if [ $? -eq 0 ]; then
    break
  fi
  echo "Waiting for sshd to start..."
  sleep 5
done

set -e
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "rm -rf ${DST_DIR}"
parallel-scp -t 0 -r -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION ${ROOT_DIR} ${DST_DIR}

# force to pull the latest image
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker pull ${CONTAINERIMAGE}"
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker run --rm -itd --privileged --net=host --ipc=host --gpus=all \
  -w /root -v ${DST_DIR}:/root/mscclpp -v /opt/microsoft:/opt/microsoft --name=mscclpp-test \
  --entrypoint /bin/bash ${CONTAINERIMAGE}"
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker exec -t --user root mscclpp-test bash '/root/mscclpp/test/deploy/setup.sh'"

