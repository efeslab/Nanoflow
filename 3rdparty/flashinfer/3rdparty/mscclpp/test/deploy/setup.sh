set -e

mkdir -p /root/.ssh
mv /root/mscclpp/sshkey.pub /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
mv /root/mscclpp/test/deploy/config /root/.ssh/config
chown root:root /root/.ssh/config
chmod 400 /root/mscclpp/sshkey
chown root:root /root/mscclpp/sshkey

nvidia-smi -pm 1
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
    nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
done

if [[ "${CUDA_VERSION}" == *"11."* ]]; then
    pip3 install -r /root/mscclpp/python/requirements_cuda11.txt
else
    pip3 install -r /root/mscclpp/python/requirements_cuda12.txt
fi

cd /root/mscclpp && pip3 install .

mkdir -p /var/run/sshd
/usr/sbin/sshd -p 22345
