docker run --gpus all --net=host --privileged -v /dev/shm:/dev/shm --name cluster_kan -v ~/framework-test:/code -v ~/kmeans:/kmeans -it nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

sysctl -w kernel.io_uring_disabled=0
sysctl -w vm.nr_hugepages=65536
apt install liburing-dev
apt update && apt install wget git
apt-get install git-lfs
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
chmod +x ./Anaconda3-2024.02-1-Linux-x86_64.sh
./Anaconda3-2024.02-1-Linux-x86_64.sh
source ~/.bashrc

conda install -c gurobi gurobi

pip install torch
pip install cmake
pip install nvtx
pip install loguru
pip install transformers
apt install pybind11-dev

cd /code/Nanoflow-python
git submodule init
git submodule update

# install nsight
NSIGHT="NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb"
if [[ ! -f "$NSIGHT" ]]; then
  wget https://developer.download.nvidia.com/devtools/nsight-systems/$NSIGHT
  dpkg -i ./$NSIGHT
fi

cd ./3rdparty/flashinfer
git submodule init
git submodule update
FLASHINFER_ENABLE_AOT=1 pip install -e . -v

cd ../../pybind
mkdir build
cd build/
cmake ..
make -j

cd ../../entry
python run_llama3.py