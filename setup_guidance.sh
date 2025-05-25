apt update && apt upgrade -y
apt install pybind11-dev
apt install liburing-dev
apt install libopenmpi-dev
apt-get install nvidia-cuda-toolkit
sysctl -w kernel.io_uring_disabled=0
sysctl -w vm.nr_hugepages=65536

git submodule init
git submodule update

conda install -c gurobi gurobi

pip install torch==2.7.0 torchvision==0.22.0+cu128 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install nvtx
pip install loguru
pip install transformers

cd ..
# install cmake 3.29.0
CMAKE_INSTALLER="cmake-3.29.0-rc2-linux-x86_64.sh"
if [[ ! -f "$CMAKE_INSTALLER" ]]; then
  wget https://github.com/Kitware/CMake/releases/download/v3.29.0-rc2/$CMAKE_INSTALLER
  chmod +x ./$CMAKE_INSTALLER
fi
./$CMAKE_INSTALLER --prefix=/usr/local --exclude-subdir

# install nsight
NSIGHT="NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb"
if [[ ! -f "$NSIGHT" ]]; then
  wget https://developer.download.nvidia.com/devtools/nsight-systems/$NSIGHT
  dpkg -i ./$NSIGHT
fi

cd Nanoflow-python

cd ./3rdparty/cutlass
git checkout main
cd ../..

# build flashinfer
cd ./3rdparty/flashinfer
git submodule init
git submodule update
FLASHINFER_ENABLE_AOT=1 pip install -e . -v
cd ../..

# build mscclpp
cd ./3rdparty/mscclpp
git reset --hard cdaf3aea3d767ba65dd3b08984d76bd50615f92e
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/mscclpp -DBUILD_PYTHON_BINDINGS=OFF ..
make -j mscclpp mscclpp_static
make install/fast
cd ../../../

# build kernels
cd ./pybind
mkdir build
cd build/
cmake ..
make -j 256
cd ../..

# run tests
cd ./entry
python run_llama3.py

