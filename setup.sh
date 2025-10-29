git submodule update --init --recursive

conda install -c gurobi gurobi

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install nvtx
pip install loguru
pip install transformers
pip install accelerate
pip install matplotlib
pip install nvmath-python
pip install flashinfer-python


cd ..
# install nsight
NSIGHT="NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb"
if [[ ! -f "$NSIGHT" ]]; then
  wget https://developer.download.nvidia.com/devtools/nsight-systems/$NSIGHT
  dpkg -i ./$NSIGHT
fi

# install cmake 3.29.0
CMAKE_INSTALLER="cmake-3.29.0-rc2-linux-x86_64.sh"
if [[ ! -f "$CMAKE_INSTALLER" ]]; then
  wget https://github.com/Kitware/CMake/releases/download/v3.29.0-rc2/$CMAKE_INSTALLER
  chmod +x ./$CMAKE_INSTALLER
fi
./$CMAKE_INSTALLER --prefix=/usr/local --exclude-subdir

cd Nanoflow-python
pip install -e . -v

# cd 3rdparty/cutlass
# git checkout main
# cd ../..

# # build flashinfer
# cd 3rdparty/flashinfer
# python -m pip install -e . -v
# # FLASHINFER_ENABLE_AOT=1 python -m pip install -v .
# cd ../..

# build mscclpp
cd 3rdparty/mscclpp
git reset --hard cdaf3aea3d767ba65dd3b08984d76bd50615f92e
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/mscclpp -DBUILD_PYTHON_BINDINGS=OFF ..
make -j mscclpp mscclpp_static
make install/fast
cd ../../../

cd nanoflow

# build kernels
cd pybind
mkdir build
cd build/
cmake ..
make -j 256
cd ../..

# load llama3-8B weights
cd core
python weightSaver.py --config_path=../config_all/llama3-8B/1024.json

# run llama3-8B model
cd ../entry
sh setup_test.sh