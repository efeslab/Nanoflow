# init submodule dependencies
git submodule init
git submodule update

# install dependencies
apt update
apt install python3
pip3 install cmake
apt install libopenmpi-dev
apt install wget
pip install torch
apt install libspdlog-dev
apt-get install libglib2.0-0
apt install pigz
pip install wget
pip install pandas
pip install seaborn
pip install mypy
pip install transformers
pip install --upgrade pydantic
pip install sentencepiece
apt-get install git-lfs
apt-get install python3-pybind11

# fix pybind header compile error
sed -i '446,486s/^/\/\//' /usr/include/pybind11/detail/type_caster_base.h

# install cmake 3.29.0
cd ..
wget https://github.com/Kitware/CMake/releases/download/v3.29.0-rc2/cmake-3.29.0-rc2-linux-x86_64.sh
chmod +x cmake-3.29.0-rc2-linux-x86_64.sh
./cmake-3.29.0-rc2-linux-x86_64.sh --prefix=/usr/local --exclude-subdir


# install nsight
NSIGHT="NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb"
if [[ ! -f "$NSIGHT" ]]; then
  wget https://developer.download.nvidia.com/devtools/nsight-systems/$NSIGHT
  dpkg -i ./$NSIGHT
fi

cd Nanoflow

# fix mscclpp
sed -i '256d' 3rdparty/mscclpp/src/bootstrap/bootstrap.cc

# fix spdlog v1.14.0 + cuda 12.1 compatibility bug
for repo in spdlog; do
  cat 3rdparty/patches/${repo}/*.patch | patch -p1 -d 3rdparty/${repo}
done


cd pipeline

# download and trace visualizer
cd utils
curl -LO https://get.perfetto.dev/trace_processor
chmod +x ./trace_processor
cd ..

# generate gemm lib
cd src/generate-gemm
python3 genGEMM.py
cd ../../

# build pllm
mkdir -p build
cd build
cmake ..
make -j 256

# set up libstdc++.so.6 directory

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

./test_compute 8 8

