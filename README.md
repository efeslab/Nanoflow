# Nanoflow

## Docker setup

``` bash
mkdir -p ~/framework-test
docker run --gpus all --net=host --privileged -v /dev/shm:/dev/shm --name nanoflow -v ~/framework-test:/code -it nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

apt update
apt install pybind11-dev
apt install liburing-dev
apt install libopenmpi-dev
sysctl -w kernel.io_uring_disabled=0
sysctl -w vm.nr_hugepages=65536
```

## Gurobi License Setup (for Docker)

Follow these steps to obtain a Gurobi license and configure it so your Docker container can use it.

### 1. Request a Gurobi License

1. Go to the Gurobi website and create an account (https://www.gurobi.com/).
2. After logging in, navigate to **My Gurobi → Get License**.
3. Choose the "WLS Academic" license type and fill out any required fields.
4. Gurobi will email you a license file named `gurobi.lic` (or provide you with a license key string).

### 2. Place the License on Your Host Machine
``` bash
mkdir -p ~/gurobi/license
mv /path/to/downloaded/gurobi.lic ~/gurobi/license/
ls ~/gurobi/license
```

## Install Dependencies

``` bash
git clone git@github.com:efeslab/Nanoflow.git
cd Nanoflow
chmod +x ./installAnaconda.sh
./installAnaconda.sh
# restart the terminal
source ~/.bashrc

# login to huggingface
mkdir -p hf
echo "export HF_HOME=$(pwd)/hf" >> ~/.bashrc
source ~/.bashrc

huggingface-cli login

# set up the environment
cd Nanoflow-python
yes | bash setup.sh
```

## Build

``` bash
cd pybind
mkdir -p build
cmake ..
make -j 256
```

## End-to-end Test

``` bash
cd entry
python run_llama3.py -load_hf_weight=True
```
