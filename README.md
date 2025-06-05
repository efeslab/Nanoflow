# Nanoflow

## docker setup

``` bash
mkdir -p ~/framework-test
docker run --gpus all --net=host --privileged -v /dev/shm:/dev/shm --name nanoflow -v ~/framework-test:/code -it nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

## install dependencies

``` bash
git clone git@github.com:serendipity-zk/pllm.git
cd Nanoflow-python
chmod +x ./installAnaconda.sh
./installAnaconda.sh
# restart the terminal
source ~/.bashrc

cd Nanoflow-python
yes | bash setup.sh
```

## build

``` bash
cd pybind
mkdir -p build
cmake ..
make -j 256
```

## end-to-end test

``` bash
cd entry
python run_llama3.py
```
