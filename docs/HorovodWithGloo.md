# How to install Horovod with gloo only

Horovod is a distributed machine learning training framework for TensorFlow, Keras and more.
See [Horovod](https://github.com/horovod/horovod).

By default Horovod relies on MPI for collective algorithms. An alternative to MPI is [Gloo](https://github.com/facebookincubator/gloo). By compiling Horovod with gloo only, you don't need MPI to be available on workers.

## Requirements
* cmake, g++-4.8 and python-dev to compile horovod and gloo.
* You also need a python environment with wheel and tensorflow installed. Caution: make sure you installed the same version of tensorflow as you'll use later. tensorflow 1.14

## Build
Gloo only works on linux for now
1. git clone https://github.com/horovod/horovod.git --recursive
2. HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_MPI=1 HOROVOD_WITH_TENSORFLOW=1 python setup.py bdist_wheel -p linux-x86_64
3. pip install dist/horovod-*.whl

## Run
python examples/collective_all_reduce_example.py 
