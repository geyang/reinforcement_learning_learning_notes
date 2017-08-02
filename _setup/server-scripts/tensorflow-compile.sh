#!/usr/bin/env bash
# Only the compilation step for tensorflow is in this script, for clarity.

git clone https://github.com/tensorflow/tensorflow
cd ./tensorflow
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-8.0/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu/
export PYTHON_BIN_PATH="$HOME/.conda/envs/gym/bin/python"
export PYTHON_LIB_PATH="$HOME/.conda/envs/gym/lib/python3.5/site-packages"
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_ENABLE_XLA=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=0
export TF_NEED_MPI=0
# MPI is not working.
export MPI_HOME="/usr/lib/openmpi/include/openmpi"
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
export CUDA_VERSION='8.0'
export CUDNN_VERSION='6'
export CUDNN_INSTALL_PATH=/usr/local/cuda
export CUDA_COMPUTE_CAPABILITIES='3.7'
export CUDA_PATH='/usr/local/cuda'
export CUDA_PATH_LINUX='/opt/cuda'
yes "" | ./configure
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k //tensorflow/tools/pip_package:build_pip_package && \
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
