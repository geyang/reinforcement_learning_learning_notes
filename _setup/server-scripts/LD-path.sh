#!/usr/bin/env bash

(echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"';\
echo 'export CUDA_HOME=/usr/local/cuda';\
cat ~/.bashrc) > ~/.bashrc.tmp
mv ~/.bashrc.tmp ~/.bashrc
