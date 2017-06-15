#!/usr/bin/env bash
# export CUDA_HOME=$SHARED_ROOT/cuda-7.5
# export LD_LIBRARY_PATH=$SHARED_ROOT/cuda-7.5/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$SHARED_ROOT/cudnn-5.0/lib64:$LD_LIBRARY_PATH
# export SYS_NAME=SLURM

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR_TMP

