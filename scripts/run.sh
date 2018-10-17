#!/bin/bash

export HOME=/data/vision/polina/projects/placenta_segmentation
export CUDA_HOME=$HOME/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"

placenta_dir=$HOME
python_exe=${placenta_dir}/venv/bin/python

###################

cd ${placenta_dir}
${python_exe} train.py "$@" & tensorboard --logdirs=logs/
