#!/bin/bash

export CUDA_HOME=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
python_exe=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/bin/python

###################

cd ${placenta_dir}

gpu=$1
args="--gpu $1"
shift
nohup ${python_exe} train.py ${args} "$@" > nohup${gpu}.out 2> nohup${gpu}.err < /dev/null &
