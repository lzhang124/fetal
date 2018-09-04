#!/bin/bash

export CUDA_HOME=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"
echo $CUDA_HOME

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
python_exe=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/bin/python

###################

cd ${placenta_dir}
${python_exe} train.py "$@"
