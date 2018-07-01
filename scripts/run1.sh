#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
python_exe=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/bin/python

###################

cd ${placenta_dir}

args="CUDA_VISIBLE_DEVICES=1"
nohup ${args} ${python_exe} train.py "$@" > nohup1.out 2> nohup1.err < /dev/null &
