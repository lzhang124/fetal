#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
python_exe=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/bin/python

###################

cd ${placenta_dir}

args="--gpu1"
nohup ${python_exe} train.py ${args} "$@" > nohup1.out 2> nohup1.err < /dev/null &
