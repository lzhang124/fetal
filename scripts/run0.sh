#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
python_exe=/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/bin/python

###################

cd ${placenta_dir}

nohup ${python_exe} train.py "$@" --gpu 0 > nohup0.out 2> nohup0.err < /dev/null &
