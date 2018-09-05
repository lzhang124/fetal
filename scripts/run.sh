#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
python_exe=${placenta_dir}/venv/bin/python

###################

cd ${placenta_dir}

gpu=$1
args="--gpu $1"
shift
nohup ${python_exe} train.py ${args} "$@" > nohup${gpu}.out 2> nohup${gpu}.err < /dev/null &
