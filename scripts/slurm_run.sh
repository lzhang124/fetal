#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation

###################

cd ${placenta_dir}
i=$1
shift
t=$1
shift
srun -p gpu -t ${t}:00:00 --mem-per-cpu 1 --gres=gpu:1 -J run${i} -o run${i}.out -e run${i}.err scripts/run.sh "$@" &
