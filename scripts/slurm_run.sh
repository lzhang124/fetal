#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation

###################

cd ${placenta_dir}
name=$1
shift
rm -rf models/${name}
rm -rf logs/${name}
srun -p gpu -t 20:00:00 --mem-per-cpu 16 --gres=gpu:1 -J ${name} -o ${name}.out -e ${name}.err scripts/run.sh "$@" --name ${name} &
