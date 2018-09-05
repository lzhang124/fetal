#!/bin/bash

placenta_dir=/data/vision/polina/projects/placenta_segmentation/

###################

n=$1
shift
cd ${placenta_dir}
for i in {1..$n}
do
    srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J part$i -o run$i.out -e run$i.err scripts/run.sh --part $i "$@" &
done
