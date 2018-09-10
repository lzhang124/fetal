#!/bin/bash

username=larryzhang
remote_machine=cumin

placenta_dir=/data/vision/polina/projects/placenta_segmentation

###################

rsync -avP ./data/$1/* ${username}@${remote_machine}.csail.mit.edu:${placenta_dir}/data/$1/
