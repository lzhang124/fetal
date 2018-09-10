#!/bin/bash

username=larryzhang
remote_machine=cumin

placenta_dir=/data/vision/polina/projects/placenta_segmentation

###################

rsync -avP ${username}@${remote_machine}.csail.mit.edu:${placenta_dir}/data/$1/* ./data/$1/
