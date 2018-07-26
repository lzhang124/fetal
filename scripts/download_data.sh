#!/bin/bash

username=larryzhang
remote_machine=aniseed

placenta_dir=/data/vision/polina/projects/placenta_segmentation/

###################

remote_ssh=${username}@${remote_machine}.csail.mit.edu

run_cmd="rsync -avP ${remote_ssh}:${placenta_dir}/data/$1/* ./data/$1/"

eval ${run_cmd}
