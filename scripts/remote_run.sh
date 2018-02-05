#!/bin/bash

username=larryzhang
remote_machine=aniseed

placenta_dir=/data/vision/polina/projects/placenta_segmentation/
startup_script=${placenta_dir}/scripts/run.sh

###################

remote_ssh=${username}@${remote_machine}.csail.mit.edu

run_cmd="ssh -t ${remote_ssh} '${startup_script}; cd ${placenta_dir}; bash -l'"

eval ${run_cmd}
