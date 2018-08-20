#!/bin/bash

username=larryzhang
remote_machine=aniseed

placenta_dir=/data/vision/polina/projects/placenta_segmentation

###################

scripts/download_data.sh nifti/$1/
python -B split_nifti.py data/nifti/$1/
scripts/upload_data.sh originals/$1/
