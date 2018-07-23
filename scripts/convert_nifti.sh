#!/bin/bash

mkdir data/nifti/$1
/data/vision/polina/shared_software/anaconda3-4.3.1/envs/keras/bin/dicom2nifti data/PlacentaData/$2 data/nifti/$1
