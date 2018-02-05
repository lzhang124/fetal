import glob
import nibabel as nib
import numpy as np
import re
from argparse import ArgumentParser
from preprocess import augment_generator

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--vols',
            dest='vol_files', help='Training volume files',
            type=str, default='data/raw/04*/*_1.nii.gz')
    parser.add_argument('--segs',
            dest='seg_files', help='Training segmentation files',
            type=str, default='data/labels/04*/*_1_placenta.nii.gz')
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    vol_path = options.vol_files.split('*/*')
    seg_path = options.seg_files.split('*/*')
    
    seg_files = glob.glob(options.seg_files)
    vol_files = [seg_file.replace(seg_path[0], vol_path[0]).replace(seg_path[1], vol_path[1])
                 for seg_file in seg_files]

    vols = np.array([volread(file) for file in vol_files])
    segs = np.array([volread(file) for file in seg_files])

    aug_gen = augment_generator(vols, segs)
    aug_vol, aug_seg = next(aug_gen)
    nib.Nifti1Image(aug_vol, np.eye(4)).to_filename('data/test/vol.nii.gz')
    nib.Nifti1Image(aug_seg, np.eye(4)).to_filename('data/test/seg.nii.gz')

def volread(filename):
    return np.squeeze(nib.load(filename).get_data())

if __name__ == '__main__':
    main()
