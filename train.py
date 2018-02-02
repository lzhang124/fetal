import glob
import nibabel as nib
import numpy as np
import re
from argparse import ArgumentParser
from preprocess import augment_generator

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--vols',
            dest='vol_files', help='Training volumes folder',
            type=str, required=True)
    parser.add_argument('--segs',
            dest='seg_files', help='Training segmentations folder',
            type=str, required=True)
    return parser

def main():
    # parser = build_parser()
    # options = parser.parse_args()

    seg_files = glob.glob('data/labels/043015/*_placenta.nii.gz')
    vol_files = []
    for seg_file in seg_files:
        p, a = re.split('/|_', seg_file)[-3:-1]
        vol_file = 'data/raw/{p}/{p}_{a}.nii.gz'.format(p=p, a=a)
        vol_files.append(vol_file)

    segs = np.array(volread(file) for file in seg_files)
    vols = np.array(volread(file) for file in vol_files)

    aug_gen = augment_generator(vols, segs)
    aug_vol, aug_seg = next(aug_gen)
    nib.save(aug_vol, 'data/test/vol.nii.gz')
    nib.save(aug_seg, 'data/test/seg.nii.gz')

def volread(filename):
    return nib.load(filename).get_data()

if __name__ == '__main__':
    main()
