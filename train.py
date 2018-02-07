import glob
import nibabel as nib
import numpy as np
import re
import time
from argparse import ArgumentParser
from preprocess import augment_generator


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--vols', dest='vol_files', help='Training volume files',
            type=str, default='data/raw/04*/*.nii.gz')
    parser.add_argument('--segs', dest='seg_files', help='Training segmentation files',
            type=str, default='data/labels/04*/*_placenta.nii.gz')
    parser.add_argument('--batch-size', dest='batch_size', help='Training batch size',
            type=int, default=32)
    return parser


def main():
    start = time.time()

    parser = build_parser()
    options = parser.parse_args()

    aug_gen = augment_data(options.vol_files, options.seg_files, options.batch_size)
    aug_vols, aug_segs = next(aug_gen)
    for i in range(aug_vols.shape[0]):
        volsave(aug_vols[i], 'data/test/vol_{}.nii.gz'.format(i))
        volsave(aug_segs[i], 'data/test/seg_{}.nii.gz'.format(i))

    end = time.time()
    print('total time:', end - start)


def augment_data(vol_files, seg_files, batch_size):
    vol_path = vol_files.split('*/*')
    seg_path = seg_files.split('*/*')
    
    seg_files = glob.glob(seg_files)
    vol_files = [seg_file.replace(seg_path[0], vol_path[0]).replace(seg_path[1], vol_path[1])
                 for seg_file in seg_files]

    vols = np.array([volread(file) for file in vol_files] * (batch_size // len(seg_files) + 1))
    segs = np.array([volread(file) for file in seg_files] * (batch_size // len(seg_files) + 1))

    return augment_generator(vols, segs, batch_size)


def volread(filename):
    return np.squeeze(nib.load(filename).get_data())


def volsave(vol, filename):
    nib.Nifti1Image(np.squeeze(vol).astype('int16'), np.eye(4)).to_filename(filename)


if __name__ == '__main__':
    main()
