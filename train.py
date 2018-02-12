import numpy as np
import time
from argparse import ArgumentParser
from preprocess import AugmentGenerator
# from models import UNet


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

    aug_gen = AugmentGenerator(options.vol_files, options.seg_files, batch_size=options.batch_size)
    print(aug_gen.shape)

    # train
    # model = UNet()

    end = time.time()
    print('total time:', end - start)


if __name__ == '__main__':
    main()
