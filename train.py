import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import time
from argparse import ArgumentParser
from preprocess import AugmentGenerator, VolumeGenerator
from models import UNet


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--vols', dest='vol_files', help='Training volume files',
                        type=str, default='data/raw/04*/*.nii.gz')
    parser.add_argument('--segs', dest='seg_files', help='Training segmentation files',
                        type=str, default='data/labels/04*/*_placenta.nii.gz')
    parser.add_argument('--batch-size', dest='batch_size', help='Training batch size',
                        type=int, default=1)
    parser.add_argument('--model', dest='model_file', help='Pretrained model file',
                        type=str)
    parser.add_argument('--train', dest='train', help='Train model',
                        action='store_true')
    parser.add_argument('--predict', dest='pred_files', help='Prediction volume files',
                        type=str)
    return parser


def main():
    start = time.time()

    parser = build_parser()
    options = parser.parse_args()

    logging.info('Creating data generator.')
    aug_gen = AugmentGenerator(options.vol_files, options.seg_files, options.batch_size, save_to_dir="data/test/")

    logging.info('Compiling model.')
    model = UNet(aug_gen.shape, options.model_file)

    if options.train:
        logging.info('Training model.')
        model.train(aug_gen)

    if options.pred_files:
        logging.info('Making predictions.')
        pred_gen = VolumeGenerator(options.pred_files, options.batch_size)
        model.predict(pred_gen)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    main()
