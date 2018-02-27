import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import time
from argparse import ArgumentParser
from data import AugmentGenerator, VolumeGenerator
from models import UNet


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train',
                        metavar=('VOL_FILES', 'SEG_FILES'), help='Train model',
                        dest='train', type=str, nargs=2)
    parser.add_argument('-p', '--predict',
                        metavar=('PRED_FILES', 'SAVE_PATH'), help='Predict segmentations',
                        dest='predict', type=str, nargs=2)
    parser.add_argument('-b', '--batch-size',
                        metavar='BATCH_SIZE', help='Training batch size',
                        dest='batch_size', type=int, default=1)
    parser.add_argument('-e', '--epochs',
                        metavar='EPOCHS', help='Training epochs',
                        dest='epochs', type=int, default=100)
    parser.add_argument('-m', '--model',
                        metavar='MODEL_FILE', help='Pretrained model file',
                        dest='model', type=str)
    return parser


def main():
    start = time.time()

    parser = build_parser()
    options = parser.parse_args()

    logging.info('Compiling model.')
    model = UNet(options.model)

    if options.train:
        logging.info('Creating data generator.')
        aug_gen = AugmentGenerator(options.train[0], options.train[1], options.batch_size)

        logging.info('Training model.')
        model.train(aug_gen, options.epochs)

    if options.predict:
        logging.info('Making predictions.')
        pred_gen = VolumeGenerator(options.predict[0], options.batch_size)
        model.predict(pred_gen, options.predict[1])

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    main()
