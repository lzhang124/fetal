import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
logging.basicConfig(level=logging.INFO)

import constants
import time
from argparse import ArgumentParser
from data import AugmentGenerator, VolSliceGenerator, VolumeGenerator
from models import UNet


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train',
                        metavar=('INPUT_FILES', 'LABEL_FILES'), help='Train model',
                        dest='train', type=str, nargs='+')
    parser.add_argument('-p', '--predict',
                        metavar=('INPUT_FILES', 'SAVE_PATH'), help='Predict segmentations',
                        dest='predict', type=str, nargs=2)
    parser.add_argument('-s', '--seed',
                        help='Seed slices',
                        dest='seed', action='store_true')
    parser.add_argument('-b', '--batch-size',
                        metavar='BATCH_SIZE', help='Training batch size',
                        dest='batch_size', type=int, default=1)
    parser.add_argument('-e', '--epochs',
                        metavar='EPOCHS', help='Training epochs',
                        dest='epochs', type=int, default=100)
    parser.add_argument('-n', '--name',
                        metavar='MODEL_NAME', help='Name of model',
                        dest='name', type=str)
    parser.add_argument('-f', '--model-file',
                        metavar='MODEL_FILE', help='Pretrained model file',
                        dest='model_file', type=str)
    return parser


def main():
    start = time.time()

    parser = build_parser()
    options = parser.parse_args()

    logging.info('Compiling model.')
    if options.seed:
        shape = tuple(list(constants.TARGET_SHAPE[:-1]) + [constants.TARGET_SHAPE[-1] + 1])
    else:
        shape = constants.TARGET_SHAPE
    model = UNet(shape, name=options.name, filename=options.model_file)

    if options.train:
        logging.info('Creating data generator.')
        labels = None if len(options.train) < 2 else options.train[1]
        generator = VolSliceGenerator if options.seed else AugmentGenerator
        aug_gen = generator(options.train[0], labels, options.batch_size)

        logging.info('Training model.')
        model.train(aug_gen, options.epochs)

    if options.predict:
        logging.info('Making predictions.')
        pred_gen = VolumeGenerator(options.predict[0], options.batch_size)
        model.predict(pred_gen, options.predict[1])

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


def seed_test():
    import numpy as np
    import nibabel as nib
    import process
    import util
    start = time.time()

    o1222 = util.shape('data/raw/122215/122215_24.nii.gz')
    h1222 = nib.load('data/raw/122215/122215_24.nii.gz').header
    o0430 = util.shape('data/raw/043015/043015_24.nii.gz')
    h0430 = nib.load('data/raw/043015/043015_24.nii.gz').header
    x1222 = process.preprocess('data/raw/122215/122215_24.nii.gz')
    x1222 = x1222[np.newaxis, :]
    x0430 = process.preprocess('data/raw/043015/043015_24.nii.gz')
    x0430 = x0430[np.newaxis, :]
    s1222 = process.preprocess('data/seeds/122215/122215_24.nii.gz', ['resize'])
    s1222 = s1222[np.newaxis, :]
    s0430 = process.preprocess('data/seeds/043015/043015_24.nii.gz', ['resize'])
    s0430 = s0430[np.newaxis, :]
    x1222_0 = np.concatenate((x1222, np.zeros(x1222.shape)), axis=-1)
    x0430_0 = np.concatenate((x0430, np.zeros(x0430.shape)), axis=-1)
    x1222_s = np.concatenate((x1222, s1222), axis=-1)
    x0430_s = np.concatenate((x0430, s0430), axis=-1)

    shape = tuple(list(constants.TARGET_SHAPE[:-1]) + [constants.TARGET_SHAPE[-1] + 1])
    m = UNet(shape, 1e-4, filename='models/UNET_SEED.760-0.1353.h5')
    p = m.model.predict(x1222_0)[0]
    util.save_vol(process.uncrop(p, o1222), 'data/predict/122215/zero_24-0.14.nii.gz', header=h1222)
    p = m.model.predict(x1222_s)[0]
    util.save_vol(process.uncrop(p, o1222), 'data/predict/122215/seed_24-0.14.nii.gz', header=h1222)
    p = m.model.predict(x0430_0)[0]
    util.save_vol(process.uncrop(p, o0430), 'data/predict/043015/zero_24-0.14.nii.gz', header=h0430)
    p = m.model.predict(x0430_s)[0]
    util.save_vol(process.uncrop(p, o0430), 'data/predict/043015/seed_24-0.14.nii.gz', header=h0430)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    main()
    # seed_test()
