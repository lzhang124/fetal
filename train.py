import os
import logging
logging.basicConfig(level=logging.INFO)

import constants
import glob
import time
from argparse import ArgumentParser
from data import AugmentGenerator, VolSliceAugmentGenerator, VolumeGenerator
from models import UNet


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train',
                        metavar='INPUT_FILES, LABEL_FILES',
                        help='Train model',
                        dest='train', type=str, nargs=2)
    parser.add_argument('-p', '--predict',
                        metavar='INPUT_FILES, [SEED_FILES,] SAVE_PATH',
                        help='Predict segmentations',
                        dest='predict', type=str, nargs='+')
    parser.add_argument('-s', '--seed',
                        help='Seed slices',
                        dest='seed', action='store_true')
    parser.add_argument('-b', '--batch-size',
                        metavar='BATCH_SIZE',
                        help='Training batch size',
                        dest='batch_size', type=int, default=1)
    parser.add_argument('-e', '--epochs',
                        metavar='EPOCHS',
                        help='Training epochs',
                        dest='epochs', type=int, default=1000)
    parser.add_argument('-n', '--name',
                        metavar='MODEL_NAME',
                        help='Name of model',
                        dest='name', type=str)
    parser.add_argument('-f', '--model-file',
                        metavar='MODEL_FILE',
                        help='Pretrained model file',
                        dest='model_file', type=str)
    parser.add_argument('--gpu',
                        metavar='GPU',
                        help='GPU to use',
                        dest='gpu', type=str, default='0')
    parser.add_argument('--sample-predict', dest='sample_predict', action='store_true')
    parser.add_argument('--sample-test', dest='sample_test', type=str)
    return parser


def main(options):
    start = time.time()

    logging.info('Compiling model.')
    if options.seed:
        shape = tuple(list(constants.TARGET_SHAPE[:-1]) + [constants.TARGET_SHAPE[-1] + 1])
    else:
        shape = constants.TARGET_SHAPE
    model = UNet(shape, name=options.name, filename=options.model_file)

    if options.train:
        logging.info('Creating data generator.')

        input_path = options.train[0].split('*')[0]
        label_path = options.train[1].split('*')[0]

        label_files = glob.glob(options.train[1])
        input_files = [label_file.replace(label_path, input_path) for label_file in label_files]

        generator = VolSliceAugmentGenerator if options.seed else AugmentGenerator
        aug_gen = generator(input_files, label_files, options.batch_size)

        logging.info('Training model.')
        model.train(aug_gen, options.epochs)

    if options.predict:
        logging.info('Making predictions.')
        
        input_files = glob.glob(options.predict[0])
        seed_files = glob.glob(options.predict[1]) if options.seed else None
        save_path = options.predict[2] if options.seed else options.predict[1]

        generator = VolSliceGenerator if options.seed else VolumeGenerator
        pred_gen = generator(input_files, seed_files, options.batch_size)
        model.predict(pred_gen, save_path)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


def seed_predict(options):
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
    m = UNet(shape, 1e-4, filename='models/UNET_SEED_1222_boundary.h5')
    p = m.model.predict(x1222_0)[0]
    util.save_vol(process.uncrop(p, o1222), 'data/predict/122215/zero-1222_boundary_24.nii.gz', header=h1222)
    p = m.model.predict(x1222_s)[0]
    util.save_vol(process.uncrop(p, o1222), 'data/predict/122215/seed-1222_boundary_24.nii.gz', header=h1222)
    p = m.model.predict(x0430_0)[0]
    util.save_vol(process.uncrop(p, o0430), 'data/predict/043015/zero-1222_boundary_24.nii.gz', header=h0430)
    p = m.model.predict(x0430_s)[0]
    util.save_vol(process.uncrop(p, o0430), 'data/predict/043015/seed-1222_boundary_24.nii.gz', header=h0430)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


def seed_test(options):
    start = time.time()

    logging.info('Compiling model.')
    if options.seed:
        shape = tuple(list(constants.TARGET_SHAPE[:-1]) + [constants.TARGET_SHAPE[-1] + 1])
    else:
        shape = constants.TARGET_SHAPE
    model = UNet(shape, name=options.name, filename=options.model_file)

    number = options.sample_test

    logging.info('Creating data generator.')
    input_files = ['data/raw/{}/{}_1.nii.gz'.format(number, number)]
    label_files = ['data/labels/{}/{}_1_placenta.nii.gz'.format(number, number)]
    aug_gen = VolSliceAugmentGenerator(input_files, label_files, options.batch_size)

    logging.info('Training model.')
    model.train(aug_gen, options.epochs)

    logging.info('Making predictions.')
    seed_files = glob.glob('data/labels/{}/{}_*.nii.gz'.format(number, number))
    predict_files = [file.replace('labels', 'raw') for file in seed_files]
    pred_gen = VolSliceGenerator(predict_files, seed_files, options.batch_size)
    model.predict(pred_gen, 'data/predict/{}/')

    logging.info('Testing model.')
    test_files = [file.replace('labels', 'labels').replace('.nii.gz', '_placenta.nii.gz')
                  for file in seed_files]
    test_gen = zip(pred_gen, VolumeGenerator(test_files, None, options.batch_size, False))
    performance = model.test(test_gen)
    logging.info(performance)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    if options.sample_predict:
        seed_predict(options)
    elif options.sample_test:
        seed_test(options)
    else:
        main(options)
