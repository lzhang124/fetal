import os
import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--name',
                    help='Name of model',
                    dest='name', type=str, required=True)
parser.add_argument('--model',
                    help='Model architecture',
                    dest='model', type=str, required=True)
parser.add_argument('--organ',
                    help='Organ to segment',
                    dest='organ', type=str, required=True)
parser.add_argument('--epochs',
                    help='Training epochs',
                    dest='epochs', type=int, default=1000)
parser.add_argument('--split',
                    help='Train and validation split',
                    dest='split', type=float, nargs=2, default=[2/3, 1/6])
parser.add_argument('--model-file',
                    help='Pretrained model file',
                    dest='model_file', type=str)
parser.add_argument('--load-files',
                    help='Load files',
                    dest='load_files', action='store_true')
parser.add_argument('--skip-training',
                    help='Skip training',
                    dest='skip_training', action='store_true')
parser.add_argument('--predict-all',
                    help='Predict all samples',
                    dest='predict_all', action='store_true')
parser.add_argument('--temporal',
                    help='Temporal segmentation using predictions from model name',
                    dest='temporal', type=str)
parser.add_argument('--good-frames',
                    help='Train using good frames from model name',
                    dest='good_frames', type=str)
options = parser.parse_args()

import constants
import datetime
import glob
import numpy as np
import time
import util
from data import DataGenerator
from models import UNet, UNetSmall, AESeg


MODELS = {
    'unet': UNet,
    'unet-small': UNetSmall,
    'aeseg': AESeg,
}

LABELS = {
    'unet': ['label'],
    'unet-small': ['label'],
    'aeseg': ['label', 'input'],
}


def main(options):
    start = time.time()

    np.random.seed(123456789)

    organ = 'all_brains' if options.organ == 'brains' else options.organ

    if options.temporal:
        logging.info('Temporal model.')
        logging.info('Splitting data.')
        samples = list(constants.GOOD_FRAMES.keys())
        n = len(samples)
        shuffled = np.random.permutation(samples)
        assert np.sum(options.split) < 1, 'Split is greater than 1.'
        train_split = int(options.split[0] * n)
        val_split = int(np.sum(options.split) * n)
        train = shuffled[:train_split]
        val = shuffled[train_split:val_split]
        test = shuffled[val_split:]
        assert len(test) > 0, 'No test data.'

        logging.info('Creating data generators.')
        label_types = LABELS[options.model]
        train_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in train},
                                  'data/raw/{s}/{s}_{n}.nii.gz',
                                  f'data/labels/{{s}}/{{s}}_{{n}}_{organ}.nii.gz',
                                  label_types=label_types,
                                  load_files=options.load_files,
                                  augment=True)
        logging.info(f'  Training generator with {len(train_gen)} samples.')
        weights = util.get_weights(train_gen.labels)

        val_gen = None
        if not options.skip_training and len(val) > 0:
            val_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in val},
                                    'data/raw/{s}/{s}_{n}.nii.gz',
                                    f'data/labels/{{s}}/{{s}}_{{n}}_{organ}.nii.gz',
                                    label_types=label_types,
                                    load_files=options.load_files,
                                    resize=True)
            logging.info(f'  Validation generator with {len(val_gen)} samples.')

        if options.predict_all:
            pred_gen = DataGenerator({s: np.arange(n) for _, (s, n) in enumerate(constants.SEQ_LENGTH.items())},
                                     'data/raw/{s}/{s}_{n}.nii.gz',
                                     load_files=False,
                                     tile_inputs=True)
            logging.info(f'  Prediction generator with {len(pred_gen)//8} samples.')
        else:
            pred_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in test},
                                     'data/raw/{s}/{s}_{n}.nii.gz',
                                     load_files=options.load_files,
                                     tile_inputs=True)
            logging.info(f'  Prediction generator with {len(pred_gen)//8} samples.')
        
        test_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in test},
                                 'data/raw/{s}/{s}_{n}.nii.gz',
                                 f'data/labels/{{s}}/{{s}}_{{n}}_{organ}.nii.gz',
                                 label_types=label_types,
                                 load_files=options.load_files,
                                 resize=True)
        logging.info(f'  Testing generator with {len(test_gen)} samples.')

        logging.info('Creating model.')
        shape = constants.SHAPE[:-1] + (3,)
        model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)
    else:
        logging.info('Non-temporal model.')
        logging.info('Splitting data.')
        input_file_format = 'data/raw/{s}/{s}_{n}.nii.gz'
        if options.good_frames:
            samples = list(constants.GOOD_FRAMES.keys())
            n = len(samples)
            shuffled = np.random.permutation(samples)
            label_file_format = f'data/predict_cleaned/{options.good_frames}/{{s}}/{{s}}_{{n}}.nii.gz'
        else:
            n = len(constants.LABELED_SAMPLES)
            shuffled = np.random.permutation(constants.LABELED_SAMPLES)
            input_file_format = 'data/raw/{s}/{s}_{n}.nii.gz'
            label_file_format = f'data/labels/{{s}}/{{s}}_{{n}}_{organ}.nii.gz'
        assert np.sum(options.split) < 1, 'Split is greater than 1.'
        train_split = int(options.split[0] * n)
        val_split = int(np.sum(options.split) * n)
        train = shuffled[:train_split]
        val = shuffled[train_split:val_split]
        test = shuffled[val_split:]
        assert len(test) > 0, 'No test data.'

        logging.info('Creating data generators.')
        label_types = LABELS[options.model]
        train_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in train},
                                  input_file_format,
                                  label_file_format,
                                  label_types=label_types,
                                  load_files=options.load_files,
                                  augment=True)
        logging.info(f'  Training generator with {len(train_gen)} samples.')
        weights = util.get_weights(train_gen.labels)

        val_gen = None
        if not options.skip_training and len(val) > 0:
            val_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in val},
                                    input_file_format,
                                    label_file_format,
                                    label_types=label_types,
                                    load_files=options.load_files,
                                    resize=True)
            logging.info(f'  Validation generator with {len(val_gen)} samples.')

        if options.predict_all:
            pred_gen = DataGenerator({s: np.arange(n) for _, (s, n) in enumerate(constants.SEQ_LENGTH.items())},
                                     'data/raw/{s}/{s}_{n}.nii.gz',
                                     load_files=False,
                                     tile_inputs=True)
            logging.info(f'  Prediction generator with {len(pred_gen)//8} samples.')
        else:
            pred_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in test},
                                     input_file_format
                                     load_files=options.load_files,
                                     tile_inputs=True)
            logging.info(f'  Prediction generator with {len(pred_gen)//8} samples.')
        
        test_gen = DataGenerator({s: [constants.LABELED_FRAME[s]] for s in test},
                                 input_file_format,
                                 label_file_format,
                                 label_types=label_types,
                                 load_files=options.load_files,
                                 resize=True)
        logging.info(f'  Testing generator with {len(test_gen)} samples.')

        logging.info('Creating model.')
        shape = constants.SHAPE
        model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)

    if not options.skip_training:
        logging.info('Training model.')
        model.train(train_gen, val_gen, options.epochs)

    logging.info('Making predictions.')
    model.predict(pred_gen)

    logging.info('Testing model.')
    metrics = model.test(test_gen)
    logging.info(metrics)
    
    dice = {}
    for i in range(len(test)):
        sample = test[i]
        dice[sample] = util.dice_coef(util.read_vol(f'data/labels/{sample}/{sample}_{str(constants.LABELED_FRAME[sample]).zfill(4)}_{organ}.nii.gz'),
                                      util.read_vol(f'data/predict/{options.name}/{sample}/{sample}_{str(constants.LABELED_FRAME[sample]).zfill(4)}.nii.gz'))
    logging.info(dice)

    end = time.time()
    logging.info(f'total time: {datetime.timedelta(seconds=(end - start))}')


if __name__ == '__main__':
    main(options)
