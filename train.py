import os
import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model',
                    help='Model architecture',
                    dest='model', type=str, required=True)
parser.add_argument('--organ',
                    help='Organ to segment',
                    dest='organ', type=str, required=True)
parser.add_argument('--epochs',
                    help='Training epochs',
                    dest='epochs', type=int, default=1000)
parser.add_argument('--name',
                    help='Name of model',
                    dest='name', type=str, required=True)
parser.add_argument('--model-file',
                    help='Pretrained model file',
                    dest='model_file', type=str)
parser.add_argument('--load-files',
                    help='Load files',
                    dest='load_files', action='store_true')
parser.add_argument('--skip-training',
                    help='Skip training',
                    dest='skip_training', action='store_true')
parser.add_argument('--validate',
                    help='Create validation set',
                    dest='validate', action='store_true')
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
        pass
        # logging.info('Temporal model.')
        # logging.info('Splitting data.')
        # samples = list(constants.GOOD_FRAMES.keys())
        # n = len(samples)
        # shuffled = np.random.permutation(samples)
        # if options.validate:
        #     train = shuffled[:(2*n)//3]
        #     val = shuffled[(2*n)//3:(5*n)//6]
        #     test = shuffled[(5*n)//6:]
        # else:
        #     train = shuffled[:(9*n)//10]
        #     test = shuffled[(9*n)//10:]

        # logging.info('Creating data generators.')
        # label_types = LABELS[options.model]
        # train_for = []
        # train_rev = []
        # train_label_for = []
        # train_label_rev = []
        # weight_labels = []
        # for s in train:
        #     frames = constants.GOOD_FRAMES[s]
        #     train_for.extend([f'data/raw/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames if i != 0])
        #     train_rev.extend([f'data/raw/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames if i != 0])
        #     train_label_for.extend([f'data/predict_cleaned/{options.temporal}/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames if i != 0])
        #     train_label_rev.extend([f'data/predict_cleaned/{options.temporal}/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames if i != 0])
        #     weight_labels.extend(glob.glob(f'data/labels/{s}/{s}_{constants.LABELED_FRAME[s]}_{organ}.nii.gz'))
        # train_gen = AugmentGenerator(train_for + train_rev,
        #                              label_files=train_label_for + train_label_rev,
        #                              concat_files=[train_rev + train_for, train_label_rev + train_label_for],
        #                              label_types=label_types,
        #                              load_files=options.load_files)
        # weights = util.get_weights(weight_labels)

        # val_gen = None
        # if not options.skip_training and options.validate:
        #     val_for = []
        #     val_rev = []
        #     val_label_for = []
        #     val_label_rev = []
        #     for s in val:
        #         frames = constants.GOOD_FRAMES[s]
        #         val_for.extend([f'data/raw/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames if i != 0])
        #         val_rev.extend([f'data/raw/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames if i != 0])
        #         val_label_for.extend([f'data/predict_cleaned/{options.temporal}/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames if i != 0])
        #         val_label_rev.extend([f'data/predict_cleaned/{options.temporal}/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames if i != 0])
        #     val_gen = VolumeGenerator(val_for + val_rev,
        #                               label_files=val_label_for + val_label_rev,
        #                               concat_files=[val_rev + val_for, val_label_rev + val_label_for],
        #                               label_types=label_types,
        #                               load_files=options.load_files)

        # if not options.predict_all:
        #     test_for = []
        #     test_rev = []
        #     test_label_for = []
        #     test_label_rev = []
        #     for s in test:
        #         frames = constants.GOOD_FRAMES[s]
        #         test_for.extend([f'data/raw/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames if i != 0])
        #         test_rev.extend([f'data/raw/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames if i != 0])
        #         test_label_for.extend([f'data/predict_cleaned/{options.temporal}/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames if i != 0])
        #         test_label_rev.extend([f'data/predict_cleaned/{options.temporal}/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames if i != 0])
        #     pred_gen = VolumeGenerator(test_for + test_rev, tile_inputs=True, load_files=options.load_files)
        #     test_gen = VolumeGenerator(test_for + test_rev,
        #                                label_files=test_label_for + test_label_rev,
        #                                concat_files=[test_rev + test_for, test_label_rev + test_label_for],
        #                                label_types=label_types,
        #                                load_files=options.load_files)

        # logging.info('Creating model.')
        # shape = constants.SHAPE[:-1] + (3,)
        # model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)
    else:
        logging.info('Non-temporal model.')
        logging.info('Splitting data.')
        n = len(constants.LABELED_SAMPLES)
        shuffled = np.random.permutation(constants.LABELED_SAMPLES)
        if options.validate:
            train = shuffled[:(2*n)//3]
            val = shuffled[(2*n)//3:(5*n)//6]
            test = shuffled[(5*n)//6:]
        else:
            train = shuffled[:(9*n)//10]
            test = shuffled[(9*n)//10:]

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
        if not options.skip_training and options.validate:
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
        shape = constants.SHAPE
        model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)

    if not options.skip_training:
        logging.info('Training model.')
        model.train(train_gen, val_gen, options.epochs)

    logging.info('Making predictions.')
    model.predict(pred_gen, f'data/predict/{options.name}/')

    if not options.predict_all:
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
