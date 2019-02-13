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
parser.add_argument('--skip-training',
                    help='Skip training',
                    dest='skip_training', action='store_true')
parser.add_argument('--predict-all',
                    help='Predict all samples',
                    dest='predict_all', action='store_true')
parser.add_argument('--temporal',
                    help='Temporal segmentation',
                    dest='temporal', action='store_true')
options = parser.parse_args()

import constants
import datetime
import glob
import numpy as np
import time
import util
from data import AugmentGenerator, VolumeGenerator
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

    np.random.seed(123454321)

    organ = 'all_brains' if options.organ == 'brains' else options.organ

    if options.temporal:
        logging.info('Splitting data.')
        samples = list(constants.GOOD_FRAMES.keys())
        n = len(samples)
        shuffled = np.random.permutation(samples)
        train = shuffled[:(2*n)//3]
        val = shuffled[(2*n)//3:(5*n)//6]
        test = shuffled[(5*n)//6:]

        logging.info('Creating data generators.')
        label_types = LABELS[options.model]
        train_for = []
        train_rev = []
        train_label_for = []
        train_label_rev = []
        for s in train:
            frames = constants.GOOD_FRAMES[s]
            train_for.extend([f'data/raw/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames])
            train_rev.extend([f'data/raw/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames])
            train_label_for.extend([f'data/predict_cleaned/unet3000/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames])
            train_label_rev.extend([f'data/predict_cleaned/unet3000/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames])
        train_gen = AugmentGenerator(train_for + train_rev,
                                     label_files=train_label_for + train_label_rev,
                                     concat_files=[[train_rev + train_for], [train_label_rev + train_label_for]],
                                     label_types=label_types,
                                     load_files=False)
        weights = util.get_weights(train_gen.labels)

        if not options.skip_training:
            val_for = []
            val_rev = []
            val_label_for = []
            val_label_rev = []
            for s in val:
                frames = constants.GOOD_FRAMES[s]
                val_for.extend([f'data/raw/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames])
                val_rev.extend([f'data/raw/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames])
                val_label_for.extend([f'data/predict_cleaned/unet3000/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames])
                val_label_rev.extend([f'data/predict_cleaned/unet3000/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames])
            val_gen = VolumeGenerator(val_for + val_rev,
                                      label_files=val_label_for + val_label_rev,
                                      concat_files=[[val_rev + val_for], [val_label_rev + val_label_for]],
                                      label_types=label_types,
                                      load_files=False)

        if options.predict_all:
            pass
        else:
            test_for = []
            test_rev = []
            test_label_for = []
            test_label_rev = []
            for s in test:
                frames = constants.GOOD_FRAMES[s]
                test_for.extend([f'data/raw/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames])
                test_rev.extend([f'data/raw/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames])
                test_label_for.extend([f'data/predict_cleaned/unet3000/{s}/{s}_{str(i).zfill(4)}.nii.gz' for i in frames])
                test_label_rev.extend([f'data/predict_cleaned/unet3000/{s}/{s}_{str(i-1).zfill(4)}.nii.gz' for i in frames])
            pred_gen = VolumeGenerator(test_for + test_rev, tile_inputs=True, load_files=False)
            test_gen = VolumeGenerator(test_for + test_rev,
                                       label_files=test_label_for + test_label_rev,
                                       concat_files=[[test_rev + test_for], [test_label_rev + test_label_for]],
                                       label_types=label_types,
                                       load_files=False)

        logging.info('Creating model.')
        shape = constants.SHAPE[:-1] + (3,)
        model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)
    else:
        logging.info('Splitting data.')
        n = len(constants.SAMPLES)
        shuffled = np.random.permutation(constants.SAMPLES)
        train = shuffled[:(2*n)//3]
        val = shuffled[(2*n)//3:(5*n)//6]
        test = shuffled[(5*n)//6:]

        logging.info('Creating data generators.')
        label_types = LABELS[options.model]
        train_files = [f'data/raw/{sample}/{sample}_0000.nii.gz' for sample in train]
        train_label_files = [f'data/labels/{sample}/{sample}_0_{organ}.nii.gz' for sample in train]
        train_gen = AugmentGenerator(train_files, label_files=train_label_files, label_types=label_types)
        weights = util.get_weights(train_gen.labels)

        if not options.skip_training:
            val_files = [f'data/raw/{sample}/{sample}_0000.nii.gz' for sample in val]
            val_label_files = [f'data/labels/{sample}/{sample}_0_{organ}.nii.gz' for sample in val]
            val_gen = VolumeGenerator(val_files, label_files=val_label_files, label_types=label_types)

        if options.predict_all:
            pass
        else:
            test_files = [f'data/raw/{sample}/{sample}_0000.nii.gz' for sample in test]
            test_label_files = [f'data/labels/{sample}/{sample}_0_{organ}.nii.gz' for sample in test]
            pred_gen = VolumeGenerator(test_files, tile_inputs=True)
            test_gen = VolumeGenerator(test_files, label_files=test_label_files, label_types=label_types)

        logging.info('Creating model.')
        shape = constants.SHAPE
        model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)

    if not options.skip_training:
        logging.info('Training model.')
        model.train(train_gen, val_gen, options.epochs)

    if options.predict_all:
        for folder in glob.glob('data/raw/*'):
            try:
                sample = folder.split('/')[-1]
                logging.info(f'{sample}..............................')
                if options.temporal:
                    # TODO
                    pass
                else:
                    pred_files = glob.glob(f'data/raw/{sample}/{sample}_*.nii.gz')
                    pred_gen = VolumeGenerator(pred_files, tile_inputs=True)
                    model.predict(pred_gen, f'data/predict/{options.name}/{sample}/')
            except Exception as e:
                logging.error(f'ERROR during {sample}: {e}')
    else:
        logging.info('Making predictions.')
        model.predict(pred_gen, f'data/predict/{options.name}/')

        logging.info('Testing model.')
        metrics = model.test(test_gen)
        logging.info(metrics)
        dice = {}
        for i in range(len(test)):
            sample = test[i]
            dice[sample] = util.dice_coef(util.read_vol(test_label_files[i]), util.read_vol(f'data/predict/{options.name}/{sample}_0000.nii.gz'))
        logging.info(metrics)
        logging.info(np.mean(list(metrics.values())))

    end = time.time()
    logging.info(f'total time: {datetime.timedelta(seconds=(end - start))}')


if __name__ == '__main__':
    main(options)
