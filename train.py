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
parser.add_argument('--predict-all',
                    help='Predict all samples',
                    dest='predict_all', action='store_true')
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

    val_files = [f'data/raw/{sample}/{sample}_0000.nii.gz' for sample in val]
    val_label_files = [f'data/labels/{sample}/{sample}_0_{organ}.nii.gz' for sample in val]
    val_gen = VolumeGenerator(val_files, label_files=val_label_files, label_types=label_types)

    test_files = [f'data/raw/{sample}/{sample}_0000.nii.gz' for sample in test]
    test_label_files = [f'data/labels/{sample}/{sample}_0_{organ}.nii.gz' for sample in test]
    pred_gen = VolumeGenerator(test_files, tile_inputs=True)
    test_gen = VolumeGenerator(test_files, label_files=test_label_files, label_types=label_types)

    logging.info('Creating model.')
    shape = constants.SHAPE
    model = MODELS[options.model](shape, name=options.name, filename=options.model_file, weights=weights)

    if options.predict_all:
        for folder in glob.glob('data/raw/*'):
            try:
                sample = folder.split('/')[-1]
                logging.info(f'{sample}..............................')
                pred_files = glob.glob(f'data/raw/{sample}/{sample}_*.nii.gz')
                pred_gen = VolumeGenerator(pred_files, tile_inputs=True)
                model.predict(pred_gen, f'data/predict/{options.name}/{sample}/')
            except Exception as e:
                logging.error(f'ERROR during {sample}:')
                logging.error(e)

    else:
        logging.info('Training model.')
        model.train(train_gen, val_gen, options.epochs)

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
