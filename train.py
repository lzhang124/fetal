import os
import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--train',
                    metavar='INPUT_FILES, LABEL_FILES',
                    help='Train model',
                    dest='train', type=str, nargs=2)
parser.add_argument('--predict',
                    metavar='INPUT_FILES, SAVE_PATH',
                    help='Predict segmentations',
                    dest='predict', type=str, nargs=2)
parser.add_argument('--test',
                    metavar='INPUT_FILES, LABEL_FILES',
                    help='Test model',
                    dest='test', type=str, nargs=2)
parser.add_argument('--model',
                    metavar='Model',
                    help='Model',
                    dest='model', type=str, required=True)
parser.add_argument('--organ',
                    metavar='ORGAN',
                    help='Organ to segment',
                    dest='organ', type=str, required=True)
parser.add_argument('--epochs',
                    metavar='EPOCHS',
                    help='Training epochs',
                    dest='epochs', type=int, default=1000)
parser.add_argument('--name',
                    metavar='MODEL_NAME',
                    help='Name of model',
                    dest='name', type=str, required=True)
parser.add_argument('--model-file',
                    metavar='MODEL_FILE',
                    help='Pretrained model file',
                    dest='model_file', type=str)
parser.add_argument('--gpu',
                    metavar='GPU',
                    help='Which GPU to use',
                    dest='gpu', type=str)
parser.add_argument('--run',
                    dest='run', action='store_true')
options = parser.parse_args()

if options.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

import constants
import datetime
import glob
import numpy as np
import time
import util
from data import AugmentGenerator, VolumeGenerator
from models import UNet, UNetSmall, ACNN, AESeg


MODELS = {
    'unet': UNet,
    'unet-small': UNetSmall,
    'acnn': ACNN,
    'aeseg': AESeg,
}


def main(options):
    start = time.time()

    logging.info('Creating model.')
    shape = constants.SHAPE
    model = MODELS[options.model](shape, name=options.name, filename=options.model_file)

    if options.train:
        logging.info('Creating data generator.')

        input_path = options.train[0].split('*')[0]
        label_path = options.train[1].split('*')[0]

        label_files = glob.glob(options.train[1])
        input_files = [label_file.replace(label_path, input_path) for label_file in label_files]

        aug_gen = AugmentGenerator(input_files, label_files=label_files)
        val_gen = VolumeGenerator(input_files, label_files=label_files, load_files=True, include_labels=True)

        logging.info('Compiling model.')
        model.compile(util.get_weights(aug_gen.labels))

        logging.info('Training model.')
        model.train(aug_gen, val_gen, options.epochs)
        model.save()

    if options.predict:
        logging.info('Making predictions.')

        input_files = glob.glob(options.predict[0])
        save_path = options.predict[1]

        pred_gen = VolumeGenerator(input_files, include_labels=False)
        model.predict(pred_gen, save_path)

    if options.test:
        logging.info('Testing model.')

        input_files = glob.glob(options.test[0])
        label_files = glob.glob(options.test[1])

        test_gen = VolumeGenerator(input_files, label_files=label_files, include_labels=True)
        metrics = model.test(test_gen)
        logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(datetime.timedelta(seconds=(end - start))))


def run(options):
    np.random.seed(123454321)
    start = time.time()
    metrics = {}
    organ = 'all_brains' if options.organ == 'brains' else options.organ

    logging.info('Splitting data.')
    n = len(constants.SAMPLES)
    shuffled = np.random.permutation(constants.SAMPLES)
    train = shuffled[:(2*n)//3]
    val = shuffled[(2*n)//3:(5*n)//6]
    test = shuffled[(5*n)//6:]

    logging.info('Creating model.')
    shape = constants.SHAPE
    model = MODELS[options.model](shape, name=options.name, filename=options.model_file)

    logging.info('Creating data generators.')
    train_files = ['data/raw/{}/{}_0000.nii.gz'.format(sample, sample) for sample in train]
    train_label_files = ['data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ) for sample in train]
    train_gen = AugmentGenerator(train_files, label_files=train_label_files)
    
    val_files = ['data/raw/{}/{}_0000.nii.gz'.format(sample, sample) for sample in val]
    val_label_files = ['data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ) for sample in val]
    val_gen = VolumeGenerator(val_files, label_files=val_label_files, load_files=True, include_labels=True)

    test_files = ['data/raw/{}/{}_0000.nii.gz'.format(sample, sample) for sample in test]
    test_label_files = ['data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ) for sample in test]
    pred_gen = VolumeGenerator(test_files, include_labels=False)
    test_gen = VolumeGenerator(test_files, label_files=test_label_files, include_labels=True)

    logging.info('Compiling model.')
    model.compile(weight=util.get_weights(train_gen.labels))

    logging.info('Training model.')
    if options.model == 'acnn':
        model.train_ae(train_gen, val_gen, options.epochs)
        model.save_ae()
    model.train(train_gen, val_gen, options.epochs)

    logging.info('Saving model.')
    model.save()

    logging.info('Making predictions.')
    model.predict(pred_gen, 'data/predict/{}/'.format(options.name))

    logging.info('Testing model.')
    metrics = model.test(test_gen)
    logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(datetime.timedelta(seconds=(end - start))))


if __name__ == '__main__':
    if options.run:
        run(options)
    else:
        main(options)
