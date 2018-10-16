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
                    metavar='INPUT_FILES, SEED_FILES/LABEL_FILES, SAVE_PATH',
                    help='Predict segmentations',
                    dest='predict', type=str, nargs=3)
parser.add_argument('--test',
                    metavar='INPUT_FILES, [SEED_FILES,] LABEL_FILES',
                    help='Test model',
                    dest='test', type=str, nargs='+')
parser.add_argument('--model',
                    metavar='Model',
                    help='Model',
                    dest='model', type=str)
parser.add_argument('--organ',
                    metavar='ORGAN',
                    help='Organ to segment',
                    dest='organ', type=str, nargs=1)
parser.add_argument('--seed',
                    metavar='SEED_TYPE',
                    help='Seed slices',
                    dest='seed', type=str)
parser.add_argument('--concat',
                    metavar='INPUT_FILE, LABEL_FILE',
                    help='Concatenate first volume',
                    dest='concat', nargs=2)
parser.add_argument('--epochs',
                    metavar='EPOCHS',
                    help='Training epochs',
                    dest='epochs', type=int, default=1000)
parser.add_argument('--name',
                    metavar='MODEL_NAME',
                    help='Name of model',
                    dest='name', type=str)
parser.add_argument('--model-file',
                    metavar='MODEL_FILE',
                    help='Pretrained model file',
                    dest='model_file', type=str)
parser.add_argument('--gpu',
                    metavar='GPU',
                    help='Which GPU to use',
                    dest='gpu', type=str, nargs=1)
parser.add_argument('--run',
                    dest='run', action='store_true')
parser.add_argument('--tensorboard',
                    dest='tensorboard', action='store_true')
options = parser.parse_args()

if options.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu[0]

import constants
import glob
import numpy as np
import time
import util
from data import AugmentGenerator, VolumeGenerator
from models import UNet, UNetSmall, ACNN, AESeg


MODELS = {
    'unet': Unet,
    'unet-small': UNetSmall,
    'acnn': ACNN,
    'aeseg': AESeg,
}


def main(options):
    start = time.time()

    logging.info('Creating model.')
    shape = constants.SHAPE
    if options.seed:
        shape = tuple(list(shape[:-1]) + [shape[-1] + 1])
    if options.concat:
        shape = tuple(list(shape[:-1]) + [shape[-1] + 2])
    model = MODELS[options.model](shape, name=options.name, filename=options.model_file)

    gen_seed = (options.seed == 'slice' or options.seed == 'volume')

    if options.train:
        logging.info('Creating data generator.')

        input_path = options.train[0].split('*')[0]
        label_path = options.train[1].split('*')[0]

        label_files = glob.glob(options.train[1])
        input_files = [label_file.replace(label_path, input_path) for label_file in label_files]

        aug_gen = AugmentGenerator(input_files,
                                   label_files=label_files,
                                   seed_type=options.seed,
                                   concat_files=options.concat)
        #FIXME
        val_gen = VolumeGenerator(input_files,
                                  label_files=label_files,
                                  seed_type=options.seed,
                                  concat_files=options.concat,
                                  load_files=True,
                                  include_labels=True)

        logging.info('Compiling model.')
        model.compile(util.get_weights(aug_gen.labels))

        logging.info('Training model.')
        model.train(aug_gen, val_gen, options.epochs, tensorboard=options.tensorboard)
        model.save()

    if options.predict:
        logging.info('Making predictions.')

        input_files = glob.glob(options.predict[0])
        seed_files = None if gen_seed else glob.glob(options.predict[1])
        label_files = glob.glob(options.predict[1]) if gen_seed else None
        save_path = options.predict[2]

        pred_gen = VolumeGenerator(input_files,
                                   seed_files=seed_files,
                                   label_files=label_files,
                                   seed_type=options.seed,
                                   concat_files=options.concat,
                                   include_labels=False)
        model.predict(pred_gen, save_path)

    if options.test:
        logging.info('Testing model.')

        input_files = glob.glob(options.test[0])
        seed_files = None if gen_seed else glob.glob(options.test[1])
        label_files = glob.glob(options.test[1]) if gen_seed else glob.glob(options.test[2])

        test_gen = VolumeGenerator(input_files,
                                   seed_files=seed_files,
                                   label_files=label_files,
                                   seed_type=options.seed,
                                   concat_files=options.concat,
                                   include_labels=True)
        metrics = model.test(test_gen)
        logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


def run(options):
    np.random.seed(12345)
    start = time.time()
    metrics = {}
    organ = 'all_brains' if options.organ[0] == 'brains' else options.organ[0]

    logging.info('Splitting data.')
    n = len(constants.SAMPLES)
    shuffled = np.random.permutation(constants.SAMPLES)
    train = shuffled[:(2*n)//3]
    val = shuffled[(2*n)//3:(5*n)//6]
    test = shuffled[(5*n)//6:]

    logging.info('Creating model.')
    shape = constants.SHAPE
    model = model = MODELS[options.model](shape, name='{}_{}'.format(options.model, options.organ), filename=options.model_file)

    logging.info('Creating data generators.')
    train_files = ['data/raw/{}/{}_0000.nii.gz'.format(sample, sample) for sample in train]
    train_label_files = ['data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ) for sample in train]
    train_gen = AugmentGenerator(train_files, label_files=train_label_files)
    
    val_files = ['data/raw/{}/{}_0000.nii.gz'.format(sample, sample) for sample in val]
    val_label_files = ['data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ) for sample in val]
    val_gen = VolumeGenerator(val_files, label_files=val_label_files, load_files=True, include_labels=True)

    test_files = ['data/raw/{}/{}_0000.nii.gz'.format(sample, sample) for sample in test]
    test_label_files = ['data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ) for sample in test]
    pred_gens = [VolumeGenerator([test_file], include_labels=False) for test_file in test_files]
    test_gen = VolumeGenerator(test_files, label_files=test_label_files, include_labels=True)

    logging.info('Compiling model.')
    model.compile(util.get_weights(train_gen.labels))

    logging.info('Training model.')
    model.train(train_gen, val_gen, options.epochs, tensorboard=options.tensorboard)

    logging.info('Saving model.')
    model.save()

    logging.info('Making predictions.')
    for pred_gen in pred_gens:
        model.predict(pred_gen, 'data/predict/{}/'.format(options.model))

    logging.info('Testing model.')
    metrics = model.test(test_gen)
    logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    if options.run:
        run(options)
    else:
        main(options)
