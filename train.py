import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging
logging.basicConfig(level=logging.INFO)

import time
from argparse import ArgumentParser
from data import AugmentGenerator, VolumeGenerator
from models import AutoEncoder, UNet


MODEL_TYPE = {
    'unet': UNet,
    'autoencoder': AutoEncoder
}


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model',
                        metavar='MODEL', help='Type of model',
                        dest='model', type=str)
    parser.add_argument('-t', '--train',
                        metavar=('INPUT_FILES', 'LABEL_FILES'), help='Train model',
                        dest='train', type=str, nargs='+')
    parser.add_argument('-p', '--predict',
                        metavar=('INPUT_FILES', 'SAVE_PATH'), help='Predict segmentations',
                        dest='predict', type=str, nargs=2)
    parser.add_argument('-b', '--batch-size',
                        metavar='BATCH_SIZE', help='Training batch size',
                        dest='batch_size', type=int, default=1)
    parser.add_argument('-e', '--epochs',
                        metavar='EPOCHS', help='Training epochs',
                        dest='epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate',
                        metavar='LEARNING_RATE', help='Training learning rate',
                        dest='lr', type=float, default=1e-4)
    parser.add_argument('-f', '--model-file',
                        metavar='MODEL_FILE', help='Pretrained model file',
                        dest='model_file', type=str)
    return parser


def main():
    start = time.time()

    parser = build_parser()
    options = parser.parse_args()

    logging.info('Compiling model.')
    model = MODEL_TYPE[options.model](options.model_file)

    if options.train:
        logging.info('Creating data generator.')
        labels = None if len(options.train) < 2 else options.train[1]
        aug_gen = AugmentGenerator(options.train[0], labels, options.batch_size)

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
