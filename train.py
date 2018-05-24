import os
import logging
logging.basicConfig(level=logging.INFO)

import constants
import glob
import time
import util
from argparse import ArgumentParser
from data import AugmentGenerator, VolumeGenerator
from models import UNet, UNetSmall, UNetBig


def build_parser():
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
                        help='Test model.',
                        dest='test', type=str, nargs='+')
    parser.add_argument('--seed',
                        help='Seed slices',
                        dest='seed', type=str)
    parser.add_argument('--concat',
                        help='Concatenate first volume',
                        dest='concat', action='store_true')
    parser.add_argument('--batch-size',
                        metavar='BATCH_SIZE',
                        help='Training batch size',
                        dest='batch_size', type=int, default=1)
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
                        help='GPU to use',
                        dest='gpu', type=str, default='0')
    parser.add_argument('--size',
                        metavar='SIZE',
                        help='Size of UNet',
                        dest='size', type=str)
    parser.add_argument('--run',
                        metavar='RUN',
                        help='Which preset program to run',
                        dest='run', type=str)
    return parser


def main(options):
    start = time.time()

    logging.info('Creating model.')
    if options.seed:
        shape = tuple(list(constants.TARGET_SHAPE[:-1]) + [constants.TARGET_SHAPE[-1] + 1])
    else:
        shape = constants.TARGET_SHAPE
    if options.size == 'small':
        m = UNetSmall
    elif options.size == 'big':
        m = UNetBig
    else:
        m = UNet
    model = m(shape, name=options.name, filename=options.model_file)

    gen_seed = (options.seed == 'slice' or options.seed == 'volume')

    if options.train:
        logging.info('Creating data generator.')

        input_path = options.train[0].split('*')[0]
        label_path = options.train[1].split('*')[0]

        label_files = glob.glob(options.train[1])
        input_files = [label_file.replace(label_path, input_path) for label_file in label_files]

        aug_gen = AugmentGenerator(input_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed=options.seed,
                                   concat_first=options.concat)
        val_gen = VolumeGenerator(input_files,
                                  label_files=label_files,
                                  batch_size=options.batch_size,
                                  seed=options.seed,
                                  concat_first=options.concat,
                                  load_files=True,
                                  include_labels=True)

        logging.info('Compiling model.')
        model.compile(util.get_weights(aug_gen.labels))

        logging.info('Training model.')
        model.train(aug_gen, val_gen, options.epochs)
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
                                   batch_size=options.batch_size,
                                   seed=options.seed,
                                   concat_first=options.concat,
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
                                   batch_size=options.batch_size,
                                   seed=options.seed,
                                   concat_first=options.concat,
                                   include_labels=True)
        metrics = model.test(test_gen)
        logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


def run(options):
    start = time.time()

    metrics = {}

    all_labels = glob.glob('data/labels/*/*_placenta.nii.gz')

    # for sample in ['043015', '051215', '061715', '062515', '081315', '083115', '110214', '112614', '122115', '122215']:
    for sample in ['043015', '061715']:
        logging.info(sample)

        logging.info('Creating model.')
        if options.seed:
            shape = tuple(list(constants.TARGET_SHAPE[:-1]) + [constants.TARGET_SHAPE[-1] + 1])
        else:
            shape = constants.TARGET_SHAPE
        if options.size == 'small':
            m = UNetSmall
        elif options.size == 'big':
            m = UNetBig
        else:
            m = UNet
        model = m(shape, name='unet_test_{}'.format(sample), filename=options.model_file)

        logging.info('Creating data generator.')

        if options.run == 'one-out':
            label_files = [file for file in all_labels if not os.path.basename(file).startswith(sample)]
        elif options.run == 'single':
            label_files = glob.glob('data/labels/{}/{}_1_placenta.nii.gz'.format(sample, sample))
        elif options.run == 'sample':
            label_files = glob.glob('data/labels/{}/{}_*_placenta.nii.gz'.format(sample, sample))[:4]
        else:
            raise ValueError('Preset program not defined.')

        input_files = [file.replace('labels', 'raw').replace('_placenta', '') for file in label_files]
        aug_gen = AugmentGenerator(input_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed=options.seed,
                                   concat_first=options.concat)
        val_gen = VolumeGenerator(input_files,
                                  label_files=label_files,
                                  batch_size=options.batch_size,
                                  seed=options.seed,
                                  concat_first=options.concat,
                                  load_files=True,
                                  include_labels=True)
        a = aug_gen.next()
        print(a[0].shape)
        print(a[1].shape)
        print(options.seed)
        print(aug_gen.seed)
        assert False

        logging.info('Compiling model.')
        model.compile(util.get_weights(aug_gen.labels))

        logging.info('Training model.')
        model.train(aug_gen, val_gen, options.epochs)

        logging.info('Making predictions.')
        if options.run == 'one-out':
            label_files = glob.glob('data/labels/{}/{}_*_placenta.nii.gz'.format(sample, sample))
        elif options.run == 'single':
            label_files = [f for f in glob.glob('data/labels/{}/{}_*_placenta.nii.gz'.format(sample, sample))
                           if not os.path.basename(f).endswith('_1_placenta.nii.gz')]
        elif options.run == 'sample':
            label_files = glob.glob('data/labels/{}/{}_*_placenta.nii.gz'.format(sample, sample))[4:]
        else:
            raise ValueError('Preset program not defined.')

        predict_files = [file.replace('labels', 'raw').replace('_placenta', '') for file in label_files]
        pred_gen = VolumeGenerator(predict_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed=options.seed,
                                   concat_first=options.concat,
                                   include_labels=False)
        model.predict(pred_gen, 'data/predict/{}/'.format(sample))

        logging.info('Testing model.')
        test_gen = VolumeGenerator(predict_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed=options.seed,
                                   concat_first=options.concat,
                                   include_labels=True)
        metrics[sample] = model.test(test_gen)

    logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    if options.run:
        run(options)
    else:
        main(options)
