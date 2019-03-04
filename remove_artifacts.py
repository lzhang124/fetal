from process import remove_artifacts
import constants
import glob
import os
import numpy as np
import util

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--sample', type=str)
options = parser.parse_args()


def main(model):
    folder = options.sample if options.sample else '*'
    files = f'data/predict/{model}/{folder}/*.nii.gz'

    for file in glob.glob(files):
        sample, i = os.path.basename(file).split('_')
        i = i[:4]
        header = util.header(file)
        volume = util.read_vol(file)

        n = 2 if sample in constants.TWINS else 1
        volume = remove_artifacts(volume, n)

        os.makedirs(f'data/predict_cleaned/{model}/{sample}', exist_ok=True)
        util.save_vol(volume, f'data/predict_cleaned/{model}/{sample}/{sample}_{i}.nii.gz', header)


if __name__ == '__main__':
    main(options.model)
