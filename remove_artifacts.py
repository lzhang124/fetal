from process import remove_artifacts
import constants
import glob
import os
import numpy as np
import util

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('model', type=str)
options = parser.parse_args()


def main(model):
    old_metrics = {}
    new_metrics = {}

    for file in glob.glob(f'data/predict/{model}/*_0000.nii.gz'):
        sample = os.path.basename(file).split('_')[0]
        header = util.header(file)
        volume = util.read_vol(file)
        label = util.read_vol(glob.glob(f'data/labels/{sample}/{sample}_0_all_brains.nii.gz')[0])

        old_metrics[sample] = util.dice_coef(label, volume)

        n = 2 if sample in constants.TWINS else 1
        volume = remove_artifacts(volume, n)

        new_metrics[sample] = util.dice_coef(label, volume)

        os.makedirs(f'data/predict_cleaned/{model}/', exist_ok=True)
        util.save_vol(volume, f'data/predict_cleaned/{model}/{sample}_0000.nii.gz', header)

    print('before: ', old_metrics)
    print('after: ', new_metrics)


if __name__ == '__main__':
    main(options.model)
