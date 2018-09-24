from process import remove_artifacts
import constants
import glob
import os
import numpy as np
import util

def dice_coef(true, pred):
    true_f = true.flatten()
    pred_f = pred.flatten()
    intersection = np.sum(true_f * pred_f)
    return (2. * intersection) / (np.sum(true_f) + np.sum(pred_f))

metrics = {}

for file in glob.glob('data/predict/*_0.nii.gz'):
    sample = os.path.basname(file).split('_')[0]
    header = util.header(file)

    n = 2 if sample in constants.TWINS else 1
    volume = remove_artifacts(util.read_vol(file), n)

    label = glob.glob('data/labels/{}/{}_0_all_brains.nii.gz'.format(sample, sample))[0]
    metrics[sample] = dice_coef(util.read_vol(label), volume)

    util.save_vol(volume, 'data/predict_cleaned/{}_0.nii.gz'.format(sample, sample), header)

print(metrics)
