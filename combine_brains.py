import constants
import glob
import numpy as np
import util


for sample in constants.SAMPLES:
    files = glob.glob('data/labels/{}/{}_0_*_brain.nii.gz'.format(sample, sample))

    volume = np.zeros(util.shape('data/raw/{}/{}_0.nii.gz'.format(sample, sample)))
    header = util.header(files[0])
    for file in files:
        volume += util.read_vol(file)

    util.save_vol(volume, 'data/labels/{}/{}_0_all_brains.nii.gz'.format(sample, sample), header)
