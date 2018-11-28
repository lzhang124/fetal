import constants
import glob
import numpy as np
import util


for sample in constants.SAMPLES:
    files = glob.glob(f'data/labels/{sample}/{sample}_0_*_brain.nii.gz')

    volume = np.zeros(util.shape(f'data/raw/{sample}/{sample}_0.nii.gz'))
    header = util.header(files[0])
    for file in files:
        volume += util.read_vol(file)

    util.save_vol(volume, f'data/labels/{sample}/{sample}_0_all_brains.nii.gz', header)
