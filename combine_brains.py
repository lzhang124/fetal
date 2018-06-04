import glob
import numpy as np
import util


for sample in ['043015', '051215', '061715', '062515', '081315', '083115', '110214', '112614', '122115', '122215']:
    for n in ['1', '49', '99', '149', '199', '249']:
        files = glob.glob('data/labels/{}/{}_{}_*_brains.nii.gz'.format(sample, sample, n))

        volume = np.zeros(util.shape(files[0]))
        header = util.header(files[0])
        for file in files:
            volume += util.read_vol(file)

        util.save_vol(volume, 'data/labels/{}/{}_{}_all_brains.nii.gz'.format(sample, sample, n), header)
