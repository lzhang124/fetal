import glob
import numpy as np
import util


for sample in ['010918L', '010918S', '012115', '013018L', '013018S',
               '013118L', '013118S', '021015', '021218L', '021218S',
               '022318L', '022318S', '022415', '022618', '030217',
               '030315', '031317L', '031317T', '031516', '031615',
               '031616', '031716', '032217', '032318a', '032318b',
               '032318c', '032318d', '032818', '040218', '040417']:
        files = glob.glob('data/labels/{}/{}_0_*_brains.nii.gz'.format(sample, sample))

        volume = np.zeros(util.shape(files[0]))
        header = util.header(files[0])
        for file in files:
            volume += util.read_vol(file)

        util.save_vol(volume, 'data/labels/{}/{}_0_all_brains.nii.gz'.format(sample, sample), header)
