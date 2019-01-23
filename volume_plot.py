import constants
import glob
import numpy as np
import os
import util
import matplotlib.pyplot as plt
from scipy.ndimage import measurements


train = ['031317T', '031616', '013018S', '041318S', '050318S',
         '032318c', '032818', '022318S', '013018L', '021218S',
         '040218', '013118L', '022618', '031615', '031317L',
         '012115', '032318d', '031516', '050917', '021218L',
         '040716', '032318b', '021015', '040417', '041818',
         '022318L', '041017']

samples = [i.split('/')[-1] for i in glob.glob('data/predict_cleaned/unet3000/*')]
os.makedirs(f'data/volumes/', exist_ok=True)
var = {}

for s in samples:
    print(s)
    segs = np.array([util.read_vol(f) for f in sorted(glob.glob(f'data/predict_cleaned/unet3000/{s}/{s}_*.nii.gz'))])

    if s in constants.TWINS:
        brains = [measurements.label(seg)[0] for seg in segs]
        vols = [measurements.sum(segs[0], brains[0], [1,2])]
        for i in range(1, len(brains)):
            intersect = brains[i-1] * brains[i]
            if 1 not in intersect and 4 not in intersect:
                brains[i] = - brains[i] + 3
            vols.append(measurements.sum(segs[i], brains[i], [1,2]))
        vols = np.array(vols).T
    else:
        vols = [np.sum(segs, axis=(1, 2, 3, 4))]

    for i in range(len(vols)):
        var[f'{s}_{i}'] = np.var(vols[i])
        x = np.arange(len(segs))
        y = (vols[i] / vols[i][0] - 1) * 100
        plt.ylabel('Change in Volume (%)')
        plt.plot(x, y, marker='', linewidth=1)
        plt.savefig(f'data/volumes/{s}_{i}.png')
        plt.close()

sort = sorted(var.items(), key=lambda x: x[1])
plt.figure(figsize=(10,8))
bar = plt.bar([s[0] for s in sort], [s[1] for s in sort])
for i in range(len(sort)):
    if sort[i][0].split('_')[0] in train:
        bar[i].set_color('#d65555')
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.ylabel('Variance in Volume')
plt.savefig('data/variance.png')
plt.close()
