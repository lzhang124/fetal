import constants
import glob
import numpy as np
import os
import util
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--sample', type=str)
options = parser.parse_args()

folder = options.sample if options.sample else '*'
samples = [i.split('/')[-1] for i in glob.glob(f'data/predict_cleaned/{options.model}/{folder}')]
os.makedirs(f'data/volumes/{options.model}', exist_ok=True)
var = {}

for s in sorted(samples):
    print(s)
    segs = np.array([util.read_vol(f) for f in sorted(glob.glob(f'data/predict_cleaned/{options.model}/{s}/{s}_*.nii.gz'))])

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
        var[f'{s}_{i}'] = np.sqrt(np.var(vols[i]))
        x = np.arange(len(segs))
        y = (vols[i] / vols[i][0] - 1) * 100
        plt.ylabel('Change in Volume (%)')
        plt.plot(x, y, marker='', linewidth=1)
        plt.savefig(f'data/volumes/{options.model}/{s}_{i}.png')
        plt.close()

sort = sorted(var.items(), key=lambda x: x[1])
plt.figure(figsize=(10,8))
bar = plt.bar([s[0] for s in sort], [s[1] for s in sort])
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.ylabel('Variance in Volume')
plt.savefig(f'data/volumes/{options.model}.png')
plt.close()
