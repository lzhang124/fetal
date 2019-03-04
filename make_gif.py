import constants
import glob
import numpy as np
import os
import util
from array2gif import write_gif
from scipy.ndimage import binary_erosion
from scipy.ndimage.measurements import center_of_mass, label

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--sample', type=str)
options = parser.parse_args()

folder = options.sample if options.sample else '*'
samples = [i.split('/')[-1] for i in glob.glob(f'data/predict_cleaned/{options.model}/{folder}')]
os.makedirs(f'data/gifs/{options.model}', exist_ok=True)

for s in sorted(samples):
    print(s)
    vols = np.asarray([util.read_vol(f) for f in sorted(glob.glob(f'data/raw/{s}/{s}_*.nii.gz'))])
    vols = np.clip(np.log(1 + vols / np.percentile(vols, 95)), 0, 1)
    segs = np.asarray([util.read_vol(f) for f in sorted(glob.glob(f'data/predict_cleaned/{options.model}/{s}/{s}_*.nii.gz'))])

    if s in constants.TWINS:
        brains = [label(seg)[0] for seg in segs]
        for i in range(1, len(brains)):
            intersect = brains[i-1] * brains[i]
            if 1 not in intersect and 4 not in intersect:
                brains[i] = - brains[i] + 3
        centers = [int(c[-2]) for c in center_of_mass(segs, brains, [1,2])]
    else:
        centers = [int(center_of_mass(segs)[-2])]

    for c in centers:
        seg_slices = segs[...,c,0]
        erode = np.asarray([binary_erosion(slice) for slice in seg_slices])
        outline = seg_slices ^ erode

        slices = np.repeat(vols[...,c,:], 3, axis=-1)
        slices[outline == 1] = [1, 0, 0]
        slices = (np.moveaxis(slices, -1, 1) * 254).astype(int)
        write_gif(slices, f'data/gifs/{options.model}/{s}_{c}.gif', fps=10)
