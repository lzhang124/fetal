import util
import glob
import os
import numpy as np

files = glob.glob('data/predict_cleaned/unet3000/*')
samples = [i.split('/')[-1] for i in files]

for s in samples:
    print(s)
    f = f'data/predict_cleaned/unet3000/{s}/{s}_0000.nii.gz'
    if len(glob.glob(f)) == 0:
        continue
    header = util.header(f)
    n = len(glob.glob(f'data/predict_cleaned/unet3000/{s}/{s}_*.nii.gz'))
    h, w, d, _ = util.shape(f)
    slice_shape = (h, w, n)
    slices = [np.zeros(slice_shape) for i in range(d)]
    for i in range(n):
        a = util.read_vol(f'data/predict_cleaned/unet3000/{s}/{s}_{str(i).zfill(4)}.nii.gz')
        for j in range(d):
            slices[j][:,:,i] = a[:,:,j,0]
    for i in range(d):
        os.makedirs(f'data/slices/{s}/', exist_ok=True)
        util.save_vol(slices[i], f'data/slices/{s}/{s}_{i}.nii.gz', header)
