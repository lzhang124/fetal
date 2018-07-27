import glob
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import util

from argparse import ArgumentParser
from scipy.interpolate import interp1d

parser = ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
options = parser.parse_args()


def main(folder):
    files = glob.glob(folder + '*.nii.gz')
    vols = np.concatenate([util.read_vol(file) for file in files], axis=-1)
    axis = int(input('shape: {}\n> '.format(vols.shape)))
    vols = np.moveaxis(vols, axis, 0)
    shape = vols.shape

    even_i = np.arange(0, shape[0], 2)
    even_slices = vols[::2,...]
    even_interpolator = interp1d(even_i, even_slices, kind='linear', axis=0)
    odd_i = np.arange(1, shape[0], 2)
    odd_slices = vols[1::2,...]
    odd_interpolator = interp1d(odd_i, odd_slices, kind='linear', axis=0)

    if shape[0] % 2 == 0:
        evens = even_interpolator(np.arange(0, shape[0] - 1))
        odds = odd_interpolator(np.arange(1, shape[0]))
        evens = np.concatenate((evens, evens[np.newaxis,-1,...]))
    else:
        evens = even_interpolator(np.arange(0, shape[0]))
        odds = odd_interpolator(np.arange(1, shape[0] - 1))
        odds = np.concatenate((odds, odds[np.newaxis,-1,...]))

    odds = np.concatenate((np.zeros([1,] + list(shape[1:])), odds))

    even = evens[shape[0]//2,shape[1]//2,...]
    odd = odds[shape[0]//2,shape[1]//2,...]
    even_img = np.zeros((shape[2], shape[3] * 2))
    even_img[:,::2] = even
    even_img[:,1::2] = odd
    odd_img = np.zeros((shape[2], shape[3] * 2))
    odd_img[:,::2] = odd
    odd_img[:,1::2] = even

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 1, 1)
    plt.imshow(odd_img)
    plt.axis('off')
    fig.add_subplot(2, 1, 2)
    plt.imshow(even_img)
    plt.axis('off')
    plt.show(block=False)

    order = input('which slice first?\n  1. odd\n  2. even\n> ')
    plt.close()

    if order == '1':
        print('odd')
    elif order == '2':
        print('even')
    else:
        raise ValueError('Must be even or odd slice.')

    series = np.moveaxis(series, 0, axis)


if __name__ == '__main__':
    main(options.folder)
