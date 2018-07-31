import glob
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import util

from argparse import ArgumentParser
from scipy.interpolate import interp1d

parser = ArgumentParser()
parser.add_argument('folder', type=str, nargs=1)
options = parser.parse_args()


def main(folder):
    files = glob.glob(folder + '*.nii.gz')
    vols = np.concatenate([util.read_vol(file) for file in files], axis=-1)
    if vols.shape[0] == vols.shape[1] == vols.shape[2]:
        axis = int(input('shape: {}\n> '.format(vols.shape)))
    elif vols.shape[0] == vols.shape[1]:
        axis = 2
    elif vols.shape[0] == vols.shape[2]:
        axis = 1
    elif vols.shape[1] == vols.shape[2]:
        axis = 0
    else:
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

    even_1 = evens[shape[0]//3,shape[1]//2,...]
    even_2 = evens[shape[0]*2//3,shape[1]//2,...]
    odd_1 = odds[shape[0]//3,shape[1]//2,...]
    odd_2 = odds[shape[0]*2//3,shape[1]//2,...]
    even_img_1 = np.zeros((shape[2], shape[3] * 2))
    even_img_1[:,::2] = even_1
    even_img_1[:,1::2] = odd_1
    odd_img_1 = np.zeros((shape[2], shape[3] * 2))
    odd_img_1[:,::2] = odd_1
    odd_img_1[:,1::2] = even_1
    even_img_2 = np.zeros((shape[2], shape[3] * 2))
    even_img_2[:,::2] = even_2
    even_img_2[:,1::2] = odd_2
    odd_img_2 = np.zeros((shape[2], shape[3] * 2))
    odd_img_2[:,::2] = odd_2
    odd_img_2[:,1::2] = even_2

    fig = plt.figure(figsize=(16, 8))
    fig.add_subplot(2, 2, 1)
    plt.imshow(odd_img_1)
    plt.axis('off')
    plt.title('odd')
    fig.add_subplot(2, 2, 2)
    plt.imshow(even_img_1)
    plt.axis('off')
    plt.title('even')
    fig.add_subplot(2, 2, 3)
    plt.imshow(odd_img_2)
    plt.axis('off')
    plt.title('odd')
    fig.add_subplot(2, 2, 4)
    plt.imshow(even_img_2)
    plt.axis('off')
    plt.title('even')
    plt.show(block=False)

    order = input('1. odd\n2. even\n> ')
    plt.close()
    new_shape = list(shape)
    new_shape[-1] *= 2
    series = np.zeros(new_shape)

    if order == '1':
        series[...,::2] = odds
        series[...,1::2] = evens
    elif order == '2':
        series[...,::2] = evens
        series[...,1::2] = odds
    else:
        raise ValueError('Must be even or odd slice.')

    series = np.moveaxis(series, 0, axis)

    sample = folder.split('/')[2]
    new_folder = 'data/originals/{}/'.format(folder.split('/')[2])
    os.makedirs(new_folder, exist_ok=True)
    for i in range(new_shape[-1]):
        util.save_vol(series[...,i], new_folder + sample + '_{}.nii.gz'.format(i))


if __name__ == '__main__':
    main(options.folder[0])
