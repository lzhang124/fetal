import glob
import nibabel as nib
import numpy as np
import os
import util

from argparse import ArgumentParser
from scipy.interpolate import interp1d

parser = ArgumentParser()
parser.add_argument('sample', type=str)
parser.add_argument('--order', type=str)
options = parser.parse_args()


def main(sample, order):
    files = sorted(glob.glob(f'data/nifti/{sample}/*.nii.gz'))
    vols = np.concatenate([util.read_vol(file) for file in files], axis=-1)
    if vols.shape[0] == vols.shape[1] == vols.shape[2]:
        axis = int(input(f'shape: {vols.shape}\n> '))
    elif vols.shape[0] == vols.shape[1]:
        axis = 2
    elif vols.shape[0] == vols.shape[2]:
        axis = 1
    elif vols.shape[1] == vols.shape[2]:
        axis = 0
    else:
        axis = int(input(f'shape: {vols.shape}\n> '))
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
        evens = np.concatenate((evens, evens[-1:,...]))
    else:
        evens = even_interpolator(np.arange(0, shape[0]))
        odds = odd_interpolator(np.arange(1, shape[0] - 1))
        odds = np.concatenate((odds, odds[-1:,...]))

    odds = np.concatenate((odds[:1,...], odds))

    new_shape = list(shape)
    new_shape[-1] *= 2
    series = np.zeros(new_shape)

    if order is None:
        import matplotlib.pyplot as plt

        even_1 = evens[shape[0]//3,shape[1]//2,...]
        even_2 = evens[shape[0]*2//3,shape[1]//2,...]
        odd_1 = odds[shape[0]//3,shape[1]//2,...]
        odd_2 = odds[shape[0]*2//3,shape[1]//2,...]
        even_img_1 = np.zeros((shape[2], shape[3] * 2))
        even_img_1[:,::2] = even_1
        even_img_1[:,1::2] = odd_1
        even_img_2 = np.zeros((shape[2], shape[3] * 2))
        even_img_2[:,::2] = even_2
        even_img_2[:,1::2] = odd_2
        odd_img_1 = np.zeros((shape[2], shape[3] * 2))
        odd_img_1[:,::2] = odd_1
        odd_img_1[:,1::2] = even_1
        odd_img_2 = np.zeros((shape[2], shape[3] * 2))
        odd_img_2[:,::2] = odd_2
        odd_img_2[:,1::2] = even_2

        img_1 = np.concatenate((odd_img_1, even_img_1), axis=0)
        img_2 = np.concatenate((odd_img_2, even_img_2), axis=0)

        fig = plt.figure(figsize=(9, 9))
        fig.add_subplot(2, 1, 1)
        plt.imshow(img_1)
        plt.axis('off')
        fig.add_subplot(2, 1, 2)
        plt.imshow(img_2)
        plt.axis('off')
        plt.suptitle(sample)
        plt.show(block=False)

        order = input('0. 3D\n1. odd\n2. even\n> ')

        if order == '0':
            even = evens[shape[0]//2,...]
            odd = odds[shape[0]//2,...]
            even_img = np.zeros((shape[1], shape[2], shape[3] * 2))
            even_img[...,::2] = even
            even_img[...,1::2] = odd
            odd_img = np.zeros((shape[1], shape[2], shape[3] * 2))
            odd_img[...,::2] = odd
            odd_img[...,1::2] = even

            temp_folder = 'data/temp'
            os.makedirs(temp_folder, exist_ok=True)
            util.save_vol(even_img, temp_folder + 'even.nii.gz')
            util.save_vol(odd_img, temp_folder + 'odd.nii.gz')
            os.system(f'open {temp_folder}/even.nii.gz')
            os.system(f'open {temp_folder}/odd.nii.gz')
            order = input('1. odd\n2. even\n> ')

        plt.close()

    if order == '1' or order == 'o':
        series[...,::2] = odds
        series[...,1::2] = evens
    elif order == '2' or order == 'e':
        series[...,::2] = evens
        series[...,1::2] = odds
    else:
        raise ValueError('Must be even or odd slice.')

    series = np.moveaxis(series, 0, axis)

    new_folder = f'data/raw/{sample}'
    os.makedirs(new_folder, exist_ok=True)
    for i in range(new_shape[-1]):
        util.save_vol(series[...,i], f'{new_folder}/{sample}_{str(i).zfill(4)}.nii.gz')


if __name__ == '__main__':
    main(options.sample, options.order)
