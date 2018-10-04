import glob
import nibabel as nib
import numpy as np
import os
import util

from argparse import ArgumentParser
from scipy.interpolate import interp1d

parser = ArgumentParser()
parser.add_argument('folder', type=str, nargs=1)
parser.add_argument('--matlab', action='store_false')
options = parser.parse_args()

if options.matlab:
    import matplotlib.pyplot as plt

samples = {
    '010918L': 1,
    '010918S': 1,
    '012115': 2,
    '013018L': 2,
    '013018S': 2,
    '013118L': 2,
    '013118S': 2,
    '021015': 2,
    '021218L': 1,
    '021218S': 1,
    '022318L': 1,
    '022318S': 1,
    '022415': 1,
    '022416': 1,
    '022618': 1,
    '030217': 1,
    '030315': 1,
    '031317L': 2,
    '031317T': 2,
    '031516': 1,
    '031615': 1,
    '031616': 1,
    '031716': 1,
    '032217': 1,
    '032318a': 1,
    '032318b': 1,
    '032318c': 1,
    '032318d': 1,
    '032818': 1,
    '040218': 1,
    '040417': 1,
    '040617': 2,
    '040716': 2,
    '041017': 2,
    '041318L': 2,
    '041318S': 2,
    '041818': 2,
    '043015': 2,
    '043018': 2,
    '050318L': 1,
    '050318S': 1,
    '050917': 1,
    '051215': 1,
    '051718L': 2,
    '051718S': 2,
    '051817': 2,
    '051818': 1,
    '052218L': 1,
    '052218S': 1,
    '052418L': 2,
    '052418S': 2,
    '052516': 1,
    '061715': 1,
    '062515': 2,
    '080117': 1,
    '081315': 1,
    '083115': 1,
    '102617': 2
}

def main(folder):
    # sample = folder.split('/')[2]
    
    for sample in samples:
        print(sample)
        # files = glob.glob(folder + '*.nii.gz')
        files = glob.glob('data/nifti/{}/*.nii.gz'.format(sample))
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
        
        if options.matlab:
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

        # order = input('0. 3D\n1. odd\n2. even\n> ')
        order = samples[sample]

        if order =='0':
            even = evens[shape[0]//2,...]
            odd = odds[shape[0]//2,...]
            even_img = np.zeros((shape[1], shape[2], shape[3] * 2))
            even_img[...,::2] = even
            even_img[...,1::2] = odd
            odd_img = np.zeros((shape[1], shape[2], shape[3] * 2))
            odd_img[...,::2] = odd
            odd_img[...,1::2] = even

            temp_folder = 'data/temp/'
            os.makedirs(temp_folder, exist_ok=True)
            util.save_vol(even_img, temp_folder + 'even.nii.gz')
            util.save_vol(odd_img, temp_folder + 'odd.nii.gz')
            os.system('open {}even.nii.gz'.format(temp_folder))
            os.system('open {}odd.nii.gz'.format(temp_folder))
            order = input('1. odd\n2. even\n> ')

        if options.matlab:
            plt.close()
        new_shape = list(shape)
        new_shape[-1] *= 2
        series = np.zeros(new_shape)

        # if order == '1':
        if order == 1:
            series[...,::2] = odds
            series[...,1::2] = evens
        # elif order == '2':
        elif order == 2:
            series[...,::2] = evens
            series[...,1::2] = odds
        else:
            raise ValueError('Must be even or odd slice.')

        series = np.moveaxis(series, 0, axis)

        new_folder = 'data/raw/{}/'.format(sample)
        os.makedirs(new_folder, exist_ok=True)
        for i in range(new_shape[-1]):
            util.save_vol(series[...,i], new_folder + sample + '_{}.nii.gz'.format(str(i).zfill(4)))


if __name__ == '__main__':
    main(options.folder[0])
