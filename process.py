import constants
import glob
import numpy as np
from util import read_vol


def crop(vol, shape):
    if (vol.shape[0] < shape[0] or
        vol.shape[1] < shape[1] or
        vol.shape[2] < shape[2]):
        raise ValueError('The input shape {shape} is not supported.'.format(shape=vol.shape))

    # convert to target shape
    dx = abs(shape[0] - vol.shape[0]) // 2
    dy = abs(shape[1] - vol.shape[1]) // 2
    dz = abs(shape[2] - vol.shape[2]) // 2
    
    resized = vol[dx:-dx, dy:-dy, dz:-dz]
    if resized.shape != shape:
        raise ValueError('The resized shape {shape} '
                         'does not match the target '
                         'shape {target}'.format(shape=resized.shape,
                                                 target=shape))
    return resized


def scale(vol):
    return vol / np.max(vol)


def preprocess(file, resize=False, rescale=False):
    vol = read_vol(file)
    if resize:
        vol = crop(vol)
    if rescale:
        vol = scale(vol)
    return vol


def uncrop(vol, shape):
    if (shape[0] < vol.shape[0] or
        shape[1] < vol.shape[1] or
        shape[2] < vol.shape[2]):
        raise ValueError('The target shape {shape} is not supported.'.format(shape=shape))

    # convert to original shape
    dx = abs(shape[0] - vol.shape[0]) // 2
    dy = abs(shape[1] - vol.shape[1]) // 2
    dz = abs(shape[2] - vol.shape[2]) // 2

    resized = np.pad(vol, ((dx, dx), (dy, dy), (dz, dz), (0, 0)), 'constant')
    if resized.shape != shape:
        raise ValueError('The resized shape {shape} '
                         'does not match the target '
                         'shape {target}'.format(shape=resized.shape,
                                                 target=shape))
    return resized
