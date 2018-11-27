import constants
import glob
import numpy as np
from util import read_vol
from scipy.ndimage.measurements import label


def crop(vol):
    if (vol.shape[0] < constants.SHAPE[0] or
        vol.shape[1] < constants.SHAPE[1] or
        vol.shape[2] < constants.SHAPE[2]):
        raise ValueError('The input shape {} is not supported.'.format(vol.shape))

    # convert to target shape
    dx = (vol.shape[0] - constants.SHAPE[0]) // 2
    dy = (vol.shape[1] - constants.SHAPE[1]) // 2
    dz = (vol.shape[2] - constants.SHAPE[2]) // 2

    resized = vol[dx:(dx+constants.SHAPE[0]),
                  dy:(dy+constants.SHAPE[1]),
                  dz:(dz+constants.SHAPE[2])]
    if resized.shape != constants.SHAPE:
        raise ValueError('The resized shape {shape} '
                         'does not match the target '
                         'shape {target}'.format(shape=resized.shape,
                                                 target=constants.SHAPE))
    return resized


def preprocess(file, resize=False):
    vol = read_vol(file)
    vol = vol / np.max(vol)
    if resize:
        vol = crop(vol)
    return vol


def uncrop(vol, shape):
    if vol.shape != constants.SHAPE:
        raise ValueError('The volume shape {} is not supported.'.format(vol.shape))
    if (shape[0] < vol.shape[0] or
        shape[1] < vol.shape[1] or
        shape[2] < vol.shape[2]):
        raise ValueError('The target shape {} is not supported.'.format(shape))

    # convert to original shape
    dx = (shape[0] - vol.shape[0]) // 2
    dy = (shape[1] - vol.shape[1]) // 2
    dz = (shape[2] - vol.shape[2]) // 2

    resized = np.pad(vol, ((dx, shape[0] - vol.shape[0] - dx),
                           (dy, shape[1] - vol.shape[1] - dy),
                           (dz, shape[2] - vol.shape[2] - dz),
                           (0, 0)), 'constant')
    if resized.shape != shape:
        raise ValueError('The resized shape {shape} '
                         'does not match the target '
                         'shape {target}'.format(shape=resized.shape,
                                                 target=shape))
    return resized


def remove_artifacts(vol, n):
    assert n > 0
    cleaned = np.zeros(vol.shape)
    artifacts, _ = label(vol)
    indices = np.argsort(np.bincount(artifacts.flat))[-n-1:-1]
    for i in indices:
        cleaned += artifacts == i
    return cleaned
