import constants
import glob
import numpy as np
from util import read_vol


def resize(vol):
    if vol.shape != constants.VOL_SHAPE:
        raise ValueError('The input shape {shape} is not supported.'.format(shape=vol.shape))

    # convert to target shape
    dx = (constants.VOL_SHAPE[0] - constants.TARGET_SHAPE[0]) // 2
    dy = (constants.VOL_SHAPE[1] - constants.TARGET_SHAPE[1]) // 2
    zeros = np.zeros((constants.TARGET_SHAPE[0],
                      constants.TARGET_SHAPE[1],
                      (constants.TARGET_SHAPE[2] - constants.VOL_SHAPE[2]) // 2, 1))
    
    resized = np.concatenate([zeros, vol[dx:-dx, dy:-dy, :], zeros], axis=2)
    if resized.shape != constants.TARGET_SHAPE:
        raise ValueError('The resized shape {shape} '
                         'does not match the target '
                         'shape {target}'.format(shape=resized.shape,
                                                    target=constants.TARGET_SHAPE))
    return resized


def rescale(vol):
    return vol / constants.MAX_VALUE


def preprocess(files, funcs=[rescale, resize]):
    vol = read_vol(file)
    for f in funcs:
        vol = f(vol)
    return vol
