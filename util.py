import nibabel as nib
import numpy as np
from keras import backend as K


def read_vol(filename):
    vol = nib.load(filename).get_data()

    if vol.ndim == 3:
        vol = vol[..., np.newaxis]
    return np.asarray(vol)


def save_vol(vol, filename, header=None, scale=False):
    if type(vol) is np.ndarray:
        if vol.ndim > 4:
            vol = vol[0]
        if scale:
            vol *= 255
        else:
            vol = np.rint(vol)
        vol = nib.Nifti1Image(vol.astype('int16'), np.diag([3, 3, 3, 1]), header=header)
    vol.to_filename(filename)


def shape(filename):
    return read_vol(filename).shape


def header(filename):
    return nib.load(filename).header


def get_weights(vols):
    if vols is None:
        return None
    if type(vols[0]) is not np.ndarray:
        vols = [read_vol(vol) for vol in vols]
    weight = 0
    total = 0
    for vol in vols:
        weight += np.sum(vol)
        total += vol.size
    w = weight/total
    return (1 - w, w)


def dice_coef(true, pred):
    true_f = true.flatten()
    pred_f = pred.flatten()
    intersection = np.sum(true_f * pred_f)
    return (2. * intersection) / (np.sum(true_f) + np.sum(pred_f))
