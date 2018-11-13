import nibabel as nib
import numpy as np


def read_vol(filename):
    vol = nib.load(filename).get_data()
    
    # need to add channel axis
    if vol.ndim == 3:
        vol = vol[..., np.newaxis]
    return vol


def save_vol(vol, filename, header=None):
    if type(vol) is np.ndarray:
        if vol.ndim > 4:
            vol = vol[0]
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
    weight = 0
    total = 0
    for vol in vols:
        weight += np.sum(vol)
        total += vol.size
    w = weight/total
    print(w)
    return (1 - w, w)
