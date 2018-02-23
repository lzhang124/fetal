import nibabel as nib
import numpy as np


MAX_VALUE = 2400.


def read_vol(filename):
    vol = nib.load(filename).get_data()
    
    # need to add channel axis
    if vol.ndim == 3:
        vol = vol[..., np.newaxis]
    return vol


def save_vol(vol, filename, scale=False):
    if type(vol) is np.ndarray:
        if scale:
            vol *= MAX_VALUE
        vol = nib.Nifti1Image(vol.astype('int16'), np.eye(4))
    vol.to_filename(filename)
