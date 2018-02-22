import glob
import numpy as np
from image3d import ImageTransformer, VolSegIterator
from sklearn.preprocessing import scale
from util import read_vol


VOL_SHAPE = (150, 150, 110, 1)
TARGET_SHAPE = (128, 128, 128, 1)


def resize(vol):
    if vol.shape != VOL_SHAPE:
        raise ValueError('The input shape {shape} is not supported.'.format(shape=vol.shape))

    # convert to target shape
    dx = (VOL_SHAPE[0] - TARGET_SHAPE[0]) // 2
    dy = (VOL_SHAPE[1] - TARGET_SHAPE[1]) // 2
    zeros = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], (TARGET_SHAPE[2] - VOL_SHAPE[2]) // 2, 1))
    
    resized = np.concatenate([zeros, vol[dx:-dx, dy:-dy, :], zeros], axis=2)
    if resized.shape != TARGET_SHAPE:
        raise ValueError('The resized shape {shape} '
                         'does not match the target shape {target}'.format(shape=resized.shape,
                                                                           target=TARGET_SHAPE))
    return resized


def preprocess(files):
    vols = [read_vol(file) for file in files]
    vols = scale(vols)
    vols = [resize(vol) for vol in vols]
    return np.array(vols)


class AugmentGenerator(VolSegIterator):
    def __init__(self,
                 vol_files,
                 seg_files,
                 batch_size,
                 rotation_range=90.,
                 shift_range=0.1,
                 shear_range=0.2,
                 zoom_range=0.2,
                 fill_mode='nearest',
                 cval=0.,
                 flip=True,
                 save_to_dir=None):
        vol_path = vol_files.split('*/*')
        seg_path = seg_files.split('*/*')

        self.seg_files = glob.glob(seg_files)
        self.vol_files = [seg_file.replace(seg_path[0], vol_path[0])
                                  .replace(seg_path[1], vol_path[1])
                          for seg_file in self.seg_files]

        vols = preprocess(self.vol_files)
        segs = preprocess(self.seg_files)

        image_transformer = ImageTransformer(rotation_range=rotation_range,
                                             shift_range=shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             flip=flip)

        super(AugmentGenerator, self).__init__(vols, segs, image_transformer,
                                               batch_size=batch_size,
                                               save_to_dir=save_to_dir,
                                               x_prefix='vol',
                                               y_prefix='seg')


class VolumeGenerator:
    def __init__(self, vol_files):
        self.vol_files = glob.glob(vol_files)

    def __len__(self):
        return len(self.vol_files)

    def __iter__(self):
        return self

    def __next__(self):
        preprocess(self.vol_files)
