import glob
import numpy as np
from image3d import ImageTransformer, VolSegIterator
from keras.utils.data_utils import Sequence


VOL_SHAPE = (150, 150, 110, 1)
TARGET_SHAPE = (128, 128, 128, 1)
MAX_VALUE = 1024.


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


def rescale(vol):
    return vol / MAX_VALUE


def preprocess(file):
    vol = read_vol(file)
    vol = rescale(vol)
    vol = resize(vol)
    return vol


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

        vols = np.array([preprocess(file) for file in self.vol_files])
        segs = np.array([preprocess(file) for file in self.seg_files])

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


class VolumeGenerator(Sequence):
    def __init__(self, vol_files):
        self.vol_files = glob.glob(vol_files)
        self.n = len(self.vol_files)
        self.idx = 0

    def __len__(self):
        return len(self.vol_files)

    def __getitem__(self, idx):
        return preprocess(self.vol_files[idx])

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.idx < self.n:
            vol = preprocess(self.vol_files[self.idx])
            self.idx += 1
            return vol
        else:
            raise StopIteration()
