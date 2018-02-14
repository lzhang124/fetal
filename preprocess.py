import glob
import nibabel as nib
import numpy as np
from image3d import ImageTransformer, VolSegIterator


VOL_SHAPE = (150, 150, 110, 1)
TARGET_SHAPE = (128, 128, 128, 1)


def volread(filename):
    vol = nib.load(filename).get_data()
    
    # need to add channel axis
    if vol.ndim == 3:
        vol = vol[..., np.newaxis]
    if vol.shape != VOL_SHAPE:
        raise ValueError('The input size {size} is not supported.'.format(size=vol.shape))

    # convert to target shape
    dx = (VOL_SHAPE[0] - TARGET_SHAPE[0]) // 2
    dy = (VOL_SHAPE[1] - TARGET_SHAPE[1]) // 2
    dz = (TARGET_SHAPE[2] - VOL_SHAPE[2]) // 2
    test = np.pad(vol[dx:-dx, dy:-dy, :], ((dz, dz), (dz, dz), (0, 0)),
                  'constant', constant_values=(0, 0))
    print(test.shape)
    raise Error
    return np.pad(vol[dx:-dx, dy:-dy, :], ((dz, dz), (dz, dz), (0, 0)),
                  'constant', constant_values=(0, 0))


class AugmentGenerator(VolSegIterator):
    def __init__(self,
                 vol_files,
                 seg_files,
                 rotation_range=90.,
                 shift_range=0.1,
                 shear_range=0.2,
                 zoom_range=0.2,
                 fill_mode='nearest',
                 cval=0.,
                 flip=True,
                 batch_size=32,
                 save_to_dir=None):
        vol_path = vol_files.split('*/*')
        seg_path = seg_files.split('*/*')

        self.seg_files = glob.glob(seg_files)
        self.vol_files = [seg_file.replace(seg_path[0], vol_path[0])
                                  .replace(seg_path[1], vol_path[1])
                          for seg_file in self.seg_files]

        vols = np.array([volread(file) for file in self.vol_files])
        segs = np.array([volread(file) for file in self.seg_files])

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
