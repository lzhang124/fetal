import glob
import nibabel as nib
import numpy as np
from image3d import ImageTransformer, VolSegIterator


def volread(filename):
    return np.squeeze(nib.load(filename).get_data())


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
                          for seg_file in seg_files]

        print(self.vol_files)
        print(self.seg_files)

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
