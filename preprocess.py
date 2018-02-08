import numpy as np
from image3d import ImageDataGenerator


def augment_generator(vols,
                      segs,
                      batch_size=32,
                      rotation_range=90.,
                      shift_range=0.1,
                      shear_range=0.2,
                      zoom_range=0.2,
                      fill_mode='nearest',
                      flip=True,
                      save_dir=None):
    """
    Creates generator that performs random data augmentations.
    """
    data_gen_args = dict(rotation_range=rotation_range,
                         shift_range=shift_range,
                         shear_range=shear_range,
                         zoom_range=zoom_range,
                         fill_mode=fill_mode,
                         flip=flip)
    vol_datagen = ImageDataGenerator(**data_gen_args)
    seg_datagen = ImageDataGenerator(**data_gen_args)

    seed = 12345
    vol_generator = vol_datagen.flow(vols, seed=seed, batch_size=batch_size, shuffle=True,
                                     save_to_dir=save_dir, save_prefix='vol')
    seg_generator = seg_datagen.flow(segs, seed=seed, batch_size=batch_size, shuffle=True,
                                     save_to_dir=save_dir, save_prefix='seg')

    aug_generator = zip(vol_generator, seg_generator)
    return aug_generator
