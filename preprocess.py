import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def augment_generator(vols,
                      segs,
                      batch_size=1,
                      rotation_range=90.,
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      shear_range=0.2,
                      zoom_range=0.2,
                      fill_mode='nearest',
                      horizontal_flip=True,
                      vertical_flip=True):
    """
    Creates generator that performs random data augmentations.
    """
    data_gen_args = dict(rotation_range=rotation_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         shear_range=shear_range,
                         zoom_range=zoom_range,
                         fill_mode=fill_mode,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip)
    vol_datagen = ImageDataGenerator(**data_gen_args)
    seg_datagen = ImageDataGenerator(**data_gen_args)

    seed = 12345
    vol_generator = vol_datagen.flow(vols, seed=seed, batch_size=batch_size)
    seg_generator = seg_datagen.flow(segs, seed=seed, batch_size=batch_size)

    aug_generator = zip(vol_generator, seg_generator)
    return aug_generator
