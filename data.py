import glob
import numpy as np
from image3d import ImageTransformer, VolumeIterator
from keras.utils.data_utils import Sequence
from process import preprocess
from util import shape


class AugmentGenerator(VolumeIterator):
    def __init__(self,
                 input_files,
                 label_files,
                 batch_size,
                 rotation_range=90.,
                 shift_range=0.1,
                 shear_range=0.2,
                 zoom_range=0.2,
                 fill_mode='nearest',
                 cval=0.,
                 flip=True):
        if label_files:
            input_path = input_files.split('*/*')
            label_path = label_files.split('*/*')

            label_files = glob.glob(label_files)
            input_files = [label_file.replace(label_path[0], input_path[0])
                                     .replace(label_path[1], input_path[1])
                              for label_file in label_files]

            inputs = np.array([preprocess(file) for file in input_files])
            labels = np.array([preprocess(file, funcs=['resize']) for file in label_files])
        else:
            inputs = np.array([preprocess(file, funcs=['resize'])
                               for file in glob.glob(input_files)])
            labels = None

        image_transformer = ImageTransformer(rotation_range=rotation_range,
                                             shift_range=shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             flip=flip)

        super().__init__(inputs, labels, image_transformer, batch_size=batch_size)


class VolSliceGenerator(AugmentGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)
        new_batch_x = np.zeros(tuple(list(batch_x.shape[:-1]) + [batch_x.shape[-1] + 1]))
        for i, label in enumerate(batch_y):
            seed = np.zeros(batch_x[i].shape)
            r = np.random.choice(label.shape[0])
            while not np.any(label[r]):
                r = np.random.choice(label.shape[0])
            seed[r] = label[r]
            new_batch_x[i] = np.concatenate((batch_x[i], seed), axis=-1)
        return new_batch_x, batch_y


class VolumeGenerator(Sequence):
    def __init__(self, files, batch_size, rescale=True, seed=False):
        self.files = glob.glob(files)
        self.shape = shape(self.files[0])
        self.batch_size = batch_size
        self.funcs = ['rescale', 'resize'] if rescale else ['resize']
        self.n = len(self.files)
        self.idx = 0
        self.seed = seed

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch = []
        for file in self.files[self.batch_size * idx:self.batch_size * (idx + 1)]:
            volume = preprocess(file, self.funcs)
            if self.seed:
                zeros = np.zeros(volume.shape)
                volume = np.concatenate((volume, zeros), axis=-1)
            batch.append(volume)
        return np.array(batch)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.idx < len(self):
            batch = self[self.idx]
            self.idx += 1
            return batch
        else:
            raise StopIteration()
