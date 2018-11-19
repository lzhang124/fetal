import numpy as np
import constants
from image3d import ImageTransformer, VolumeIterator
from keras.utils.data_utils import Sequence
from process import preprocess
from util import shape

class AugmentGenerator(VolumeIterator):
    def __init__(self,
                 input_files,
                 label_files=None,
                 batch_size=1,
                 rotation_range=90.,
                 shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.1,
                 crop_size=constants.SHAPE,
                 fill_mode='nearest',
                 cval=0.,
                 flip=True):
        self.inputs = [preprocess(file, rescale=True) for file in input_files]

        if label_files is not None:
            self.labels = [preprocess(file) for file in label_files]
        else:
            self.labels = None

        image_transformer = ImageTransformer(rotation_range=rotation_range,
                                             shift_range=shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             crop_size=crop_size,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             flip=flip)

        super().__init__(self.inputs, self.labels, image_transformer, batch_size=batch_size)


class VolumeGenerator(Sequence):
    def __init__(self,
                 input_files,
                 label_files=None,
                 batch_size=1,
                 load_files=False,
                 include_labels=False,
                 rescale=True):
        self.inputs = input_files
        self.labels = label_files
        self.batch_size = batch_size
        self.load_files = load_files
        self.include_labels = include_labels
        self.rescale = rescale
        self.shapes = [shape(file) for file in input_files]
        self.n = len(input_files)
        self.idx = 0
        
        if load_files:
            self.inputs = np.array([preprocess(file, resize=True, rescale=self.rescale)
                                    for file in input_files])

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch = []
        for file in self.inputs[self.batch_size * idx:self.batch_size * (idx + 1)]:
            if self.load_files:
                volume = file
            else:
                volume = preprocess(file, resize=True, rescale=self.rescale)
            batch.append(volume)
        batch = np.array(batch)

        if self.include_labels:
            if self.labels is None:
                raise ValueError('No labels provided.')

            labels = []
            for file in self.labels[self.batch_size * idx:self.batch_size * (idx + 1)]:
                label = file if self.load_files else preprocess(file, resize=True)
                labels.append(label)
            labels = np.array(labels)
            batch = (batch, labels)
        
        return batch

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
