import numpy as np
import constants
from image3d import ImageTransformer, VolumeIterator
from keras.utils.data_utils import Sequence
from process import preprocess


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
                 flip=True,
                 label_types=None):
        self.input_files = input_files
        self.label_files = label_files
        self.inputs = [preprocess(file) for file in input_files]

        if label_files is not None:
            self.labels = [preprocess(file) for file in label_files]
        else:
            self.labels = None
            
        self.label_types = label_types

        image_transformer = ImageTransformer(rotation_range=rotation_range,
                                             shift_range=shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             crop_size=crop_size,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             flip=flip)

        super().__init__(self.inputs, self.labels, image_transformer, batch_size=batch_size)

    def _get_batches_of_transformed_samples(self, index_array):
        batch = super()._get_batches_of_transformed_samples(index_array)
        labels = None
        if self.labels is not None:
            batch, labels = batch

        all_labels = []
        for label_type in self.label_types:
            if label_type == 'label':
                if labels is None:
                    raise ValueError('Labels not provided.')
                all_labels.append(labels)
            elif label_type == 'input':
                all_labels.append(batch)
            else:
                raise ValueError(f'Label type {label_type} is not supported.')
        return (batch, all_labels) if len(all_labels) > 0 else batch


class VolumeGenerator(Sequence):
    def __init__(self,
                 input_files,
                 label_files=None,
                 batch_size=1,
                 label_types=None,
                 tile_inputs=False):
        self.input_files = input_files
        self.label_files = label_files
        self.inputs = np.array([preprocess(file, resize=True, tile=tile_inputs) for file in input_files])
        self.inputs = np.reshape(self.inputs, (-1,) + self.inputs.shape[2:])
        if label_files is not None:
            self.labels = np.array([preprocess(file, resize=True, tile=tile_inputs) for file in label_files])
            self.labels = np.reshape(self.labels, (-1,) + self.labels.shape[2:])
        else:
            self.labels = None

        self.batch_size = batch_size
        self.label_types = label_types
        self.tile_inputs = tile_inputs
        self.n = len(self.inputs)
        self.idx = 0
        
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch = np.array(self.inputs[self.batch_size * idx:self.batch_size * (idx + 1)])

        if self.label_types:
            if self.labels is None:
                raise ValueError('No labels provided.')

            all_labels = []
            for label_type in self.label_types:
                if label_type == 'label':
                    all_labels.append(np.array(self.labels[self.batch_size * idx:self.batch_size * (idx + 1)]))
                elif label_type == 'input':
                    all_labels.append(batch)
                else:
                    raise ValueError(f'Label type {label_type} is not supported.')
            if len(all_labels) > 0:
                batch = (batch, all_labels)
        
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
