import numpy as np
import constants
from image3d import ImageTransformer, VolumeIterator
from keras.utils.data_utils import Sequence
from process import preprocess


class AugmentGenerator(VolumeIterator):
    def __init__(self,
                 input_files,
                 label_files=None,
                 concat_files=None,
                 batch_size=1,
                 rotation_range=90.,
                 shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.1,
                 crop_size=constants.SHAPE,
                 fill_mode='nearest',
                 cval=0.,
                 flip=True,
                 label_types=None,
                 load_files=True):
        self.input_files = input_files
        self.label_files = label_files
        self.concat_files = concat_files
        self.inputs = input_files
        self.labels = label_files

        if load_files:
            self.inputs = [preprocess(file) for file in input_files]
            if concat_files is not None:
                concats = [[preprocess(file) for file in channel] for channel in concat_files]
                self.inputs = np.concatenate((self.inputs, *concats), axis=-1)
            if label_files is not None:
                self.labels = [preprocess(file) for file in label_files]
        elif concat_files is not None:
            self.inputs = np.stack((self.inputs, *concat_files), axis=1)

        self.label_types = label_types
        self.load_files = load_files

        image_transformer = ImageTransformer(rotation_range=rotation_range,
                                             shift_range=shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             crop_size=crop_size,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             flip=flip)

        super().__init__(self.inputs, self.labels, image_transformer, batch_size=batch_size)

    def _get_batches_of_transformed_samples(self, index_array, load_fn=None):
        if not self.load_files:
            def load(sample):
                if isinstance(sample, str):
                    return preprocess(sample)
                else:
                    return np.concatenate([preprocess(s) for s in sample], axis=-1)
            load_fn = load
        batch = super()._get_batches_of_transformed_samples(index_array, load_fn=load_fn)    
        labels = None
        if self.label_types is not None:
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
                 concat_files=None,
                 batch_size=1,
                 label_types=None,
                 load_files=True,
                 tile_inputs=False):
        self.input_files = input_files
        self.label_files = label_files
        self.concat_files = concat_files
        self.inputs = input_files
        self.labels = label_files

        if load_files:
            self.inputs = np.array([preprocess(file, resize=True, tile=tile_inputs) for file in self.inputs])
            self.inputs = np.reshape(self.inputs, (-1,) + self.inputs.shape[-4:])

            if concat_files is not None:
                concats = [[preprocess(file, resize=True, tile=tile_inputs) for file in channel] for channel in self.concat_files]
                concats = np.reshape(concats, (-1,) + self.inputs.shape[-4:-1] + (len(concats),))
                self.inputs = np.concatenate((self.inputs, *concats), axis=-1)

            if label_files is not None:
                self.labels = np.array([preprocess(file, resize=True, tile=tile_inputs) for file in self.labels])
                self.labels = np.reshape(self.labels, (-1,) + self.labels.shape[-4:])

        self.batch_size = batch_size
        self.label_types = label_types
        self.load_files = load_files
        self.tile_inputs = tile_inputs
        self.n = len(self.inputs)
        self.idx = 0

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch_start = self.batch_size * idx
        batch_end = self.batch_size * (idx + 1)

        if self.load_files:
            batch = np.array(self.inputs[batch_start:batch_end])
        else:
            batch = np.array([preprocess(file, resize=True, tile=self.tile_inputs) for file in self.inputs[batch_start:batch_end]])
            batch = np.reshape(batch, (-1,) + batch.shape[-4:])
            if self.concat_files is not None:
                concats = [[preprocess(file, resize=True, tile=self.tile_inputs) for file in channel[batch_start:batch_end]] for channel in self.concat_files]
                concats = np.reshape(concats, (-1,) + batch.shape[-4:-1] + (len(concats),))
                batch = np.concatenate((batch, *concats), axis=-1)

        if self.label_types:
            if self.labels is None:
                raise ValueError('No labels provided.')

            all_labels = []
            for label_type in self.label_types:
                if label_type == 'label':
                    if self.load_files:
                        label = np.array(self.labels[batch_start:batch_end])
                    else:
                        label = np.array([preprocess(file, resize=True, tile=self.tile_inputs) for file in self.labels[batch_start:batch_end]])
                        label = np.reshape(label, (-1,) + label.shape[-4:])
                    all_labels.append(label)
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
