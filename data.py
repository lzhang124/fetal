import constants
import numpy as np
from image3d import ImageTransformer, Iterator
from process import preprocess


class DataGenerator(Iterator):
    def __init__(self,
                 input_files,
                 label_files=None,
                 label_types=None,
                 load_files=True,
                 augment=False,
                 resize=False,
                 tile_inputs=False,
                 batch_size=1,
                 seed=None):
        self.input_files = input_files
        self.label_files = label_files
        self.label_types = label_types
        self.load_files = load_files
        self.augment = augment
        self.resize = resize
        self.tile_inputs = tile_inputs
        
        self.inputs = input_files
        self.labels = label_files

        if load_files:
            self.inputs = [preprocess(file, resize=self.resize, tile=self.tile_inputs) for file in self.input_files]
            if label_files is not None:
                self.labels = [preprocess(file, resize=self.resize, tile=self.tile_inputs) for file in self.label_files]
            if self.tile_inputs:
                print(np.asarray(self.inputs).shape)
                print(np.asarray(self.labels).shape)
                self.inputs = np.reshape(self.inputs, (-1,) + np.asarray(self.inputs).shape[-4:])
                self.labels = np.reshape(self.labels, (-1,) + np.asarray(self.labels).shape[-4:])
        if augment:
            self.image_transformer = ImageTransformer(rotation_range=90.,
                                             shift_range=0.1,
                                             shear_range=0.1,
                                             zoom_range=0.1,
                                             crop_size=constants.SHAPE,
                                             fill_mode='nearest',
                                             cval=0,
                                             flip=True)

        super().__init__(len(input_files), batch_size, self.augment, seed)

    def _get_batch(self, index_array):
        batch = []
        if self.label_types is None:
            for _, i in enumerate(index_array):
                if self.load_files:
                    x = self.inputs[i]
                else:
                    x = preprocess(self.inputs[i], resize=self.resize, tile=self.tile_inputs)
                if self.augment:
                    x = self.image_transformer.random_transform(x, seed=self.seed)
                batch.append(x)
            return np.asarray(batch)

        labels = []
        for _, i in enumerate(index_array):
            if self.load_files:
                x = self.inputs[i]
                y = self.labels[i]
            else:
                x = preprocess(self.inputs[i], resize=self.resize, tile=self.tile_inputs)
                y = preprocess(self.labels[i], resize=self.resize, tile=self.tile_inputs)
            if self.augment:
                x, y = self.image_transformer.random_transform(x, y, seed=self.seed)
            batch.append(x)
            labels.append(y)

        all_labels = []
        for label_type in self.label_types:
            if label_type == 'label':
                if self.labels is None:
                    raise ValueError('Labels not provided.')
                all_labels.append(labels)
            elif label_type == 'input':
                all_labels.append(batch)
            else:
                raise ValueError(f'Label type {label_type} is not supported.')
        if len(all_labels) == 1:
            all_labels = all_labels[0]
        return (np.asarray(batch), np.asarray(all_labels))


class VolumeGenerator(Iterator):
    def __init__(self,
                 input_files,
                 label_files=None,
                 label_types=None,
                 tile_inputs=False,
                 batch_size=1,
                 shuffle=False,
                 seed=None):
        self.input_files = input_files
        self.label_files = label_files
        self.label_types = label_types
        self.tile_inputs = tile_inputs
        
        self.inputs = input_files
        self.labels = label_files

        self.inputs = np.asarray([preprocess(file, resize=True, tile=tile_inputs) for file in input_files])
        self.inputs = np.reshape(self.inputs, (-1,) + self.inputs.shape[-4:])
        if label_files is not None:
            self.labels = np.asarray([preprocess(file, resize=True, tile=tile_inputs) for file in label_files])
            self.labels = np.reshape(self.labels, (-1,) + self.labels.shape[-4:])

        super().__init__(len(self.inputs), batch_size, shuffle, seed)

    def _get_batch(self, index_array):
        batch = []
        labels = []
        for _, i in enumerate(index_array):
            batch.append(self.inputs[i])
            labels.append(self.labels[i])

        all_labels = []
        for label_type in self.label_types:
            if label_type == 'label':
                if self.labels is None:
                    raise ValueError('Labels not provided.')
                all_labels.append(labels)
            elif label_type == 'input':
                all_labels.append(batch)
            else:
                raise ValueError(f'Label type {label_type} is not supported.')
        if len(all_labels) == 1:
            all_labels = all_labels[0]
        if len(all_labels) > 0:
            return (np.asarray(batch), np.asarray(all_labels))
        return np.asarray(batch)


class FrameGenerator(Iterator):
    def __init__(self, good_frames):
        pass

    def _get_batch(self, index_array):
        pass
