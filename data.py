import numpy as np
from image3d import ImageTransformer, VolumeIterator
from keras.utils.data_utils import Sequence
from process import preprocess
from util import shape


class AugmentGenerator(VolumeIterator):
    def __init__(self,
                 input_files,
                 label_files=None,
                 batch_size=1,
                 seed_type=None,
                 concat_files=None,
                 rotation_range=90.,
                 shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.1,
                 fill_mode='nearest',
                 cval=0.,
                 flip=True):
        self.inputs = np.array([preprocess(file) for file in input_files])

        if label_files is not None:
            self.labels = np.array([preprocess(file, funcs=['resize']) for file in label_files])
        else:
            self.labels = None

        self.seed_type = seed_type

        if concat_files is not None:
            concat = np.concatenate((preprocess(concat_files[0]),
                                     preprocess(concat_files[1], funcs=['resize'])), axis=-1)
            new_inputs = []
            for vol in self.inputs:
                new_inputs.append(np.concatenate((vol, concat), axis=-1))
            self.inputs = np.array(new_inputs)

        image_transformer = ImageTransformer(rotation_range=rotation_range,
                                             shift_range=shift_range,
                                             shear_range=shear_range,
                                             zoom_range=zoom_range,
                                             fill_mode=fill_mode,
                                             cval=cval,
                                             flip=flip)

        super().__init__(self.inputs, self.labels, image_transformer, batch_size=batch_size)

    def _get_batches_of_transformed_samples(self, index_array):
        batch = super()._get_batches_of_transformed_samples(index_array)
        
        if self.seed_type is not None:
            if self.labels is None:
                raise ValueError('No labels to generate slices.')
            batch_x, batch_y = batch
            
            new_batch_x = np.zeros(tuple(list(batch_x.shape[:-1]) + [batch_x.shape[-1] + 1]))
            for i, label in enumerate(batch_y):
                if self.seed_type == 'slice':
                    seed = np.zeros(batch_x[i].shape)
                    r = np.random.choice(label.shape[0])
                    while not np.any(label[r]):
                        r = np.random.choice(label.shape[0])
                    seed[r] = label[r]
                elif self.seed_type == 'volume':
                    seed = label.copy()
                new_batch_x[i] = np.concatenate((batch_x[i], seed), axis=-1)
            batch = (new_batch_x, batch_y)

        return batch


class VolumeGenerator(Sequence):
    def __init__(self,
                 input_files,
                 seed_files=None,
                 label_files=None,
                 batch_size=1,
                 seed_type=None,
                 concat_files=None,
                 load_files=False,
                 include_labels=False,
                 rescale=True):
        self.inputs = input_files
        self.seeds = seed_files
        self.labels = label_files
        self.batch_size = batch_size
        self.seed_type = seed_type
        self.concat = None
        self.load_files = load_files
        self.include_labels = include_labels
        self.funcs = ['rescale', 'resize'] if rescale else ['resize']
        self.shape = shape(input_files[0])
        self.n = len(input_files)
        self.idx = 0
        
        if concat_files is not None:
            self.concat = np.concatenate((preprocess(concat_files[0]),
                                          preprocess(concat_files[1], funcs=['resize'])), axis=-1)

        if load_files:
            self.inputs = np.array([preprocess(file, self.funcs) for file in input_files])
            if self.concat is not None:
                new_inputs = []
                for vol in self.inputs:
                    new_inputs.append(np.concatenate((vol, self.concat), axis=-1))
                self.inputs = np.array(new_inputs)
            if seed_files is not None:
                self.seeds = np.array([preprocess(file, ['resize']) for file in seed_files])
            if label_files is not None:
                self.labels = np.array([preprocess(file, ['resize']) for file in label_files])

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch = []
        for file in self.inputs[self.batch_size * idx:self.batch_size * (idx + 1)]:
            volume = file if self.load_files else preprocess(file, self.funcs)
            if self.concat is not None:
                volume = np.concatenate((volume, self.concat), axis=-1)
            batch.append(volume)
        batch = np.array(batch)

        if self.seeds is not None:
            seeds = []
            for file in self.seeds[self.batch_size * idx:self.batch_size * (idx + 1)]:
                seed = file if self.load_files else preprocess(file, ['resize'])
                seeds.append(seed)
            batch = np.concatenate((batch, np.array(seeds)), axis=-1)

        if self.seed_type is not None:
            if self.labels is None:
                raise ValueError('No labels to generate slices.')
            if self.seeds is not None:
                raise ValueError('Seeds already exist.')

            new_batch = np.zeros(tuple(list(batch.shape[:-1]) + [batch.shape[-1] + 1]))
            for i, file in enumerate(self.labels[self.batch_size * idx:self.batch_size * (idx + 1)]):
                label = file if self.load_files else preprocess(file, ['resize'])
                if self.seed_type == 'slice':
                    seed = np.zeros(batch[i].shape)
                    r = np.random.choice(label.shape[0])
                    while not np.any(label[r]):
                        r = np.random.choice(label.shape[0])
                    seed[r] = label[r]
                elif self.seed_type == 'volume':
                    seed = label.copy()
                new_batch[i] = np.concatenate((batch[i], seed), axis=-1)
            batch = new_batch

        if self.include_labels:
            if self.labels is None:
                raise ValueError('No labels provided.')

            labels = []
            for file in self.labels[self.batch_size * idx:self.batch_size * (idx + 1)]:
                label = file if self.load_files else preprocess(file, ['resize'])
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
