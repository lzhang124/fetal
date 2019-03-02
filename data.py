import constants
import numpy as np
from image3d import ImageTransformer, Iterator
from process import preprocess


def _format(file_formats, s, n):
    if isinstance(file_formats, str):
        return file_formats.format(s=s, n=str(n).zfill(4), p=str(max(0, n-1)).zfill(4))
    else:
        files = []
        for f in file_formats:
            files.append(f.format(s=s, n=str(n).zfill(4), p=str(max(0, n-1)).zfill(4)))
        return files


class DataGenerator(Iterator):
    def __init__(self,
                 frames,
                 input_file_format,
                 label_file_format=None,
                 label_types=None,
                 load_files=True,
                 random_gen=False,
                 augment=False,
                 resize=False,
                 tile_inputs=False,
                 batch_size=1,
                 seed=None):
        self.frames = frames
        self.samples = list(self.frames.keys())
        self.input_file_format = input_file_format
        self.label_file_format = label_file_format
        self.input_files = []
        self.label_files = None if self.label_file_format is None else []
        self.label_types = label_types
        self.load_files = load_files
        self.random_gen = random_gen
        self.augment = augment
        self.resize = resize
        self.tile_inputs = tile_inputs
        
        if not self.random_gen:
            for s in self.frames:
                for n in self.frames[s]:
                    self.input_files.append(_format(self.input_file_format, s, n))
                    if self.label_file_format:
                        self.label_files.append(_format(self.label_file_format, s, n))

        self.inputs = self.input_files
        self.labels = self.label_files

        if self.load_files:
            if self.random_gen:
                raise ValueError('Input sampling is only supported if files are not preloaded.')
            self.inputs = [preprocess(file, resize=self.resize, tile=self.tile_inputs) for file in self.input_files]
            if self.label_files is not None:
                self.labels = [preprocess(file, resize=self.resize, tile=self.tile_inputs) for file in self.label_files]
            if self.tile_inputs:
                self.inputs = np.reshape(self.inputs, (-1,) + np.asarray(self.inputs).shape[-4:])
                if self.label_files is not None:
                    self.labels = np.reshape(self.labels, (-1,) + np.asarray(self.labels).shape[-4:])
        elif self.tile_inputs:
            self.inputs = np.repeat(self.inputs, 8, axis=0)
            if self.label_files is not None:
                self.labels = np.repeat(self.labels, 8, axis=0)

        if self.augment:
            if self.tile_inputs:
                raise ValueError('Augmentation not supported if inputs are tiled.')
            self.image_transformer = ImageTransformer(rotation_range=90.,
                                                      shift_range=0.1,
                                                      shear_range=0.1,
                                                      zoom_range=0.1,
                                                      crop_size=constants.SHAPE,
                                                      fill_mode='nearest',
                                                      cval=0,
                                                      flip=True)

        super().__init__(max(len(self.samples), len(self.inputs)), batch_size, self.augment, seed)

    def _get_batch(self, index_array):
        batch = []
        if self.label_types is None:
            for _, i in enumerate(index_array):
                if self.load_files:
                    x = self.inputs[i]
                elif self.tile_inputs:
                    x = preprocess(self.inputs[i], tile=self.tile_inputs)[i%8]
                elif self.random_gen:
                    s = self.samples[i]
                    n = np.random.choice(self.frames[s])
                    x = preprocess(_format(self.input_file_format, s, n), resize=self.resize, tile=self.tile_inputs)
                else:
                    x = preprocess(self.inputs[i], resize=self.resize)
                if self.augment:
                    x = self.image_transformer.random_transform(x, seed=self.seed)
                batch.append(x)
            return np.asarray(batch)

        labels = []
        for _, i in enumerate(index_array):
            if self.load_files:
                x = self.inputs[i]
                y = self.labels[i]
            elif self.tile_inputs:
                x = preprocess(self.inputs[i], tile=self.tile_inputs)[i%8]
                y = preprocess(self.labels[i], tile=self.tile_inputs)[i%8]
            elif self.random_gen:
                s = self.samples[i]
                n = np.random.choice(self.frames[s])
                x = preprocess(_format(self.input_file_format, s, n), resize=self.resize, tile=self.tile_inputs)
                y = preprocess(_format(self.label_file_format, s, n), resize=self.resize, tile=self.tile_inputs)
            else:
                x = preprocess(self.inputs[i], resize=self.resize)
                y = preprocess(self.labels[i], resize=self.resize)
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
