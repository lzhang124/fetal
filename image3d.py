"""Fairly basic set of tools for real-time data augmentation on 3D image data.

Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
import nibabel as nib

from keras import backend as K
from keras.utils.data_utils import Sequence


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 3D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    final_affine_matrix = transform_matrix[:-1, :-1]
    final_offset = transform_matrix[:-1, -1]
    x = ndi.interpolation.affine_transform(
        x,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x):
    return nib.Nifti1Image(x.astype('int16'), np.eye(4))


def img_to_array(img):
    x = np.squeeze(img.get_data()).astype(K.floatx())
    if len(x.shape) != 3:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        rotation_range: degrees (0 to 180).
        shift_range: fraction of each dimension.
        shear_range: shear intensity (fraction).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        flip: whether to randomly flip images along its axes.
    """

    def __init__(self,
                 rotation_range=0.,
                 shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 flip=False):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.flip = flip

        self.x_axis = 1
        self.y_axis = 2
        self.z_axis = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

        if shear_range > 1:
            raise ValueError('`shear_range` should be a float. '
                             'Received arg: ', shear_range)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='nii.gz'):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    # def flow_from_directory(self, directory,
    #                         target_size=(256, 256), color_mode='rgb',
    #                         classes=None, class_mode='categorical',
    #                         batch_size=32, shuffle=True, seed=None,
    #                         save_to_dir=None,
    #                         save_prefix='',
    #                         save_format='png',
    #                         follow_links=False,
    #                         interpolation='nearest'):
    #     return DirectoryIterator(
    #         directory, self,
    #         target_size=target_size, color_mode=color_mode,
    #         classes=classes, class_mode=class_mode,
    #         batch_size=batch_size, shuffle=shuffle, seed=seed,
    #         save_to_dir=save_to_dir,
    #         save_prefix=save_prefix,
    #         save_format=save_format,
    #         follow_links=follow_links,
    #         interpolation=interpolation)

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        transform_matrix = None

        if self.rotation_range:
            rx, ry, rz = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range, 3))
            Rx = np.array([[1, 0, 0, 0],
                           [0, np.cos(rx), -np.sin(rx), 0],
                           [0, np.sin(rx), np.cos(rx), 0],
                           [0, 0, 0, 1]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                           [0, 1, 0, 0],
                           [-np.sin(ry), 0, np.cos(ry), 0],
                           [0, 0, 0, 1]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                           [np.sin(rz), np.cos(rz), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
            rotation_matrix = np.dot(np.dot(Rx, Ry), Rz)
            transform_matrix = rotation_matrix

        if self.shift_range:
            tx, ty, tz = np.random.uniform(-self.shift_range, self.shift_range, 3)
            if self.shift_range < 1:
                tx *= x.shape[img_x_axis]
                ty *= x.shape[img_y_axis]
                tz *= x.shape[img_z_axis]
            shift_matrix = np.array([[1, 0, 0, tx],
                                     [0, 1, 0, ty],
                                     [0, 0, 1, tz],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if self.shear_range:
            sxy, sxz, syx, syz, szx, szy = np.random.uniform(-self.shear_range, self.shear_range, 6)
            shear_matrix = np.array([[1, sxy, sxz, 0],
                                     [syx, 1, syz, 0], 
                                     [szx, szy, 1, 0],
                                     [0, 0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
            zoom_matrix = np.array([[zx, 0, 0, 0],
                                    [0, zy, 0, 0],
                                    [0, 0, zz, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            x = apply_transform(x, transform_matrix, fill_mode=self.fill_mode, cval=self.cval)

        if self.flip:
            for axis in range(1, 4):
                if np.random.random() < 0.5:
                    x = flip_axis(x, axis)

        return x


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('x (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: x.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i])
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.to_filename(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


# def _count_valid_files_in_directory(directory, white_list_formats, follow_links):
#     """Count files with extension in `white_list_formats` contained in directory.

#     # Arguments
#         directory: absolute path to the directory
#             containing files to be counted
#         white_list_formats: set of strings containing allowed extensions for
#             the files to be counted.
#         follow_links: boolean.

#     # Returns
#         the count of files with extension in `white_list_formats` contained in
#         the directory.
#     """
#     def _recursive_list(subpath):
#         return sorted(os.walk(subpath, followlinks=follow_links), key=lambda x: x[0])

#     samples = 0
#     for _, _, files in _recursive_list(directory):
#         for fname in files:
#             is_valid = False
#             for extension in white_list_formats:
#                 if fname.lower().endswith('.tiff'):
#                     warnings.warn('Using \'.tiff\' files with multiple bands will cause distortion. '
#                                   'Please verify your output.')
#                 if fname.lower().endswith('.' + extension):
#                     is_valid = True
#                     break
#             if is_valid:
#                 samples += 1
#     return samples


# def _list_valid_filenames_in_directory(directory, white_list_formats,
#                                        class_indices, follow_links):
#     """List paths of files in `subdir` with extensions in `white_list_formats`.

#     # Arguments
#         directory: absolute path to a directory containing the files to list.
#             The directory name is used as class label and must be a key of `class_indices`.
#         white_list_formats: set of strings containing allowed extensions for
#             the files to be counted.
#         class_indices: dictionary mapping a class name to its index.
#         follow_links: boolean.

#     # Returns
#         classes: a list of class indices
#         filenames: the path of valid files in `directory`, relative from
#             `directory`'s parent (e.g., if `directory` is "dataset/class1",
#             the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
#     """
#     def _recursive_list(subpath):
#         return sorted(os.walk(subpath, followlinks=follow_links), key=lambda x: x[0])

#     classes = []
#     filenames = []
#     subdir = os.path.basename(directory)
#     basedir = os.path.dirname(directory)
#     for root, _, files in _recursive_list(directory):
#         for fname in sorted(files):
#             is_valid = False
#             for extension in white_list_formats:
#                 if fname.lower().endswith('.' + extension):
#                     is_valid = True
#                     break
#             if is_valid:
#                 classes.append(class_indices[subdir])
#                 # add filename relative to directory
#                 absolute_path = os.path.join(root, fname)
#                 filenames.append(os.path.relpath(absolute_path, basedir))
#     return classes, filenames


# class DirectoryIterator(Iterator):
#     """Iterator capable of reading images from a directory on disk.

#     # Arguments
#         directory: Path to the directory to read images from.
#             Each subdirectory in this directory will be
#             considered to contain images from one class,
#             or alternatively you could specify class subdirectories
#             via the `classes` argument.
#         image_data_generator: Instance of `ImageDataGenerator`
#             to use for random transformations and normalization.
#         target_size: tuple of integers, dimensions to resize input images to.
#         color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
#         classes: Optional list of strings, names of subdirectories
#             containing images from each class (e.g. `["dogs", "cats"]`).
#             It will be computed automatically if not set.
#         class_mode: Mode for yielding the targets:
#             `"binary"`: binary targets (if there are only two classes),
#             `"categorical"`: categorical targets,
#             `"sparse"`: integer targets,
#             `"input"`: targets are images identical to input images (mainly
#                 used to work with autoencoders),
#             `None`: no targets get yielded (only input images are yielded).
#         batch_size: Integer, size of a batch.
#         shuffle: Boolean, whether to shuffle the data between epochs.
#         seed: Random seed for data shuffling.
#         data_format: String, one of `channels_first`, `channels_last`.
#         save_to_dir: Optional directory where to save the pictures
#             being yielded, in a viewable format. This is useful
#             for visualizing the random transformations being
#             applied, for debugging purposes.
#         save_prefix: String prefix to use for saving sample
#             images (if `save_to_dir` is set).
#         save_format: Format to use for saving sample images
#             (if `save_to_dir` is set).
#         interpolation: Interpolation method used to resample the image if the
#             target size is different from that of the loaded image.
#             Supported methods are "nearest", "bilinear", and "bicubic".
#             If PIL version 1.1.3 or newer is installed, "lanczos" is also
#             supported. If PIL version 3.4.0 or newer is installed, "box" and
#             "hamming" are also supported. By default, "nearest" is used.
#     """

#     def __init__(self, directory, image_data_generator,
#                  target_size=(256, 256), color_mode='rgb',
#                  classes=None, class_mode='categorical',
#                  batch_size=32, shuffle=True, seed=None,
#                  data_format=None, save_to_dir=None,
#                  save_prefix='', save_format='png',
#                  follow_links=False, interpolation='nearest'):
#         if data_format is None:
#             data_format = K.image_data_format()
#         self.directory = directory
#         self.image_data_generator = image_data_generator
#         self.target_size = tuple(target_size)
#         if color_mode not in {'rgb', 'grayscale'}:
#             raise ValueError('Invalid color mode:', color_mode,
#                              '; expected "rgb" or "grayscale".')
#         self.color_mode = color_mode
#         self.data_format = data_format
#         if self.color_mode == 'rgb':
#             if self.data_format == 'channels_last':
#                 self.image_shape = self.target_size + (3,)
#             else:
#                 self.image_shape = (3,) + self.target_size
#         else:
#             if self.data_format == 'channels_last':
#                 self.image_shape = self.target_size + (1,)
#             else:
#                 self.image_shape = (1,) + self.target_size
#         self.classes = classes
#         if class_mode not in {'categorical', 'binary', 'sparse',
#                               'input', None}:
#             raise ValueError('Invalid class_mode:', class_mode,
#                              '; expected one of "categorical", '
#                              '"binary", "sparse", "input"'
#                              ' or None.')
#         self.class_mode = class_mode
#         self.save_to_dir = save_to_dir
#         self.save_prefix = save_prefix
#         self.save_format = save_format
#         self.interpolation = interpolation

#         white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

#         # first, count the number of samples and classes
#         self.samples = 0

#         if not classes:
#             classes = []
#             for subdir in sorted(os.listdir(directory)):
#                 if os.path.isdir(os.path.join(directory, subdir)):
#                     classes.append(subdir)
#         self.num_classes = len(classes)
#         self.class_indices = dict(zip(classes, range(len(classes))))

#         pool = multiprocessing.pool.ThreadPool()
#         function_partial = partial(_count_valid_files_in_directory,
#                                    white_list_formats=white_list_formats,
#                                    follow_links=follow_links)
#         self.samples = sum(pool.map(function_partial,
#                                     (os.path.join(directory, subdir)
#                                      for subdir in classes)))

#         print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

#         # second, build an index of the images in the different class subfolders
#         results = []

#         self.filenames = []
#         self.classes = np.zeros((self.samples,), dtype='int32')
#         i = 0
#         for dirpath in (os.path.join(directory, subdir) for subdir in classes):
#             results.append(pool.apply_async(_list_valid_filenames_in_directory,
#                                             (dirpath, white_list_formats,
#                                              self.class_indices, follow_links)))
#         for res in results:
#             classes, filenames = res.get()
#             self.classes[i:i + len(classes)] = classes
#             self.filenames += filenames
#             i += len(classes)
#         pool.close()
#         pool.join()
#         super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

#     def _get_batches_of_transformed_samples(self, index_array):
#         batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
#         grayscale = self.color_mode == 'grayscale'
#         # build batch of image data
#         for i, j in enumerate(index_array):
#             fname = self.filenames[j]                           interpolation=self.interpolation)
#             x = img_to_array(img, data_format=self.data_format)
#             x = self.image_data_generator.random_transform(x)
#             batch_x[i] = x
#         # optionally save augmented images to disk for debugging purposes
#         if self.save_to_dir:
#             for i, j in enumerate(index_array):
#                 img = array_to_img(batch_x[i], self.data_format, scale=True)
#                 fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                   index=j,
#                                                                   hash=np.random.randint(1e7),
#                                                                   format=self.save_format)
#                 img.save(os.path.join(self.save_to_dir, fname))
#         # build batch of labels
#         if self.class_mode == 'input':
#             batch_y = batch_x.copy()
#         elif self.class_mode == 'sparse':
#             batch_y = self.classes[index_array]
#         elif self.class_mode == 'binary':
#             batch_y = self.classes[index_array].astype(K.floatx())
#         elif self.class_mode == 'categorical':
#             batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
#             for i, label in enumerate(self.classes[index_array]):
#                 batch_y[i, label] = 1.
#         else:
#             return batch_x
#         return batch_x, batch_y

#     def next(self):
#         """For python 2.x.

#         # Returns
#             The next batch.
#         """
#         with self.lock:
#             index_array = next(self.index_generator)
#         # The transformation of images is not under thread lock
#         # so it can be done in parallel
#         return self._get_batches_of_transformed_samples(index_array)
