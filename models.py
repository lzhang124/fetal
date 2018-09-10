import os
import tensorflow as tf
import util
from datetime import datetime
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import layers
from process import uncrop
from time import time


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weighted_crossentropy(weight=None, boundary_weight=None, pool=3):
    w = (.5, .5) if weight is None else weight
    epsilon = K.epsilon()

    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy = K.stack([-(y_true * K.log(y_pred)), -((1 - y_true) * K.log(1 - y_pred))],
                                axis=-1)
        loss = cross_entropy * w

        if boundary_weight is not None:
            y_true_avg = K.pool3d(y_true, pool_size=(pool,)*3, padding='same', pool_mode='avg')
            boundaries = K.cast(y_true_avg >= epsilon, 'float32') \
                         * K.cast(y_true_avg <= 1 - epsilon, 'float32')
            loss += cross_entropy * K.stack([boundaries, boundaries], axis=-1) * boundary_weight

        return K.mean(K.sum(loss, axis=-1))
    return loss_fn


class BaseModel:
    def __init__(self, input_size, name=None, filename=None):
        self.input_size = input_size
        self.name = name if name else self.__class__.__name__.lower()
        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)

    def _new_model(self):
        raise NotImplementedError()

    def save(self):
        self.model.save('models/{}_weights.{}.h5'.format(self.name, datetime.now().strftime('%m.%d.%y-%H:%M:%S')))

    def compile(self, weight):
        raise NotImplementedError()

    def train(self, generator, val_gen, epochs, tensorboard=False):
        callbacks = [TensorBoard(log_dir='./logs/{}'.format(time()))] if tensorboard else []
        self.model.fit_generator(generator,
                                 epochs=epochs,
                                 validation_data=val_gen,
                                 verbose=1,
                                 callbacks=callbacks)

    def predict(self, generator, path):
        preds = self.model.predict_generator(generator, verbose=1)
        # FIXME
        for i in range(preds.shape[0]):
            fname = generator.inputs[i].split('/')[-1]
            header = util.header(generator.inputs[i])
            util.save_vol(uncrop(preds[i], generator.shape), os.path.join(path, fname), header)

    def test(self, generator):
        return self.model.evaluate_generator(generator)


class UNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        conc6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = layers.concatenate([up9, conv1])
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self, weight=None):
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss=weighted_crossentropy(weight=weight, boundary_weight=2.),
                           metrics=['accuracy', dice_coef])


class UNetSmall(UNet):
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)

        up5 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
        conc5 = layers.concatenate([up5, conv3])
        conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc5)
        conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        conc6 = layers.concatenate([up6, conv2])
        conv6 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv1])
        conv7 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

        self.model = Model(inputs=inputs, outputs=outputs)


class UNetBig(UNet):
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)
        pool5 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv5)

        conv6 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool5)
        conv6 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv5])
        conv7 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = layers.concatenate([up8, conv4])
        conv8 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = layers.concatenate([up9, conv3])
        conv9 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)

        up10 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9)
        conc10 = layers.concatenate([up10, conv2])
        conv10 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc10)
        conv10 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv10)

        up11 = layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10)
        conc11 = layers.concatenate([up11, conv1])
        conv11 = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conc11)
        conv11 = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv11)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv11)

        self.model = Model(inputs=inputs, outputs=outputs)
