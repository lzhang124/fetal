import nibabel as nib
import os
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import layers
from process import uncrop
from util import get_weights, save_vol


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weighted_crossentropy(weight=None, boundary_weight=None, pool=3):
    w = (.5, .5) if weight is None else weight

    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        cross_entropy = K.stack([-(y_true * K.log(y_pred)), -((1 - y_true) * K.log(1 - y_pred))])
        loss = cross_entropy * w

        if boundary_weight is not None:
            y_true_avg = K.pool3d(y_true, pool_size=(pool,)*3, padding='same', pool_mode='avg')
            boundaries = K.cast(y_true_avg >= K.epsilon(), 'float32') \
                         * K.cast(y_true_avg <= 1 - K.epsilon(), 'float32')
            loss += cross_entropy * boundary_weight * boundaries

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

    def _compile(self, weight):
        raise NotImplementedError()

    def train(self, generator, epochs):
        self._compile(get_weights(generator.labels))
        fname = 'models/{}_weights'.format(self.name)
        model_checkpoint = ModelCheckpoint(fname + '.{epoch:03d}-{loss:.4f}-{dice_coef:.4f}.h5',
                                           monitor='loss',
                                           save_best_only=True,
                                           save_weights_only=True)
        self.model.fit_generator(generator, epochs=epochs, callbacks=[model_checkpoint], verbose=1)

    def predict(self, generator, path):
        preds = self.model.predict_generator(generator, verbose=1)
        for i in range(preds.shape[0]):
            fname = generator.files[i].split('/')[-1]
            # FIXME
            header = nib.load(generator.files[i]).header
            save_vol(uncrop(preds[i], generator.shape), os.path.join(path, fname), header)


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

    def _compile(self, weight):
        self.model.compile(optimizer=Adam(lr=1e-5),
                           loss=weighted_crossentropy(weight=weight, boundary_weight=.2),
                           metrics=['accuracy', dice_coef])
