import constants
import os
from keras.models import Model
from keras.layers import concatenate, Conv3D, Conv3DTranspose, Dense, Input, MaxPooling3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from process import postprocess
from util import save_vol


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class BaseModel:
    def __init__(self, filename=None):
        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)
        self._compile()

    def _new_model(self):
        raise NotImplementedError()

    def _compile(self):
        raise NotImplementedError()

    def train(self, generator, epochs):
        fname = 'models/{}_weights'.format(self.__class__.__name__.lower())
        model_checkpoint = ModelCheckpoint(fname + '.{epoch:02d}-{loss:.4f}.h5',
                                           monitor='loss',
                                           save_best_only=True,
                                           save_weights_only=True)

        self.model.fit_generator(generator, epochs=epochs, callbacks=[model_checkpoint])

    def predict(self, generator, path):
        preds = self.model.predict_generator(generator)
        for i in range(preds.shape[0]):
            fname = generator.files[i].split('/')[-1]
            save_vol(postprocess(preds[i]), os.path.join(path, fname))


class UNet(BaseModel):
    def _new_model(self):
        inputs = Input(shape=constants.TARGET_SHAPE)

        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        conc6 = concatenate([up6, conv4])
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = concatenate([up7, conv3])
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = concatenate([up8, conv2])
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = concatenate([up9, conv1])
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss=dice_coef_loss,
                           metrics=[dice_coef])


class AutoEncoder(BaseModel):
    def _new_model(self):
        inputs = Input(shape=constants.TARGET_SHAPE)

        conv1 = Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)

        conv2 = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(conv1)
        conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)

        conv3 = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(conv2)
        conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)

        conv4 = Conv3D(1, (3, 3, 3), strides=(3, 3, 3), activation='relu', padding='same')(conv3)
        embed = Dense(64)(conv4)
        dense = Dense(128, activation='relu')(embed)

        up5 = Conv3DTranspose(64, (7, 7, 7), strides=(3, 3, 3), activation='relu', padding='same')(dense)
        conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)

        up6 = Conv3DTranspose(32, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same')(conv5)
        conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)

        up7 = Conv3DTranspose(16, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same')(conv6)
        conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up7)

        up8 = Conv3DTranspose(16, (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same')(conv7)
        outputs = Conv3D(1, (3, 3, 3), padding='same')(up8)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def encode(self):
        raise NotImplementedError()
