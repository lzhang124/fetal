from keras.models import load_model, Model
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class Model:
    def __init__(self, input_shape, filename=None):
        self.input_shape = input_shape
        self.model = self._new_model() if filename is None else load_model(filename)

    def _new_model(self):
        raise NotImplementedError()

    def train(self, generator):
        raise NotImplementedError()

    def predict(self, inputs):
        raise NotImplementedError()


class UNet(Model):
    def _new_model(self):
        inputs = Input(shape=self.input_shape)

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
        conc6 = concatenate([up6, conv4], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = concatenate([up7, conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = concatenate([up8, conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = concatenate([up9, conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizers=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

        return model

    def train(self, generator, epochs=10):
        model_checkpoint = ModelCheckpoint('unet_weights.{epoch:02d}-{loss:.2f}.hdf5',
                                           monitor='loss',
                                           save_best_only=True)

        self.model.fit_generator(generator, epochs=epochs, callbacks=[model_checkpoint])

    def predict(self, inputs):
        raise NotImplementedError()
