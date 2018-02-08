from keras.models import load_model
import keras.layers as layers


class Model:
    def __init__(self, input_shape, filename=None):
        self.input_shape = input_shape
        self.model = self._new_model() if filename is None else load_model(filename)

    def _new_model(self):
        raise NotImplementedError()

    def train(self, generator):
        raise NotImplementedError()


class UNet(Model):
    def _new_model(self):
        inputs = layers.Input(shape=self.input_shape)

    def train(self, generator):
        raise NotImplementedError()
