import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as backend

# rescale the output of RNN (always ranges from -1 to 1) to the time-varying parameters (any value)
class rescale_layer(keras.layers.Layer):
    def __init__(self, units, data, k, name=None, regularizers=None) -> None:
        super(rescale_layer, self).__init__(name=name, activity_regularizer=regularizers)
        self.units = units
        self.state_size = units
        self.data = data
        self.k = k

    def build(self, input_shape):
        self.scale_weight = self.add_weight(
            shape=(self.units, self.k),
            initializer='RandomNormal',
            name='scale_weight', regularizer=keras.regularizers.l1(0.01))

    def call(self, inputs):
        scaled_h = backend.dot(inputs, self.scale_weight)

        return scaled_h


# linear function with time-varyinG parameters
class linear_layer(keras.layers.Layer):
    def __init__(self, units, data,name=None) -> None:
        super(linear_layer, self).__init__(name=name)
        self.units = units
        self.state_size = units
        self.data = data

    def build(self, input_shape):
        pass


    def call(self, inputs, y_fix):
        out_put = y_fix[:, :, 0] + tf.math.reduce_sum(tf.math.multiply(self.data, inputs), 2)

        return out_put

