import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(seq_len, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, seq_len, d_model):
        angle_rads = self.get_angles(np.arange(seq_len)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len, 'd_model': self.d_model})
        return config
