#!/usr/bin/env python3

import tensorflow as tf


def Sample(x, depth=128):

    # this expression causes a bug
    last_channel = x.shape[-1]

    basic_conv = tf.keras.layers.Conv2D(depth, 3)

    same_conv = tf.keras.layers.Conv2D(depth, 1)

    twice_conv = tf.keras.layers.Conv2D(depth, 1)

    y = basic_conv(x)
    skip = y
    y = same_conv(y)
    # You can not use "+" operator in Functional API
    # y = y + skip
    y = tf.keras.layers.Add()([y, skip])
    y = twice_conv(y)

    model = tf.keras.Model(x, y)
    model.summary()
    return model


class SampleUseLayer(tf.keras.Model):
    def build(self, input_shape):
        self.sample = self.sample_func(tf.keras.layers.Input(input_shape[1:]))

    def __init__(self, sample_func=lambda x: Sample(x, depth=128)):
        super().__init__()
        self.sample_func = sample_func

    def call(self, x):
        return self.sample(x)


if __name__ == "__main__":
    sampleuselayer = SampleUseLayer()
    x = tf.keras.layers.Input([None, None, 3])
    y = sampleuselayer(x)
    tf.keras.Model(x, y).summary()
