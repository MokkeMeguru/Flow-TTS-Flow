#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import tensorflow as tf

from TFGENZOO.flows.inv1x1conv import regular_matrix_init
from TFGENZOO.flows.flowbase import FlowComponent


class Inv1x1Conv2D(FlowComponent):
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):
        _, t, c = input_shape
        self.t = t
        self.c = c
        self.W = self.add_weight(
            name="W",
            shape=(c, c),
            regularizer=tf.keras.regularizers.l2(0.01),
            initializer=regular_matrix_init,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        config.update(config_update)
        return config

    def forward(self, x: tf.Tensor, **kwargs):
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(W, [1, self.c, self.c])
        z = tf.nn.conv1d(x, _W, [1, 1, 1], "SAME")
        # scalar
        log_det_jacobian = tf.cast(
            tf.linalg.slogdet(tf.cast(W, tf.float64))[1] * self.t, tf.float32,
        )
        # expand as batch
        log_det_jacobian = tf.broadcast_to(log_det_jacobian, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, **kwargs):
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(tf.linalg.inv(W), [1, self.c, self.c])
        x = tf.nn.conv1d(z, _W, [1, 1, 1], "SAME")

        inverse_log_det_jacobian = tf.cast(
            -1 * tf.linalg.slogdet(tf.cast(W, tf.float64))[1] * self.t, tf.float32,
        )

        inverse_log_det_jacobian = tf.broadcast_to(
            inverse_log_det_jacobian, tf.shape(z)[0:1]
        )
        return x, inverse_log_det_jacobian
