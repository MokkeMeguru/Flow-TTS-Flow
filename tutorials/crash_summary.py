#!/usr/bin/env python3
import tensorflow as tf
from TFGENZOO.flows.flowbase import FlowBase


class Inv1x1Conv2D(FlowBase):
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):
        _, t, c = input_shape
        self.c = c
        self.W = self.add_weight(
            name="W", shape=(c, c), regularizer=tf.keras.regularizers.l2(0.01)
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        config.update(config_update)
        return config

    def call(self, x, inverse=False):
        if inverse:
            return self.inverse(x)
        else:
            return self.forward(x)

    def forward(self, x: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        """
        Args:
            x    (tf.Tensor): base input tensor [B, T, C]
            mask (tf.Tensor): mask input tensor [B, T]

        Returns:
            z    (tf.Tensor): latent variable tensor [B, T, C]
            ldj  (tf.Tensor): log det jacobian [B]

        Notes:
            * mask's example
                | [[True, True, True, False],
                |  [True, False, False, False],
                |  [True, True, True, True],
                |  [True, True, True, True]]
        """
        _, t, _ = x.shape
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(W, [1, self.c, self.c])
        z = tf.nn.conv1d(x, _W, [1, 1, 1], "SAME")

        # scalar
        # tf.math.log(tf.abs(tf.linalg.det(W))) == tf.linalg.slogdet(W)[1]
        log_det_jacobian = tf.cast(
            tf.linalg.slogdet(tf.cast(W, tf.float64))[1], tf.float32,
        )

        # expand as batch
        if mask is not None:
            # mask -> mask_tensor: [B, T] -> [B, T, 1]
            mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), [-1])
            z = z * mask_tensor
            log_det_jacobian = log_det_jacobian * tf.reduce_sum(mask, axis=[-1])
        else:
            log_det_jacobian = log_det_jacobian * t
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        _, t, _ = z.shape
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(tf.linalg.inv(W), [1, self.c, self.c])
        x = tf.nn.conv1d(z, _W, [1, 1, 1], "SAME")

        inverse_log_det_jacobian = tf.cast(
            -1 * tf.linalg.slogdet(tf.cast(W, tf.float64))[1], tf.float32,
        )

        if mask is not None:
            # mask -> mask_tensor: [B, T] -> [B, T, 1]
            mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), [-1])
            x = x * mask_tensor
            inverse_log_det_jacobian = inverse_log_det_jacobian * tf.reduce_sum(
                tf.cast(mask, tf.float32), axis=[-1]
            )
        else:
            inverse_log_det_jacobian = tf.broadcast_to(
                inverse_log_det_jacobian * t, tf.shape(z)[0:1]
            )
        return x, inverse_log_det_jacobian


class SubClass(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(128)
        self.dense2 = tf.keras.layers.Dense(64)
        super().build(input_shape)

    def init(self):
        super().__init__()

    def call(self, x):
        return self.dense2(self.dense1(x))


class SubSubClass(tf.keras.layers.Layer):
    def init(self):
        super().__init__()

    def build(self, input_shape):
        self.subc = [SubClass()]
        super().build(input_shape)

    def call(self, x, inverse=False):
        if inverse:
            return self.inverse(x)
        else:
            return self.forward(x)

    def forward(self, x):
        for l in reversed(self.subc):

            x = l(x)
        return x

    def inverse(self, x):
        for l in self.subc:
            x = l(x)
        return x


class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mylayer = Inv1x1Conv2D()  # SubSubClass()

    def call(self, x):
        return self.mylayer(x)


if __name__ == "__main__":
    model = CustomModel()
    model(tf.random.normal([32, 32, 12]))
    model.summary()
