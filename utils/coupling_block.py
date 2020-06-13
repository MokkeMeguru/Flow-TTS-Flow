#!/usr/bin/env python3

import tensorflow as tf


class GTU(tf.keras.layers.Layer):
    """GTU layer proposed in Flow-TTS

    Notes:

        * formula
            .. math::

                z = tanh(W_{f, k} \star y) \odot sigmoid(W_{g, k} \star c)
    """

    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):

        self.conv_first = tf.keras.layers.Conv1D(
            input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
        )
        self.conv_last = tf.keras.layers.Conv1D(
            input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
        )

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        config.update(config_update)
        return config

    def call(self, y: tf.Tensor, c: tf.Tensor, **kwargs):
        """
        Args:

            y (tf.Tensor): input contents tensor [B, T, C]
            c (tf.Tensor): input conditional tensor [B, T, C'] where C' can be different with C

        Returns:

            tf.Tensor: [B, T, C]
        """

        right = tf.nn.tanh(self.conv_first(y))
        left = tf.nn.sigmoid(self.conv_last(c))
        z = right * left
        return z


def CouplingBlock(x: tf.Tensor, c: tf.Tensor, depth, **kwargs):
    """
    Args:

        x (tf.Tensor): input contents tensor [B, T, C]
        c (tf.Tensor): input conditional tensor [B, T, C'] where C' can be different with C
    Returns:

        tf.keras.Model: CouplingBlock
                        reference: Flow-TTS
    Examples:

        >>> import tensorflow as tf
        >>> from utils.coupling_block import CouplingBlock
        >>> x = tf.keras.Input([16, 32])
        >>> c = tf.keras.Input([16, 128])
        >>> cp = CouplingBlock(x, c, depth=128)
        >>> cp.summary()
        Model: "model"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        input_1 (InputLayer)            [(None, 16, 32)]     0
        __________________________________________________________________________________________________
        conv1d_2 (Conv1D)               (None, 16, 128)      4096        input_1[0][0]
        __________________________________________________________________________________________________
        input_2 (InputLayer)            [(None, 16, 128)]    0
        __________________________________________________________________________________________________
        gtu_1 (GTU)                     (None, 16, 128)      33024       conv1d_2[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_AddV2 (TensorFlowOp [(None, 16, 128)]    0           gtu_1[0][0]
                                                                        conv1d_2[0][0]
        __________________________________________________________________________________________________
        conv1d_3 (Conv1D)               (None, 16, 32)       4128        tf_op_layer_AddV2[0][0]
        ==================================================================================================
        Total params: 41,248
        Trainable params: 41,248
        Non-trainable params: 0
        __________________________________________________________________________________________________
        >>> cp([x, c])
        <tf.Tensor 'model_3/Identity:0' shape=(None, 16, 32) dtype=float32>
    """
    input_shape = x.shape

    conv1x1_1 = tf.keras.layers.Conv1D(
        depth,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        activation="relu",
    )
    conv1x1_2 = tf.keras.layers.Conv1D(
        input_shape[-1],
        kernel_size=1,
        strides=1,
        padding="same",
        data_format="channels_last",
        kernel_initializer="zeros",
    )
    gtu = GTU()

    y = x
    y = conv1x1_1(y)
    skip_connection = y
    y = gtu(y, c)
    y += skip_connection
    y = conv1x1_2(y)
    return tf.keras.Model([x, c], y)
