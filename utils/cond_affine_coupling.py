#!/usr/bin/env python3

from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling


class ConditionalAffineCouplingWithMask(ConditionalAffineCoupling):
    """Conditional Affine Coupling Layer with mask

    Sources:
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    Note:
        * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN([x1, c])
            | scale = sigmoid(log_scale + 2.0)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat([z1, z2])
            | LogDetJacobian = sum(log(scale))

        * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN([z1, c])
            | scale = sigmoid(log_scale + 2.0)
            | x1 = z1
            | x2 = z2 / scale - shift
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale))

        * implementation notes
           | in Glow's Paper, scale is calculated by exp(log_scale),
           | but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
           | where c is the conditional input for WaveGlow or cINN
           | https://arxiv.org/abs/1907.02392

        * TODO notes
           | cINN uses double coupling, but our coupling is single coupling
    """

    def forward(self, x: tf.Tensor, cond: tf.Tensor, mask: tf.Tensor = None, **kwargs):
