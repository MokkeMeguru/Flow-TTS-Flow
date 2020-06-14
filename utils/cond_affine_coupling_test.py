#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from TFGENZOO.flows import AffineCoupling, AffineCouplingMask
from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling
from coupling_block import CouplingBlock


class AffineCouplingTest(tf.test.TestCase):
    def setUp(self):
        super(AffineCouplingTest, self).setUp()
        x = tf.keras.Input([None, 16])
        c = tf.keras.Input([None, 128])
        self.cb = lambda x: CouplingBlock(x, c, depth=128)
        self.caf = ConditionalAffineCoupling(
            mask_type=AffineCouplingMask.ChannelWise, scale_shift_net_template=self.cb
        )
        self.caf.build(x.shape)

    def testAffineCouplingOutputShape(self):
        x = tf.random.normal([1024, 16, 16])
        c = tf.random.normal([1024, 16, 128])
        z, ldj = self.caf(x, cond=c)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testAffineCouplingOutput(self):
        x = tf.random.normal([1024, 16, 16])
        c = tf.random.normal([1024, 16, 128])
        z, ldj = self.caf(x, cond=c)
        rev_x, ildj = self.caf(x, cond=c, inverse=True)
        self.assertAllClose(x, rev_x)
        self.assertAllClose(ldj + ildj, np.zeros(x.shape[0:1]))
