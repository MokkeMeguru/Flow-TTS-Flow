#!/usr/bin/env python3

import tensorflow as tf
from decoder import build_flow_step
import numpy as np


class BuildFlowStepTest(tf.test.TestCase):
    def setUp(self):
        self.flow_step = build_flow_step(
            step_num=4,
            coupling_depth=128,
            conditional_input=tf.keras.layers.Input([None, 128]),
        )
        # base input x
        self.inputs = tf.keras.layers.Input([None, 32])
        # conditional input c
        self.cond = tf.keras.layers.Input([None, 128])

    def testFlowStepOutputShape(self):
        z, ldj = self.flow_step(self.inputs, cond=self.cond)
