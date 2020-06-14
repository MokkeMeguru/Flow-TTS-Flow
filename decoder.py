#!/usr/bin/env python3

import tensorflow as tf
from TFGENZOO.flows import FactorOut, Squeeze
from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling
from utils.inv1x1conv2D import Inv1x1Conv2D
from utils.coupling_block import GTU, CouplingBlock


class FlowTTSDecoder(tf.keras.Model):
    pass
