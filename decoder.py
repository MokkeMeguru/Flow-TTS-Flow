#!/usr/bin/env python3

from typing import Dict
import tensorflow as tf
from utils.inv1x1conv2D import Inv1x1Conv2D
from utils.coupling_block import GTU, CouplingBlock
from TFGENZOO.flows.flowbase import ConditionalFlowModule
from utils.cond_affine_coupling import ConditionalAffineCouplingWithMask
from utils.flow_step import build_flow_step


class FlowTTSDecoder(tf.keras.Model):
    def __init__(self, hparams: Dict, **kwargs):
        self.hparams = hparams
        self.build_model()

    def build_model():
        pass
