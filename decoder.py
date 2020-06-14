#!/usr/bin/env python3

from typing import Dict
import tensorflow as tf
from TFGENZOO.flows import FactorOut, Squeeze
from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling
from utils.inv1x1conv2D import Inv1x1Conv2D
from utils.coupling_block import GTU, CouplingBlock
from TFGENZOO.flows.flowbase import ConditionalFlowModule


class FlowTTSDecoder(tf.keras.Model):
    def __init__(self, hparams: Dict, **kwargs):
        self.hparams = hparams
        self.build_model()

    def build_model():
        pass

    def build_flow_step(
        step_num: int,
        coupling_depth: int,
        conditional_input: tf.keras.layers.Input,
        scale_type: str = "safe_exp",
    ):
        flows = []

        cfml = []
        for i in range(step_num):
            inv1x1 = Inv1x1Conv2D()

            couplingBlock_template = lambda x: CouplingBlock(
                x, cond=conditional_input, depth=coupling_depth
            )

            conditionalAffineCoupling = ConditionalAffineCoupling(
                scale_shift_net_template=couplingBlock_template, scale_type=scale_type
            )
            cfml.append(inv1x1)
            cfml.append(conditionalAffineCoupling)
        return ConditionalFlowModule(cfml)
