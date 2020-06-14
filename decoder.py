#!/usr/bin/env python3

from typing import Dict
import tensorflow as tf
from TFGENZOO.flows import FactorOut, Squeeze
from utils.inv1x1conv2D import Inv1x1Conv2D
from utils.coupling_block import GTU, CouplingBlock
from TFGENZOO.flows.flowbase import ConditionalFlowModule
from utils.cond_affine_coupling import ConditionalAffineCouplingWithMask


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
        """utility function to construct step-of-flow

        Sources:

            Flow-TTS's Figure 1

        """

        flows = []

        cfml = []
        for i in range(step_num):

            # Sources:
            #
            #    FLow-TTS's Figure 1 (b)

            # Inv1x1Conv
            inv1x1 = Inv1x1Conv2D()

            # CouplingBlock
            couplingBlock_template = lambda x: CouplingBlock(
                x, cond=conditional_input, depth=coupling_depth
            )

            # Affine_xform + Coupling Block
            #
            # Notes:
            #
            #     * forward formula
            #         |  where x is source input [B, T, C]
            #         |        c is conditional input [B, T, C'] where C' can be difference with C
            #         |
            #         |  x_1, x_2 = split(x)
            #         |  z_1 = x_1
            #         |  [logs, shift] = NN(x_1, c)
            #         |  z_2 = (x_2 + shift) * exp(logs)
            #    * Coupling Block formula
            #         |
            #         |  where x_1', x_1'' is [B, T, C''] where C'' can be difference with C and C'
            #         |        logs, shift is [B, T, C]
            #         |
            #         |  x_1' =  1x1Conv_1(x_1)
            #         |  x_1'' = GTU(x_1', c)
            #         |  [logs, shift] = 1x1Conv_2(x_1'' + x')
            #         |
            #         |
            #
            conditionalAffineCoupling = ConditionalAffineCouplingWithMask(
                scale_shift_net_template=couplingBlock_template, scale_type=scale_type
            )
            cfml.append(inv1x1)
            cfml.append(conditionalAffineCoupling)
        return ConditionalFlowModule(cfml)
