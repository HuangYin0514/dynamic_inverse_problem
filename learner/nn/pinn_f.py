# encoding: utf-8

import os
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from integrator import ODEIntegrate
from utils import batched_jacobian, initialize_class, tensors_to_numpy
from integrator import RK4, RK4_high_order
from tqdm import tqdm


from learner.metric.dynamics_metric import (
    calculate_dynamics_metrics,
    plot_dynamics_metrics,
)
from utils import batched_jacobian, initialize_class, tensors_to_numpy

from .utils_nn import MLPBlock, SinActivation


#######################################################################
#
# force F function
#
#######################################################################
class F_Net(nn.Module):
    def __init__(self, config, logger):
        super(F_Net, self).__init__()

        self.device = config.device
        self.dtype = config.dtype

        self.unitnum = config.unitnum
        self.F_a = config.F_a
        # self.F_f = config.F_f
        self.F_r = config.F_r

        self.F_f = BackboneNet(config)

        self.logger = logger

    # Compute the force F(t, coords)
    def forward(self, t, coords):
        bs, num_states = coords.shape
        t = t.reshape(-1, 1)
        F = torch.zeros(
            bs, 12 * (self.unitnum + 1), device=self.device, dtype=self.dtype
        )

        F_f = self.F_f(t)

        F[:, 0:1] = -self.F_a * F_f
        F[:, 1:2] = self.F_a * F_f

        F[:, 3:4] = -self.F_a * F_f
        F[:, 4:5] = -self.F_a * F_f

        F[:, 6:7] = self.F_a * F_f
        F[:, 7:8] = -self.F_a * F_f

        F[:, 9:10] = self.F_a * F_f
        F[:, 10:11] = self.F_a * F_f

        F[:, 12 * self.unitnum + 2] = -self.F_r
        F[:, 12 * self.unitnum + 5] = -self.F_r
        F[:, 12 * self.unitnum + 8] = -self.F_r
        F[:, 12 * self.unitnum + 11] = -self.F_r

        return F


class BackboneNet(nn.Module):
    """
    Custom backbone neural network module.

    This module contains multiple MLPBlocks with a specified number of layers.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output.
        layers_num (int): Number of hidden layers in the network.
    """

    def __init__(self, config):
        super(BackboneNet, self).__init__()

        input_dim = config.BackboneNet_input_dim
        hidden_dim = config.BackboneNet_hidden_dim
        output_dim = config.BackboneNet_output_dim
        layers_num = config.BackboneNet_layers_num

        # activation = SinActivation()
        # activation = nn.ELU()
        activation = nn.LeakyReLU()

        # Input layer
        input_layer = MLPBlock(input_dim, hidden_dim, activation)

        # Hidden layers
        hidden_layers = nn.ModuleList(
            [MLPBlock(hidden_dim, hidden_dim, activation) for _ in range(layers_num)]
        )

        # Output layer
        output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

        layers = []
        layers.extend([input_layer])
        layers.extend(hidden_layers)
        layers.extend([output_layer])

        # Create the sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the BackboneNet.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        out = self.net(x)
        return out


class PINN_F(nn.Module):
    def __init__(self, config, logger, *args, **kwargs):
        super(PINN_F, self).__init__()

        self.config = config
        self.logger = logger

        self.device = config.device
        self.dtype = config.dtype

        self.f_net = F_Net(config, logger)
        self.rk4_high_order = RK4_high_order(None, 0, 1, 1, config.y0)

        try:
            class_name = self.config.dynamic_class
            kwargs = {"config": config, "logger": logger}
            self.right_term_net = initialize_class("dynamics", class_name, **kwargs)
            self.right_term_net.f_net = self.f_net
        except ValueError as e:
            raise RuntimeError("class '{}' is not available".format(class_name))

    def get_q_qt_qtt(self, t, q, qt):
        yt_hat = self.right_term_net(t, torch.cat([q, qt], dim=-1))
        qt_hat, qtt_hat, lambdas = torch.tensor_split(
            yt_hat, (self.config.dof, self.config.dof * 2), dim=-1
        )
        return q, qt, qtt_hat

    def criterion(self, current_iterations, y0, ti, dt):
        dof = self.config.dof
        func = self.right_term_net
        F = self.right_term_net.f_net.F_f(ti)

        yi = self.rk4_high_order.step_with_lam(func, ti, y0, dt, dof=dof)

        q_hat, qt_hat, lambdas = torch.tensor_split(yi, (dof, dof * 2), dim=-1)

        loss_res = torch.mean((qt_hat[:, -1] - 0.3) ** 2)

        error = torch.abs(qt_hat[:, -1] - 0.3)
        
        ################################
        #
        # log
        #
        ################################
        flag_output = "============>"
        current_iterations_output = "{}: {}".format(
            "current_iterations", current_iterations
        )
        loss_res_output = "{}: {:.3e}".format("loss_res", loss_res)
        qt_output = "{}: {:.3e}".format("qt", qt_hat[:, -1].item())
        F_output = "{}: {:.3e}".format("F", F.item())
        err_output = "{}: {:.3e}".format("error", error.item())
        keys_values_pairs = [
            flag_output,
            current_iterations_output,
            loss_res_output,
            qt_output,
            F_output,
            err_output,
        ]
        full_output = "|".join(keys_values_pairs)
        self.logger.debug(full_output)

        return (
            self.config.loss_physic_weight * loss_res,
            qt_hat[:, -1].item(),
            F.item(),
            error.item(),
            yi,
        )

    def evaluate(self, data, output_dir="", current_iterations=None, *args, **kwargs):
        return None
