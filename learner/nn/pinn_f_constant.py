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

        self.logger.debug(F_f)
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

    def criterion(self, data, current_iterations, *args, **kwargs):
        y0, train_y, train_yt, train_t, physics_t = data
        q, qt = torch.chunk(train_y, 2, dim=-1)
        qt, qtt = torch.chunk(train_yt, 2, dim=-1)
        q, qt, qtt_hat = self.get_q_qt_qtt(train_t, q, qt)

        loss_res = torch.mean((qtt[:,-1] - qtt_hat[:,-1]) ** 2)

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
        keys_values_pairs = [
            flag_output,
            current_iterations_output,
            loss_res_output,
        ]
        full_output = "|".join(keys_values_pairs)
        self.logger.debug(full_output)

        return self.config.loss_physic_weight * loss_res

    def evaluate(self, data, output_dir="", current_iterations=None, *args, **kwargs):
        ################################
        #
        # Unpack data
        #
        ################################
        y0, y, yt, data_t, physics_t = data

        # Split tensors
        q, qt = torch.chunk(y, 2, dim=-1)
        qt, qtt = torch.chunk(yt, 2, dim=-1)

        ################################
        #
        # Predict using the physics model
        #
        ################################
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(data_t,q, qt)

        ################################
        #
        # Calculate the error
        #
        ################################
        # Calculate mean squared error
        mse_error = torch.mean((qtt - qtt_hat) ** 2)

        # Convert tensors to numpy arrays
        data_t = tensors_to_numpy(data_t)
        q, qt, qtt = tensors_to_numpy(q, qt, qtt)
        q_hat, qt_hat, qtt_hat = tensors_to_numpy(q_hat, qt_hat, qtt_hat)

        # Calculate energy and other terms using the physics model
        (
            metric_value,
            metric_error_value,
            output_log_string,
        ) = calculate_dynamics_metrics(
            calculator=self.right_term_net.calculator,
            pred_data=[q_hat, qt_hat, qtt_hat],
            gt_data=[q, qt, qtt],
        )
        iteration_output = f"iteration: {current_iterations}"
        MSE_y_output = f"MSE_y: {mse_error:.4e}"
        output_metric = " | ".join([iteration_output, MSE_y_output] + output_log_string)
        self.logger.debug(output_metric)

        ################################
        #
        # Plot the error
        #
        ################################
        # plot_dynamics_metrics(
        #     config=self.config,
        #     pred_data=[q_hat, qt_hat, qtt_hat],
        #     gt_data=[q, qt, qtt],
        #     t=data_t,
        # )
        # save_path = os.path.join(output_dir, f"iter_{current_iterations}.png")
        # plt.savefig(save_path)
        # plt.close()

        return mse_error
