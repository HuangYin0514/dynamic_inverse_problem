# encoding: utf-8

import os
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learner.metric.dynamics_metric import (
    calculate_dynamics_metrics,
    plot_dynamics_metrics,
)
from utils import batched_jacobian, initialize_class, tensors_to_numpy

from .utils_nn import MLPBlock, SinActivation


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

        activation = SinActivation()

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


class PINN(nn.Module):
    def __init__(self, config, logger, *args, **kwargs):
        super(PINN, self).__init__()

        self.config = config
        self.logger = logger

        self.device = config.device
        self.dtype = config.dtype

        self.backboneNet = BackboneNet(config)

        try:
            class_name = self.config.dynamic_class
            kwargs = {"config": config, "logger": logger}
            self.right_term_net = initialize_class("dynamics", class_name, **kwargs)
        except ValueError as e:
            raise RuntimeError("class '{}' is not available".format(class_name))

    def forward(self, t):
        t = t.reshape(-1, 1)
        out = self.backboneNet(t)
        return out

    def get_q_qt_qtt(self, t):
        q_hat = self(t)
        qt_hat = batched_jacobian(
            q_hat, t, device=self.device, dtype=self.dtype
        ).squeeze(-1)
        qtt_hat = batched_jacobian(
            qt_hat, t, device=self.device, dtype=self.dtype
        ).squeeze(-1)
        return q_hat, qt_hat, qtt_hat

    def criterion(self, data, current_iterations, *args, **kwargs):
        y0, train_y, train_yt, train_t, physics_t = data
        q, qt = torch.chunk(train_y, 2, dim=-1)
        qt, qtt = torch.chunk(train_yt, 2, dim=-1)

        ################################
        #
        # Loss for initial condition (y0)
        #
        ################################
        t0 = torch.zeros(1, 1, device=self.device, dtype=self.dtype)
        t0 = t0.requires_grad_(True)
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(t0)
        y0_hat = torch.cat([q_hat, qt_hat], dim=-1)
        loss_y0 = F.mse_loss(y0_hat, y0)
        ################################

        ################################
        #
        # Loss for training data
        #
        ################################
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(train_t)
        loss_data = 0
        loss_data += F.mse_loss(q_hat, q)
        loss_data += F.mse_loss(qt_hat, qt)
        loss_data += F.mse_loss(qtt_hat, qtt)

        ################################
        #
        # Loss for physics data
        #
        ################################
        q_left_hat, qt_left_hat, qtt_left_hat = self.get_q_qt_qtt(physics_t)
        yt_right_hat = self.right_term_net(
            physics_t, torch.cat([q_left_hat, qt_left_hat], dim=-1)
        )
        qt_right_hat, qtt_right_hat, lambdas = torch.tensor_split(
            yt_right_hat, (self.config.dof, self.config.dof * 2), dim=-1
        )
        loss_res = torch.mean((qtt_left_hat - qtt_right_hat) ** 2)

        # if loss_data > 1e-2 or 10 * loss_data < loss_res:
        #     loss_res = (loss_data / (loss_data + loss_res)) * loss_res

        ################################
        #
        # log
        #
        ################################
        flag_output = "============>"
        current_iterations_output = "{}: {}".format(
            "current_iterations", current_iterations
        )
        loss_y0_output = "{}: {:.3e}".format("loss_y0", loss_y0)
        loss_data_output = "{}: {:.3e}".format("loss_data", loss_data)
        loss_res_output = "{}: {:.3e}".format("loss_res", loss_res)
        keys_values_pairs = [
            flag_output,
            current_iterations_output,
            loss_y0_output,
            loss_data_output,
            loss_res_output,
        ]
        full_output = "|".join(keys_values_pairs)
        self.logger.debug(full_output)

        return (
            self.config.loss_y0_weight * loss_y0
            + self.config.loss_data_weight * loss_data
            + self.config.loss_physic_weight * loss_res
        )

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
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(data_t)

        ################################
        #
        # Calculate the error
        #
        ################################
        # Calculate mean squared error
        mse_error = F.mse_loss(q_hat, q) + F.mse_loss(qt_hat, qt)

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
        plot_dynamics_metrics(
            config=self.config,
            pred_data=[q_hat, qt_hat, qtt_hat],
            gt_data=[q, qt, qtt],
            t=data_t,
        )
        save_path = os.path.join(output_dir, f"iter_{current_iterations}.png")
        plt.savefig(save_path)
        plt.close()

        return mse_error
