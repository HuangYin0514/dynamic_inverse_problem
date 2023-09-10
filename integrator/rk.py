# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm

from .base import SolverBase


class RK4(SolverBase):

    def __init__(self, func, t0, t1, dt, y0, *args, **kwargs):
        super(RK4, self).__init__(func, t0, t1, dt, y0, *args, **kwargs)

    def step(self, func, t0, y0, dt, *args, **kwargs):
        k1 = dt * func(t0, y0)
        k2 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k1)
        k3 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k2)
        k4 = dt * func(t0 + dt, y0 + k3)
        y1 = y0 + (k1 + k2 * 2 + k3 * 2 + k4) / 6
        return y1


class RK4_high_order(SolverBase):
    """
    仅限于输出为q_ori, dq_ori, lam的情况
    """

    def __init__(self, func, t0, t1, dt, y0, *args, **kwargs):
        super(RK4_high_order, self).__init__(func, t0, t1, dt, y0, *args, **kwargs)

    def step_with_lam(self, func, t0, y0, dt, dof, *args, **kwargs):
        q_ori, dq_ori, lam = torch.tensor_split(y0, (dof, dof * 2), dim=-1)

        k1 = q_ori
        dk1 = dq_ori
        dqddqlam = func(t0, torch.cat([k1, dk1, lam], dim=-1))
        dq, ddk1, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        k2 = k1 + 0.5 * dt * dk1
        dk2 = dk1 + 0.5 * dt * ddk1
        dqddqlam = func(t0, torch.cat([k2, dk2, lam], dim=-1))
        dq, ddk2, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        k3 = k1 + 0.5 * dt * dk2
        dk3 = dk1 + 0.5 * dt * ddk2
        dqddqlam = func(t0, torch.cat([k3, dk3, lam], dim=-1))
        dq, ddk3, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        k4 = k1 + dt * dk3
        dk4 = dk1 + dt * ddk3
        dqddqlam = func(t0, torch.cat([k4, dk4, lam], dim=-1))
        dq, ddk4, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        q_next = k1 + dt * dk1 + (dt ** 2) / 6 * (ddk1 + ddk2 + ddk3)
        dq_next = dk1 + dt / 6 * (ddk1 + 2 * ddk2 + 2 * ddk3 + ddk4)

        return torch.cat([q_next, dq_next, lam], dim=-1)

    def step(self, func, t0, y0, dt, *args, **kwargs):
        q_ori, dq_ori = torch.chunk(y0, 2, dim=-1)

        k1 = q_ori
        dk1 = dq_ori
        dqddq = func(t0, torch.cat([k1, dk1], dim=-1))
        dq, ddk1 = torch.chunk(y0, 2, dim=-1)

        k2 = k1 + 0.5 * dt * dk1
        dk2 = dk1 + 0.5 * dt * ddk1
        dqddq = func(t0, torch.cat([k2, dk2], dim=-1))
        dq, ddk2 = torch.chunk(dqddq, 2, dim=-1)

        k3 = k1 + 0.5 * dt * dk2
        dk3 = dk1 + 0.5 * dt * ddk2
        dqddq = func(t0, torch.cat([k3, dk3], dim=-1))
        dq, ddk3 = torch.chunk(dqddq, 2, dim=-1)

        k4 = k1 + dt * dk3
        dk4 = dk1 + dt * ddk3
        dqddq = func(t0, torch.cat([k4, dk4], dim=-1))
        dq, ddk4 = torch.chunk(dqddq, 2, dim=-1)

        q_next = k1 + dt * dk1 + (dt ** 2) / 6 * (ddk1 + ddk2 + ddk3)
        dq_next = dk1 + dt / 6 * (ddk1 + 2 * ddk2 + 2 * ddk3 + ddk4)

        return torch.cat([q_next, dq_next], dim=-1)

    def solve(self, func, t, y0, dt, *args, **kwargs):
        sol = torch.empty(len(t), *y0.shape, device=self.device, dtype=self.dtype)
        sol[0] = y0

        pbar = tqdm(range(len(self.t) - 1), desc='solving ODE...')
        for idx in pbar:
            temp = self.step_with_lam(self.func, self.t[idx], sol[idx], self.dt, *args, **kwargs)
            if sum((temp[0, 0:3] - temp[0, 15:18]) * (temp[0, 12:15] - temp[0, 3:6])) < 0:  # 16与52的夹角
                print('stop time is ', self.t[idx].cpu().detach().numpy())
                sol = sol[:idx + 1]
                t = t[:idx + 1]
                break
            else:
                sol[idx + 1] = temp
        sol = sol.permute(1, 0, 2)
        return t, sol
