# encoding: utf-8

import numpy as np
import torch
from tqdm import tqdm


class SolverBase:
    def __init__(self, func, t0, t1, dt, y0, device=None, dtype=None, *args, **kwargs):
        self.device = device
        self.dtype = dtype

        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.y0 = y0
        self.dt = dt
        t = np.arange(t0, t1 + dt, dt)
        self.t = torch.tensor(t, device=device, dtype=dtype)

    def get_results(self, *args, **kwargs):
        return self.solve(self.func, self.t, self.y0, self.dt, *args, **kwargs)

    def solve(self, func, t, y0, dt, *args, **kwargs):
        sol = torch.empty(len(t), *y0.shape, device=self.device, dtype=self.dtype)
        sol[0] = y0

        pbar = tqdm(range(len(self.t) - 1), desc="solving ODE...")
        for idx in pbar:
            sol[idx + 1] = self.step(
                self.func, self.t[idx], sol[idx], self.dt, *args, **kwargs
            ).clone().detach()
            if torch.any(torch.isnan(sol)):  # 执行停止程序的操作，例如引发异常
                print(self.t[idx])
                sol = sol[: idx + 1]
                t = t[: idx + 1]
                break
        sol = sol.permute(1, 0, 2)
        return t, sol
