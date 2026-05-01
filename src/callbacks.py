import os
import sys

sys.path.append("./src")

import numpy as np
import torch
from neurodiffeq.callbacks import ActionCallback
from mh_solver import MHSolver2D

class DoSchedulerStep(ActionCallback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def __call__(self, solver):
        self.scheduler.step()

class SaveCallback:
    def __init__(self, n, save_dir="./model_solutions"):
        self.n = n
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, model):
        epoch = getattr(model, "global_epoch", model.local_epoch)
        if epoch % self.n != 0:
            return
        path = os.path.join(self.save_dir, f"solution_epoch_{epoch}.pt")

        save_nets(path, model)

class MetricsCallback:
    def __init__(
        self,
        x,
        t,
        r,
        convergence_value,
        n_heads,
        period=250,
        reference_fns_container=None,
        reference_solutions=None,
    ):
        self.reference_fns_container = reference_fns_container
        self.reference_solutions = reference_solutions
        self.convergence_value = convergence_value
        self.period = period

        self.training_time = None

        self.x = x
        self.t = t
        self.r_values = torch.as_tensor(r, dtype=x.dtype, device=x.device).reshape(-1)
        if self.r_values.numel() == 1:
            self.r_values = self.r_values.repeat(n_heads)
        elif self.r_values.numel() != n_heads:
            raise ValueError(
                f"MetricsCallback expected 1 or {n_heads} parameter values, "
                f"got {self.r_values.numel()}"
            )

        self.predictions = [[] for _ in range(n_heads)]
        self.mse = [[] for _ in range(n_heads)]
        self.mae = [[] for _ in range(n_heads)]
        self.l2_rel = [[] for _ in range(n_heads)]
        self.convergence_epoch = [0 for _ in range(n_heads)]

    def _get_head_r_value(self, head):
        return self.r_values[head]

    def _get_head_r_tensor(self, head):
        head_r = self._get_head_r_value(head).detach()
        return torch.ones_like(self.x, requires_grad=True) * head_r

    def _get_reference_solution(self, head):
        if self.reference_fns_container is not None:
            head_r = self._get_head_r_tensor(head)
            try:
                return self.reference_fns_container(self.x, self.t, head, r=head_r)
            except TypeError:
                return self.reference_fns_container(self.x, self.t, head)

        if self.reference_solutions is not None:
            u_ref = self.reference_solutions[head]
            if not torch.is_tensor(u_ref):
                u_ref = torch.tensor(u_ref, dtype=self.x.dtype, device=self.x.device)
            else:
                u_ref = u_ref.to(device=self.x.device, dtype=self.x.dtype)
            return u_ref

        raise ValueError(
            "Either reference_fns_container or reference_solutions must be provided."
        )

    def __call__(self, solver: MHSolver2D):
        if solver.local_epoch % self.period != 0:
            return

        for head in range(solver.n_heads):
            sol = solver.get_solution(head=head, copy=False)

            r_eval = self._get_head_r_tensor(head)
            u_ref = self._get_reference_solution(head)
            u_pred = sol(self.x, self.t, r_eval)

            u_ref_np = u_ref.detach().cpu().numpy()
            u_pred_np = u_pred.detach().cpu().numpy()

            diff = u_pred_np - u_ref_np

            mse = np.mean(diff**2)
            mae = np.mean(np.abs(diff))
            l2 = np.linalg.norm(diff) / np.linalg.norm(u_ref_np)

            if mae < self.convergence_value and self.convergence_epoch[head] == 0:
                self.convergence_epoch[head] = solver.local_epoch

            self.mse[head].append(mse)
            self.mae[head].append(mae)
            self.l2_rel[head].append(l2)