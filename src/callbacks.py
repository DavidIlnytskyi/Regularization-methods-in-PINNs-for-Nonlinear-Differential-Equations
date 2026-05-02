"""Training callbacks for saving models, stepping schedulers, and collecting metrics.

This module contains small callable classes used by ``MHSolver2D.fit``. Each
callback receives the active solver instance after an epoch and can perform a
side effect such as advancing a scheduler, writing a checkpoint, or recording
evaluation metrics.
"""

import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

sys.path.append("./src")

from neurodiffeq.callbacks import ActionCallback
from mh_solver import MHSolver2D
from io_model import save_nets, load_nets


class DoSchedulerStep(ActionCallback):
    """Callback that advances a learning-rate scheduler once per epoch."""

    def __init__(self, scheduler: Any) -> None:
        """Store the scheduler object used during training."""
        super().__init__()
        self.scheduler = scheduler

    def __call__(self, solver: MHSolver2D) -> None:
        """Step the scheduler after the solver completes an epoch."""
        self.scheduler.step()


class SaveCallback:
    """Callback that periodically saves solver networks to disk."""

    def __init__(self, n: int, save_dir: str = "./model_solutions") -> None:
        """Create the save directory and remember the checkpoint period."""
        self.period = n
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, model: MHSolver2D) -> None:
        """Save the model when the current epoch is divisible by ``n``."""
        epoch = getattr(model, "global_epoch", model.local_epoch)
        if epoch % self.period != 0:
            return
        path = os.path.join(self.save_dir, f"solution_epoch_{epoch}.pt")

        save_nets(path, model)


class MetricsCallback:
    """Callback that evaluates each solver head against reference solutions."""

    def __init__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor | float | Sequence[float],
        convergence_value: float,
        n_heads: int,
        period: int = 250,
        reference_fns_container: Callable[..., torch.Tensor] | None = None,
        reference_solutions: Sequence[torch.Tensor | np.ndarray[Any, Any]] | None = None,
    ) -> None:
        """Configure evaluation points, references, and metric storage."""
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

    def _get_head_r_value(self, head: int) -> torch.Tensor:
        """Return the scalar parameter value associated with one head."""
        return self.r_values[head]

    def _get_head_r_tensor(self, head: int) -> torch.Tensor:
        """Return a parameter tensor matching the evaluation grid shape."""
        head_r = self._get_head_r_value(head).detach()
        return torch.ones_like(self.x, requires_grad=True) * head_r

    def _get_reference_solution(self, head: int) -> torch.Tensor:
        """Return the reference solution for one solver head."""
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

    def __call__(self, solver: MHSolver2D) -> None:
        """Evaluate solver heads and append MSE, MAE, and relative L2 metrics."""
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
