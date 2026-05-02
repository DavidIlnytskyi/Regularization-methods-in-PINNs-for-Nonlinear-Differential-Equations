"""Multi-head 2D solver with optional PINN regularization losses.

This module extends ``neurodiffeq``'s ``Solver2D`` for a set of heads that share
training logic but use different condition/network pairs. It also adds optional
regularization terms used by the experiments in this repository.
"""

import sys
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

sys.path.append("./src")

from mh_solver import *

import numpy as np
import torch
from tqdm import tqdm

from neurodiffeq.neurodiffeq import diff
from neurodiffeq.solvers import Solver2D
from neurodiffeq.solvers import _requires_closure

from neurodiffeqq import SolutionBundle2D


class MHSolver2D(Solver2D):
    r"""A solver class for solving ODEs (single-input differential equations)
    , or a bundle of ODEs for different values of its parameters and/or conditions

    :param ode_system:
        The ODE system to solve, which maps a torch.Tensor or a tuple of torch.Tensors, to a tuple of ODE residuals,
        both the input and output must have shape (n_samples, 1).
    :type ode_system: callable
    :param conditions:
        List of conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param t_min:
        Lower bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_min: float, optional
    :param t_max:
        Upper bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_max: float, optional
    :param theta_min:
        Lower bound of input (parameters and/or conditions). If conditions are included in the bundle,
        the order should match the one inferred by the values of the ``bundle_param_lookup`` input
        in the ``neurodiffeq.conditions.BundleIVP``.
        Defaults to None.
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type theta_min: float or tuple, optional
    :param theta_max:
        Upper bound of input (parameters and/or conditions). If conditions are included in the bundle,
        the order should match the one inferred by the values of the ``bundle_param_lookup`` input
        in the ``neurodiffeq.conditions.BundleIVP``.
        Defaults to None.
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type theta_max: float or tuple, optional
    :param eq_param_index:
        Index (or indices) of bundle parameter that appears in the equation.
        E.g., if there are 5 bundle parameters generated and the first (index 0) and last (index 4) parameters are used
        in the equation, then eq_param_index should be (0, 4).
        The signature of the original equation should have, in addition to functions to solve for and coordinates,
        2 more parameters, corresponding to bundle parameters indexed at 0 and 4, in that order.
    :type eq_param_index: int or tuple[int], optional
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length of ``nets`` must equal that of ``conditions``
    :type nets: list[torch.nn.Module], optional
    :param train_generator:
        Generator for sampling training points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``train_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
        If provided, the generator must generate bundle parameters too.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        Generator for sampling validation points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``valid_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
        If provided, the generator must generate bundle parameters too.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytical solutions to be compared with neural net solutions.
        It maps a torch.Tensor to a tuple of function values.
        Output shape should match that of ``nets``.
    :type analytic_solutions: callable, optional
    :param optimizer:
        Optimizer to be used for training.
        Defaults to a ``torch.optim.Adam`` instance that trains on all parameters of ``nets``.
    :type optimizer: ``torch.nn.optim.Optimizer``, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``ode_system``.

    :type metrics: dict[str, callable], optional
    :param n_output_units:
        Number of output units for each neural network.
        Ignored if ``nets`` is specified.
        Defaults to 1.
    :type n_output_units: int, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    def __init__(
        self,
        pde_system: Sequence[Callable[..., Sequence[torch.Tensor]]],
        conditions_list: np.ndarray[Any, Any],
        all_nets: np.ndarray[Any, Any],
        theta_min: float | tuple[float, ...] | None = None,
        theta_max: float | tuple[float, ...] | None = None,
        eq_param_index: Sequence[int] = (),
        n_samplings: int = 32,
        method: str = "equally-spaced-noisy",
        equations_number: int | None = None,
        flatten: bool = False,
        UR_period: int = 100,
        regularization: str | None = None,
        regularization_lambda: float = 0,
        t_min: float = 0,
        t_max: float = 1,
        t_sampling: int = 64,
        x_min: float = 0,
        x_max: float = 1,
        x_sampling: int = 64,
        param_min: float = 0,
        param_max: float = 1,
        param_sampling: int = 64,
        *args,
        **kwargs,
    ) -> None:
        """Initialize heads, conditions, equation wrappers, and solver state."""

        # Pop and set all kwargs as attributes of the instance
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.equations = pde_system
        self.equations_number = equations_number
        self.UR_period = UR_period
        self.TL = flatten

        self.regularization = regularization
        self.regularization_lambda = regularization_lambda

        self.t_min = t_min
        self.t_max = t_max
        self.t_sampling = t_sampling

        self.x_min = x_min
        self.x_max = x_max
        self.x_sampling = x_sampling

        self.param_min = param_min
        self.param_max = param_max
        self.param_sampling = param_sampling

        self.all_conditions = conditions_list
        self.conditions = self.all_conditions[:, 0]
        N_FUNCTIONS = len(self.conditions)
        # 1 coordinate (time) for ODEs, 2 for PDEs
        N_COORDS = 2

        # Note: It is intentionally design in this way where `eq_param_index` and `self.eq_param_index`
        # both contain values offset by `N_FUNCTIONS + N_COORDS`. The first one (not bound to `self`) is used for the
        # `eq_param_filter` closure, while the second one (bound to `self`) is used for `_get_internal_variables()`.
        eq_param_index = tuple(N_FUNCTIONS + N_COORDS + idx for idx in eq_param_index)
        self.eq_param_index = eq_param_index

        def _diff_eqs_wrapper(*variables: torch.Tensor) -> Sequence[torch.Tensor]:
            """Call the PDE system with only the selected equation parameters."""
            funcs_and_coords = variables[: N_FUNCTIONS + N_COORDS]
            eq_params = tuple(variables[idx] for idx in eq_param_index)
            return pde_system(*funcs_and_coords, *eq_params)

        # Prepare the variables for the Multihead #
        self.n_heads = len(self.all_conditions[0, :])  # Define the number of heads #
        self.best_nets_list = np.ones(self.n_heads, dtype=object)
        self.all_nets = all_nets  # Define all the nets to iterate #
        self.nets = self.all_nets[:, 0]
        self.pde_list = []
        for head in range(
            self.n_heads
        ):  # Define the generators and append them on the list #
            pde_system = self.equations[head]
            super().__init__(pde_system=pde_system, *args, **kwargs)
            self.pde_list.append(self.diff_eqs)

        self.metrics_history["r2_loss"] = []
        self.metrics_history["add_loss"] = []

    def custom_epoch(self, key: str) -> None:
        """Run one train or validation epoch across all solver heads."""
        if self.n_batches[key] <= 0:
            return

        def make_closure(
            batch: Sequence[torch.Tensor],
            head: int,
        ) -> Callable[
            [bool, bool | None],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]:
            """Build an optimizer closure for one batch and head."""

            def closure(
                zero_grad: bool = True,
                backward: bool | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Compute total, residual, and regularization losses."""
                if backward is None:
                    backward = key == "train"

                if key == "train" and zero_grad:
                    self.optimizer.zero_grad()

                funcs = [
                    self.compute_func_val(n, c, *batch)
                    for n, c in zip(self.nets, self.conditions)
                ]

                residuals = self.diff_eqs(*funcs, *batch)
                residuals = torch.cat(residuals, dim=1)

                try:
                    r_loss = self.loss_fn(residuals, funcs, batch)

                    if self.regularization is not None:
                        if self.regularization == "UR":
                            add_loss = self.ur_loss(residuals, funcs, batch, head)
                        elif self.regularization == "L1":
                            add_loss = self.l1_regularization(head)
                        elif self.regularization == "L2":
                            add_loss = self.l2_regularization(head)
                        elif self.regularization == "J":
                            add_loss = self.jacobian_frobenius_reg_exact(head)
                        elif self.regularization == "gPINN":
                            add_loss = self.gpinn_loss(residuals, batch, head)
                    else:
                        add_loss = torch.tensor(
                            0.0, device=r_loss.device, dtype=r_loss.dtype
                        )
                    total_loss = r_loss + add_loss
                except TypeError as e:
                    warnings.warn(
                        "You might need to update your code. "
                        "Since v0.4.0; both `criterion` and `additional_loss` require three inputs: "
                        "`residual`, `funcs`, and `coords`. See documentation for more.",
                        FutureWarning,
                    )
                    raise e

                if backward:
                    total_loss.backward()

                return total_loss, r_loss, add_loss

            return closure

        self._phase = key

        epoch_total_loss = 0.0
        epoch_residual_loss = 0.0
        epoch_add_loss = 0.0

        requires_closure = _requires_closure(self.optimizer)

        if key == "train" and requires_closure:
            head_batches = [
                (head, [self._generate_batch(key) for _ in range(self.n_batches[key])])
                for head in range(self.n_heads)
            ]

            def optimizer_closure() -> torch.Tensor:
                """Accumulate losses for closure-based optimizers."""
                self.optimizer.zero_grad()
                accumulated_loss = None

                for head, batches in head_batches:
                    self.nets = self.all_nets[:, head]
                    self.conditions = self.all_conditions[:, head]
                    self.diff_eqs = self.pde_list[head]

                    for batch in batches:
                        batch_total_loss, _, _ = make_closure(batch, head)(
                            zero_grad=False,
                            backward=False,
                        )
                        accumulated_loss = (
                            batch_total_loss
                            if accumulated_loss is None
                            else accumulated_loss + batch_total_loss
                        )

                accumulated_loss.backward()
                return accumulated_loss

            self.optimizer.step(optimizer_closure)

            for head, batches in head_batches:
                self.nets = self.all_nets[:, head]
                self.conditions = self.all_conditions[:, head]
                self.diff_eqs = self.pde_list[head]

                for batch in batches:
                    batch_total_loss, batch_r_loss, batch_add_loss = make_closure(
                        batch,
                        head,
                    )(zero_grad=False, backward=False)

                    epoch_total_loss += batch_total_loss.detach().item()
                    epoch_residual_loss += batch_r_loss.detach().item()
                    epoch_add_loss += batch_add_loss.detach().item()

            self.metrics_history["r2_loss"].append(epoch_residual_loss)
            self.metrics_history["add_loss"].append(epoch_add_loss)

            if key == "valid" or self.n_batches["valid"] == 0:
                self._update_best(key)

            self._update_history(epoch_total_loss, "loss", key)
            return

        if key == "train" and not requires_closure:
            self.optimizer.zero_grad()

        for head in range(self.n_heads):
            head_total_loss = 0.0
            head_residual_loss = 0.0
            head_add_loss = 0.0

            self.nets = self.all_nets[:, head]
            self.conditions = self.all_conditions[:, head]
            self.diff_eqs = self.pde_list[head]

            for batch_id in range(self.n_batches[key]):
                batch = self._generate_batch(key)
                closure = make_closure(batch, head)

                if key == "train":
                    batch_total_loss, batch_r_loss, batch_add_loss = closure(
                        zero_grad=False, backward=True
                    )
                else:
                    with torch.no_grad():
                        batch_total_loss, batch_r_loss, batch_add_loss = closure(
                            zero_grad=False, backward=False
                        )

                head_total_loss += batch_total_loss.detach().item()
                head_residual_loss += batch_r_loss.detach().item()
                head_add_loss += batch_add_loss.detach().item()

            epoch_total_loss += head_total_loss
            epoch_residual_loss += head_residual_loss
            epoch_add_loss += head_add_loss

        if key == "train" and not requires_closure:
            self._do_optimizer_step()

        self.metrics_history["r2_loss"].append(epoch_residual_loss)
        self.metrics_history["add_loss"].append(epoch_add_loss)

        if key == "valid" or self.n_batches["valid"] == 0:
            self._update_best(key)

        self._update_history(epoch_total_loss, "loss", key)

    def run_custom_epoch(self) -> None:
        r"""Run a training epoch, update history, and perform gradient descent."""
        self.custom_epoch("train")

    def _update_best(self, key: str) -> None:
        """Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history["r2_loss"][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            for i in range(self.n_heads):
                self.best_nets_list[i] = deepcopy(self.all_nets[:, i])

    def fit(
        self,
        max_epochs: int,
        callbacks: Sequence[Callable[["MHSolver2D"], None]] = (),
        tqdm_file: Any = "default",
        **kwargs: Any,
    ) -> None:
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training, validating and updating best model.

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]
        :param tqdm_file:
            File to write tqdm progress bar. If set to None, tqdm is not used at all.
            Defaults to ``sys.stderr``.
        :type tqdm_file: io.StringIO or _io.TextIOWrapper

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs
        if not hasattr(self, "global_epoch"):
            self.global_epoch = 0

        self.callbacks = callbacks

        monitor = kwargs.pop("monitor", None)
        if monitor:
            warnings.warn(
                "Passing `monitor` is deprecated, "
                "use a MonitorCallback and pass a list of callbacks instead"
            )
            callbacks = [monitor.to_callback()] + list(callbacks)
        if kwargs:
            raise ValueError(
                f"Unknown keyword argument(s): {list(kwargs.keys())}"
            )  # pragma: no cover

        flag = False
        if str(tqdm_file) == "default":
            bar = tqdm(
                total=max_epochs,
                desc="Training Progress",
                colour="blue",
                dynamic_ncols=True,
            )
        elif tqdm_file is not None:
            bar = tqdm_file
        else:
            flag = True

        for local_epoch in range(max_epochs):
            # stop training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register cumulative epoch so repeated optimizer phases keep callback numbering monotonic
            self.local_epoch += 1
            self.run_custom_epoch()
            self.run_valid_epoch()
            for cb in callbacks:
                cb(self)
            if not flag:
                bar.update(1)

    def get_solution(
        self,
        copy: bool = True,
        best: bool = True,
        head: int | None = None,
    ) -> SolutionBundle2D:
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        if head is not None:
            nets = [np.array([self.all_nets[0, head]], dtype=object)]
            conditions = [self.all_conditions[0, head]]

        else:
            if best:
                nets = self.best_nets_list
            else:
                nets = np.ones(self.n_heads, dtype=object)
                for i in range(self.n_heads):
                    nets[i] = deepcopy(self.all_nets[:, i])

            conditions = self.all_conditions[0]

        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return SolutionBundle2D(nets, conditions)

    def _get_internal_variables(self) -> dict[str, Any]:
        """Return solver configuration variables used for serialization."""
        available_variables = super(BundleSolver2D, self)._get_internal_variables()
        available_variables.update(
            {
                "r_min": self.r_min,
                "r_max": self.r_max,
                "eq_param_index": self.eq_param_index,
            }
        )
        return available_variables

    def l1_regularization(self, head: int) -> torch.Tensor:
        """Return L1 penalty on the final body layer for transfer learning."""
        if (
            head != self.n_heads - 1
            or self.local_epoch % self.UR_period != 0
            or not self.TL
        ):
            return torch.tensor([0.0])

        loss_sum = torch.zeros((), device=next(self.all_nets[0][0].parameters()).device)
        for idx in range(len(self.nets)):

            body_model = self.all_nets[idx][0].H_model
            last_layer = body_model.NN[-1]
            l1 = last_layer.weight.abs().sum()
            loss_sum = loss_sum + l1

        return loss_sum * self.regularization_lambda

    def l2_regularization(self, head: int) -> torch.Tensor:
        """Return L2 penalty on the final body layer for transfer learning."""
        if (
            head != self.n_heads - 1
            or self.local_epoch % self.UR_period != 0
            or not self.TL
        ):
            return torch.tensor([0.0])

        loss_sum = torch.zeros((), device=next(self.all_nets[0][0].parameters()).device)
        for idx in range(len(self.nets)):

            body_model = self.all_nets[idx][0].H_model
            last_layer = body_model.NN[-1]
            l2 = last_layer.weight.pow(2).sum()
            loss_sum = loss_sum + l2

        return loss_sum * self.regularization_lambda

    def jacobian_frobenius_reg_exact(self, head: int) -> torch.Tensor:
        """Return exact Frobenius-norm Jacobian regularization for body models."""
        if (
            head != self.n_heads - 1
            or self.local_epoch % self.UR_period != 0
            or not self.TL
        ):
            return torch.tensor(
                0.0, device=self.device if hasattr(self, "device") else None
            )

        nu_list = torch.linspace(
            self.param_min,
            self.param_max,
            self.param_sampling,
            device=self.device if hasattr(self, "device") else None,
        )
        t_list = torch.linspace(
            self.t_min,
            self.t_max,
            self.t_sampling,
            device=self.device if hasattr(self, "device") else None,
        )
        x_list = torch.linspace(
            self.x_min,
            self.x_max,
            self.x_sampling,
            device=self.device if hasattr(self, "device") else None,
        )

        T, X, N = torch.meshgrid(t_list, x_list, nu_list, indexing="ij")

        inp = torch.cat(
            [T.reshape(-1, 1), X.reshape(-1, 1), N.reshape(-1, 1)], dim=1
        ).requires_grad_(True)

        jacobian_loss = inp.new_tensor(0.0)

        for idx in range(len(self.nets)):
            H = self.nets[idx].H_model(inp)  # [P, M], P = number of sampled points

            grads = []
            M = H.shape[1]

            for j in range(M):
                g = torch.autograd.grad(
                    outputs=H[:, j].sum(),
                    inputs=inp,
                    create_graph=True,
                    retain_graph=True,
                )[
                    0
                ]  # [P, 3]

                grads.append(g)

            J = torch.stack(grads, dim=1)  # [P, M, 3]
            jacobian_loss = jacobian_loss + (J**2).mean()

        return self.regularization_lambda * jacobian_loss

    def ur_loss(
        self,
        r: torch.Tensor,
        f: Sequence[torch.Tensor],
        x: Sequence[torch.Tensor],
        head: int,
    ) -> torch.Tensor:
        """Return uniform regularization loss over sampled input parameters."""
        if (
            head != self.n_heads - 1
            or self.local_epoch % self.UR_period != 0
            or not self.TL
        ):
            device = self.device if hasattr(self, "device") else None
            return torch.tensor(0.0, device=device)

        param0 = next(self.nets[0].parameters())
        device = param0.device

        flat_metric = float(len(self.nets))

        nu_list = torch.linspace(
            self.param_min, self.param_max, self.param_sampling, device=device
        )
        t_list = torch.linspace(self.t_min, self.t_max, self.t_sampling, device=device)
        x_list = torch.linspace(self.x_min, self.x_max, self.x_sampling, device=device)

        T, X, N = torch.meshgrid(t_list, x_list, nu_list, indexing="ij")
        inp = torch.stack((T, X, N), dim=-1).reshape(-1, 3).requires_grad_(True)

        g_det = torch.ones_like(T)

        inp_dim = inp.shape[1]
        eye = (
            torch.eye(inp_dim, device=device).unsqueeze(0).expand(inp.shape[0], -1, -1)
        )

        for net in self.nets:
            H = net.H_model(inp)

            h_partials = [
                diff(H[:, j], inp, shape_check=False) for j in range(H.shape[1])
            ]
            h_partials = torch.stack(h_partials, dim=1)

            partial_omega = torch.cat((eye, h_partials), dim=1)
            g_MAT = partial_omega.transpose(1, 2) @ partial_omega

            g_det = g_det + torch.sqrt(torch.linalg.det(g_MAT)).reshape_as(T)

        ur_loss = torch.sum(g_det - flat_metric)
        return ur_loss * self.regularization_lambda

    def gpinn_loss(
        self,
        residuals: torch.Tensor,
        batch: Sequence[torch.Tensor],
        head: int,
    ) -> torch.Tensor:
        """Return gradient-enhanced PINN loss from residual derivatives."""
        if (
            head != self.n_heads - 1
            or self.local_epoch % self.UR_period != 0
            or not self.TL
        ):
            device = self.device if hasattr(self, "device") else None
            return torch.tensor(0.0, device=device)

        loss = 0.0
        for j in range(residuals.shape[1]):
            rj = residuals[:, j : j + 1]

            grads = torch.autograd.grad(
                outputs=rj,
                inputs=batch,
                grad_outputs=torch.ones_like(rj),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )

            for g in grads:
                loss = loss + (g**2).mean()

        return self.regularization_lambda * loss
