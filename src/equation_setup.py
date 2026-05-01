import os
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from neurodiffeq.neurodiffeq import diff
from neurodiffeq.generators import PredefinedGenerator

from helpers import set_seed
from neurodiffeqq import BundleIBVP1D
from models import NET, build_body_model, build_head_model
from mh_solver import MHSolver2D
from callbacks import DoSchedulerStep, SaveCallback, MetricsCallback
from numerical_solvers import solve_burgers, solve_allen_cahn, solve_fisher_kpp

from helpers import save_run_config_txt, sync_if_needed

from data_classes import *

def load_matching_state_dict(model, state_dict):
    model_dict = model.state_dict()

    filtered_state = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    skipped = {
        k: {"checkpoint": tuple(v.shape), "model": tuple(model_dict[k].shape)}
        for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape != v.shape
    }

    unexpected = {
        k: tuple(v.shape) for k, v in state_dict.items() if k not in model_dict
    }

    model.load_state_dict(filtered_state, strict=False)

    return {"skipped_shape_mismatch": skipped, "unexpected_keys": unexpected}


def _normalize_optimizer_checkpoint_name(name):
    if name is None:
        return None

    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "l_bfgs": "lbfgs",
        "torch.optim.adam": "adam",
        "torch.optim.lbfgs": "lbfgs",
    }
    return aliases.get(normalized, normalized)


def get_optimizer_checkpoint_name(solver):
    optimizer = getattr(solver, "optimizer", None)
    explicit_name = getattr(solver, "optimizer_name", None)
    if explicit_name is not None:
        return _normalize_optimizer_checkpoint_name(explicit_name)
    if optimizer is None:
        return None
    return _normalize_optimizer_checkpoint_name(optimizer.__class__.__name__)


def _resolve_optimizer_for_checkpoint(solver, checkpoint, optimizer_lookup=None):
    saved_name = _normalize_optimizer_checkpoint_name(checkpoint.get("optimizer_name"))
    current_optimizer = getattr(solver, "optimizer", None)
    current_name = _normalize_optimizer_checkpoint_name(getattr(solver, "optimizer_name", None))
    if current_name is None and current_optimizer is not None:
        current_name = _normalize_optimizer_checkpoint_name(current_optimizer.__class__.__name__)

    if saved_name is None:
        return current_optimizer, current_name

    candidates = {}
    if optimizer_lookup:
        candidates.update(
            {
                _normalize_optimizer_checkpoint_name(name): optimizer
                for name, optimizer in optimizer_lookup.items()
                if optimizer is not None
            }
        )

    named_attrs = {
        "adam": getattr(solver, "adam_optimizer", None),
        "lbfgs": getattr(solver, "lbfgs_optimizer", None),
    }
    for name, optimizer in named_attrs.items():
        if optimizer is not None:
            candidates[name] = optimizer

    if current_optimizer is not None and current_name is not None:
        candidates.setdefault(current_name, current_optimizer)

    resolved_optimizer = candidates.get(saved_name)
    if resolved_optimizer is None:
        available = ", ".join(sorted(candidates)) or "none"
        raise ValueError(
            f"Checkpoint was saved with optimizer '{saved_name}', but no matching "
            f"optimizer is available on the current solver. Available: {available}"
        )

    return resolved_optimizer, saved_name


def save_nets(path, solver, save_optimizer=True):
    nets = solver.all_nets
    n_rows, n_cols = nets.shape

    H_state = []
    head_state = []

    for i in range(n_rows):
        H_state.append(nets[i, 0].H_model.state_dict())
        for j in range(n_cols):
            head_state.append(nets[i, j].head_model.state_dict())

    best_nets_state = []
    best_nets_list = solver.best_nets_list

    if isinstance(best_nets_list, (list, tuple, np.ndarray)):
        for row in best_nets_list:
            if not isinstance(row, (list, tuple, np.ndarray)):
                continue

            row_state = []
            for net in row:
                if not hasattr(net, "H_model") or not hasattr(net, "head_model"):
                    continue

                row_state.append(
                    {
                        "H_model": net.H_model.state_dict(),
                        "head_model": net.head_model.state_dict(),
                    }
                )

            best_nets_state.append(row_state)

    checkpoint = {
        "H_state": H_state,
        "head_state": head_state,
        "best_nets_state": best_nets_state,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "global_epoch": solver.global_epoch,
        "train_loss": solver.metrics_history["train_loss"],
        "add_loss": solver.metrics_history["add_loss"],
    }

    if save_optimizer and hasattr(solver, "optimizer") and solver.optimizer is not None:
        checkpoint["optimizer_name"] = get_optimizer_checkpoint_name(solver)
        checkpoint["optimizer_state"] = solver.optimizer.state_dict()

    torch.save(checkpoint, path)


def load_nets(
    path,
    solver,
    load_heads=True,
    load_optimizer=False,
    load_best_nets=False,
    map_location="cpu",
    optimizer_lookup=None,
):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    H_state = checkpoint["H_state"]
    head_state = checkpoint["head_state"]
    n_rows, n_cols = solver.all_nets.shape

    ckpt_rows = checkpoint.get("n_rows")
    ckpt_cols = checkpoint.get("n_cols")

    if ckpt_rows is not None and ckpt_cols is not None:
        if ckpt_rows != n_rows or ckpt_cols != n_cols:
            print(
                f"Checkpoint shape {(ckpt_rows, ckpt_cols)} "
                f"does not match solver shape {(n_rows, n_cols)}"
            )

    if load_optimizer:
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is None:
            raise ValueError("Checkpoint does not contain optimizer_state")
        optimizer, optimizer_name = _resolve_optimizer_for_checkpoint(
            solver,
            checkpoint,
            optimizer_lookup=optimizer_lookup,
        )
        optimizer.load_state_dict(optimizer_state)
        solver.optimizer = optimizer
        solver.optimizer_name = optimizer_name

    if checkpoint.get("train_loss") is not None:
        solver.metrics_history["train_loss"] = checkpoint["train_loss"]

    if checkpoint.get("add_loss") is not None:
        solver.metrics_history["add_loss"] = checkpoint["add_loss"]

    max_rows = min(n_rows, len(H_state))

    for i in range(max_rows):
        load_matching_state_dict(solver.all_nets[i, 0].H_model, H_state[i])

        if load_heads:
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx >= len(head_state):
                    continue

                load_matching_state_dict(
                    solver.all_nets[i, j].head_model, head_state[idx]
                )

    if load_best_nets:
        best_nets_state = checkpoint.get("best_nets_state")
        if best_nets_state is not None:
            if len(best_nets_state) != len(solver.best_nets_list):
                raise ValueError("best_nets_list outer size does not match checkpoint")

            for i, row_state in enumerate(best_nets_state):
                if len(row_state) != len(solver.best_nets_list[i]):
                    raise ValueError(f"best_nets_list inner size mismatch at row {i}")

                for j, net_state in enumerate(row_state):
                    solver.best_nets_list[i][j].H_model.load_state_dict(
                        net_state["H_model"]
                    )
                    if load_heads:
                        solver.best_nets_list[i][j].head_model.load_state_dict(
                            net_state["head_model"]
                        )

    return solver, checkpoint["global_epoch"]


# =========================
# Allen-Cahn
# =========================

def allen_cahn(x, idx):
    return x**2 * torch.cos(np.pi * x + idx * (torch.pi / 6))


def allen_cahn_ic(idx=0):
    def ic(x):
        return allen_cahn(x, idx)

    return ic


def make_allen_cahn_bc_left(x_left: float, idx=0):
    def bc_left(t):
        x = torch.full_like(t, x_left)
        return allen_cahn(x, idx)

    return bc_left


def make_allen_cahn_bc_right(x_right: float, idx=0):
    def bc_right(t):
        x = torch.full_like(t, x_right)
        return allen_cahn(x, idx)

    return bc_right


# =========================
# Fisher-KPP
# =========================


def fisher_ic(idx):
    idx += 1
    idx = torch.as_tensor(idx)

    def ic(x):
        x = torch.as_tensor(x, device=idx.device)
        result = (torch.exp(-idx * x) - torch.exp(-idx)) / (1 - torch.exp(-idx))
        return result

    return ic


def make_fisher_bc_left(x_left: float, idx=0):
    def bc_left(t):
        return torch.ones_like(t)

    return bc_left


def make_fisher_bc_right(x_right: float, idx=0):
    def bc_right(t):
        return torch.zeros_like(t)

    return bc_right


# =========================
# Burgers equation
# =========================


def burgers_ic(idx):
    def ic(x):
        return -torch.sin(torch.pi * x + idx * (torch.pi / 6))

    return ic


def make_burgers_bc_left(x_left: float = 0.0, idx=None):
    def bc_left(t):
        return torch.zeros_like(t)

    return bc_left


def make_burgers_bc_right(x_right: float = 1.0, idx=None):
    def bc_right(t):
        return torch.zeros_like(t)

    return bc_right


def burgers_equation(head):
    def equation(u, x, t, nu):
        term1 = diff(u, t)
        term2 = u * diff(u, x)
        term3 = -nu * diff(u, x, order=2)
        return [term1 + term2 + term3]

    equation.head = head
    equation.name = "burgers"
    return equation


def allen_cahn_equation(head):
    def equation(u, x, t, eps):
        term1 = diff(u, t)
        term2 = -eps * diff(u, x, order=2)
        term3 = 5 * u**3 - 5 * u
        return [term1 + term2 + term3]

    equation.head = head
    equation.name = "allen_cahn"
    return equation


def fisher_kpp_equation(head):
    def equation(u, x, t, r, D=1):
        term1 = diff(u, t)
        term2 = -D * diff(u, x, order=2)
        term3 = -r * u * (1 - u)
        return [term1 + term2 + term3]

    equation.head = head
    equation.name = "fisher_kpp"
    return equation


def generate_equations(n_heads, equation_factory):
    eq_dict = {f"equations_{head}": equation_factory(head) for head in range(n_heads)}
    return eq_dict

def build_ib_conditions(config: RunConfig):
    eq = config.equation
    d = config.domain
    return [
        BundleIBVP1D(
            t_min=d.t_min,
            t_min_val=eq.ic_fn(head),
            x_min=d.x_min,
            x_min_val=eq.bc_left_builder(d.x_min, idx=head),
            x_max=d.x_max,
            x_max_val=eq.bc_right_builder(d.x_max, idx=head),
        )
        for head in range(eq.ib_conditions_size)
    ]


def build_eval_tensors(config: RunConfig):
    d = config.domain
    e = config.eval

    x = torch.arange(d.x_min, d.x_max + e.dx, e.dx, requires_grad=True)
    t = torch.full_like(x, e.t_i, requires_grad=True)
    p = torch.linspace(
        d.param_min,
        d.param_max,
        config.equation.ib_conditions_size,
        dtype=x.dtype,
        device=x.device,
    )

    return x, t, p


def build_regularization_setups(equation_name, reg_lambdas):
    setups = []

    for regtype, lambdas in reg_lambdas.items():
        for lambda_v in lambdas:

            setups.append(
                RegularizationSetup(
                    name=f"{equation_name}_{regtype}_lambda_{lambda_v}",
                    reg_type=regtype,
                    reg_lambda=lambda_v,
                    use_regularization=regtype is not None,
                )
            )

    return setups


def build_save_dir(root_dir: str, model_name: str):
    save_dir = os.path.join(root_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def build_callbacks(
    config: RunConfig,
    *,
    scheduler_cb,
    save_dir: str,
    x_eval,
    t_eval,
    p_eval,
):
    d = config.domain
    e = config.eval
    callbacks = []

    if scheduler_cb is not None:
        callbacks.append(scheduler_cb)

    callbacks.append(SaveCallback(n=config.train.save_period, save_dir=save_dir))

    metric_cb = None
    if config.equation.exact_fns_container is not None:
        metric_cb = MetricsCallback(
            x=x_eval,
            t=t_eval,
            r=p_eval,
            reference_fns_container=config.equation.exact_fns_container,
            period=config.eval.period,
            convergence_value=config.eval.convergence_value,
            n_heads=config.equation.ib_conditions_size,
        )
    else:
        reference_solutions = []
        for head_idx in range(config.equation.ib_conditions_size):
            head_param = float(p_eval[head_idx].detach().cpu())
            x_num, u0, u_num = config.equation.numerical_solution_func(
                r=head_param,
                dt=1e-4,
                T=e.t_i,
                idx=head_idx,
                x_left=d.x_min,
                x_right=d.x_max,
                dx=e.dx,
            )
            reference_solutions.append(u_num)

        metric_cb = MetricsCallback(
            x=x_eval,
            t=t_eval,
            r=p_eval,
            convergence_value=config.eval.convergence_value,
            n_heads=config.equation.ib_conditions_size,
            period=config.eval.period,
            reference_solutions=reference_solutions,
        )
    print(metric_cb)
    callbacks.append(metric_cb)

    return callbacks


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unique_trainable_parameters(modules):
    seen = set()
    for module in modules:
        for p in module.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            yield p


def to_jsonable(obj):
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def save_experiments_to_json(results, total_time, save_path=None):
    if save_path is None:
        raise ValueError("save_path must be defined")

    summary = {}

    for run in results:
        model_name = run["model_name"]
        metric_cb = run["metric_cb"]
        model_solver = run["solver"]

        equation_name = run["equation_name"]
        if equation_name is None:
            equation_name = getattr(
                run["regularization_setup"], "equation_name", "unknown_equation"
            )

        if equation_name not in summary:
            summary[equation_name] = {}

        summary[equation_name][model_name] = {
            "convergence_value": to_jsonable(metric_cb.convergence_value),
            "predictions": to_jsonable(metric_cb.predictions),
            "mse": to_jsonable(metric_cb.mse),
            "mae": to_jsonable(metric_cb.mae),
            "l2_rel": to_jsonable(metric_cb.l2_rel),
            "r2_loss": run["solver"].metrics_history["r2_loss"],
            "add_loss": run["solver"].metrics_history["add_loss"],
            "convergence_epoch": to_jsonable(metric_cb.convergence_epoch),
            "eval_period": to_jsonable(metric_cb.period),
            "regularization_period": to_jsonable(model_solver.UR_period),
            "train_time": to_jsonable(metric_cb.training_time),
        }
    summary[equation_name]["total_time"] = total_time

    with open(f"{save_path}/{equation_name}_data.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    return summary

def normalize_optimizer_mode(mode):
    normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "adam_lbfgs": "adam_then_lbfgs",
        "adam_then_l_bfgs": "adam_then_lbfgs",
        "l_bfgs": "lbfgs",
    }
    normalized = aliases.get(normalized, normalized)

    valid_modes = {"adam", "lbfgs", "adam_then_lbfgs"}
    if normalized not in valid_modes:
        supported = ", ".join(sorted(valid_modes))
        raise ValueError(f"Unsupported optimizer mode '{mode}'. Use one of: {supported}.")

    return normalized


def make_baseline_regularization_setup(equation_name):
    return RegularizationSetup(
        name=f"{equation_name}_None_lambda_0.0",
        reg_type=None,
        reg_lambda=0.0,
        use_regularization=False,
    )


def setup_PINN_2D_from_config(
    config: RunConfig,
    *,
    init_cond,
    regularization_setup: Optional[RegularizationSetup] = None,
):
    d = config.domain
    tcfg = config.train
    ncfg = config.network
    ocfg = config.optimizer
    scfg = config.scheduler
    bcfg = config.solver_behavior
    io = config.io
    eq = config.equation
    optimizer_mode = normalize_optimizer_mode(ocfg.mode)

    if regularization_setup is None:
        regularization_setup = make_baseline_regularization_setup(eq.name)

    nu_list = torch.linspace(d.param_min, d.param_max, d.param_num)
    t_list = torch.linspace(d.t_min, d.t_max, d.t_num)
    x_list = torch.linspace(d.x_min, d.x_max, d.x_num)

    T, X, N = torch.meshgrid(t_list, x_list, nu_list, indexing="ij")
    train_generator = PredefinedGenerator(X, T, N)

    init_cond = np.array(init_cond, dtype=object).reshape(1, -1)
    n_heads = int(len(init_cond[0, :]))

    basis_length = ncfg.basis_length if ncfg.basis_length is not None else n_heads

    H = [
        build_body_model(
            ncfg,
            n_input_units=3,
            basis_length=basis_length,
        )
        for _ in range(eq.equations_number)
    ]

    heads = [
        [
            build_head_model(
                ncfg,
                basis_length=basis_length,
            )
            for __ in range(n_heads)
        ]
        for _ in range(len(H))
    ]

    nets = np.empty((len(H), n_heads), dtype=object)
    for i in range(len(H)):
        for j in range(n_heads):
            nets[i, j] = NET(H[i], heads[i][j])

    eq_dict = generate_equations(n_heads, eq.equation_factory)
    equation_list = [eq_dict[f"equations_{head}"] for head in range(n_heads)]

    optimizer_params = list(unique_trainable_parameters(nets.reshape(-1)))

    adam = None
    if optimizer_mode in {"adam", "adam_then_lbfgs"}:
        adam = torch.optim.Adam(
            optimizer_params,
            lr=ocfg.lr,
            amsgrad=ocfg.amsgrad,
        )

    lbfgs = None
    if optimizer_mode in {"lbfgs", "adam_then_lbfgs"} and ocfg.lbfgs_epochs > 0:
        lbfgs = torch.optim.LBFGS(
            optimizer_params,
            lr=ocfg.lbfgs_lr,
            max_iter=ocfg.lbfgs_max_iter,
            max_eval=ocfg.lbfgs_max_eval,
            tolerance_grad=ocfg.lbfgs_tolerance_grad,
            tolerance_change=ocfg.lbfgs_tolerance_change,
            history_size=ocfg.lbfgs_history_size,
            line_search_fn=ocfg.lbfgs_line_search_fn,
        )

    scheduler = None
    scheduler_cb = None
    if adam is not None:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            adam,
            start_factor=scfg.warmup_start_factor,
            total_iters=scfg.warmup_iters,
        )

        main_scheduler = torch.optim.lr_scheduler.StepLR(
            adam,
            step_size=scfg.step_size,
            gamma=scfg.gamma,
        )

        scheduler_milestone = (
            scfg.warmup_iters
            if scfg.sequential_milestone is None
            else scfg.sequential_milestone
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            adam,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[scheduler_milestone],
        )
        scheduler_cb = DoSchedulerStep(scheduler=scheduler)

    solver = MHSolver2D(
        pde_system=equation_list,
        equations_number=eq.equations_number,
        conditions=init_cond,
        conditions_list=init_cond,
        all_nets=nets,
        train_generator=train_generator,
        valid_generator=None,
        xy_min=(d.x_min, d.t_min),
        xy_max=(d.x_max, d.t_max),
        t_min=d.t_min,
        t_max=d.t_max,
        t_sampling=d.t_num,
        x_min=d.x_min,
        x_max=d.x_max,
        x_sampling=d.x_num,
        param_min=d.param_min,
        param_max=d.param_max,
        param_sampling=d.param_num,
        eq_param_index=(0,),
        n_batches_valid=bcfg.n_batches_valid,
        n_batches_train=bcfg.n_batches_train,
        optimizer=adam if adam is not None else lbfgs,
        UR_period=tcfg.regularization_period,
        flatten=regularization_setup.use_regularization,
        regularization=regularization_setup.reg_type,
        regularization_lambda=regularization_setup.reg_lambda,
    )

    if regularization_setup is not None:
        if hasattr(solver, "regularization_setup"):
            solver.regularization_setup = regularization_setup

    solver.adam_optimizer = adam
    solver.lbfgs_optimizer = lbfgs
    solver.optimizer_name = get_optimizer_checkpoint_name(solver)

    if io.TL_path:
        load_nets(
            io.TL_path,
            solver=solver,
            load_heads=io.load_heads,
        )

    return solver, nets, adam, lbfgs, scheduler, scheduler_cb, train_generator




def run_experiment(config: RunConfig):
    if config.io.models_save_dir is None:
        raise ValueError("config.io.models_save_dir must be set")

    if config.domain.param_num != config.equation.ib_conditions_size:
        raise ValueError(
            "IB conditions and equation parameters should have the same size"
        )

    set_seed(config.equation.seed)
    os.makedirs(config.io.models_save_dir, exist_ok=True)
    optimizer_mode = normalize_optimizer_mode(config.optimizer.mode)

    ib_conditions = build_ib_conditions(config)
    x_eval, t_eval, p_eval = build_eval_tensors(config)

    regularization_setups = build_regularization_setups(
        config.equation.name,
        config.regularizations,
    )

    enabled_regularizations = [
        setup for setup in regularization_setups if setup.reg_type is not None
    ]
    if enabled_regularizations:
        regularization_applications = (
            config.train.epochs // config.train.regularization_period
        )
        if regularization_applications < 10:
            warnings.warn(
                f"Regularization will run only {regularization_applications} time(s) "
                f"in {config.train.epochs} epochs. Decrease train.regularization_period "
                "if this run is meant to measure a regularization effect.",
                RuntimeWarning,
            )

    results = []
    total_start = time.perf_counter()

    init_checkpoint_path = os.path.join(
        config.io.models_save_dir, "shared_initialization.pt"
    )

    for run_idx, regularization_setup in enumerate(regularization_setups):
        save_dir = build_save_dir(config.io.models_save_dir, regularization_setup.name)
        save_run_config_txt(config, save_dir, regularization_setup)

        solver, nets, adam, lbfgs, scheduler, scheduler_cb, train_generator = (
            setup_PINN_2D_from_config(
                config,
                init_cond=ib_conditions,
                regularization_setup=regularization_setup,
            )
        )
        if run_idx == 0:
            save_nets(
                init_checkpoint_path,
                solver,
                save_optimizer=False,
            )
        else:
            load_nets(
                init_checkpoint_path,
                solver=solver,
                load_heads=True,
                load_optimizer=False,
                load_best_nets=False,
            )
        parameters_number = count_params(nets[0, 0])

        callbacks = build_callbacks(
            config,
            scheduler_cb=scheduler_cb,
            save_dir=save_dir,
            x_eval=x_eval,
            t_eval=t_eval,
            p_eval=p_eval,
        )

        sync_if_needed()
        train_start = time.perf_counter()

        if adam is not None:
            solver.optimizer = adam
            solver.optimizer_name = "adam"
            solver.fit(config.train.epochs, callbacks=callbacks)

        if lbfgs is not None:
            solver.optimizer = lbfgs
            solver.optimizer_name = "lbfgs"
            lbfgs_callbacks = [cb for cb in callbacks if cb is not scheduler_cb]
            solver.fit(config.optimizer.lbfgs_epochs, callbacks=lbfgs_callbacks)

        # if adam is None and lbfgs is None:
        #     raise ValueError(f"Optimizer mode '{optimizer_mode}' did not create an optimizer")

        sync_if_needed()
        training_time = time.perf_counter() - train_start

        metric_cb = callbacks[-1] if callbacks else None
        if metric_cb is not None and hasattr(metric_cb, "training_time"):
            metric_cb.training_time = training_time

        results.append(
            {
                "equation_name": config.equation.name,
                "model_name": regularization_setup.name,
                "save_dir": save_dir,
                "regularization_setup": regularization_setup,
                "solver": solver,
                "nets": nets,
                "optimizer": adam,
                "lbfgs_optimizer": lbfgs,
                "scheduler": scheduler,
                "scheduler_cb": scheduler_cb,
                "metric_cb": metric_cb,
                "train_generator": train_generator,
                "param_num": parameters_number,
                "init_checkpoint_path": init_checkpoint_path,
            }
        )

    sync_if_needed()
    total_time = time.perf_counter() - total_start

    save_experiments_to_json(results, total_time, save_path=config.io.models_save_dir)

    return results


EQUATION_REGISTRY = {}


def register_equation(spec: EquationSpec):
    EQUATION_REGISTRY[spec.name] = spec


def make_run_config(
    *,
    equation_name: str,
    models_save_dir: str,
    domain: Optional[DomainConfig] = None,
    network: Optional[NetworkConfig] = None,
    optimizer: Optional[OptimizerConfig] = None,
    scheduler: Optional[SchedulerConfig] = None,
    solver_behavior: Optional[SolverBehaviorConfig] = None,
    train: Optional[TrainConfig] = None,
    eval_cfg: Optional[EvalConfig] = None,
    io: Optional[IOConfig] = None,
    regularizations: Optional[dict[str, Any]] = None,
):
    if equation_name not in EQUATION_REGISTRY:
        raise ValueError(f"Unknown equation '{equation_name}'")

    os.makedirs(models_save_dir, exist_ok=True)

    eq_spec = EQUATION_REGISTRY[equation_name]

    if regularizations is None:
        regularizations = {None: [0.0]}

    final_io = io or IOConfig()
    final_io.equation_name = equation_name
    final_io.models_save_dir = models_save_dir

    return RunConfig(
        domain=domain or DomainConfig(),
        network=network or NetworkConfig(),
        optimizer=optimizer or OptimizerConfig(),
        scheduler=scheduler or SchedulerConfig(),
        solver_behavior=solver_behavior or SolverBehaviorConfig(),
        train=train or TrainConfig(),
        eval=eval_cfg or EvalConfig(),
        io=final_io,
        equation=eq_spec,
        regularizations=regularizations or {},
    )