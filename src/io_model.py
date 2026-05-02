import numpy as np
import torch

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


def resolve_optimizer_for_checkpoint(solver, checkpoint, optimizer_lookup=None):
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
        optimizer, optimizer_name = resolve_optimizer_for_checkpoint(
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
