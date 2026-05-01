import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

def ema(x, alpha=0.05):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def plot_specific_runs(
    data,
    equation_name,
    models,
    metric="mse",
    head_idx=(0,),
    use_ema=False,
    start_from=None,
    title=None,
):
    data = data[equation_name]

    if not isinstance(head_idx, (list, tuple)):
        head_idx = [head_idx]

    def get_metric_data(model, head):
        if metric in {"mae", "mse", "l2_rel"}:
            return data[model][metric][head]

        if metric == "add_loss":
            if "None" in model:
                return None
            return [v for v in data[model][metric] if v != 0.0]

        return data[model][metric]

    plt.figure(figsize=(8, 5))

    for head in head_idx:
        for model in models:
            model_data = get_metric_data(model, head)

            if model_data is None:
                continue

            if start_from is not None:
                model_data = model_data[start_from:]

            x = np.arange(len(model_data))

            if use_ema:
                plt.plot(
                    x,
                    ema(model_data, alpha=0.2),
                    linestyle="--",
                    label=f"EMA {model} (head {head})",
                )

            plt.plot(
                x,
                model_data,
                linestyle="-",
                label=f"Raw {model} (head {head})",
            )

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    plot_title = title or f"{metric.upper()} comparison: {equation_name}, Heads {head_idx}"
    plt.title(plot_title)

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_solution_times(
    num_solver,
    times=(0.0, 0.2, 0.5, 0.8),
    r=0.01,
    dt=0.001,
    dx=0.0001,
    x_left=0.0,
    x_right=1.0,
    title="Exact solution",
):
    plt.figure(figsize=(8, 5))

    for t in times:
        x, _, u_ref = num_solver(r, dt, t, 0, x_left, x_right, dx)

        x = x.cpu().detach().numpy() if torch.is_tensor(x) else np.asarray(x)
        u_ref = (
            u_ref.cpu().detach().numpy()
            if torch.is_tensor(u_ref)
            else np.asarray(u_ref)
        )

        plt.plot(x, u_ref, linewidth=2, label=f"Solution at t = {t}")

    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_1d_solution_comparison_few(
    x,
    reference,
    regularized,
    vanilla,
    reference_label="Reference",
    regularized_label="Regularized",
    vanilla_label="Vanilla",
    error_label="Absolute Error",
    title_solution="Solution",
    title_error="Pointwise Error",
    filename=None,
    save_dir="."
):
    x = x.cpu().detach().numpy() if torch.is_tensor(x) else np.asarray(x)
    reference = reference.cpu().detach().numpy() if torch.is_tensor(reference) else np.asarray(reference)
    regularized = regularized.cpu().detach().numpy() if torch.is_tensor(regularized) else np.asarray(regularized)
    vanilla = vanilla.cpu().detach().numpy() if torch.is_tensor(vanilla) else np.asarray(vanilla)

    regularized_error = np.abs(regularized - reference)
    vanilla_error = np.abs(vanilla - reference)

    regularized_max_error = regularized_error.max()
    vanilla_max_error = vanilla_error.max()

    regularized_l2re = np.sqrt(np.sum((regularized - reference) ** 2) / np.sum(reference ** 2))
    vanilla_l2re = np.sqrt(np.sum((vanilla - reference) ** 2) / np.sum(reference ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, reference, "k-", linewidth=2, label=reference_label, alpha=0.8)
    ax1.plot(x, regularized, "r--", linewidth=2, label=regularized_label, alpha=0.7)
    ax1.plot(x, vanilla, "g-.", linewidth=2, label=vanilla_label, alpha=0.7)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("u(x)", fontsize=11)
    ax1.set_title(
        f"Regularized L2RE: {regularized_l2re:.2e} | Vanilla L2RE: {vanilla_l2re:.2e}",
        fontsize=11
    )
    ax1.legend(loc="best", frameon=False)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, regularized_error, "r--", linewidth=1.5, label=f"{regularized_label} error", alpha=0.8)
    ax2.plot(x, vanilla_error, "g-.", linewidth=1.5, label=f"{vanilla_label} error", alpha=0.8)
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel(error_label, fontsize=11)
    ax2.set_title(
        f"Regularized max: {regularized_max_error:.2e} | Vanilla max: {vanilla_max_error:.2e}",
        fontsize=11
    )
    ax2.legend(loc="best", frameon=False)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

    plt.show()


def plot_1d_solution_comparison(
    x,
    pred,
    exact,
    pred_label="Predicted",
    exact_label="Exact",
    error_label="|Error|",
    title_solution="Solution",
    title_error="Pointwise Error",
    filename=None,
    save_dir="."
):
    x = x.cpu().detach().numpy() if torch.is_tensor(x) else np.asarray(x)
    pred = pred.cpu().detach().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    exact = exact.cpu().detach().numpy() if torch.is_tensor(exact) else np.asarray(exact)

    error = np.abs(pred - exact)
    max_error = error.max()
    l2re = np.sqrt(np.sum((pred - exact) ** 2) / np.sum(exact ** 2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, exact, "k-", linewidth=2, label=exact_label, alpha=0.8)
    ax1.plot(x, pred, "r--", linewidth=2, label=pred_label, alpha=0.7)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("u(x)", fontsize=11)
    ax1.set_title(f"L2RE: {l2re:.2e}", fontsize=11)
    ax1.legend(loc="best", frameon=False)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, error, "b-", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel(error_label, fontsize=11)
    ax2.set_title(f"Max: {max_error:.2e}", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

    plt.show()


def plot_solution_heatmap(
    exact_solution,
    prediction,
    x,
    t,
    pred_label="Prediction",
    exact_label="Exact solution",
    error_label="Absolute Error",
):
    exact_solution = (
        exact_solution.detach().cpu().numpy()
        if torch.is_tensor(exact_solution)
        else np.asarray(exact_solution)
    )
    prediction = (
        prediction.detach().cpu().numpy()
        if torch.is_tensor(prediction)
        else np.asarray(prediction)
    )
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    t = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

    err = np.abs(exact_solution - prediction)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plt.subplots_adjust(wspace=0.28)

    vmin_sol = min(exact_solution.min(), prediction.min())
    vmax_sol = max(exact_solution.max(), prediction.max())

    im1 = axes[0].imshow(
        prediction,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=vmin_sol,
        vmax=vmax_sol,
    )
    axes[0].set_title(pred_label)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(
        exact_solution,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=vmin_sol,
        vmax=vmax_sol,
    )
    axes[1].set_title(exact_label)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(
        err,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap="hot",
    )
    axes[2].set_title(error_label)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(t.min(), t.max())

    plt.tight_layout()
    plt.show()