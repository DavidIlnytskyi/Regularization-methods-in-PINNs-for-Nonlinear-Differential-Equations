import sys
import platform
import subprocess
from datetime import datetime

import torch
import numpy as np
import seaborn as sns
import neurodiffeq
import matplotlib as mpl
import random
import os

from data_classes import RunConfig

def get_version(module):
    return getattr(module, "__version__", "builtin / unknown")


def save_reproducibility_snapshot(filename="reproducibility_snapshot.txt"):
    with open(filename, "w") as f:
        f.write("===== REPRODUCIBILITY SNAPSHOT =====\n\n")

        f.write("Timestamp:\n")
        f.write(f"{datetime.now()}\n\n")

        f.write("Python:\n")
        f.write(f"{sys.version}\n\n")

        f.write("Platform:\n")
        f.write(f"{platform.platform()}\n")
        f.write(f"Processor: {platform.processor()}\n\n")

        f.write("Core Libraries:\n")
        f.write(f"torch: {get_version(torch)}\n")
        f.write(f"numpy: {get_version(np)}\n")
        f.write(f"scipy: {get_version(__import__('scipy'))}\n")
        f.write(f"neurodiffeq: {get_version(neurodiffeq)}\n")
        f.write(f"matplotlib: {get_version(__import__('matplotlib'))}\n")
        f.write(f"seaborn: {get_version(sns)}\n")
        f.write("\n")

        f.write("CUDA:\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        f.write(f"CUDA version: {torch.version.cuda}\n")
        f.write(f"cuDNN version: {torch.backends.cudnn.version()}\n")

        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")

        f.write("\n")

        f.write("Installed packages (pip freeze):\n")
        try:
            packages = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"]
            ).decode()
            f.write(packages)
        except Exception as e:
            f.write(f"Could not retrieve pip freeze: {e}\n")

def is_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False

def set_paper_style():
    mpl.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "figure.dpi": 300,
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "text.usetex": False,
        }
    )

def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def build_regularization_values(
    regnames=("UR", "J", "gPINN", "L1", "L2"),
    ks=(2, 4, 7),
    powers=range(-6, 6),
    include_baseline=True,
):
    def generate_lambdas():
        return [k * (10**p) for k in ks for p in powers]

    values = {name: generate_lambdas() for name in regnames}

    if include_baseline:
        values[None] = [0.0]

    return values

def _callable_name(value):
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if qualname is None:
        return repr(value)
    if module and module != "__main__":
        return f"{module}.{qualname}"
    return qualname


def _format_config_value(value, indent=0):
    child_indent = indent + 2
    child_pad = " " * child_indent
    simple_types = (str, int, float, bool, type(None))

    if hasattr(value, "__dataclass_fields__"):
        lines = [f"{value.__class__.__name__}:"]
        for field_name in value.__dataclass_fields__:
            field_value = getattr(value, field_name)
            formatted = _format_config_value(field_value, child_indent)
            if "\n" in formatted:
                lines.append(f"{child_pad}{field_name}:")
                lines.extend(
                    f"{' ' * (child_indent + 2)}{line}"
                    for line in formatted.splitlines()
                )
            else:
                lines.append(f"{child_pad}{field_name}: {formatted}")
        return "\n".join(lines)

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for key in sorted(value, key=lambda item: str(item)):
            formatted = _format_config_value(value[key], child_indent)
            if "\n" in formatted:
                lines.append(f"{child_pad}{repr(key)}:")
                lines.extend(
                    f"{' ' * (child_indent + 2)}{line}"
                    for line in formatted.splitlines()
                )
            else:
                lines.append(f"{child_pad}{repr(key)}: {formatted}")
        return "\n".join(lines)

    if isinstance(value, (list, tuple, set)):
        if not value:
            return repr(value)
        if all(isinstance(item, simple_types) for item in value):
            return repr(value)
        lines = []
        for item in value:
            formatted = _format_config_value(item, child_indent)
            if "\n" in formatted:
                lines.append(f"{child_pad}-")
                lines.extend(
                    f"{' ' * (child_indent + 2)}{line}"
                    for line in formatted.splitlines()
                )
            else:
                lines.append(f"{child_pad}- {formatted}")
        return "\n".join(lines)

    if callable(value):
        return _callable_name(value)

    return repr(value)


def save_run_config_txt(config: RunConfig, save_dir: str, regularization_setup=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "run_config.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("RUN CONFIGURATION\n")
        f.write(f"Saved at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Save dir: {save_dir}\n\n")

        if regularization_setup is not None:
            f.write("Run-specific regularization setup:\n")
            f.write(_format_config_value(regularization_setup))
            f.write("\n\n")

        f.write("Full RunConfig:\n")
        f.write(_format_config_value(config))
        f.write("\n")

    return path