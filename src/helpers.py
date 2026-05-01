import sys
import platform
import subprocess
from datetime import datetime

import torch
import numpy as np
import seaborn as sns
import neurodiffeq

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