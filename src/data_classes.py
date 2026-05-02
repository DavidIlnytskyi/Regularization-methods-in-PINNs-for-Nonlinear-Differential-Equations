"""Configuration dataclasses for PINN experiment setup.

The classes in this module group domain, network, optimizer, scheduler,
training, evaluation, I/O, equation, and regularization options into typed
containers that are passed through the experiment-building pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class DomainConfig:
    """Spatial, temporal, and equation-parameter sampling ranges."""

    t_min: float = 0.0
    t_max: float = 1.0
    t_num: int = 64

    x_min: float = 0.0
    x_max: float = 1.0
    x_num: int = 64

    param_min: float = 1.0
    param_max: float = 1.0
    param_num: int = 1


@dataclass
class NetworkConfig:
    """Neural network architecture and initialization settings."""

    body_units: tuple[int, ...] = (20, 20, 20, 20, 20)
    head_units: tuple[int, ...] = (20, 20)
    basis_length: Optional[int] = 20
    use_bias: bool = True
    use_bias_in_body: Optional[bool] = None
    use_bias_in_heads: Optional[bool] = None
    small_normal_std: float = 1e-2
    weights_init_method: str = "default"
    input_encoding: str = "identity"
    fourier_mapping_size: int = 32
    fourier_scale: float = 3.0
    fourier_include_input: bool = False
    fourier_trainable: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer mode and hyperparameters for Adam and L-BFGS phases."""

    mode: str = "adam_then_lbfgs"  # one of: adam, lbfgs, adam_then_lbfgs
    lr: float = 1e-3
    amsgrad: bool = False
    lbfgs_epochs: int = 0
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_max_eval: Optional[int] = None
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: Optional[str] = "strong_wolfe"


@dataclass
class SchedulerConfig:
    """Learning-rate warmup and step scheduler settings."""

    warmup_start_factor: float = 1e-3
    warmup_iters: int = 1000
    step_size: int = 1000
    gamma: float = 0.985
    sequential_milestone: Optional[int] = None


@dataclass
class SolverBehaviorConfig:
    """Batch counts used by the solver during train and validation epochs."""

    n_batches_train: int = 1
    n_batches_valid: int = 0


@dataclass
class TrainConfig:
    """Training loop duration and checkpoint/regularization cadence."""

    epochs: int = 10_000
    save_period: int = 10_000
    regularization_period: int = 100


@dataclass
class EvalConfig:
    """Evaluation grid, target time/parameter, and convergence threshold."""

    period: int = 5
    dx: float = 0.0001
    t_i: float = 0.2
    param_i: float = 1.0
    convergence_value: float = 1e-3


@dataclass
class IOConfig:
    """Input/output paths and transfer-learning loading options."""

    equation_name: Optional[str] = None
    models_save_dir: Optional[str] = None
    TL_path: Optional[str] = None
    load_heads: bool = False


@dataclass
class EquationSpec:
    """Equation callbacks, boundary/initial conditions, and metadata."""

    name: str
    equation_factory: Callable[..., Any]
    ic_fn: Callable[..., Any]
    bc_left_builder: Callable[[float], Callable]
    bc_right_builder: Callable[[float], Callable]
    exact_fns_container: Optional[Callable[..., Any]] = None
    numerical_solution_func: Optional[Callable[..., Any]] = None
    equations_number: int = 1
    ib_conditions_size: int = 1
    seed: int = 42


@dataclass
class RegularizationSetup:
    """Single regularization experiment name, type, weight, and enable flag."""

    name: str
    reg_type: Optional[str]
    reg_lambda: float
    use_regularization: bool = True


@dataclass
class RunConfig:
    """Top-level configuration bundle for a complete experiment run."""

    domain: DomainConfig
    network: NetworkConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    solver_behavior: SolverBehaviorConfig
    train: TrainConfig
    eval: EvalConfig
    io: IOConfig
    equation: EquationSpec
    regularizations: dict[str, Any] = field(default_factory=dict)
