from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class DomainConfig:
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
    warmup_start_factor: float = 1e-3
    warmup_iters: int = 1000
    step_size: int = 1000
    gamma: float = 0.985
    sequential_milestone: Optional[int] = None


@dataclass
class SolverBehaviorConfig:
    n_batches_train: int = 1
    n_batches_valid: int = 0


@dataclass
class TrainConfig:
    epochs: int = 10_000
    save_period: int = 10_000
    regularization_period: int = 100


@dataclass
class EvalConfig:
    period: int = 5
    dx: float = 0.0001
    t_i: float = 0.2
    param_i: float = 1.0
    convergence_value: float = 1e-3


@dataclass
class IOConfig:
    equation_name: Optional[str] = None
    models_save_dir: Optional[str] = None
    TL_path: Optional[str] = None
    load_heads: bool = False


@dataclass
class EquationSpec:
    name: str
    equation_factory: Callable
    ic_fn: Callable
    bc_left_builder: Callable[[float], Callable]
    bc_right_builder: Callable[[float], Callable]
    exact_fns_container: Optional[Callable] = None
    numerical_solution_func: Optional[Callable] = None
    equations_number: int = 1
    ib_conditions_size: int = 1
    seed: int = 42


@dataclass
class RegularizationSetup:
    name: str
    reg_type: Optional[str]
    reg_lambda: float
    use_regularization: bool = True


@dataclass
class RunConfig:
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
