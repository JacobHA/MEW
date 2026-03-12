import dataclasses
from typing import Optional, Union


@dataclasses.dataclass(frozen=True)
class ASACTorchConfig:
    env_id_str: str
    gamma: float
    ent_coef: Union[float, str]
    batch_size: int
    buffer_size: int
    gradient_steps: int
    train_freq: int
    learning_rate: float
    learning_starts: int
    target_update_interval: int
    tau: float
    rho_learning_rate: float
    eval_freq: int
    total_timesteps: int
    value_based_centering: bool
    use_min_aggregation: bool
    rho_unbiased_step_size: bool
    reward_centering: bool
    recompute_td: bool
    max_grad_norm: Optional[float]

