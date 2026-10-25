"""Paper defaults and explicitly smaller engineering smoke presets."""
from dataclasses import asdict, dataclass, fields
import json
import math


@dataclass
class Config:
    seed: int = 0
    threads: int = 2
    history_window: int = 20
    latent_dim: int = 64
    embedding_dim: int = 32  # Not specified in the manuscript.
    hidden_dim: int = 256
    horizons: tuple = (1, 3, 5)
    gamma: float = .99
    lambda_length: float = .5
    lambda_successor: float = .1
    encoder_ema: float = .995
    ensemble_size: int = 5  # Distinct from the evaluation exit threshold M=0.
    diffusion_steps: int = 20
    uncertainty_samples: int = 4
    lambda_reward: float = .1
    lambda_value: float = .1
    feedback_loss_weight: float = 1.
    terminal_loss_weight: float = 1.
    support_k: int = 50
    support_threshold: float = 1e-5
    rollout_horizon: int = 5
    rollout_batch: int = 64
    learning_rate: float = 3e-4
    target_tau: float = .005
    batch_size: int = 256
    representation_epochs: int = 50
    world_epochs: int = 50
    # Zero means use epoch budget for representation/world model stages.
    representation_steps: int = 0
    world_steps: int = 0
    behavior_steps: int = 1000
    value_steps: int = 1000
    policy_steps: int = 1000  # Manuscript does not specify this budget.
    no_repeat: bool = True
    log_every: int = 25

    def validate(self):
        defaults = type(self)()
        for name, value in self.to_dict().items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if isinstance(getattr(defaults, name), int) and name != "no_repeat":
                if type(value) is not int:
                    raise ValueError(f"{name} must be an integer")
        if type(self.no_repeat) is not bool:
            raise ValueError("no_repeat must be boolean")
        for name in ("history_window", "latent_dim", "embedding_dim", "hidden_dim",
                     "ensemble_size", "diffusion_steps", "uncertainty_samples", "support_k",
                     "rollout_horizon", "rollout_batch", "batch_size", "threads", "log_every",
                     "behavior_steps", "value_steps", "policy_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.ensemble_size < 2 or self.diffusion_steps < 2:
            raise ValueError("FORUM requires >=2 ensemble members and diffusion steps")
        if not self.horizons or any(type(k) is not int or k < 1 for k in self.horizons):
            raise ValueError("Invalid future horizons")
        if not 0 <= self.gamma < 1 or not 0 <= self.encoder_ema < 1 or not 0 < self.target_tau <= 1:
            raise ValueError("Invalid discount/EMA coefficients")
        if not 0 <= self.support_threshold <= 1 or self.learning_rate <= 0:
            raise ValueError("Invalid support threshold or learning rate")
        for name in ("lambda_length", "lambda_successor", "lambda_reward", "lambda_value",
                     "feedback_loss_weight", "terminal_loss_weight", "representation_steps", "world_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.representation_epochs < 1 or self.world_epochs < 1:
            raise ValueError("Epoch budgets must be positive")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            values = json.load(f)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if "horizons" in values:
            values["horizons"] = tuple(values["horizons"])
        return cls(**values).validate()
