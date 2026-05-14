from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RenderConfig:
    image_size: int = 224
    num_views: int = 18
    sigma: float = 1e-4
    faces_per_pixel: int = 60


@dataclass
class TrainConfig:
    steps: int = 220
    lr: float = 0.08
    log_every: int = 10
    save_every: int = 25
    seed: int = 2026


@dataclass
class LossConfig:
    w_rgb: float = 1.0
    w_lap: float = 0.6
    w_edge: float = 1.0
    w_normal: float = 0.05


@dataclass
class ExperimentConfig:
    mode: str = "silhouette"
    work_root: Path = Path("outputs/work6")
    render: RenderConfig = field(default_factory=RenderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def build_config(mode: str, steps: int, image_size: int, num_views: int, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.mode = mode
    cfg.train.steps = steps
    cfg.train.seed = seed
    cfg.render.image_size = image_size
    cfg.render.num_views = num_views

    if mode == "textured":
        cfg.train.lr = 0.06
        cfg.loss.w_rgb = 1.2
        cfg.loss.w_lap = 0.45
    elif mode == "both":
        cfg.train.lr = 0.05
        cfg.loss.w_rgb = 1.1
        cfg.loss.w_lap = 0.5
        cfg.train.save_every = 20
    return cfg
