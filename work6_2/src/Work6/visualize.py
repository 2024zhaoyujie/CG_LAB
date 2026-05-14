import json
from pathlib import Path
from typing import Dict, List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import save_obj


def ensure_dirs(run_dir: Path):
    for sub in ["images", "plots", "meshes", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def save_step_image(path: Path, image: torch.Tensor):
    img = image.detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_losses(path: Path, history: Dict[str, List[float]]):
    plt.figure(figsize=(8, 4))
    for key, values in history.items():
        if values:
            plt.plot(values, label=key)
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_mesh(path: Path, mesh):
    verts = mesh.verts_packed().detach().cpu()
    faces = mesh.faces_packed().detach().cpu()
    save_obj(str(path), verts=verts, faces=faces)


def save_metrics(path: Path, metrics: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_gif(path: Path, image_paths: List[Path], fps: int = 5):
    frames = []
    for p in image_paths:
        frames.append(imageio.v2.imread(p))
    if frames:
        imageio.mimsave(path, frames, fps=fps)
