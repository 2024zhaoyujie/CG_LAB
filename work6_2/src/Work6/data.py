from pathlib import Path
from typing import Tuple

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform


def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0, keepdim=True)
    scale = (verts - center).norm(dim=1).max()
    verts = (verts - center) / scale
    mesh = mesh.offset_verts(verts - mesh.verts_packed())
    return mesh


def load_target_mesh(device: torch.device, obj_path: str = "data/cow_mesh/cow.obj"):
    path = Path(obj_path)
    if not path.exists():
        raise FileNotFoundError(f"target mesh not found: {path}")
    mesh = load_objs_as_meshes([str(path)], device=device)
    return normalize_mesh(mesh)


def build_cameras(num_views: int, device: torch.device) -> FoVPerspectiveCameras:
    azim = torch.linspace(-180, 180, num_views, device=device)
    elev = torch.linspace(-25, 25, num_views, device=device)
    dist = torch.full((num_views,), 2.8, device=device)
    r, t = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return FoVPerspectiveCameras(device=device, R=r, T=t)


def ensure_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def timestamp_name(prefix: str) -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
