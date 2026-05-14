import torch
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency


def silhouette_loss(pred, target):
    return torch.mean((pred[..., 3] - target[..., 3]) ** 2)


def rgb_loss(pred, target):
    return torch.mean((pred[..., :3] - target[..., :3]) ** 2)


def regularization_loss(mesh, w_lap: float, w_edge: float, w_normal: float):
    lap = mesh_laplacian_smoothing(mesh, method="uniform")
    edge = mesh_edge_loss(mesh)
    normal = mesh_normal_consistency(mesh)
    total = w_lap * lap + w_edge * edge + w_normal * normal
    return total, {"lap": lap.item(), "edge": edge.item(), "normal": normal.item()}
