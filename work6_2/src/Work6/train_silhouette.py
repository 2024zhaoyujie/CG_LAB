from pathlib import Path

import torch
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .data import build_cameras, ensure_seed, load_target_mesh, pick_device, timestamp_name
from .losses import regularization_loss, silhouette_loss
from .renderers import build_silhouette_renderer
from .visualize import ensure_dirs, save_gif, save_losses, save_mesh, save_metrics, save_step_image


def run_silhouette(cfg):
    ensure_seed(cfg.train.seed)
    device = pick_device()

    target_mesh = load_target_mesh(device=device)
    cameras = build_cameras(cfg.render.num_views, device)
    renderer = build_silhouette_renderer(
        cameras=cameras,
        image_size=cfg.render.image_size,
        sigma=cfg.render.sigma,
        faces_per_pixel=cfg.render.faces_per_pixel,
    )

    with torch.no_grad():
        target_images = renderer(target_mesh.extend(cfg.render.num_views))

    source_mesh = ico_sphere(4, device)
    deform_verts = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=cfg.train.lr)

    run_dir = Path(cfg.work_root) / timestamp_name("silhouette")
    ensure_dirs(run_dir)
    frame_paths = []

    history = {"total": [], "silhouette": [], "reg": []}
    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        deformed_mesh = source_mesh.offset_verts(deform_verts)
        batch_mesh = deformed_mesh.extend(cfg.render.num_views)
        pred_images = renderer(batch_mesh)

        sil = silhouette_loss(pred_images, target_images)
        reg, reg_items = regularization_loss(
            deformed_mesh,
            w_lap=cfg.loss.w_lap,
            w_edge=cfg.loss.w_edge,
            w_normal=cfg.loss.w_normal,
        )
        loss = sil + reg
        loss.backward()
        optimizer.step()

        history["total"].append(float(loss.item()))
        history["silhouette"].append(float(sil.item()))
        history["reg"].append(float(reg.item()))

        if step % cfg.train.save_every == 0 or step == cfg.train.steps:
            image_path = run_dir / "images" / f"silhouette_step_{step:04d}.png"
            save_step_image(image_path, pred_images[0, ..., :3])
            frame_paths.append(image_path)

        if step % cfg.train.log_every == 0:
            print(
                f"[silhouette] step={step:04d} "
                f"total={loss.item():.6f} sil={sil.item():.6f} "
                f"lap={reg_items['lap']:.6f} edge={reg_items['edge']:.6f} normal={reg_items['normal']:.6f}"
            )

    final_mesh = source_mesh.offset_verts(deform_verts.detach())
    save_mesh(run_dir / "meshes" / "silhouette_final.obj", final_mesh)
    save_losses(run_dir / "plots" / "silhouette_losses.png", history)
    save_gif(run_dir / "images" / "silhouette_turntable.gif", frame_paths)
    save_metrics(
        run_dir / "logs" / "metrics.json",
        {
            "mode": "silhouette",
            "seed": cfg.train.seed,
            "steps": cfg.train.steps,
            "final_total": history["total"][-1],
            "final_silhouette": history["silhouette"][-1],
        },
    )
    return run_dir
