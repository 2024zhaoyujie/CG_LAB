from pathlib import Path

import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.utils import ico_sphere

from .data import build_cameras, ensure_seed, load_target_mesh, pick_device, timestamp_name
from .losses import regularization_loss, rgb_loss, silhouette_loss
from .renderers import build_rgb_renderer, build_silhouette_renderer
from .visualize import ensure_dirs, save_gif, save_losses, save_mesh, save_metrics, save_step_image


def run_textured(cfg):
    ensure_seed(cfg.train.seed)
    device = pick_device()

    target_mesh = load_target_mesh(device=device)
    cameras = build_cameras(cfg.render.num_views, device)
    sil_renderer = build_silhouette_renderer(
        cameras=cameras,
        image_size=cfg.render.image_size,
        sigma=cfg.render.sigma,
        faces_per_pixel=cfg.render.faces_per_pixel,
    )
    rgb_renderer = build_rgb_renderer(
        cameras=cameras,
        image_size=cfg.render.image_size,
        sigma=cfg.render.sigma,
        faces_per_pixel=cfg.render.faces_per_pixel,
        device=device,
    )

    with torch.no_grad():
        target_sil = sil_renderer(target_mesh.extend(cfg.render.num_views))
        target_rgb = rgb_renderer(target_mesh.extend(cfg.render.num_views))

    source_mesh = ico_sphere(4, device)
    init_color = 0.5 * torch.ones_like(source_mesh.verts_packed())
    source_mesh = source_mesh.update_padded(
        source_mesh.verts_padded()
    )
    source_mesh.textures = TexturesVertex(verts_features=init_color[None])

    deform_verts = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    color_param = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts, color_param], lr=cfg.train.lr)

    run_dir = Path(cfg.work_root) / timestamp_name("textured")
    ensure_dirs(run_dir)
    frame_paths = []
    history = {"total": [], "silhouette": [], "rgb": [], "reg": []}

    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()

        deformed = source_mesh.offset_verts(deform_verts)
        colors = torch.sigmoid(init_color + color_param)[None]
        deformed.textures = TexturesVertex(verts_features=colors)

        pred_sil = sil_renderer(deformed.extend(cfg.render.num_views))
        pred_rgb = rgb_renderer(deformed.extend(cfg.render.num_views))

        sil = silhouette_loss(pred_sil, target_sil)
        rgb = rgb_loss(pred_rgb, target_rgb)
        reg, reg_items = regularization_loss(
            deformed,
            w_lap=cfg.loss.w_lap,
            w_edge=cfg.loss.w_edge,
            w_normal=cfg.loss.w_normal,
        )
        loss = sil + cfg.loss.w_rgb * rgb + reg
        loss.backward()
        optimizer.step()

        history["total"].append(float(loss.item()))
        history["silhouette"].append(float(sil.item()))
        history["rgb"].append(float(rgb.item()))
        history["reg"].append(float(reg.item()))

        if step % cfg.train.save_every == 0 or step == cfg.train.steps:
            image_path = run_dir / "images" / f"rgb_step_{step:04d}.png"
            save_step_image(image_path, pred_rgb[0, ..., :3])
            frame_paths.append(image_path)

        if step % cfg.train.log_every == 0:
            print(
                f"[textured] step={step:04d} total={loss.item():.6f} "
                f"sil={sil.item():.6f} rgb={rgb.item():.6f} "
                f"lap={reg_items['lap']:.6f} edge={reg_items['edge']:.6f} normal={reg_items['normal']:.6f}"
            )

    final_mesh = source_mesh.offset_verts(deform_verts.detach())
    final_mesh.textures = TexturesVertex(verts_features=torch.sigmoid(init_color + color_param).unsqueeze(0).detach())

    save_mesh(run_dir / "meshes" / "textured_final.obj", final_mesh)
    torch.save(torch.sigmoid(init_color + color_param).detach().cpu(), run_dir / "meshes" / "vertex_rgb.pt")
    save_losses(run_dir / "plots" / "textured_losses.png", history)
    save_gif(run_dir / "images" / "textured_turntable.gif", frame_paths)
    save_metrics(
        run_dir / "logs" / "metrics.json",
        {
            "mode": "textured",
            "seed": cfg.train.seed,
            "steps": cfg.train.steps,
            "final_total": history["total"][-1],
            "final_silhouette": history["silhouette"][-1],
            "final_rgb": history["rgb"][-1],
        },
    )
    return run_dir
