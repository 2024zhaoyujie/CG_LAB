"""
可微渲染实验：射线投射球体 + Leaky Lambertian + Adam 优化光源位置
依赖: pip install taichi

运行（CPU，适合含中文路径的 Windows）:
  python differentiable_raycasting_adam.py

指定 GPU/SPIR-V（需英文路径等环境时可试）:
  set TI_ARCH=cuda&& python differentiable_raycasting_adam.py

选做① 联合优化光源 + 物体颜色:
  python differentiable_raycasting_adam.py --joint-color

选做② Blinn-Phong + 可微 shininess（漫反射仍为 Leaky Lambert）:
  python differentiable_raycasting_adam.py --blinn-phong

选做①+②一并开启（推荐写进实验报告演示）:
  python differentiable_raycasting_adam.py --extras

说明：含 Blinn-Phong 时 shininess 从错误初值爬升到目标（如 32）较慢，默认 ITERS=1400；
若未贴近可再加 `--iters 2200`。

快速试跑（步数）:
  python differentiable_raycasting_adam.py --no-gui --iters 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import taichi as ti
from PIL import Image


def init_ti():
    name = os.environ.get("TI_ARCH", "cpu").strip().lower()
    if name in ("cuda", "vulkan", "metal", "opengl"):
        ti.init(arch=getattr(ti, name))
    else:
        ti.init(arch=ti.cpu)


init_ti()

# -------------------- 场景与渲染参数 --------------------
W, H = 384, 384
SPHERE_C = ti.Vector([0.5, 0.5, 0.5])
SPHERE_R = 0.3
LEAK_ALPHA = 0.1

TARGET_LIGHT = (0.8, 0.8, 0.2)
INIT_LIGHT = (0.2, 0.2, 0.8)

# 平行光视图：原点侧向 +Z 投射，成像平面位于 z=0（与球心间视野匹配）
CAM_Z = -1.25
RAY_DIR = ti.Vector([0.0, 0.0, 1.0])

# Adam
LR = 0.03
# Blinn–Phong 的 shininess 进入指数，灵敏度低，单独放大步长
LR_SHININESSSCALE = 10.0
ITERS = 1400
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8

# 选做①：仅标量 kd 会令 RGB 有多种等价解；加一点对 (0.85,0.85,0.85) 的正则迫使三色一致收敛
ALBEDO_REG = 8e-2

@ti.func
def ray_sphere_intersect(o, d, center, rad):
    oc = o - center
    b = 2.0 * oc.dot(d)
    c_oc = oc.dot(oc) - rad * rad
    disc = b * b - 4.0 * c_oc
    t_hit = -1.0
    if disc >= 0.0:
        sd = ti.sqrt(disc)
        t0 = (-b - sd) * 0.5
        t1 = (-b + sd) * 0.5
        if t0 > 1e-5:
            t_hit = t0
        elif t1 > 1e-5:
            t_hit = t1
    return t_hit


@ti.func
def leaky_lambertian(ndotl, alpha):
    return ti.max(alpha * ndotl, ndotl)


@ti.func
def shade_pixel(u, v, light_pos, surf_color, shininess_in, use_phong):
    ray_o = ti.Vector([u, v, CAM_Z])
    t = ray_sphere_intersect(ray_o, RAY_DIR, SPHERE_C, SPHERE_R)
    out = ti.cast(0.0, ti.f32)
    if t > 0.0:
        hit = ray_o + t * RAY_DIR
        n = ti.math.normalize(hit - SPHERE_C)
        # 法线朝相机
        if n.dot(-RAY_DIR) < 0.0:
            n = -n
        lv = light_pos - hit
        l = ti.math.normalize(lv)

        nl = ti.math.dot(n, l)
        # 标量渲染：用三通道均值，使 joint-color 时 RGB 均可收到梯度
        kd = (surf_color.x + surf_color.y + surf_color.z) * (1.0 / 3.0)
        if use_phong != 0:  # Blinn–Phong：漫反射 kd·LeakyLambert；高光 ks·n_h^shiny，并用 nl 泄漏门控以保持背光梯度
            vdir = ti.math.normalize(-RAY_DIR)
            h = ti.math.normalize(l + vdir)
            nh = ti.math.max(ti.math.dot(n, h), 0.0)
            diffuse = kd * leaky_lambertian(nl, LEAK_ALPHA)
            nl_gate = leaky_lambertian(nl, LEAK_ALPHA)
            spec = ti.cast(0.28, ti.f32) * ti.pow(nh + 1e-4, shininess_in) * nl_gate
            out = diffuse + spec
        else:
            shade = leaky_lambertian(nl, LEAK_ALPHA)
            out = kd * shade
    return out


surf_albedo = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
shininess_field = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


def run_experiment(
    *,
    joint_color: bool,
    blinn_phong: bool,
    gui: bool,
    save_gt: bool = True,
    gt_path: str | None = None,
):
    light_pos = ti.Vector.field(3, ti.f32, shape=(), needs_grad=True)
    fixed_light = ti.Vector.field(3, ti.f32, shape=())

    img = ti.field(ti.f32, shape=(W, H), needs_grad=True)
    target_img = ti.field(ti.f32, shape=(W, H))
    loss = ti.field(ti.f32, (), needs_grad=True)

    use_phong_flag = ti.field(ti.i32, ())
    joint_active = ti.field(ti.i32, ())

    @ti.kernel
    def forward_render_and_loss():
        for i, j in img:
            inv_n = ti.static(1.0 / float(W * H))
            u = (i + 0.5) / W
            v = (j + 0.5) / H
            lp = ti.Vector([light_pos[None].x, light_pos[None].y, light_pos[None].z])
            sh = shininess_field[None]
            ph = use_phong_flag[None]
            c = surf_albedo[None]
            shade = shade_pixel(u, v, lp, c, sh, ph)
            img[i, j] = shade
            d = shade - target_img[i, j]
            pix = inv_n * (d * d)
            if joint_active[None] != 0:
                # 单色 GT 下设为对灰目标 RGB 对齐（仅在一次循环里累加正则，等价于 λ‖a−tgt‖²）
                if i == 0 and j == 0:
                    a = surf_albedo[None]
                    g = ti.cast(0.85, ti.f32)
                    dr = a.x - g
                    dg = a.y - g
                    db = a.z - g
                    pix += ALBEDO_REG * (dr * dr + dg * dg + db * db) / 3.0
            ti.atomic_add(loss[None], pix)

    @ti.kernel
    def render_to_img():
        for i, j in img:
            u = (i + 0.5) / W
            v = (j + 0.5) / H
            lp = ti.Vector([light_pos[None].x, light_pos[None].y, light_pos[None].z])
            sh = shininess_field[None]
            ph = use_phong_flag[None]
            c = surf_albedo[None]
            img[i, j] = shade_pixel(u, v, lp, c, sh, ph)

    @ti.kernel
    def copy_fixed_light_to_target():
        for i, j in target_img:
            u = (i + 0.5) / W
            v = (j + 0.5) / H
            fl = fixed_light[None]
            lp = ti.Vector([fl.x, fl.y, fl.z])
            sh = shininess_field[None]
            ph = use_phong_flag[None]
            c = surf_albedo[None]
            target_img[i, j] = shade_pixel(u, v, lp, c, sh, ph)

    # --------- 纹理 / 常量初始化 ---------
    use_phong_flag[None] = 1 if blinn_phong else 0
    joint_active[None] = 1 if joint_color else 0
    shininess_field[None] = 32.0

    # 灰白物体色（GT）；若启用 --joint-color 会在 GT 之后改为错误初始色
    surf_albedo[None] = (0.85, 0.85, 0.85)
    surf_albedo.grad.fill(0)

    light_pos.grad.fill(0)
    img.grad.fill(0)

    # --------- Ground Truth ----------
    lx, ly, lz = TARGET_LIGHT
    fixed_light[None] = (lx, ly, lz)
    copy_fixed_light_to_target()
    if save_gt:
        arr = target_img.to_numpy()
        root = os.path.dirname(os.path.abspath(__file__))
        out_p = gt_path if os.path.isabs(gt_path) else os.path.join(root, gt_path)
        # field[i,j]: i→水平 u, j→垂直 v → 存盘用 (height, width)
        gsave = np.clip(np.asarray(arr, dtype=np.float32).T, 0.0, 1.0)
        Image.fromarray((gsave * 255.0 + 0.5).astype(np.uint8), mode="L").save(out_p)

    # --------- 可优化变量初值 ---------
    ix, iy, iz = INIT_LIGHT
    light_pos[None] = (ix, iy, iz)

    if joint_color:
        surf_albedo[None] = (0.35, 0.2, 0.95)  # 错误初始色，向 GT（灰）与目标光收敛

    if blinn_phong:
        shininess_field[None] = 4.0  # 错误初始 shininess（GT 在上方以 32 生成）

    m_l = np.zeros(3, dtype=np.float64)
    v_l = np.zeros(3, dtype=np.float64)
    m_a = np.zeros(3, dtype=np.float64)
    v_a = np.zeros(3, dtype=np.float64)
    m_s = 0.0
    v_s = 0.0

    gui_handle = None
    if gui:
        gui_handle = ti.GUI("Target (左) vs Current (右) — Differentiable Ray Casting", res=(W * 2, H))

    for step in range(1, ITERS + 1):
        # 梯度清零
        img.grad.fill(0)
        light_pos.grad.fill(0)
        if joint_color:
            surf_albedo.grad.fill(0)
        if blinn_phong:
            shininess_field.grad.fill(0)
        loss.grad.fill(0)

        loss[None] = 0.0
        with ti.ad.Tape(loss):
            forward_render_and_loss()

        # --------- Adam ----------
        lg = light_pos.grad[None]
        g_l = np.array([float(lg.x), float(lg.y), float(lg.z)], dtype=np.float64)
        m_l = BETA1 * m_l + (1.0 - BETA1) * g_l
        v_l = BETA2 * v_l + (1.0 - BETA2) * (g_l**2)
        hat_m = m_l / (1.0 - BETA1**step)
        hat_v = v_l / (1.0 - BETA2**step)
        upd_l = LR * hat_m / (np.sqrt(hat_v) + EPS)
        lp = light_pos[None]
        lp_np = np.array([float(lp.x), float(lp.y), float(lp.z)], dtype=np.float64) - upd_l
        lp_np = np.clip(lp_np, 0.001, 0.999)
        light_pos[None] = tuple(lp_np.astype(np.float32))

        if joint_color:
            ag = surf_albedo.grad[None]
            g_a = np.array([float(ag.x), float(ag.y), float(ag.z)], dtype=np.float64)
            m_a = BETA1 * m_a + (1.0 - BETA1) * g_a
            v_a = BETA2 * v_a + (1.0 - BETA2) * (g_a**2)
            hat_ma = m_a / (1.0 - BETA1**step)
            hat_va = v_a / (1.0 - BETA2**step)
            upd_a = LR * hat_ma / (np.sqrt(hat_va) + EPS)
            sa = surf_albedo[None]
            albedo_np = np.array([float(sa.x), float(sa.y), float(sa.z)], dtype=np.float64) - upd_a
            albedo_np = np.clip(albedo_np, 0.03, 0.999)
            surf_albedo[None] = tuple(albedo_np.astype(np.float32))

        if blinn_phong:
            g_s = float(shininess_field.grad[None])
            m_s = BETA1 * m_s + (1.0 - BETA1) * g_s
            v_s = BETA2 * v_s + (1.0 - BETA2) * (g_s**2)
            hat_ms = m_s / (1.0 - BETA1**step)
            hat_vs = v_s / (1.0 - BETA2**step)
            s_cur = float(shininess_field[None])
            s_new = s_cur - (LR * LR_SHININESSSCALE) * hat_ms / (np.sqrt(hat_vs) + EPS)
            shininess_field[None] = float(np.clip(s_new, 1.01, 128.0))

        if gui_handle:
            render_to_img()
            left = np.clip(target_img.to_numpy(), 0.0, 1.0)
            cur = np.clip(img.to_numpy(), 0.0, 1.0)
            # ti.GUI(res=(宽,高)) 与 set_image 的 numpy 形如 (宽度, 高度) 一致
            duo = np.zeros((W * 2, H), dtype=np.float32)
            duo[:W, :] = left
            duo[W:, :] = cur
            gui_handle.set_image(duo)
            if not gui_handle.running:
                print("[INFO] GUI 未运行，退出优化循环。")
                break
            try:
                gui_handle.show()
            except RuntimeError as e:
                # 关闭窗口时 Taichi 会抛 RuntimeError（非程序错误）
                if "close" in str(e).lower() or "Window" in str(e):
                    print("[INFO] 已关闭窗口，提前结束优化。")
                    break
                raise

        if step == 1 or step % 50 == 0 or step == ITERS:
            lf = loss[None]
            lx, ly, lz = float(light_pos[None].x), float(light_pos[None].y), float(light_pos[None].z)
            msg = f"step {step:4d}  loss={lf:.6e}  light=({lx:.4f},{ly:.4f},{lz:.4f})"
            if joint_color:
                ab = surf_albedo[None]
                msg += f"  albedo=({float(ab.x):.3f},{float(ab.y):.3f},{float(ab.z):.3f})"
            if blinn_phong:
                msg += f"  shininess={float(shininess_field[None]):.3f}"
            print(msg)

    if gui_handle:
        try:
            gui_handle.close()
        except RuntimeError:
            pass


def main():
    global ITERS

    p = argparse.ArgumentParser()
    p.add_argument("--joint-color", action="store_true", help="选做①：联合优化光源 + 漫反射颜色")
    p.add_argument("--blinn-phong", action="store_true", help="选做②：Blinn-Phong，shininess 可微")
    p.add_argument(
        "--extras",
        action="store_true",
        help="同时启用选做①②（等价于联合指定 --joint-color 与 --blinn-phong）",
    )
    p.add_argument("--no-gui", action="store_true", help="无窗口，仅训练与日志")
    p.add_argument(
        "--gt-path",
        default=None,
        help="保存 GT PNG 的路径；不设时按模式自动命名（lambert / phong / optional）",
    )
    p.add_argument("--iters", type=int, default=0, metavar="N", help="覆盖默认训练步数 ITERS（0 表示用全局默认）")
    args = p.parse_args()
    if args.iters > 0:
        ITERS = args.iters

    joint = args.joint_color or args.extras
    phong = args.blinn_phong or args.extras

    if args.gt_path is not None:
        gt_out = args.gt_path
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        if joint and phong:
            gt_fname = "target_gt_optional_full.png"
        elif phong:
            gt_fname = "target_gt_blinn_phong.png"
        elif joint:
            gt_fname = "target_gt_joint_color.png"
        else:
            gt_fname = "target_render.png"
        gt_out = os.path.join(base, gt_fname)

    run_experiment(
        joint_color=joint,
        blinn_phong=phong,
        gui=not args.no_gui,
        save_gt=True,
        gt_path=gt_out,
    )


if __name__ == "__main__":
    main()
