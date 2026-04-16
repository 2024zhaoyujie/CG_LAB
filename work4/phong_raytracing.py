"""
Phong 局部光照 + 光线投射（球 + 有限圆锥）+ Taichi UI 滑块
依赖: uv sync  （见 pyproject.toml，需 Python 3.10~3.12，Taichi 尚无 3.14 轮子）
运行: uv run python phong_raytracing.py
"""

import os

import taichi as ti

# 默认 CPU：在 Windows + 含中文路径时，Vulkan 后端常在 canvas.set_image 处无法加载 .spv
# 若项目放在纯英文路径且需 GPU，可在运行前设置环境变量：set TI_ARCH=vulkan 或 TI_ARCH=cuda
def _init_ti():
    name = os.environ.get("TI_ARCH", "cpu").strip().lower()
    if name in ("cuda", "vulkan", "metal", "opengl"):
        ti.init(arch=getattr(ti, name))
    elif name == "cpu":
        ti.init(arch=ti.cpu)
    else:
        ti.init(arch=ti.cpu)


_init_ti()

# ---------- 场景常量（在 @ti.func 内用 ti.Vector 字面量，保证可被编译）----------
WIDTH, HEIGHT = 800, 600
img = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))
params = ti.Vector.field(4, ti.f32, shape=())  # Ka, Kd, Ks, Shininess

# 缩小并略向视口中心靠拢（原：球约 x=-1.2、锥约 x=+1.2）
SPHERE_R = 0.5
SPHERE_CX = -0.45
SPHERE_CY = -0.05
SPHERE_CZ = 0.0

CONE_AX = 0.45
CONE_AY = 0.55
CONE_AZ = 0.0
CONE_BASE_Y = -0.55
CONE_R_BASE = 0.5
CONE_H = CONE_AY - CONE_BASE_Y
CONE_K = CONE_R_BASE / CONE_H


@ti.func
def ray_sphere(o, d, center, rad):
    oc = o - center
    b = 2.0 * oc.dot(d)
    c = oc.dot(oc) - rad * rad
    disc = b * b - 4.0 * c
    t_hit = -1.0
    if disc >= 0.0:
        s = ti.sqrt(disc)
        t0 = (-b - s) * 0.5
        t1 = (-b + s) * 0.5
        if t0 > 1e-4:
            t_hit = t0
        elif t1 > 1e-4:
            t_hit = t1
    return t_hit


@ti.func
def try_cone_t(o, d, t, t_best):
    """若 t 有效且在有限锥段上，更新最小 t"""
    res = t_best
    if t > 1e-4:
        p = o + t * d
        yp = p.y - CONE_AY
        if yp <= 1e-2 and yp >= -CONE_H - 1e-2:
            if res < 0.0 or t < res:
                res = t
    return res


@ti.func
def ray_cone_finite(o, d):
    """有限圆锥：顶点 (CONE_AX,CONE_AY,CONE_AZ)，轴 -Y；底面 y=CONE_BASE_Y"""
    ax, ay, az = CONE_AX, CONE_AY, CONE_AZ
    ox = o.x - ax
    oy = o.y - ay
    oz = o.z - az
    dx, dy, dz = d.x, d.y, d.z

    a = dx * dx + dz * dz - CONE_K * CONE_K * dy * dy
    b = 2.0 * (ox * dx + oz * dz - CONE_K * CONE_K * oy * dy)
    c = ox * ox + oz * oz - CONE_K * CONE_K * oy * oy

    t_best = -1.0
    disc = b * b - 4.0 * a * c
    if ti.abs(a) < 1e-8:
        if ti.abs(b) > 1e-8:
            t0 = -c / b
            t_best = try_cone_t(o, d, t0, t_best)
    elif disc >= 0.0:
        s = ti.sqrt(disc)
        t0 = (-b - s) / (2.0 * a)
        t1 = (-b + s) / (2.0 * a)
        t_best = try_cone_t(o, d, t0, t_best)
        t_best = try_cone_t(o, d, t1, t_best)

    if ti.abs(d.y) > 1e-8:
        t_cap = (CONE_BASE_Y - o.y) / d.y
        if t_cap > 1e-4:
            p = o + t_cap * d
            if (p.x - ax) ** 2 + (p.z - az) ** 2 <= CONE_R_BASE * CONE_R_BASE + 1e-3:
                if t_best < 0.0 or t_cap < t_best:
                    t_best = t_cap

    return t_best


@ti.func
def cone_side_normal(p):
    u = p - ti.Vector([CONE_AX, CONE_AY, CONE_AZ])
    g = ti.Vector(
        [
            2.0 * u.x,
            -2.0 * CONE_K * CONE_K * u.y,
            2.0 * u.z,
        ]
    )
    return g.normalized()


@ti.func
def cone_normal_at(p, hit_cap: ti.i32):
    n_cap = ti.Vector([0.0, -1.0, 0.0])
    n_side = cone_side_normal(p)
    return ti.select(hit_cap != 0, n_cap, n_side)


@ti.func
def phong_color(p, n, base_color, ka, kd, ks, shininess):
    cam = ti.Vector([0.0, 0.0, 5.0])
    light_pos = ti.Vector([2.0, 3.0, 4.0])
    light_color = ti.Vector([1.0, 1.0, 1.0])

    L = (light_pos - p).normalized()
    V = (cam - p).normalized()
    n_use = n
    if n_use.dot(V) < 0.0:
        n_use = -n_use

    amb = ka * light_color * base_color
    ndl = ti.max(0.0, n_use.dot(L))
    diff = kd * ndl * light_color * base_color
    R = (2.0 * n_use.dot(L) * n_use - L).normalized()
    rv = ti.max(0.0, R.dot(V))
    spec = ks * (rv**shininess) * light_color
    col = amb + diff + spec
    # 与 ti.math.clamp 等价，兼容未带 ti.math 的旧版
    return ti.min(ti.max(col, 0.0), 1.0)


@ti.kernel
def render():
    ka = params[None][0]
    kd = params[None][1]
    ks = params[None][2]
    sh = params[None][3]

    sphere_c = ti.Vector([SPHERE_CX, SPHERE_CY, SPHERE_CZ])
    c_sphere = ti.Vector([0.8, 0.1, 0.1])
    c_cone = ti.Vector([0.6, 0.2, 0.8])
    bg = ti.Vector([0.02, 0.12, 0.15])
    cam = ti.Vector([0.0, 0.0, 5.0])

    for i, j in img:
        u = (i + 0.5) / WIDTH - 0.5
        v = (j + 0.5) / HEIGHT - 0.5
        aspect = WIDTH / HEIGHT
        sx = u * 2.0 * aspect
        sy = v * 2.0
        target = ti.Vector([sx, sy, 0.0])
        d = (target - cam).normalized()
        o = cam

        t_s = ray_sphere(o, d, sphere_c, SPHERE_R)
        t_c = ray_cone_finite(o, d)

        t_min = -1.0
        use_sphere = 0
        if t_s > 0.0 and t_c > 0.0:
            if t_s < t_c:
                t_min = t_s
                use_sphere = 1
            else:
                t_min = t_c
                use_sphere = 0
        elif t_s > 0.0:
            t_min = t_s
            use_sphere = 1
        elif t_c > 0.0:
            t_min = t_c
            use_sphere = 0

        if t_min < 0.0:
            img[i, j] = bg
        else:
            p = o + t_min * d
            col = ti.Vector([0.0, 0.0, 0.0])
            if use_sphere == 1:
                n = (p - sphere_c).normalized()
                col = phong_color(p, n, c_sphere, ka, kd, ks, sh)
            else:
                hit_cap = 0
                if ti.abs(p.y - CONE_BASE_Y) < 2e-2:
                    hit_cap = 1
                n = cone_normal_at(p, hit_cap)
                col = phong_color(p, n, c_cone, ka, kd, ks, sh)
            img[i, j] = col


def main_taichi_ui():
    """ti.ui 窗口（需 Vulkan；项目路径含中文时 Windows 上可能无法加载 GGUI 着色器）。"""
    params[None] = [0.2, 0.7, 0.5, 32.0]

    window = ti.ui.Window("Phong Ray Tracing (Taichi)", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()

    while window.running:
        render()
        canvas.set_image(img)

        with window.get_gui() as gui:
            p = params[None]
            ka = gui.slider_float("Ka (ambient)", p[0], 0.0, 1.0)
            kd = gui.slider_float("Kd (diffuse)", p[1], 0.0, 1.0)
            ks = gui.slider_float("Ks (specular)", p[2], 0.0, 1.0)
            shin = gui.slider_float("Shininess", p[3], 1.0, 128.0)
            params[None] = [ka, kd, ks, shin]

        window.show()


def _cwd_has_non_ascii() -> bool:
    try:
        return not all(ord(c) < 128 for c in os.getcwd())
    except Exception:
        return True


def main_matplotlib_ui():
    """备用：Matplotlib + 滑块，避免 GGUI/Vulkan 在非 ASCII 路径下的 spv 加载失败。"""
    import matplotlib

    matplotlib.use("TkAgg")
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Slider

    params[None] = [0.2, 0.7, 0.5, 32.0]
    render()
    rgb = np.transpose(img.to_numpy(), (1, 0, 2))

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phong_preview.png")
    try:
        plt.imsave(out_png, np.clip(rgb, 0.0, 1.0))
        print("已保存预览图:", out_png)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.28)
    im_artist = ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title("Phong Ray Tracing (Taichi CPU + Matplotlib UI)")
    ax.axis("off")

    ax_ka = fig.add_axes([0.15, 0.20, 0.70, 0.03])
    ax_kd = fig.add_axes([0.15, 0.15, 0.70, 0.03])
    ax_ks = fig.add_axes([0.15, 0.10, 0.70, 0.03])
    ax_sh = fig.add_axes([0.15, 0.05, 0.70, 0.03])

    s_ka = Slider(ax_ka, "Ka", 0.0, 1.0, valinit=params[None][0])
    s_kd = Slider(ax_kd, "Kd", 0.0, 1.0, valinit=params[None][1])
    s_ks = Slider(ax_ks, "Ks", 0.0, 1.0, valinit=params[None][2])
    s_sh = Slider(ax_sh, "Shininess", 1.0, 128.0, valinit=params[None][3])

    def on_change(_):
        params[None] = [s_ka.val, s_kd.val, s_ks.val, s_sh.val]
        render()
        im_artist.set_data(np.transpose(img.to_numpy(), (1, 0, 2)))
        fig.canvas.draw_idle()

    s_ka.on_changed(on_change)
    s_kd.on_changed(on_change)
    s_ks.on_changed(on_change)
    s_sh.on_changed(on_change)

    try:
        mgr = fig.canvas.manager
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("Phong Lab — 拖动下方滑块实时更新")
    except Exception:
        pass

    print("已打开 Matplotlib 窗口（若看不到，请检查任务栏是否被最小化）。")
    plt.show(block=True)


def main():
    # 中文路径下 Taichi GGUI 几乎必挂，直接走 Matplotlib，避免先弹空窗
    if os.environ.get("PHONG_FORCE_TAICHI_UI", "").strip() != "1" and _cwd_has_non_ascii():
        print(
            "当前工程路径含非 ASCII 字符，已直接使用 Matplotlib(TkAgg) 界面（含四个滑块）。"
        )
        main_matplotlib_ui()
        return

    try:
        main_taichi_ui()
    except Exception as e:
        err = str(e)
        if "spv" in err or "set_image" in err or "failed to open" in err or "Vulkan" in err:
            print(
                "提示: Taichi GGUI(Vulkan) 无法加载着色器，已改用 Matplotlib 滑块界面。"
            )
            main_matplotlib_ui()
        else:
            raise


if __name__ == "__main__":
    main()
