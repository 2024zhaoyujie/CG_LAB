"""
Taichi 贝塞尔曲线：De Casteljau 采样 + GPU 光栅化 + GGUI 交互
左键添加控制点，C 清空画布。
"""

import numpy as np
import taichi as ti

W = H = 800
NUM_SEGMENTS = 1000
NUM_CURVE_POINTS = NUM_SEGMENTS + 1
MAX_CONTROL_POINTS = 100

# 初始化：优先 GPU，失败则回退
try:
    ti.init(arch=ti.gpu)
except Exception:
    try:
        ti.init(arch=ti.vulkan)
    except Exception:
        ti.init(arch=ti.cpu)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_CURVE_POINTS,))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS,))

MAX_LINE_VERTS = 2 * (MAX_CONTROL_POINTS - 1)
line_vertices = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_LINE_VERTS,))

BG = np.array([0.07, 0.12, 0.22], dtype=np.float32)


def de_casteljau(points, t: float) -> np.ndarray:
    """纯 Python De Casteljau，返回 shape (2,) 的 float32。"""
    if len(points) == 0:
        return np.zeros(2, dtype=np.float32)
    pts = [np.array(p[:2], dtype=np.float64) for p in points]
    while len(pts) > 1:
        nxt = []
        for i in range(len(pts) - 1):
            nxt.append((1.0 - t) * pts[i] + t * pts[i + 1])
        pts = nxt
    return pts[0].astype(np.float32)


def sample_bezier_curve(control_points: list) -> np.ndarray:
    """返回 (NUM_CURVE_POINTS, 2) float32，归一化坐标 [0,1]。"""
    n = len(control_points)
    out = np.zeros((NUM_CURVE_POINTS, 2), dtype=np.float32)
    if n < 2:
        return out
    pts = [np.array(p, dtype=np.float32) for p in control_points]
    for i in range(NUM_CURVE_POINTS):
        t = i / float(NUM_SEGMENTS)
        out[i] = de_casteljau(pts, t)
    return out


@ti.kernel
def clear_screen_kernel():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([BG[0], BG[1], BG[2]])


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        u = curve_points_field[i][0]
        v = curve_points_field[i][1]
        xi = ti.cast(u * ti.cast(W - 1, ti.f32), ti.i32)
        yi = ti.cast(v * ti.cast(H - 1, ti.f32), ti.i32)
        if 0 <= xi < W and 0 <= yi < H:
            pixels[xi, yi] = ti.Vector([0.0, 1.0, 0.0])


def sync_gui_points_pool(control_points: list):
    pool = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    k = len(control_points)
    if k > 0:
        pool[:k] = np.array(control_points, dtype=np.float32)
    gui_points.from_numpy(pool)


def sync_line_vertices(control_points: list):
    n = len(control_points)
    if n < 2:
        lv = np.full((MAX_LINE_VERTS, 2), -10.0, dtype=np.float32)
        line_vertices.from_numpy(lv)
        return
    pairs = []
    for i in range(n - 1):
        pairs.append(control_points[i])
        pairs.append(control_points[i + 1])
    arr = np.array(pairs, dtype=np.float32)
    lv = np.full((MAX_LINE_VERTS, 2), -10.0, dtype=np.float32)
    lv[: len(arr)] = arr
    line_vertices.from_numpy(lv)


def main():
    window = ti.ui.Window("Bezier (De Casteljau)", (W, H), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color(tuple(BG.tolist()))

    control_points: list = []
    prev_lmb = False
    prev_c = False

    while window.running:
        lmb = window.is_pressed(ti.ui.LMB)
        if lmb and not prev_lmb and len(control_points) < MAX_CONTROL_POINTS:
            cx, cy = window.get_cursor_pos()
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            control_points.append([cx, cy])
        prev_lmb = lmb

        c_down = window.is_pressed("c")
        if c_down and not prev_c:
            control_points.clear()
        prev_c = c_down

        clear_screen_kernel()

        if len(control_points) >= 2:
            curve_np = sample_bezier_curve(control_points)
            curve_points_field.from_numpy(curve_np)
            draw_curve_kernel(NUM_CURVE_POINTS)

        sync_gui_points_pool(control_points)
        sync_line_vertices(control_points)

        canvas.set_image(pixels)

        if len(control_points) >= 2:
            canvas.lines(line_vertices, width=0.002, color=(0.55, 0.55, 0.58))

        if len(control_points) > 0:
            canvas.circles(gui_points, radius=0.015, color=(1.0, 0.15, 0.15))

        gui = window.get_gui()
        with gui.sub_window("hint", 0.02, 0.90, 0.75, 0.07):
            gui.text("LMB: 添加控制点   C: 清空", color=(0.85, 0.9, 0.95))

        window.show()


if __name__ == "__main__":
    main()
