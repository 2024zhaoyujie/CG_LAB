"""
离线生成贝塞尔曲线演示 GIF（不依赖 Taichi GGUI / 无需显示窗口）。
光标跟随曲线“笔尖”移动，模拟绘制过程。
用法: python gen_gif.py [--frames 72] [--out bezier_cursor.gif]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W = H = 800
NUM_SEGMENTS = 1000
NUM_CURVE_POINTS = NUM_SEGMENTS + 1

BG = (18, 31, 56)  # 与 main 中 0.07,0.12,0.22 对应
GREEN = (0, 255, 80)
GRAY = (140, 140, 150)
RED = (255, 40, 40)
CURSOR_FILL = (248, 248, 255)
CURSOR_OUTLINE = (40, 40, 55)


def de_casteljau(points, t: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(2, dtype=np.float32)
    pts = [np.array(p[:2], dtype=np.float64) for p in points]
    while len(pts) > 1:
        nxt = []
        for i in range(len(pts) - 1):
            nxt.append((1.0 - t) * pts[i] + t * pts[i + 1])
        pts = nxt
    return pts[0].astype(np.float32)


def sample_bezier(control_points: list) -> np.ndarray:
    out = np.zeros((NUM_CURVE_POINTS, 2), dtype=np.float32)
    pts = [np.array(p, dtype=np.float32) for p in control_points]
    for i in range(NUM_CURVE_POINTS):
        t = i / float(NUM_SEGMENTS)
        out[i] = de_casteljau(pts, t)
    return out


def norm_to_px(u: float, v: float) -> tuple[int, int]:
    x = int(np.clip(u, 0.0, 1.0) * (W - 1))
    y = int((1.0 - np.clip(v, 0.0, 1.0)) * (H - 1))
    return x, y


def draw_arrow_cursor(draw: ImageDraw.ImageDraw, tip_x: int, tip_y: int) -> None:
    """经典箭头鼠标指针，尖端在 (tip_x, tip_y)，整体向右下延伸。"""
    # 相对尖角的局部多边形（与系统默认指针形状类似）
    poly = [
        (tip_x, tip_y),
        (tip_x, tip_y + 16),
        (tip_x + 4, tip_y + 12),
        (tip_x + 10, tip_y + 18),
        (tip_x + 12, tip_y + 16),
        (tip_x + 6, tip_y + 10),
        (tip_x + 14, tip_y + 10),
        (tip_x + 14, tip_y + 2),
    ]
    draw.polygon(poly, fill=CURSOR_FILL, outline=CURSOR_OUTLINE)


def render_frame(
    curve_full: np.ndarray,
    control_points: list,
    progress: float,
) -> Image.Image:
    """progress in [0,1]：已绘制的曲线比例（从起点向终点生长）。"""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # 控制多边形
    if len(control_points) >= 2:
        for i in range(len(control_points) - 1):
            x0, y0 = norm_to_px(*control_points[i][:2])
            x1, y1 = norm_to_px(*control_points[i + 1][:2])
            draw.line((x0, y0, x1, y1), fill=GRAY, width=2)

    # 曲线（按 progress 截断采样点）
    n_draw = max(1, int(np.ceil(progress * (NUM_CURVE_POINTS - 1))) + 1)
    n_draw = min(n_draw, NUM_CURVE_POINTS)
    for i in range(n_draw):
        u, v = float(curve_full[i, 0]), float(curve_full[i, 1])
        x, y = norm_to_px(u, v)
        if 0 <= x < W and 0 <= y < H:
            img.putpixel((x, y), GREEN)

    # 控制点（小圆）
    r = 4
    for p in control_points:
        cx, cy = norm_to_px(float(p[0]), float(p[1]))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=RED, outline=RED)

    # 光标：位于当前已绘制曲线末端（“笔尖”）
    u_tip = float(curve_full[n_draw - 1, 0])
    v_tip = float(curve_full[n_draw - 1, 1])
    tx, ty = norm_to_px(u_tip, v_tip)
    tx = int(np.clip(tx, 2, W - 18))
    ty = int(np.clip(ty, 2, H - 22))
    draw_arrow_cursor(draw, tx, ty)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--out", type=str, default="bezier_cursor.gif")
    args = ap.parse_args()

    # 固定一组有代表性的控制点（与课堂「依次点击」效果类似）
    control_points = [
        [0.12, 0.75],
        [0.35, 0.25],
        [0.65, 0.72],
        [0.88, 0.28],
    ]

    curve_full = sample_bezier(control_points)
    frames = int(args.frames)
    out_path = Path(__file__).resolve().parent / args.out

    imgs = []
    for f in range(frames):
        p = (f + 1) / float(frames)
        im = render_frame(curve_full, control_points, p)
        im.info["disposal"] = 2
        imgs.append(im)

    duration_ms = 50
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
