# 计算机图形学实验报告：Phong 局部光照与光线投射（Taichi）

**课程主页：** <https://zhanghongwen.cn/cg>  
**实验名称：** Phong 光照模型（环境光 / 漫反射 / 镜面高光）与光线投射  
**实现语言：** Python 3 + Taichi  
**工程文件：** `phong_raytracing.py`、`pyproject.toml`、`run_phong.bat`  

**说明：** 为避免部分浏览器或 GitHub 客户端在「数学公式」模式下把内容拆成单字换行，本报告**不使用** `$` 与 `$$` 定界符；公式以等宽代码块与 Unicode 符号书写，在 GitHub 网页上可稳定阅读。

---

## 一、实验目的

1. **理论：** 理解 Phong 模型中环境光、Lambert 漫反射与镜面高光三项的含义，以及三者叠加得到表面颜色的思路。  
2. **数学：** 掌握向量归一化、点积、反射方向与法向量；理解最近正交点与遮挡关系。  
3. **工程：** 用 Taichi 编写逐像素 Kernel，完成射线与隐式几何求交、深度竞争与 Phong 着色；通过滑块调节 **Ka、Kd、Ks** 与高光指数 **n**，观察画面变化。

---

## 二、实验原理（纯文本公式）

总光强为三项之和：

```text
I = I_ambient + I_diffuse + I_specular
```

**环境光（Ambient）** — 近似均匀背景光，对 RGB 各分量分别缩放：

```text
I_ambient = Ka · C_light · C_object
```

（上式中 **·** 表示三个颜色向量按分量相乘。）

**漫反射（Lambert）** — 与法向 **N** 和指向光的方向 **L** 的点积成正比，背光一侧截断为 0：

```text
I_diffuse = Kd · max(0, N·L) · C_light · C_object
```

**镜面高光（Phong）**：

```text
I_specular = Ks · max(0, R·V)^n · C_light
```

其中 **N、L、V** 均为单位向量；**L** 从表面指向光源，**V** 从表面指向相机。理想反射方向：

```text
R = normalize( 2·(N·L)·N − L )
```

**n** 为 Shininess，越大高光越集中。实现中对 RGB 做 **clamp 到 [0,1]**，并对 **N·L**、**R·V** 做非负截断；若 **N·V < 0** 则翻转法线，减轻背面异常。

---

## 三、实验环境与依赖

| 项目 | 说明 |
|------|------|
| Python | 3.10～3.12（见 `pyproject.toml`） |
| 包管理 | `uv`（`uv sync` / `uv run`） |
| 核心库 | `taichi`、`matplotlib`（备用 UI） |
| Taichi 后端 | 默认 CPU；纯英文路径可设 `TI_ARCH=vulkan` 或 `TI_ARCH=cuda` |

---

## 四、场景与几何参数（与代码一致）

本实验不加载外部模型，在 Kernel 内用隐式曲面求交。

| 对象 | 说明 |
|------|------|
| 相机 | 位置 (0, 0, 5) |
| 点光源 | 位置 (2, 3, 4)，白色 (1, 1, 1) |
| 背景 | RGB 约 (0.02, 0.12, 0.15) |
| 红球 | 半径 0.5，球心约 (−0.45, −0.05, 0)，常量 `SPHERE_*` |
| 紫圆锥 | 顶点约 (0.45, 0.55, 0)；底面 y = −0.55（`CONE_BASE_Y`），底半径 0.5（`CONE_R_BASE`） |
| 锥形比 | `CONE_K = CONE_R_BASE / CONE_H`，侧壁为二次方程并在高度上裁剪 |

球与锥在 x 轴两侧大致对称，略向视口中心收拢。

---

## 五、算法与实现要点

### 5.1 光线投射

对每个像素计算视平面上一点 `target`（z = 0），相机为 **o** = (0,0,5)，射线方向：

```text
d = normalize(target − o)
```

### 5.2 求交

- **球：** 由 **‖o + t·d − C‖² = R²** 得一元二次方程，取最小正根 **t**。  
- **有限圆锥：** 在顶点局部系下写侧壁二次型，系数由 `CONE_K` 与高度约束确定；用 `try_cone_t` 限制 **t** 在顶点到底面之间；底面与 **y = CONE_BASE_Y**（−0.55）求交，再判断是否在底圆内。

### 5.3 深度竞争

分别得到球的 **t_s** 与锥的 **t_c**，取较小正距离作为可见交点，等价于光线上的最近表面（Z-buffer 思想）。

### 5.4 法向量与着色

- 球面法线：**normalize(P − C)**。  
- 锥侧壁：对隐式函数梯度归一化；底面法线 **(0, −1, 0)**。  
- 在 `phong_color` 中合成环境、漫反射、镜面三项，并 **min(max(col,0),1)** 输出。

---

## 六、交互界面与参数

| 滑块 | 范围 | 默认值 | 含义 |
|------|------|--------|------|
| Ka | 0.0～1.0 | 0.2 | 环境光系数 |
| Kd | 0.0～1.0 | 0.7 | 漫反射系数 |
| Ks | 0.0～1.0 | 0.5 | 镜面系数 |
| Shininess | 1.0～128.0 | 32.0 | 高光指数 n |

参数写入 Taichi 字段 `params`，每帧 `render()` 读取。

### 6.1 双前端（Windows）

工作目录含**中文等非 ASCII 路径**时，Taichi GGUI（Vulkan）常无法加载 `.spv` 或 `set_image` 失败。默认走 **Matplotlib（TkAgg）+ Slider**，并保存 `phong_preview.png`。纯英文路径可尝试 `ti.ui`，或设置 **`PHONG_FORCE_TAICHI_UI=1`** 强制 Taichi 窗口。

---

## 七、运行方式

在 `work4` 目录下执行：

```bash
uv sync
uv run python phong_raytracing.py
```

或在 Windows 下双击 **`run_phong.bat`**。演示动图见 [`demo.gif`](demo.gif)。

---

## 八、现象简述

- **Ka** 增大：整体变亮，暗部抬升。  
- **Kd** 增大：漫反射对比增强（随 **N·L** 变化）。  
- **Ks** 增大：高光更亮。  
- **Shininess** 增大：高光更尖、更窄。

---

## 九、总结

本实验在 Taichi 上实现了射线生成、球与有限圆锥求交、最近表面选择与 Phong 着色，并针对中文路径提供 Matplotlib 备用 UI，保证可复现。可在同一框架上扩展 Blinn-Phong 或阴影等。

---

## 附录：文件列表

| 文件 | 作用 |
|------|------|
| `phong_raytracing.py` | 主程序 |
| `pyproject.toml` | 依赖与 Python 版本 |
| `run_phong.bat` | Windows 启动脚本 |
| `demo.gif` | 演示（可选） |
| `phong_preview.png` | Matplotlib 模式下的静态预览 |
