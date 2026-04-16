# 计算机图形学实验报告：Phong 局部光照与光线投射（Taichi）

**课程主页：** [https://zhanghongwen.cn/cg](https://zhanghongwen.cn/cg)  
**实验名称：** Phong 光照模型（环境光 / 漫反射 / 镜面高光）与光线投射  
**实现语言：** Python 3 + Taichi  
**工程文件：** `phong_raytracing.py`、`pyproject.toml`、`run_phong.bat`  

> **公式排版：** 本文在 GitHub 上使用 `$…$`（行内）与单独成段的 `$$…$$`（块公式）。行内公式与中文括号之间保留空格，避免渲染歧义。

---

## 一、实验目的

1. **理论：** 理解 Phong 局部光照模型中环境光、Lambert 漫反射与 Phong 镜面项的含义，以及三者线性叠加得到表面颜色的思路。  
2. **数学：** 熟练运用向量归一化、点积、反射方向与法向量；理解「最近正交点」与遮挡关系。  
3. **工程：** 使用 Taichi 编写逐像素 Kernel，完成射线与隐式几何求交、深度竞争与 Phong 着色；通过滑块实时调节 $K_a$、$K_d$、$K_s$ 与高光指数 $n$，观察画面变化。

---

## 二、实验原理

总光强为三项之和：

$$
I = I_{ambient} + I_{diffuse} + I_{specular}
$$

**环境光：**

$$
I_{ambient} = K_a \cdot C_{light} \odot C_{object}
$$

**漫反射（Lambert）：**

$$
I_{diffuse} = K_d \cdot \max(0,\,\mathbf{N}\cdot\mathbf{L}) \cdot C_{light} \odot C_{object}
$$

**镜面高光（Phong）：**

$$
I_{specular} = K_s \cdot \max(0,\,\mathbf{R}\cdot\mathbf{V})^{n} \cdot C_{light}
$$

其中 $\mathbf{N}$、$\mathbf{L}$、$\mathbf{V}$ 均为单位向量；$\mathbf{L}$ 由表面指向光源，$\mathbf{V}$ 由表面指向相机；理想反射方向取

$$
\mathbf{R} = \mathrm{normalize}\bigl(2(\mathbf{N}\cdot\mathbf{L})\mathbf{N}-\mathbf{L}\bigr)
$$

$n$ 为 Shininess，越大高光越集中。实现中对 RGB 做分量 clamp 到 $[0,1]$，并对 $\mathbf{N}\cdot\mathbf{L}$、$\mathbf{R}\cdot\mathbf{V}$ 做非负截断；若 $\mathbf{N}\cdot\mathbf{V}<0$ 则将法线翻向观察者，减轻背面异常。

---

## 三、实验环境与依赖

| 项目 | 说明 |
|------|------|
| Python | **3.10～3.12**（`pyproject.toml` 约束；Taichi 暂无稳定 3.14 轮子） |
| 包管理 | `uv`（`uv sync` / `uv run`） |
| 核心库 | `taichi`、`matplotlib`（后者为备用 UI） |
| Taichi 架构 | 默认 **CPU**；纯英文路径下可设环境变量 `TI_ARCH=vulkan` 或 `TI_ARCH=cuda` 尝试 GPU |

---

## 四、场景与几何参数（与代码一致）

本实验**不加载外部模型**，在 Kernel 内用隐式曲面求交。

| 对象 | 参数 | 数值 / 常量名 |
|------|------|----------------|
| 相机 | 位置 | $(0,0,5)$ |
| 点光源 | 位置 / 颜色 | $(2,3,4)$，白色 $(1,1,1)$ |
| 背景色 | RGB | $(0.02,0.12,0.15)$ |
| 红球 | 半径 / 球心 | `SPHERE_R=0.5`，圆心 $(-0.45,-0.05,0)$ |
| 紫圆锥 | 顶点 | $(0.45,0.55,0)$ |
| 紫圆锥 | 底面 | 平面 $y=-0.55$（`CONE_BASE_Y`），底半径 `CONE_R_BASE=0.5` |
| 圆锥形状 | 斜率 | `CONE_K = CONE_R_BASE / CONE_H`，侧壁用二次方程 + 高度裁剪 |

球与锥在 $x$ 上大致对称并略向视口中心收拢，便于同屏观察遮挡与高光。

---

## 五、算法与实现要点

### 5.1 光线投射

对每个像素 $(i,j)$，将像素映射到归一化设备坐标，再映射到 $z=0$ 视平面上的点 `target`，射线为

$$
\mathbf{o}=(0,0,5),\quad \mathbf{d}=\mathrm{normalize}(\mathbf{target}-\mathbf{o})
$$

与课程常见的「过像素射向成像平面」一致。

### 5.2 求交

- **球：** 将 $\lVert \mathbf{o}+t\mathbf{d}-\mathbf{C}\rVert^2=R^2$ 化为关于 $t$ 的一元二次方程，取**最小正根**作为入射前表面交点。  
- **有限圆锥：** 以顶点为原点、轴沿 $-y$，将圆锥写为 $x'^2+z'^2=k^2y'^2$ 形式（$k=\tan\theta$ 由底半径与锥高导出），对射线解二次方程；用 `try_cone_t` 将 $t$ 限制在顶点到底面之间；**底面圆盘**与 $y=\texttt{CONE\_BASE\_Y}$ 求交并检验落在底圆内。

### 5.3 深度竞争

分别得到球的 $t_s$ 与锥的 $t_c$ 后，取**较小正距离**对应表面着色，等价于光线意义下的最近表面 / Z-buffer 思想，保证遮挡正确。

### 5.4 法向量与 Phong

- 球面：$\mathbf{N}=\mathrm{normalize}(\mathbf{P}-\mathbf{C})$。  
- 锥面：对 $F=x'^2+z'^2-k^2y'^2$ 求梯度并归一化；底面用法向量 $(0,-1,0)$。  
- 在 `phong_color` 中计算 $\mathbf{L}$、$\mathbf{V}$、$\mathbf{R}$ 与三项光照，最后 `min(max(col,0),1)` 输出。

---

## 六、交互界面与参数

| 滑块 | 范围 | 默认值 | 含义 |
|------|------|--------|------|
| Ka | $[0,1]$ | $0.2$ | 环境光系数 |
| Kd | $[0,1]$ | $0.7$ | 漫反射系数 |
| Ks | $[0,1]$ | $0.5$ | 镜面系数 |
| Shininess | $[1,128]$ | $32$ | 高光指数 $n$ |

参数存放在 Taichi 标量场 `params` 中，由 UI 回写后每帧调用 `render()`。

### 6.1 双前端策略（Windows 实践）

在 **工作目录含中文等非 ASCII 路径** 时，Taichi **GGUI（Vulkan）** 常出现 `.spv` 着色器加载或 `set_image` 失败。本工程默认策略为：

- **路径含非 ASCII：** 直接使用 **Matplotlib（TkAgg）+ Slider**，避免先弹不可用窗口；启动后保存一帧 `phong_preview.png` 便于核对渲染。  
- **路径为纯英文：** 可走 `ti.ui`；也可通过环境变量 **`PHONG_FORCE_TAICHI_UI=1`** 强制尝试 Taichi 窗口。  
- 默认 Taichi 后端为 **CPU**（`TI_ARCH` 未设置时），利于在受限环境下稳定跑通实验。

---

## 七、编译与运行

在 `work4` 目录下：

```bash
uv sync
uv run python phong_raytracing.py
```

或在 Windows 下双击 **`run_phong.bat`**。

**演示动图：** 仓库内同目录的 [`demo.gif`](demo.gif)（若本地另有录屏，可替换后更新本说明中的文件名）。

---

## 八、现象与分析（简要）

- **增大 Ka：** 整体底光抬升，暗部仍可见细节，画面趋「平」。  
- **增大 Kd：** 漫反射增强，明暗随 $\mathbf{N}\cdot\mathbf{L}$ 变化更明显。  
- **增大 Ks：** 镜面项变强，高光更亮。  
- **增大 Shininess：** 高光更尖锐、范围更窄。

---

## 九、总结

本实验在 Taichi 上实现了「射线生成 → 球与有限圆锥求交 → 最近表面选择 → Phong 三项叠加 → 交互调参」的完整链路，并针对 **中文路径 + Windows** 下的 GGUI 限制给出了 **Matplotlib 备用 UI**，保证实验可复现、可展示。后续可在同一框架上扩展 Blinn-Phong 或阴影射线等选做内容。

---

## 附录：文件说明

| 文件 | 作用 |
|------|------|
| `phong_raytracing.py` | 主程序：求交、着色、UI 分支 |
| `pyproject.toml` | 依赖与 Python 版本约束 |
| `run_phong.bat` | Windows 一键运行 |
| `demo.gif` | 效果演示（可选） |
| `phong_preview.png` | 首次 Matplotlib 启动时自动保存的静态预览 |
