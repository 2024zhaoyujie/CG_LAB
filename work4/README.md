# 实验报告：Phong 局部光照模型与光线投射（Taichi）

## 1. 实验信息

- **实验主题**：Phong 光照模型（Ambient + Diffuse + Specular）与光线投射（Ray Casting）
- **实现方式**：Taichi Kernel 逐像素发射射线，隐式定义几何体（球体 + 有限圆锥），最近交点深度竞争（Z-buffer 逻辑），Phong 着色；UI 通过滑块实时调参
- **代码文件**：`phong_raytracing.py`
- **依赖管理**：`uv` + `pyproject.toml`
- **运行演示**：`C:\CG-Lab\屏幕录制 2026-04-16 182653.gif`

---

## 2. 实验目标

- **理论理解**：理解环境光（Ambient）、漫反射（Diffuse）、镜面高光（Specular）的物理含义与计算方式。
- **数学基础**：掌握三维向量运算：归一化、点积、反射向量计算，法向量的求取与方向处理。
- **工程实践**：用 Taichi 实现交互式渲染，实时修改材质参数，观察各项参数对视觉效果的影响。

---

## 3. 实验原理

Phong 光照模型将反射分为三部分并相加：

\[
I = I_{ambient} + I_{diffuse} + I_{specular}
\]

### 3.1 环境光（Ambient）

\[
I_{ambient} = K_a \cdot C_{light} \cdot C_{object}
\]

用于模拟场景中多次反射后的均匀背景光。

### 3.2 漫反射（Diffuse）

\[
I_{diffuse} = K_d \cdot \max(0, \mathbf{N}\cdot\mathbf{L}) \cdot C_{light}\cdot C_{object}
\]

满足 Lambert 定律：入射角越正对（\(\mathbf{N}\cdot\mathbf{L}\) 越大），亮度越强；背光侧截断为 0。

### 3.3 镜面高光（Specular）

\[
I_{specular} = K_s \cdot \max(0, \mathbf{R}\cdot\mathbf{V})^n \cdot C_{light}
\]

其中：
- \(\mathbf{N}\)：交点处单位法向量  
- \(\mathbf{L}\)：交点指向光源的单位向量  
- \(\mathbf{V}\)：交点指向相机的单位向量  
- \(\mathbf{R}\)：理想反射方向，\(\mathbf{R} = 2(\mathbf{N}\cdot\mathbf{L})\mathbf{N} - \mathbf{L}\)  
- \(n\)：高光指数（Shininess），越大高光越“尖锐”

### 3.4 数值稳定性与约束

- \(\mathbf{N},\mathbf{L},\mathbf{V}\) 必须归一化，否则亮度会失真、甚至出现全黑/噪点。
- \(\mathbf{N}\cdot\mathbf{L}\) 与 \(\mathbf{R}\cdot\mathbf{V}\) 需要用 `max(0,·)` 截断，避免背光侧出现非法高光。
- 颜色累加后可能超过 1.0，最终输出前需 clamp 到 \([0,1]\)。

---

## 4. 场景与参数设置

### 4.1 摄像机与光源

- **摄像机位置**：\((0,0,5)\)
- **点光源位置**：\((2,3,4)\)
- **光源颜色**：白色 \((1,1,1)\)
- **背景颜色**：深青色 \((0.02, 0.12, 0.15)\)

### 4.2 几何体（隐式定义）

本实验不加载外部模型，采用数学隐式方程求交：

1) **球体（红色）**  
- 半径：`SPHERE_R = 0.5`  
- 圆心：\((SPHERE\_CX,SPHERE\_CY,SPHERE\_CZ)=(-0.45,-0.05,0)\)  
- 基础颜色：\((0.8,0.1,0.1)\)

2) **有限圆锥（紫色）**  
- 顶点：\((CONE\_AX,CONE\_AY,CONE\_AZ)=(0.45,0.55,0)\)  
- 底面平面：\(y = CONE\_BASE\_Y = -0.55\)  
- 底面半径：`CONE_R_BASE = 0.5`  
- 基础颜色：\((0.6,0.2,0.8)\)

> 说明：为了画面更紧凑，本次将球和圆锥**缩小并向中心靠拢**（相较实验建议的 ±1.2 布置）。

---

## 5. 实现方法与关键步骤

### 5.1 光线投射（Ray Casting）

对每个像素 \((i,j)\)：

1. 计算归一化屏幕坐标 \((u,v)\) 并换算成视平面点 `target=(sx,sy,0)`  
2. 射线：  
   - 起点 \( \mathbf{o} = (0,0,5)\)  
   - 方向 \( \mathbf{d} = \text{normalize}(\mathbf{target}-\mathbf{o})\)

### 5.2 求交与深度竞争（Z-buffer 逻辑）

对同一条射线分别计算：
- `t_s`：球体最近正交点距离  
  - 解二次方程，取最小的正根  
- `t_c`：有限圆锥最近正交点距离  
  - **侧壁**：解二次方程并限制在顶点到基底之间的有限高度范围  
  - **底面圆盘**：与平面 \(y=CONE\_BASE\_Y\) 求交，并判断是否落在圆盘半径内

最后取 `min(t_s, t_c)` 作为真正可见的交点（若两者都无交点则使用背景色），实现正确遮挡关系。

### 5.3 法向量计算

- **球面法线**：\(\mathbf{N}=\text{normalize}(\mathbf{P}-\mathbf{C})\)
- **圆锥侧壁法线**：对隐式函数  
  \[
  F(x',y',z')=x'^2+z'^2-k^2y'^2
  \]
  使用梯度作为法向量：
  \[
  \nabla F=(2x',-2k^2y',2z')
  \]
  再归一化得到单位法向量。
- **圆锥底面法线**：\((0,-1,0)\)

此外，为了避免背面导致高光方向异常，若 \(\mathbf{N}\cdot\mathbf{V}<0\) 则翻转法线（让法线朝向观察者）。

### 5.4 Phong 着色

对最近交点 \(\mathbf{P}\)：

- \(\mathbf{L}=\text{normalize}(\mathbf{LightPos}-\mathbf{P})\)
- \(\mathbf{V}=\text{normalize}(\mathbf{Cam}-\mathbf{P})\)
- \(\mathbf{R}=\text{normalize}(2(\mathbf{N}\cdot\mathbf{L})\mathbf{N}-\mathbf{L})\)

计算 `ambient + diffuse + specular`，并 clamp 到 \([0,1]\) 输出到像素。

---

## 6. UI 交互与运行展示

### 6.1 滑块参数

提供 4 个滑块并与材质参数绑定（范围与默认值符合实验要求）：

- **Ka**：0.0 ~ 1.0，默认 0.2  
- **Kd**：0.0 ~ 1.0，默认 0.7  
- **Ks**：0.0 ~ 1.0，默认 0.5  
- **Shininess**：1.0 ~ 128.0，默认 32.0  

### 6.2 UI 实现说明（Windows 兼容）

在 Windows + **中文路径**下，Taichi `ti.ui`（GGUI/Vulkan）存在着色器 `.spv` 文件加载失败的问题。为保证实验可运行，本实现采取：

- 若检测到当前工作目录包含非 ASCII 字符，则自动使用 **Matplotlib(TkAgg) + Slider** 作为交互界面；
- 启动时会输出并保存一帧预览 `phong_preview.png` 以便核验渲染正确性；
- 若将工程移动到纯英文路径，可切回 `ti.ui` 界面。

### 6.3 运行结果（GIF 说明）

录屏 `屏幕录制 2026-04-16 182653.gif` 显示：

- 画面包含红色球体与紫色圆锥，具备清晰的漫反射明暗过渡与镜面高光点/条带；
- 拖动 **Ka/Kd/Ks/Shininess** 滑块后，亮度与高光形状实时变化，符合 Phong 模型预期：
  - Ka 增大 → 整体更亮（阴影处也抬升）
  - Kd 增大 → 明暗对比增强（受 \(N\cdot L\) 控制）
  - Ks 增大 → 高光更强更明显
  - Shininess 增大 → 高光更尖锐、更集中

---

## 7. 环境与运行方式

### 7.1 环境说明

- Python：3.12（原因：Taichi 暂不提供 Python 3.14 的可用发行包）
- 依赖：`taichi`, `matplotlib`

### 7.2 运行命令

在项目目录执行：

```bash
uv sync
uv run python phong_raytracing.py
```

也可直接双击 `run_phong.bat`。

---

## 8. 总结

本实验使用 Taichi 实现了一个完整的“逐像素光线投射 + 最近交点深度测试 + Phong 着色 + 交互调参”的渲染流程。通过对球体与有限圆锥的隐式求交、法线计算和 Phong 三项分量叠加，能够直观观察材质参数对渲染结果的影响，并通过滑块实现实时交互。

