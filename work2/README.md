# Work2：Taichi 3D 变换作业

## 作业内容

本作业基于 `main.py` 实现了一个简单的 3D 三角形变换与显示流程，包含：

- 模型变换（绕 Z 轴旋转）
- 视图变换（相机平移到原点）
- 透视投影变换（平截头体到标准立方体）
- 透视除法与视口映射（NDC -> 屏幕坐标）

程序使用 `Taichi GUI` 实时绘制三角形三条边，并支持键盘交互旋转。

## 运行环境

- Python 3.10+
- taichi

安装依赖示例：

```bash
pip install taichi
```

## 运行方式

在 `work2` 目录下执行：

```bash
python main.py
```

## 交互说明

- `A`：逆时针旋转（角度 +10）
- `D`：顺时针旋转（角度 -10）
- `ESC`：退出程序

## 动图演示
![triangle_mvp_120f](https://github.com/user-attachments/assets/7ac500e4-9a02-432f-9fda-e18d0786b20a)

![cube_mvp](https://github.com/user-attachments/assets/1c8e16a3-69c1-46af-b3fd-a8a443bd3977)


