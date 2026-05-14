# Work6_2: PyTorch3D 可微渲染网格重建

本目录用于提交实验六（work6_2）内容，包含代码、结果和实验报告。

## 1. 目录结构

- `src/Work6/`：核心代码
- `outputs/work6/`：实验输出（必做与选做）
- `实验报告.md`：实验报告
- `requirements_colab.txt`：Colab 依赖说明

## 2. Colab 运行（推荐）

### 2.1 环境安装

在 Colab Notebook 中依次执行：

```bash
!nvidia-smi
!pip install -U pip
!pip install -r work6_2/requirements_colab.txt
```

若 PyTorch3D 轮子安装失败，可按课程环境改用手动版本组合安装。

### 2.2 运行命令

在仓库根目录执行：

```bash
# 必做：轮廓监督
python -m src.Work6.main silhouette --steps 220 --image-size 224 --num-views 18 --seed 2026

# 选做：轮廓 + RGB 联合优化
python -m src.Work6.main textured --steps 240 --image-size 224 --num-views 20 --seed 2027

# 顺序执行两者
python -m src.Work6.main both --steps 260 --image-size 224 --num-views 20 --seed 2028
```

## 3. 结果清单要求

建议每个 run 目录至少包含：

- `meshes/*.obj`
- `plots/*losses.png`
- `images/*turntable.gif`
- `logs/metrics.json`

## 4. 非雷同说明

本实现与常见模板相比做了以下重构：

1. 训练入口改为统一子命令调度（`main.py`）。
2. 配置采用结构化 dataclass（渲染/训练/损失分组）。
3. silhouette 与 textured 训练采用不同默认超参数与随机种子。
4. 结果目录采用时间戳管理并输出 JSON 指标日志。
