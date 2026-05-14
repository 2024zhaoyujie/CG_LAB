# Colab 提交指南（零本地环境）

## 1) 准备仓库

```bash
!git clone https://github.com/2024zhaoyujie/CG_LAB.git
%cd CG_LAB
```

将本目录 `work6_2` 上传/复制到 Colab 的 `CG_LAB` 根目录后再执行下面步骤。

## 2) 安装依赖

```bash
!python -m pip install -U pip
!python -m pip install -r work6_2/requirements_colab.txt
```

## 3) 运行实验

```bash
# 必做
!python -m work6_2.src.Work6.main silhouette --steps 220 --image-size 224 --num-views 18 --seed 2026

# 选做
!python -m work6_2.src.Work6.main textured --steps 240 --image-size 224 --num-views 20 --seed 2027
```

## 4) 整理结果

将 `outputs/work6/` 里的真实 run 目录复制到 `work6_2/outputs/work6/`，保持 `silhouette_时间戳` 与 `textured_时间戳` 命名。

## 5) 提交

```bash
!git add work6_2
!git commit -m "add work6_2 differential rendering submission"
!git push
```
