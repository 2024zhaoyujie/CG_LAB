# outputs/work6

此目录用于保存实验实际运行产物。建议保留两类 run：

- `silhouette_YYYYmmdd_HHMMSS`
- `textured_YYYYmmdd_HHMMSS`

每个 run 至少包含：

- `images/`：中间过程图与 turntable gif
- `plots/`：loss 曲线图
- `meshes/`：最终 obj（textured 还应包含 `vertex_rgb.pt`）
- `logs/metrics.json`：最终指标与参数记录

如果你使用 Colab 运行，请在训练结束后将 run 目录复制回本仓库并提交。
