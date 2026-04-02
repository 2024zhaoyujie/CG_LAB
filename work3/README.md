# Work3：贝塞尔曲线（De Casteljau + 光栅化）

## 演示录屏

以下为本地运行 `main.py` 时的交互录屏（原文件：屏幕录制 2026-04-02 190535.gif，已重命名为仓库内文件名以便引用）：

![贝塞尔曲线 De Casteljau 交互演示](./screen_record_2026-04-02.gif)

## 运行

安装依赖后，在 `work3` 目录执行：

```bash
pip install taichi numpy
python main.py
```

或在资源管理器中双击 [`run_bezier.bat`](run_bezier.bat)（使用仓库内 `work2\.venv` 的 Python）。

## 操作

- **鼠标左键**：在画布上添加控制点（最多 100 个）
- **C**：清空控制点与曲线

