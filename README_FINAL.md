# InfiniDepth Final Far-Depth Inference Version

这是一份适合上传 GitHub 的整理版仓库。它保留原 InfiniDepth 源码，并加入本次针对超远景深度估计和 GS 点生成的最终推理阶段修改。

## 本仓库做了什么

原始流程在 `mount`、`vally` 这类超远景图片中，远处山体或背景容易缺点、纸板化或被采样范围过滤。本版本的核心修改是把两个阶段拆开：

1. `DepthSensor` 阶段主要负责近景和中景深度补全。
2. `GS` 阶段允许极远景使用第一阶段 `raw dense depth` 参与三维点生成。
3. `smart` 策略会先用 2000m 上限做一次 `probe`，判断陌生图片是否属于超远景。

## 重要英文术语

`inference`：推理，意思是用已有模型权重生成结果，不重新训练。

`DepthSensor`：深度补全网络，主要修正近景和中景。

`GS`：Gaussian Splatting，高斯点渲染，把图像和深度转换成三维高斯点。

`raw dense depth`：第一阶段模型直接输出的原始稠密深度。

`profile`：运行配置档，例如 `auto`、`final`、`smart`。

`probe`：探测，先跑一次粗判断，再决定使用哪套策略。

`checkpoint`：模型权重文件，本仓库不直接提交权重。

## 关键文件

```text
run_final.py
inference_gs.py
InfiniDepth/model/model.py
InfiniDepth/utils/sampling_utils.py
InfiniDepth/utils/gs_utils.py
InfiniDepth/utils/io_utils.py
code_snapshots/nice_latest/
docs/
```

## 运行方式

先进入服务器上的原项目环境：

```bash
cd /home/liuyanb/InfiniDepth
source /tmp/liuyanb-infinidepth/venv/bin/activate
```

运行 campus：

```bash
python run_final.py /home/liuyanb/InfiniDepth/example_data/image/campus.jpg --profile auto
```

运行 mount：

```bash
python run_final.py /home/liuyanb/InfiniDepth/example_data/image/mount.png --profile final
```

运行 vally：

```bash
python run_final.py /home/liuyanb/InfiniDepth/example_data/image/vally.png --profile final
```

运行陌生图片：

```bash
python run_final.py /path/to/your/image.jpg
```

陌生图片默认使用 `smart`，它会自动判断是否需要超远景策略。

## GitHub 注意事项

本仓库没有包含模型权重和 Git LFS 历史缓存。需要的权重文件请参考 [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md)。

本仓库已排除：

```text
.git/
checkpoints/
__pycache__/
assets/demo.gif
assets/supersplat-viewer/index.js.map
```

## 报告文档

项目汇报 PDF：

```text
docs/InfiniDepth_project_report.pdf
```

算法说明 PDF：

```text
docs/InfiniDepth_final_algorithm_explanation.pdf
```

