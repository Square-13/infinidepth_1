# InfiniDepth Far-Depth Final Version

这是一个基于官方 InfiniDepth 修改的远景适配版本。目标不是重新训练模型，而是在推理阶段让近景和远景都能参与深度估计与 3D Gaussian Splatting 生成，解决原始流程在山体、远处建筑、远景背景中容易出现的“纸板感”和远景缺点问题。

官方项目地址：

- InfiniDepth: https://github.com/zju3dv/InfiniDepth
- Project page: https://zju3dv.github.io/InfiniDepth/
- Paper: https://arxiv.org/abs/2601.03252

本仓库保留官方目录结构和大部分源码，只在推理流程、远景采样、深度上限和 GS 点生成策略上做修改。权重文件不随 GitHub 上传，需要单独下载后放入对应目录。

## 这个版本解决什么问题

官方 InfiniDepth 的推理流程对普通场景效果很好，但在非常远的场景里会遇到两个问题：

1. 深度估计阶段默认有效距离偏短，极远区域可能被压到上限附近。
2. GS 点生成阶段的 sparse prompt 和 mesh/sample pruning 更偏向近中景，远处点数不足时会像一张有起伏的纸板。

本版本的核心思想是把“深度补全”和“GS 点生成”分开处理：

1. 第一遍先做 dense depth，也就是完整的稠密深度估计。
2. 根据场景自动或手动设置远景上限，例如 200、500、2000。
3. 生成 sparse prompt 时优先保留近景稳定点，再对远处空白区域补点。
4. 对输入给 DepthSensor GS 的 prompt 做压缩，避免远景数值过大破坏近景结构。
5. 在 GS 阶段允许使用更大的采样深度上限，并对远景点的尺度做控制。

这样做的原因是：直接把所有远景深度点喂给模型会让网络把远处区域拉伸或变成大片平面；完全不喂远景点又会导致远处没有点、变成纸板。本版本保留近景精度，同时让远景以受控方式进入第二阶段。

## 推荐入口

推荐使用：

```bash
python run_final.py INPUT_IMAGE [OUTPUT_NAME]
```

`run_final.py` 会完成四步：

1. 恢复最终测试过的 nice/V29 基线代码快照。
2. 运行第一遍 InfiniDepth dense depth。
3. 根据 dense depth 生成近景优先、远景补全的 sparse prompt。
4. 调用 DepthSensor GS 生成最终 `.ply` 和可选视频。

不推荐直接运行旧的 `.sh` 脚本，因为本版本已经把最终参数和流程整理进了 Python 入口。

## 环境准备

建议在 Linux 服务器上运行。示例使用默认运行目录 `/tmp/infinidepth`：

```bash
git clone https://github.com/Square-13/infinidepth_1.git
cd infinidepth_1

python3 -m venv /tmp/infinidepth/venv
/tmp/infinidepth/venv/bin/pip install --upgrade pip
/tmp/infinidepth/venv/bin/pip install -r requirements.txt
```

如果你已有自己的虚拟环境，可以运行时指定：

```bash
python run_final.py example_data/image/mount.png --venv /path/to/your/venv
```

## 权重放置

权重文件不在仓库里，需要单独下载。默认情况下，`run_final.py` 会读取以下路径：

```text
/tmp/infinidepth/checkpoints/infinidepth.ckpt
/tmp/infinidepth/checkpoints/moge/model.pt
checkpoints/depth/infinidepth_depthsensor.ckpt
checkpoints/gs/infinidepth_depthsensor_gs.ckpt
checkpoints/sky/skyseg.onnx
```

需要手动创建目录：

```bash
mkdir -p /tmp/infinidepth/checkpoints/moge
mkdir -p checkpoints/depth checkpoints/gs checkpoints/sky
```

权重来源：

| 用途 | 文件名 | 放置位置 |
| --- | --- | --- |
| 第一遍 RGB 深度估计 | `infinidepth.ckpt` | `/tmp/infinidepth/checkpoints/infinidepth.ckpt` |
| MoGe 尺度恢复 | `model.pt` | `/tmp/infinidepth/checkpoints/moge/model.pt` |
| DepthSensor 深度模型 | `infinidepth_depthsensor.ckpt` | `checkpoints/depth/infinidepth_depthsensor.ckpt` |
| DepthSensor GS 模型 | `infinidepth_depthsensor_gs.ckpt` | `checkpoints/gs/infinidepth_depthsensor_gs.ckpt` |
| 天空分割，可选但推荐 | `skyseg.onnx` | `checkpoints/sky/skyseg.onnx` |

如果不想使用 `/tmp/infinidepth`，可以用 `--tmp-root` 修改运行目录：

```bash
python run_final.py example_data/image/mount.png --tmp-root /your/runtime/root
```

此时第一遍深度模型和 MoGe 需要放到：

```text
/your/runtime/root/checkpoints/infinidepth.ckpt
/your/runtime/root/checkpoints/moge/model.pt
```

## 运行命令

陌生图片优先使用默认命令。默认是 `smart` 策略，会先探测场景深度，再决定是否启用远景处理。

```bash
python run_final.py example_data/image/mount.png
```

也可以显式指定 profile。

普通近中景，例如 campus：

```bash
python run_final.py example_data/image/campus.jpg campus --profile auto
```

远景或山体场景，例如 mount、vally：

```bash
python run_final.py example_data/image/mount.png mount --profile final --far-cap 2000
python run_final.py example_data/image/vally.png vally --profile final --far-cap 2000
```

其他示例图：

```bash
python run_final.py example_data/image/ice.png ice --profile auto
python run_final.py example_data/image/runway.jpg runway
python run_final.py example_data/image/bird.png bird
```

如果只想生成 `.ply`，不生成视频：

```bash
python run_final.py example_data/image/mount.png mount --profile final --far-cap 2000 --no-render-video
```

运行前检查参数，不真正执行：

```bash
python run_final.py example_data/image/mount.png mount --profile final --far-cap 2000 --dry-run
```

## profile 说明

| profile | 适合场景 | 说明 |
| --- | --- | --- |
| `smart` | 陌生图片 | 默认策略，先用较大深度上限探测，再决定走普通场景还是远景场景。 |
| `auto` | campus 这类普通近中景 | 更接近原来 `AUTO_SCENE=1` 的行为，避免短距离场景被过大的远景上限拉伸。 |
| `final` | mount、vally 这类极远景 | 使用最终测试稳定的远景策略，默认允许到 2000 左右的深度范围。 |

经验规则：

- 如果场景普通，远处不超过几百，先用 `--profile auto`。
- 如果画面里有山、远处大背景、远景明显超过 1000，用 `--profile final --far-cap 2000`。
- 如果不确定，用默认 `smart`。

## 输出文件

默认输出在 `/tmp/infinidepth/outputs/` 下：

```text
/tmp/infinidepth/outputs/self_prompt_<name>_nice_prompt_compress/
/tmp/infinidepth/outputs/gs_<name>_nice_prompt_compress/
```

常用结果：

| 文件 | 含义 |
| --- | --- |
| `<name>_dense_vis.png` | 第一遍 dense depth 可视化 |
| `<name>_dense_prompt_compressed_vis.png` | prompt 压缩后的深度可视化 |
| `<name>_final_sparse_preview.png` | 最终 sparse prompt 点分布 |
| `<name>_nice_prompt_compress_sparse.npy` | 输入给 DepthSensor GS 的 sparse depth |
| `<name>_nice_prompt_compress.ply` | 最终 3DGS 点云/高斯结果 |
| `<name>_nice_prompt_compress_novel.mp4` | 可选的 orbit 视频 |

脚本结束时会自动打印 Windows PowerShell 的 `scp` 下载命令。也可以手动下载，例如：

```powershell
scp server:/tmp/infinidepth/outputs/gs_mount_nice_prompt_compress/mount_nice_prompt_compress.ply "$env:USERPROFILE\Desktop\mount_nice_prompt_compress.ply"
```

其中 `server` 换成你的 SSH 主机名。

## 算法思想

### 1. 深度估计阶段

官方模型本质上是先由 encoder 提取图像特征，再由 decoder 输出深度。这里的 encoder 可以理解为“看图并提取多尺度语义和几何线索”的部分，decoder 是“根据这些特征恢复每个像素深度”的部分。

本版本没有重新训练 encoder 或 decoder，而是在推理阶段调整深度上限：

- `INFINIDEPTH_MAX_DEPTH` 控制第一遍深度估计的最大范围。
- `INFINIDEPTH_MAX_PROMPT_DEPTH` 控制 sparse prompt 被模型接收的最大范围。
- `--far-cap` 控制当前场景允许的远景上限。

这样可以让远处区域先被 dense depth 估计出来，而不是一开始就被距离上限过滤掉。

### 2. sparse prompt 生成阶段

sparse prompt 指输入给 DepthSensor 分支的稀疏深度提示。它不是把整张 dense depth 全部喂进去，而是选择一部分点作为约束。

本版本采用近景优先策略：

- 近景和中景保留更多稳定点，保证主体和边缘不被破坏。
- 远景只在原来没有点的区域补充点，避免远景点覆盖近景结构。
- 对远景 prompt 做深度压缩，例如把过大的远景距离压到更温和的范围。

这样做的目的是让模型知道“远处确实有东西”，但不要让巨大深度值直接支配第二阶段。

### 3. GS 点生成阶段

GS 是 Gaussian Splatting 的缩写，这里指最终生成 3D Gaussian 点。原流程容易在远景处点数不足，或者 mesh/sample pruning 把远处过滤掉。

本版本做了几类修改：

- 放开 GS 的最大采样深度，使远处点可以进入生成阶段。
- 对远景 Gaussian 尺度做控制，避免远处点过大造成糊成一片。
- 记录中间 trace 图，方便判断是 dense depth、sparse prompt 还是 GS 采样出了问题。

最终效果是：近景仍然保持原来的精度，远处不再简单变成一张空白纸板。

## 主要修改文件

| 文件 | 作用 |
| --- | --- |
| `run_final.py` | 最终统一入口，替代原来的长 `.sh` 脚本。 |
| `inference_gs.py` | 增加远景深度控制、trace 输出、GS 阶段远景处理参数。 |
| `InfiniDepth/model/model.py` | 让最大深度和 GS 采样上限可配置。 |
| `InfiniDepth/utils/io_utils.py` | sparse depth prompt 支持更灵活的最大深度。 |
| `InfiniDepth/utils/sampling_utils.py` | 修改 GS 采样和远景 pruning 策略。 |
| `InfiniDepth/utils/gs_utils.py` | 增加远景 Gaussian 尺度控制。 |
| `InfiniDepth/utils/inference_utils.py` | 调整推理阶段过滤相关工具。 |
| `code_snapshots/nice_latest/` | 保存最终验证过的基线代码快照，`run_final.py` 默认会恢复它。 |

如果你正在开发这些文件，不希望 `run_final.py` 覆盖当前源码，可以加：

```bash
python run_final.py example_data/image/mount.png --no-restore
```

## 常见问题

### 1. 报错 `ModuleNotFoundError`

通常是没有进入正确环境，或者 `--venv` 指错了。检查：

```bash
/tmp/infinidepth/venv/bin/python3 - <<'PY'
for m in ["h5py", "jaxtyping", "cv2", "open3d", "sklearn", "torch"]:
    try:
        __import__(m)
        print(m, "ok")
    except Exception as e:
        print(m, "missing:", e)
PY
```

### 2. 报错找不到 checkpoint

检查权重是否放到了 README 上面的默认路径。尤其注意：第一遍 RGB 深度模型默认在 `/tmp/infinidepth/checkpoints/infinidepth.ckpt`，而 DepthSensor GS 权重默认在仓库的 `checkpoints/` 目录下。

### 3. 近景被拉伸

一般是远景上限设置太大。短距离图片不要直接用 `--far-cap 2000`，优先：

```bash
python run_final.py example_data/image/campus.jpg campus --profile auto
```

### 4. 远景还是像纸板

先看中间文件：

```text
<name>_dense_vis.png
<name>_final_sparse_preview.png
<name>_stage2_refined_dense_depth.png
```

如果 dense depth 本身已经有远景，但 final sparse preview 远景点少，说明 sparse prompt 还需要增加远景补点；如果 sparse prompt 有点但最终仍然糊，说明 GS 采样点数或远景尺度需要继续调。

## 说明

本仓库是在官方 InfiniDepth 基础上的推理阶段改造版本。训练代码保留官方结构，方便完整运行和对照，但本版本的主要贡献集中在 `run_final.py` 和远景 GS 推理相关代码上。

如果需要引用原方法，请引用 InfiniDepth 官方论文和项目。
