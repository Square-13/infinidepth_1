# infinidepth_1

Inference-only InfiniDepth fork for single-image far-depth reconstruction.

This repository keeps the runtime code needed for depth estimation and Gaussian Splatting, then adds `run_final.py` as the main entry point for scenes with both near foreground and very distant background.

## What Changed

The original inference pipeline can miss distant scene content in two places:

1. The depth estimation stage may be limited by a short maximum depth.
2. The sparse prompt and GS point-generation stage may filter out far pixels before they become 3D points.

This version changes the inference pipeline, not the training code:

1. Depth range can be set at runtime.
2. Sparse prompt generation keeps selected far-depth pixels instead of discarding them.
3. GS point sampling uses configurable depth limits.
4. Far GS points can be generated from the first-pass dense depth when the refined DepthSensor result flattens distant scenery.
5. `run_final.py` provides `auto`, `final`, and `smart` profiles.

## Repository Layout

```text
run_final.py                 main runnable script
inference_depth.py           dense depth inference
inference_gs.py              Gaussian Splatting inference
InfiniDepth/                 runtime model and utility code
code_snapshots/nice_latest/  final patched files restored by run_final.py
example_data/image/          example input images
requirements.txt             inference dependencies
```

This repository does not include model checkpoints.

## Environment

Use Python 3.10 or 3.11 on a CUDA-capable Linux server.

Install PyTorch first with the CUDA version that matches the server, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If the server already has a working InfiniDepth virtual environment, activate it and run the code directly.

## Checkpoints

Place checkpoints at the default paths below, or pass them explicitly on the command line.

Default paths:

```text
<tmp-root>/checkpoints/infinidepth.ckpt
<tmp-root>/checkpoints/moge/model.pt
<repo>/checkpoints/depth/infinidepth_depthsensor.ckpt
<repo>/checkpoints/gs/infinidepth_depthsensor_gs.ckpt
<repo>/checkpoints/sky/skyseg.onnx
```

The default `<tmp-root>` is:

```text
/tmp/infinidepth
```

Explicit checkpoint example:

```bash
python run_final.py path/to/image.png \
  --depth-ckpt path/to/infinidepth.ckpt \
  --moge-ckpt path/to/moge/model.pt \
  --depthsensor-ckpt path/to/infinidepth_depthsensor.ckpt \
  --depthsensor-gs-ckpt path/to/infinidepth_depthsensor_gs.ckpt \
  --sky-ckpt path/to/skyseg.onnx
```

## Run

For an unknown image, start with the smart profile:

```bash
python run_final.py path/to/image.jpg
```

Examples:

```bash
python run_final.py example_data/image/campus.jpg --profile auto
python run_final.py example_data/image/ice.png --profile auto
python run_final.py example_data/image/mount.png --profile final --far-cap 2000
python run_final.py example_data/image/vally.png --profile final
```

Use an existing virtual environment:

```bash
python run_final.py example_data/image/mount.png --venv /path/to/venv
```

Disable video rendering if `gsplat` is not installed:

```bash
python run_final.py example_data/image/mount.png --profile final --no-render-video
```

## Profiles

`auto` is for ordinary near and mid-range scenes. It avoids forcing a very large far-depth range onto short scenes.

`final` is for far scenes such as mountains or valleys. It allows far points to enter sparse prompt generation and GS point generation.

`smart` runs a probe first, then chooses between the near/mid-range policy and the far-scene policy.

## Output

Outputs are written under:

```text
<tmp-root>/outputs/
```

Typical outputs:

```text
dense depth visualization
sparse prompt preview
PLY point cloud
novel-view video, if enabled
```

`run_final.py` also prints `scp` commands for downloading the generated files.

## Development History

The `main` branch is the clean runnable repository.

The `py-history` branch keeps a focused history of the key Python files that were changed:

```text
https://github.com/Square-13/infinidepth_1/tree/py-history
```
