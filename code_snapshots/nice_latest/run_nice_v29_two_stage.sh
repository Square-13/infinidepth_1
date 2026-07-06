#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:?Usage: bash run_nice_v29_two_stage.sh /path/to/image.png output_name}"
NAME="${2:-$(basename "${IMAGE_PATH%.*}")}"

REPO="/home/liuyanb/InfiniDepth"
VENV="/tmp/liuyanb-infinidepth/venv"
TMP_ROOT="/tmp/liuyanb-infinidepth"
NICE="${REPO}/code_snapshots/nice_latest"

DEPTH_CKPT="${TMP_ROOT}/checkpoints/infinidepth.ckpt"
MOGE_CKPT="${TMP_ROOT}/checkpoints/moge/model.pt"
DEPTHSENSOR_CKPT="${REPO}/checkpoints/depth/infinidepth_depthsensor.ckpt"
DEPTHSENSOR_GS_CKPT="${REPO}/checkpoints/gs/infinidepth_depthsensor_gs.ckpt"

SELF_OUT="${TMP_ROOT}/outputs/self_prompt_${NAME}_v29"
GS_OUT="${TMP_ROOT}/outputs/gs_${NAME}_v29_nice"
NICE_MAX_DEPTH="${NICE_MAX_DEPTH:-500}"
NICE_GS_SAMPLE_POINTS="${NICE_GS_SAMPLE_POINTS:-3000000}"
NICE_FAR_PROMPT_SAMPLES="${NICE_FAR_PROMPT_SAMPLES:-12000}"
NICE_TARGET_PROMPT_POINTS="${NICE_TARGET_PROMPT_POINTS:-36000}"
NICE_FAR_SCALE_MULTIPLIER="${NICE_FAR_SCALE_MULTIPLIER:-0.60}"

cd "$REPO"
source "${VENV}/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMP_ROOT}/tmp"
export INFINIDEPTH_MAX_DEPTH="${INFINIDEPTH_MAX_DEPTH:-$NICE_MAX_DEPTH}"
export INFINIDEPTH_MAX_PROMPT_DEPTH="${INFINIDEPTH_MAX_PROMPT_DEPTH:-$INFINIDEPTH_MAX_DEPTH}"
mkdir -p "$TMPDIR" "$SELF_OUT" "$GS_OUT"
echo "[config] INFINIDEPTH_MAX_DEPTH=${INFINIDEPTH_MAX_DEPTH}"
echo "[config] INFINIDEPTH_MAX_PROMPT_DEPTH=${INFINIDEPTH_MAX_PROMPT_DEPTH}"
echo "[config] NICE_GS_SAMPLE_POINTS=${NICE_GS_SAMPLE_POINTS}"
echo "[config] NICE_FAR_PROMPT_SAMPLES=${NICE_FAR_PROMPT_SAMPLES}"
echo "[config] NICE_TARGET_PROMPT_POINTS=${NICE_TARGET_PROMPT_POINTS}"
echo "[config] NICE_FAR_SCALE_MULTIPLIER=${NICE_FAR_SCALE_MULTIPLIER}"

echo "[1/4] Restore nice/V29 baseline code"
for f in InfiniDepth/utils/gs_utils.py InfiniDepth/utils/sampling_utils.py InfiniDepth/model/model.py inference_gs.py; do
  if [ -f "$NICE/$f" ]; then
    cp "$NICE/$f" "$f"
    echo "restored $f"
  else
    echo "warning: missing $NICE/$f"
  fi
done

echo "[1.5/4] Force inference_gs.py to preserve sparse prompt"
python3 - <<'PY'
from pathlib import Path
import re

p = Path("inference_gs.py")
s = p.read_text()

if '"preserve_sparse": True' not in s:
    old = """gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=args.input_depth_path,
        input_size=args.input_size,
        image=image,
        device=device,
        moge2_pretrained=args.moge2_pretrained,
    )"""

    new = """gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=args.input_depth_path,
        input_size=args.input_size,
        image=image,
        device=device,
        moge2_pretrained=args.moge2_pretrained,
        depth_load_kwargs={
            "preserve_sparse": True,
            "max_prompt": float(os.environ.get("INFINIDEPTH_MAX_PROMPT_DEPTH", os.environ.get("INFINIDEPTH_MAX_DEPTH", "500"))),
            "num_samples": 100000000,
        },
    )"""

    if old not in s:
        raise RuntimeError("Cannot find prepare_metric_depth_inputs block in inference_gs.py")
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("patched inference_gs.py: preserve_sparse=True")
else:
    print("checked inference_gs.py: preserve_sparse=True")
PY


echo "[1.6/4] Check V29 GS far-depth sampling config"
python3 - <<'PY2'
from pathlib import Path

p = Path("InfiniDepth/model/model.py")
s = p.read_text()
if "gs_sample_filter_mode" not in s:
    raise RuntimeError("model.py does not expose gs_sample_filter_mode")
print("checked model.py: sample_filter_mode is configurable")
PY2

python3 -m py_compile InfiniDepth/utils/gs_utils.py InfiniDepth/utils/sampling_utils.py InfiniDepth/model/model.py inference_gs.py

echo "[2/4] First-pass InfiniDepth dense depth"
python3 - <<PY
from pathlib import Path
from inference_depth import DepthInferenceArgs, load_depth_model, run_depth_inference
from InfiniDepth.utils.io_utils import save_depth_array, plot_depth

out = Path("${SELF_OUT}")
out.mkdir(parents=True, exist_ok=True)

args = DepthInferenceArgs(
    input_image_path="${IMAGE_PATH}",
    model_type="InfiniDepth",
    depth_model_path="${DEPTH_CKPT}",
    moge2_pretrained="${MOGE_CKPT}",
    input_size=(768, 1024),
    output_size=(768, 1024),
    output_resolution_mode="specific",
    save_pcd=False,
    enable_skyseg_model=False,
)

model, device = load_depth_model(args)
result = run_depth_inference(args, model=model, device=device)

save_depth_array(result.pred_depthmap, str(out / "${NAME}_dense.npy"))
plot_depth(result.org_img, result.pred_depthmap, str(out / "${NAME}_dense_vis.png"))

print("saved dense:", out / "${NAME}_dense.npy")
print("saved vis:", out / "${NAME}_dense_vis.png")
PY

echo "[3/4] Build V29 sparse depth prompt"
python3 - <<PY
from pathlib import Path
import os
import numpy as np
from PIL import Image

img_path = Path("${IMAGE_PATH}")
dense_path = Path("${SELF_OUT}/${NAME}_dense.npy")
out_dir = Path("${SELF_OUT}")
out_dir.mkdir(parents=True, exist_ok=True)

depth = np.load(dense_path).astype(np.float32)
depth = np.squeeze(depth)
if depth.ndim != 2:
    raise ValueError(f"Expected 2D dense depth, got {depth.shape}")

max_depth = float(os.environ.get("INFINIDEPTH_MAX_PROMPT_DEPTH", os.environ.get("INFINIDEPTH_MAX_DEPTH", "500")))
far_prompt_samples = int(os.environ.get("NICE_FAR_PROMPT_SAMPLES", "12000"))
H, W = depth.shape
img = Image.open(img_path).convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
rgb = np.asarray(img).astype(np.float32) / 255.0
gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

valid = np.isfinite(depth) & (depth > 1.0) & (depth < max_depth)
if int(valid.sum()) < 256:
    raise RuntimeError(f"Too few valid dense-depth pixels: {int(valid.sum())}")

logd = np.log(np.clip(depth, 1e-3, None))
gx = np.zeros_like(logd)
gy = np.zeros_like(logd)
gx[:, 1:] = np.abs(logd[:, 1:] - logd[:, :-1])
gy[1:, :] = np.abs(logd[1:, :] - logd[:-1, :])
depth_edge = np.maximum(gx, gy)

ix = np.zeros_like(gray)
iy = np.zeros_like(gray)
ix[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
iy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
rgb_edge = ix + iy

stable = valid.copy()
stable &= depth_edge < np.quantile(depth_edge[valid], 0.82)
stable &= rgb_edge < np.quantile(rgb_edge[valid], 0.92)

rng = np.random.default_rng(20260623)
sparse = np.zeros_like(depth, dtype=np.float32)

bins = [
    (1, 10, 3600),
    (10, 25, 4800),
    (25, 55, 4800),
    (55, 90, 7600),
    (90, 120, 4530),
]
if max_depth > 120:
    if max_depth <= 220:
        bins.append((120, max_depth, far_prompt_samples))
    else:
        far_each = max(1, far_prompt_samples // 3)
        mid1 = min(220, max_depth)
        mid2 = min(350, max_depth)
        bins.append((120, mid1, far_each))
        if max_depth > mid1:
            bins.append((mid1, mid2, far_each))
        if max_depth > mid2:
            bins.append((mid2, max_depth, max(1, far_prompt_samples - 2 * far_each)))

for lo, hi, n in bins:
    mask = stable & (depth >= lo) & (depth < hi)
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        continue
    take = rng.choice(idx, size=min(n, idx.size), replace=False)
    sparse.reshape(-1)[take] = depth.reshape(-1)[take]

target_min = int(os.environ.get("NICE_TARGET_PROMPT_POINTS", "36000"))
current = int((sparse > 0).sum())
if current < target_min:
    remain = valid & (sparse <= 0)
    idx = np.flatnonzero(remain.reshape(-1))
    if idx.size > 0:
        take = rng.choice(idx, size=min(target_min - current, idx.size), replace=False)
        sparse.reshape(-1)[take] = depth.reshape(-1)[take]

np.save(out_dir / "${NAME}_v29_sparse.npy", sparse)

vis = np.zeros((H, W, 3), dtype=np.uint8)
lo, hi = np.percentile(depth[valid], [2, 98])
norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
vis[..., :] = (norm[..., None] * 255).astype(np.uint8)
vis[sparse > 0] = [255, 0, 0]
Image.fromarray(vis).save(out_dir / "${NAME}_v29_sparse_preview.png")

print("dense shape:", depth.shape)
print("max depth cap:", max_depth)
print("sparse valid:", int((sparse > 0).sum()))
print("saved sparse:", out_dir / "${NAME}_v29_sparse.npy")
print("saved preview:", out_dir / "${NAME}_v29_sparse_preview.png")
PY

echo "[4/4] Second-pass DepthSensor GS"
python3 inference_gs.py \
  --input-image-path="$IMAGE_PATH" \
  --input-depth-path="${SELF_OUT}/${NAME}_v29_sparse.npy" \
  --model-type=InfiniDepth_DepthSensor \
  --depth-model-path="$DEPTHSENSOR_CKPT" \
  --gs-model-path="$DEPTHSENSOR_GS_CKPT" \
  --moge2-pretrained="$MOGE_CKPT" \
  --output-ply-dir="$GS_OUT" \
  --output-ply-name="${NAME}_v29_nice.ply" \
  --novel-video-path="${GS_OUT}/${NAME}_v29_nice_novel.mp4" \
  --sample-point-num="$NICE_GS_SAMPLE_POINTS" \
  --max-prompt-depth="$INFINIDEPTH_MAX_PROMPT_DEPTH" \
  --gs-max-sample-depth="$INFINIDEPTH_MAX_DEPTH" \
  --gs-sample-filter-mode=max_depth \
  --gs-far-detail-min-depth=45 \
  --gs-far-detail-edge-quantile=0.65 \
  --gs-far-detail-area-boost=10 \
  --gs-far-scale-min-depth=80 \
  --gs-far-scale-multiplier="$NICE_FAR_SCALE_MULTIPLIER" \
  --novel-trajectory=orbit \
  --novel-num-frames=120 \
  --novel-video-fps=30 \
  --render-size 720 1280

echo ""
echo "Done."
echo "Download commands:"
echo "scp Point:${SELF_OUT}/${NAME}_dense_vis.png \"\$env:USERPROFILE\\Desktop\\${NAME}_dense_vis.png\""
echo "scp Point:${SELF_OUT}/${NAME}_v29_sparse_preview.png \"\$env:USERPROFILE\\Desktop\\${NAME}_v29_sparse_preview.png\""
echo "scp Point:${GS_OUT}/${NAME}_v29_nice.ply \"\$env:USERPROFILE\\Desktop\\${NAME}_v29_nice.ply\""
echo "scp Point:${GS_OUT}/${NAME}_v29_nice_novel.mp4 \"\$env:USERPROFILE\\Desktop\\${NAME}_v29_nice_novel.mp4\""
