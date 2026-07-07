#!/usr/bin/env python3
"""Pure Python entrypoint for the final InfiniDepth far-depth pipeline."""

from __future__ import annotations

import argparse
import math
import os
import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_ENV: Dict[str, str] = {
    "AUTO_SCENE": "0",
    "NICE_CAP": "220",
    "FAR_MIN": "180",
    "FAR_CAP": "2000",
    "NICE_TARGET": "36000",
    "NICE_PER_TILE": "14",
    "NICE_TILE": "16",
    "FAR_TARGET": "0",
    "FAR_TILE": "32",
    "FAR_PER_TILE": "2",
    "SKY_DILATE": "1",
    "GS_SAMPLE_POINTS": "6000000",
    "GS_FAR_SCALE_MULTIPLIER": "0.70",
    "FUSE_TWO_STAGE_FAR": "0",
    "FUSE_FAR_TARGET": "48000",
    "FUSE_FAR_OVERWRITE": "1",
    "FUSE_FAR_USE_RAW": "1",
    "FAR_RELATIVE_PROMPT": "0",
    "FAR_RELATIVE_PROMPT_MIN": "140",
    "FAR_RELATIVE_PROMPT_MAX": "220",
    "FAR_RELATIVE_Q_LOW": "5",
    "FAR_RELATIVE_Q_HIGH": "95",
    "FAR_RELATIVE_GAIN": "1.0",
    "FAR_RELATIVE_CENTER": "0.5",
    "FAR_RAW_GS_OVERRIDE": "1",
    "FAR_RAW_GS_BLEND": "1.0",
    "FAR_RAW_GS_APPLY_DENSE": "0",
    "FAR_RESCUE": "auto",
    "FAR_RESCUE_PROMPT": "1",
    "FAR_RESCUE_PROTECT_NICE_PX": "18",
    "FAR_RESCUE_Y_MIN": "0.10",
    "FAR_RESCUE_Y_MAX": "0.86",
    "FAR_RESCUE_FOREGROUND_Y_MAX": "0.76",
    "MOUNTAIN_RESCUE": "0",
    "MOUNTAIN_RESCUE_TARGET": "24000",
    "MOUNTAIN_RESCUE_TILE": "16",
    "MOUNTAIN_RESCUE_PER_TILE": "8",
    "MOUNTAIN_RESCUE_Y_MIN": "0.10",
    "MOUNTAIN_RESCUE_Y_MAX": "0.86",
    "MOUNTAIN_OBJECT_UNIFORM": "1",
    "MOUNTAIN_UNIFORM_SPACING": "0",
    "MOUNTAIN_RELIEF": "0",
    "MOUNTAIN_RELIEF_METERS": "22",
    "MOUNTAIN_RELIEF_TO_GS": "0",
    "MOUNTAIN_RELIEF_BLEND": "1.0",
    "MOUNTAIN_PROMPT": "0",
    "MOUNTAIN_PROTECT_NICE_PX": "18",
    "FAR_COMPRESS_TO_GS": "0",
    "FAR_COMPRESS_START": "120",
    "FAR_COMPRESS_RATIO": "0.35",
    "FAR_COMPRESS_BLEND": "1.0",
    "FAR_SHAPE_PRIOR": "off",
    "FAR_SHAPE_MAX_METERS": "10",
    "FAR_SHAPE_FLAT_RANGE": "4",
    "PROMPT_COMPRESS_DEPTH": "1",
    "PROMPT_COMPRESS_START": "100",
    "PROMPT_COMPRESS_RATIO": "0.35",
    "PROMPT_COMPRESS_MAX": "180",
    "RENDER_VIDEO": "1",
    "ALLOW_FAR_PROMPT": "0",
}

AUTO_PROFILE_ENV: Dict[str, str] = {
    **DEFAULT_ENV,
    "AUTO_SCENE": "1",
    "NICE_CAP": "180",
    "FAR_MIN": "180",
    "FAR_CAP": "220",
    "GS_SAMPLE_POINTS": "2000000",
    "GS_FAR_SCALE_MULTIPLIER": "1.0",
    "FAR_RAW_GS_OVERRIDE": "0",
    "MOUNTAIN_RESCUE": "auto",
    "MOUNTAIN_PROMPT": "1",
    "PROMPT_COMPRESS_MAX": "160",
    "RENDER_VIDEO": "1",
}

SMART_PROFILE_ENV: Dict[str, str] = {
    **AUTO_PROFILE_ENV,
    "SMART_PROBE_FAR_CAP": "2000",
}

PROFILE_ENVS: Dict[str, Dict[str, str]] = {
    "final": DEFAULT_ENV,
    "auto": AUTO_PROFILE_ENV,
    "smart": SMART_PROFILE_ENV,
}


@dataclass
class Paths:
    repo: Path
    venv: Path
    tmp_root: Path
    nice: Path
    self_out: Path
    gs_out: Path
    trace_out: Path
    depth_ckpt: Path
    moge_ckpt: Path
    depthsensor_ckpt: Path
    depthsensor_gs_ckpt: Path
    sky_ckpt: Path
    python_exe: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final InfiniDepth far-depth pipeline without a shell script."
    )
    parser.add_argument("image", help="Input image path on the server.")
    parser.add_argument(
        "name",
        nargs="?",
        help="Output name. Defaults to the image file name without extension.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="InfiniDepth repo path. Defaults to this file's directory.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_ENVS),
        default=None,
        help="Configuration profile. `smart` auto-selects; `final` is for mount/vally; `auto` matches AUTO_SCENE=1 shell behavior.",
    )
    parser.add_argument("--tmp-root", default="/tmp/liuyanb-infinidepth")
    parser.add_argument("--venv", default="/tmp/liuyanb-infinidepth/venv")
    parser.add_argument("--far-cap", default=None)
    parser.add_argument("--far-min", default=None)
    parser.add_argument("--nice-cap", default=None)
    parser.add_argument("--blend", default=None, help="Far raw GS blend ratio.")
    parser.add_argument("--gs-sample-points", default=None)
    parser.add_argument("--gs-far-scale-multiplier", default=None)
    parser.add_argument("--auto-scene", dest="auto_scene", action="store_true", default=None)
    parser.add_argument("--no-auto-scene", dest="auto_scene", action="store_false")
    parser.add_argument("--render-video", dest="render_video", action="store_true", default=None)
    parser.add_argument("--no-render-video", dest="render_video", action="store_false")
    parser.add_argument(
        "--download-host",
        default="Point",
        help="SSH host name used when printing Windows PowerShell scp download commands.",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do not copy code_snapshots/nice_latest into the active source tree.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def compute_prompt_cap(cfg: Dict[str, str]) -> str:
    if (
        cfg["ALLOW_FAR_PROMPT"] == "1"
        or cfg["MOUNTAIN_PROMPT"] == "1"
        or cfg["FUSE_TWO_STAGE_FAR"] == "1"
        or cfg["FAR_RELATIVE_PROMPT"] == "1"
    ):
        return cfg["FAR_CAP"]
    return cfg["NICE_CAP"]


def make_config(args: argparse.Namespace) -> Dict[str, str]:
    profile_name = args.profile
    if profile_name is None:
        env_auto_scene = os.environ.get("AUTO_SCENE") == "1"
        profile_name = "auto" if args.auto_scene is True or env_auto_scene else "smart"

    cfg = PROFILE_ENVS[profile_name].copy()
    cfg["RUN_PROFILE"] = profile_name
    for key in list(cfg):
        if key in os.environ:
            cfg[key] = os.environ[key]

    optional = {
        "FAR_CAP": args.far_cap,
        "FAR_MIN": args.far_min,
        "NICE_CAP": args.nice_cap,
        "FAR_RAW_GS_BLEND": args.blend,
        "GS_SAMPLE_POINTS": args.gs_sample_points,
        "GS_FAR_SCALE_MULTIPLIER": args.gs_far_scale_multiplier,
    }
    for key, value in optional.items():
        if value is not None:
            cfg[key] = str(value)

    if args.auto_scene is not None:
        cfg["AUTO_SCENE"] = "1" if args.auto_scene else "0"
    if args.render_video is not None:
        cfg["RENDER_VIDEO"] = "1" if args.render_video else "0"

    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"ERROR: --env must use KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"ERROR: --env key cannot be empty: {item}")
        cfg[key] = value

    if "MOUNTAIN_RESCUE" not in cfg or cfg["MOUNTAIN_RESCUE"] == "":
        cfg["MOUNTAIN_RESCUE"] = cfg["FAR_RESCUE"]
    if "MOUNTAIN_PROMPT" not in cfg or cfg["MOUNTAIN_PROMPT"] == "":
        cfg["MOUNTAIN_PROMPT"] = cfg["FAR_RESCUE_PROMPT"]
    return cfg


def resolve_paths(args: argparse.Namespace, name: str) -> Paths:
    if args.repo:
        repo = Path(args.repo)
    else:
        repo = Path(__file__).resolve().parent

    tmp_root = Path(args.tmp_root)
    venv = Path(args.venv)
    nice = repo / "code_snapshots" / "nice_latest"
    if not nice.is_dir():
        nice = tmp_root / "code_snapshots" / "nice_latest"

    self_out = tmp_root / "outputs" / f"self_prompt_{name}_nice_prompt_compress"
    gs_out = tmp_root / "outputs" / f"gs_{name}_nice_prompt_compress"
    trace_out = self_out / "trace_stage2"
    python_exe = venv / "bin" / "python3"

    return Paths(
        repo=repo,
        venv=venv,
        tmp_root=tmp_root,
        nice=nice,
        self_out=self_out,
        gs_out=gs_out,
        trace_out=trace_out,
        depth_ckpt=tmp_root / "checkpoints" / "infinidepth.ckpt",
        moge_ckpt=tmp_root / "checkpoints" / "moge" / "model.pt",
        depthsensor_ckpt=repo / "checkpoints" / "depth" / "infinidepth_depthsensor.ckpt",
        depthsensor_gs_ckpt=repo / "checkpoints" / "gs" / "infinidepth_depthsensor_gs.ckpt",
        sky_ckpt=repo / "checkpoints" / "sky" / "skyseg.onnx",
        python_exe=python_exe,
    )


def maybe_reexec_in_venv(args: argparse.Namespace, paths: Paths) -> None:
    if args.dry_run:
        return
    if not paths.python_exe.exists():
        raise SystemExit(f"ERROR: venv python not found: {paths.python_exe}")
    current = Path(sys.executable).resolve()
    target = paths.python_exe.resolve()
    if current == target:
        return

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(paths.venv)
    env["PATH"] = f"{paths.venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    os.execve(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def prepare_runtime(paths: Paths, cfg: Dict[str, str], name: str) -> None:
    os.chdir(paths.repo)
    sys.path.insert(0, str(paths.repo))

    prompt_cap = compute_prompt_cap(cfg)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TMPDIR"] = str(paths.tmp_root / "tmp")
    depth_cap = cfg.get("SMART_PROBE_FAR_CAP", cfg["FAR_CAP"]) if cfg.get("RUN_PROFILE") == "smart" else cfg["FAR_CAP"]
    os.environ["INFINIDEPTH_MAX_DEPTH"] = os.environ.get("INFINIDEPTH_MAX_DEPTH", depth_cap)
    os.environ["INFINIDEPTH_MAX_PROMPT_DEPTH"] = os.environ.get("INFINIDEPTH_MAX_PROMPT_DEPTH", prompt_cap)

    for path in [Path(os.environ["TMPDIR"]), paths.self_out, paths.gs_out, paths.trace_out]:
        path.mkdir(parents=True, exist_ok=True)

    refresh_official_env(paths, cfg, name, depth_cap_override=depth_cap)


def refresh_official_env(
    paths: Paths,
    cfg: Dict[str, str],
    name: str,
    depth_cap_override: Optional[str] = None,
    prompt_cap_override: Optional[str] = None,
) -> None:
    prompt_cap = compute_prompt_cap(cfg)
    os.environ["INFINIDEPTH_MAX_DEPTH"] = str(depth_cap_override or cfg["FAR_CAP"])
    os.environ["INFINIDEPTH_MAX_PROMPT_DEPTH"] = str(prompt_cap_override or prompt_cap)
    os.environ["OFFICIAL_SAFE_TRACE_OUT"] = str(paths.trace_out)
    os.environ["OFFICIAL_SAFE_TRACE_NAME"] = name
    os.environ["OFFICIAL_SAFE_RELIEF_TO_GS"] = cfg["MOUNTAIN_RELIEF_TO_GS"]
    os.environ["OFFICIAL_SAFE_RELIEF_MASK_PATH"] = str(paths.self_out / f"{name}_mountain_rescue_candidate_mask.npy")
    os.environ["OFFICIAL_SAFE_RELIEF_METERS"] = cfg["MOUNTAIN_RELIEF_METERS"]
    os.environ["OFFICIAL_SAFE_RELIEF_FAR_CAP"] = cfg["FAR_CAP"]
    os.environ["OFFICIAL_SAFE_RELIEF_BLEND"] = cfg["MOUNTAIN_RELIEF_BLEND"]
    os.environ["OFFICIAL_SAFE_FAR_COMPRESS_TO_GS"] = cfg["FAR_COMPRESS_TO_GS"]
    os.environ["OFFICIAL_SAFE_FAR_COMPRESS_START"] = cfg["FAR_COMPRESS_START"]
    os.environ["OFFICIAL_SAFE_FAR_COMPRESS_RATIO"] = cfg["FAR_COMPRESS_RATIO"]
    os.environ["OFFICIAL_SAFE_FAR_COMPRESS_BLEND"] = cfg["FAR_COMPRESS_BLEND"]
    os.environ["OFFICIAL_SAFE_FAR_SHAPE_PRIOR"] = cfg["FAR_SHAPE_PRIOR"]
    os.environ["OFFICIAL_SAFE_FAR_SHAPE_MAX_METERS"] = cfg["FAR_SHAPE_MAX_METERS"]
    os.environ["OFFICIAL_SAFE_FAR_SHAPE_FLAT_RANGE"] = cfg["FAR_SHAPE_FLAT_RANGE"]
    os.environ["OFFICIAL_SAFE_FAR_RELATIVE_GAIN"] = cfg["FAR_RELATIVE_GAIN"]
    os.environ["OFFICIAL_SAFE_FAR_RELATIVE_CENTER"] = cfg["FAR_RELATIVE_CENTER"]
    if cfg["FAR_RELATIVE_PROMPT"] == "1":
        os.environ["OFFICIAL_SAFE_FAR_RELATIVE_MAP"] = str(paths.self_out / f"{name}_far_relative_mapping.npz")
    else:
        os.environ["OFFICIAL_SAFE_FAR_RELATIVE_MAP"] = ""
    if cfg["FAR_RAW_GS_OVERRIDE"] == "1":
        os.environ["OFFICIAL_SAFE_FAR_RAW_GS_DEPTH_PATH"] = str(paths.self_out / f"{name}_dense_raw.npy")
    else:
        os.environ["OFFICIAL_SAFE_FAR_RAW_GS_DEPTH_PATH"] = ""
    os.environ["OFFICIAL_SAFE_FAR_RAW_GS_MIN"] = cfg["FAR_MIN"]
    os.environ["OFFICIAL_SAFE_FAR_RAW_GS_MAX"] = cfg["FAR_CAP"]
    os.environ["OFFICIAL_SAFE_FAR_RAW_GS_BLEND"] = cfg["FAR_RAW_GS_BLEND"]
    os.environ["OFFICIAL_SAFE_FAR_RAW_GS_APPLY_DENSE"] = cfg["FAR_RAW_GS_APPLY_DENSE"]


def restore_snapshot(paths: Paths) -> None:
    print(f"[1/4] Restore nice/V29 baseline code from {paths.nice}")
    files = [
        "InfiniDepth/utils/gs_utils.py",
        "InfiniDepth/utils/sampling_utils.py",
        "InfiniDepth/model/model.py",
        "inference_gs.py",
    ]
    for rel in files:
        src = paths.nice / rel
        dst = paths.repo / rel
        if not src.exists():
            raise FileNotFoundError(f"missing snapshot file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"restored {rel}")


def compile_files(paths: Paths) -> None:
    for rel in [
        "InfiniDepth/utils/gs_utils.py",
        "InfiniDepth/utils/sampling_utils.py",
        "InfiniDepth/model/model.py",
        "inference_gs.py",
    ]:
        py_compile.compile(str(paths.repo / rel), doraise=True)


def run_dense_depth(image_path: str, paths: Paths, name: str) -> None:
    print("[2/4] First-pass InfiniDepth dense depth")
    from inference_depth import DepthInferenceArgs, load_depth_model, run_depth_inference
    from InfiniDepth.utils.io_utils import plot_depth, save_depth_array

    paths.self_out.mkdir(parents=True, exist_ok=True)
    args = DepthInferenceArgs(
        input_image_path=image_path,
        model_type="InfiniDepth",
        depth_model_path=str(paths.depth_ckpt),
        moge2_pretrained=str(paths.moge_ckpt),
        input_size=(768, 1024),
        output_size=(768, 1024),
        output_resolution_mode="specific",
        save_pcd=False,
        enable_skyseg_model=False,
    )
    model, device = load_depth_model(args)
    result = run_depth_inference(args, model=model, device=device)

    save_depth_array(result.pred_depthmap, str(paths.self_out / f"{name}_dense.npy"))
    plot_depth(result.org_img, result.pred_depthmap, str(paths.self_out / f"{name}_dense_vis.png"))
    print("saved dense:", paths.self_out / f"{name}_dense.npy")
    print("saved dense vis:", paths.self_out / f"{name}_dense_vis.png")


def apply_adaptive_policy(paths: Paths, cfg: Dict[str, str], name: str) -> None:
    if cfg["AUTO_SCENE"] != "1":
        print("[adaptive] disabled by AUTO_SCENE=0")
        return

    print("[2.5/4] Adaptive scene/depth policy")
    import numpy as np

    dense_path = paths.self_out / f"{name}_dense.npy"
    depth = np.load(dense_path).astype(np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D dense depth, got {depth.shape}")

    h, _ = depth.shape
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    valid_all = np.isfinite(depth) & (depth > 1.0)
    valid_scene = valid_all & (y >= 0.12) & (y <= 0.88)
    if int(valid_scene.sum()) < 512:
        valid_scene = valid_all
    if int(valid_scene.sum()) < 256:
        raise RuntimeError(f"Too few valid depth pixels for adaptive policy: {int(valid_scene.sum())}")

    vals = depth[valid_scene].astype(np.float32)
    p50, p85, p95, p98, p99, p995 = [float(v) for v in np.percentile(vals, [50, 85, 95, 98, 99, 99.5])]
    far220 = float(np.mean(vals >= 220.0))
    far500 = float(np.mean(vals >= 500.0))
    far1000 = float(np.mean(vals >= 1000.0))

    def nice_round(value: float, step: float = 10.0) -> int:
        return int(max(step, math.ceil(float(value) / step) * step))

    settings: Dict[str, object] = {}
    if p98 < 90.0:
        scene = "close"
        cap = nice_round(max(80.0, min(140.0, p99 * 1.25)))
        settings.update(
            NICE_CAP=cap,
            FAR_MIN=cap,
            FAR_CAP=cap,
            NICE_TARGET=28000,
            MOUNTAIN_RESCUE=0,
            MOUNTAIN_PROMPT=0,
            ALLOW_FAR_PROMPT=0,
            PROMPT_COMPRESS_DEPTH=0,
            GS_SAMPLE_POINTS=1800000,
            GS_FAR_SCALE_MULTIPLIER=1.0,
        )
    elif p99 < 280.0 and far500 < 0.01:
        scene = "near_mid"
        cap = nice_round(max(180.0, min(240.0, p99 * 1.10)))
        settings.update(
            NICE_CAP=cap,
            FAR_MIN=min(180, cap),
            FAR_CAP=cap,
            NICE_TARGET=36000,
            MOUNTAIN_RESCUE=0,
            MOUNTAIN_PROMPT=0,
            ALLOW_FAR_PROMPT=0,
            PROMPT_COMPRESS_DEPTH=1,
            PROMPT_COMPRESS_START=100,
            PROMPT_COMPRESS_RATIO=0.40,
            PROMPT_COMPRESS_MAX=min(170, max(140, cap - 20)),
            GS_SAMPLE_POINTS=2200000,
            GS_FAR_SCALE_MULTIPLIER=1.0,
        )
    elif p99 < 750.0 and far1000 < 0.01:
        scene = "wide_mid"
        cap = nice_round(max(300.0, min(700.0, p995 * 1.12)))
        settings.update(
            NICE_CAP=220,
            FAR_MIN=180,
            FAR_CAP=cap,
            NICE_TARGET=42000,
            MOUNTAIN_RESCUE="auto",
            MOUNTAIN_PROMPT=1,
            MOUNTAIN_RESCUE_TARGET=18000,
            ALLOW_FAR_PROMPT=0,
            PROMPT_COMPRESS_DEPTH=1,
            PROMPT_COMPRESS_START=100,
            PROMPT_COMPRESS_RATIO=0.35,
            PROMPT_COMPRESS_MAX=170,
            GS_SAMPLE_POINTS=3000000,
            GS_FAR_SCALE_MULTIPLIER=0.85,
        )
    elif p99 < 1500.0:
        scene = "far"
        cap = nice_round(max(900.0, min(1500.0, p995 * 1.08)), step=50.0)
        settings.update(
            NICE_CAP=220,
            FAR_MIN=180,
            FAR_CAP=cap,
            NICE_TARGET=46000,
            MOUNTAIN_RESCUE="auto",
            MOUNTAIN_PROMPT=1,
            MOUNTAIN_RESCUE_TARGET=30000,
            ALLOW_FAR_PROMPT=0,
            PROMPT_COMPRESS_DEPTH=1,
            PROMPT_COMPRESS_START=100,
            PROMPT_COMPRESS_RATIO=0.30,
            PROMPT_COMPRESS_MAX=180,
            GS_SAMPLE_POINTS=4000000,
            GS_FAR_SCALE_MULTIPLIER=0.75,
        )
    else:
        scene = "extreme_far"
        cap = nice_round(max(1600.0, min(2200.0, p995 * 1.05)), step=100.0)
        settings.update(
            NICE_CAP=220,
            FAR_MIN=180,
            FAR_CAP=cap,
            NICE_TARGET=50000,
            MOUNTAIN_RESCUE="auto",
            MOUNTAIN_PROMPT=1,
            MOUNTAIN_RESCUE_TARGET=42000,
            ALLOW_FAR_PROMPT=0,
            PROMPT_COMPRESS_DEPTH=1,
            PROMPT_COMPRESS_START=100,
            PROMPT_COMPRESS_RATIO=0.25,
            PROMPT_COMPRESS_MAX=180,
            GS_SAMPLE_POINTS=5000000,
            GS_FAR_SCALE_MULTIPLIER=0.65,
        )

    if cfg.get("RUN_PROFILE") == "smart":
        # The smart profile starts from the original AUTO_SCENE policy, then
        # switches only very deep scenes to the validated mount/vally far-GS
        # override. Normal campus-like scenes keep the old AUTO_SCENE behavior.
        use_far_raw_gs = scene == "extreme_far" or p995 >= 1500.0 or far1000 >= 0.02
        if use_far_raw_gs:
            settings.update(
                NICE_CAP=220,
                FAR_MIN=180,
                FAR_CAP=2000,
                MOUNTAIN_RESCUE=0,
                MOUNTAIN_PROMPT=0,
                FUSE_TWO_STAGE_FAR=0,
                FAR_RELATIVE_PROMPT=0,
                FAR_RAW_GS_OVERRIDE=1,
                FAR_RAW_GS_BLEND=1.0,
                FAR_RAW_GS_APPLY_DENSE=0,
                PROMPT_COMPRESS_DEPTH=1,
                PROMPT_COMPRESS_START=100,
                PROMPT_COMPRESS_RATIO=0.35,
                PROMPT_COMPRESS_MAX=180,
                GS_SAMPLE_POINTS=6000000,
                GS_FAR_SCALE_MULTIPLIER=0.70,
            )
            settings["SMART_STAGE"] = "far_raw_gs"
        else:
            settings.update(
                FAR_RAW_GS_OVERRIDE=0,
                FAR_RAW_GS_APPLY_DENSE=0,
            )
            settings["SMART_STAGE"] = "auto_scene"

    settings["AUTO_SCENE_LABEL"] = scene
    settings["AUTO_DEPTH_P99"] = f"{p99:.3f}"
    settings["AUTO_DEPTH_P995"] = f"{p995:.3f}"
    settings["AUTO_FAR500_RATIO"] = f"{far500:.6f}"

    for key, value in settings.items():
        cfg[key] = str(value)

    env_path = paths.self_out / f"{name}_adaptive_scene.env"
    env_path.write_text(
        "\n".join(f'export {key}="{str(value).replace(chr(34), chr(92) + chr(34))}"' for key, value in settings.items()) + "\n",
        encoding="utf-8",
    )
    print(
        "[adaptive]",
        f"scene={scene}",
        f"p50={p50:.2f}",
        f"p85={p85:.2f}",
        f"p95={p95:.2f}",
        f"p99={p99:.2f}",
        f"p99.5={p995:.2f}",
        f"far>=220={far220:.4f}",
        f"far>=500={far500:.4f}",
        f"far>=1000={far1000:.4f}",
    )
    print("[adaptive] env:", env_path)
    for key in sorted(settings):
        print(f"[adaptive] {key}={settings[key]}")


def build_sparse_prompt(image_path: str, paths: Paths, cfg: Dict[str, str], name: str) -> None:
    print("[3/4] Build nice-priority sparse prompt, then fill far blank only")
    import numpy as np
    from PIL import Image

    img_path = Path(image_path)
    dense_path = paths.self_out / f"{name}_dense.npy"
    out_dir = paths.self_out
    out_dir.mkdir(parents=True, exist_ok=True)

    nice_cap = float(cfg["NICE_CAP"])
    far_min = float(cfg["FAR_MIN"])
    far_cap = float(cfg["FAR_CAP"])
    nice_target = int(cfg["NICE_TARGET"])
    nice_per_tile = int(cfg["NICE_PER_TILE"])
    nice_tile = int(cfg["NICE_TILE"])
    far_target = int(cfg["FAR_TARGET"])
    far_tile = int(cfg["FAR_TILE"])
    far_per_tile = int(cfg["FAR_PER_TILE"])
    sky_dilate = int(cfg["SKY_DILATE"])
    allow_far_prompt = cfg["ALLOW_FAR_PROMPT"] == "1"
    mountain_prompt = cfg["MOUNTAIN_PROMPT"] == "1"
    mountain_protect_nice_px = int(cfg["MOUNTAIN_PROTECT_NICE_PX"])
    mountain_rescue = cfg["MOUNTAIN_RESCUE"]
    mountain_rescue_target = int(cfg["MOUNTAIN_RESCUE_TARGET"])
    mountain_rescue_tile = int(cfg["MOUNTAIN_RESCUE_TILE"])
    mountain_rescue_per_tile = int(cfg["MOUNTAIN_RESCUE_PER_TILE"])
    mountain_y_min = float(cfg["MOUNTAIN_RESCUE_Y_MIN"])
    mountain_y_max = float(cfg["MOUNTAIN_RESCUE_Y_MAX"])
    far_rescue_foreground_y_max = float(cfg["FAR_RESCUE_FOREGROUND_Y_MAX"])
    mountain_object_uniform = int(cfg["MOUNTAIN_OBJECT_UNIFORM"]) != 0
    mountain_uniform_spacing = int(cfg["MOUNTAIN_UNIFORM_SPACING"])
    mountain_relief = int(cfg["MOUNTAIN_RELIEF"]) != 0
    mountain_relief_meters = float(cfg["MOUNTAIN_RELIEF_METERS"])
    prompt_compress_depth = int(cfg["PROMPT_COMPRESS_DEPTH"]) != 0
    prompt_compress_start = float(cfg["PROMPT_COMPRESS_START"])
    prompt_compress_ratio = float(cfg["PROMPT_COMPRESS_RATIO"])
    prompt_compress_max = float(cfg["PROMPT_COMPRESS_MAX"])
    fuse_two_stage_far = cfg["FUSE_TWO_STAGE_FAR"] == "1"
    fuse_far_target = int(cfg["FUSE_FAR_TARGET"])
    fuse_far_overwrite = int(cfg["FUSE_FAR_OVERWRITE"]) != 0
    fuse_far_use_raw = int(cfg["FUSE_FAR_USE_RAW"]) != 0
    far_relative_prompt = cfg["FAR_RELATIVE_PROMPT"] == "1"
    far_relative_prompt_min = float(cfg["FAR_RELATIVE_PROMPT_MIN"])
    far_relative_prompt_max = float(cfg["FAR_RELATIVE_PROMPT_MAX"])
    far_relative_q_low = float(cfg["FAR_RELATIVE_Q_LOW"])
    far_relative_q_high = float(cfg["FAR_RELATIVE_Q_HIGH"])

    depth = np.load(dense_path).astype(np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D dense depth, got {depth.shape}")
    h, w = depth.shape

    raw_depth = depth.copy()
    prompt_depth = raw_depth.copy()
    if prompt_compress_depth:
        prompt_compress_ratio = float(np.clip(prompt_compress_ratio, 0.02, 1.0))
        far = raw_depth > prompt_compress_start
        compressed = prompt_compress_start + np.maximum(raw_depth - prompt_compress_start, 0.0) * prompt_compress_ratio
        if prompt_compress_max > prompt_compress_start:
            compressed = np.minimum(compressed, prompt_compress_max)
        prompt_depth[far] = compressed[far]

    np.save(out_dir / f"{name}_dense_raw.npy", raw_depth)
    np.save(out_dir / f"{name}_dense_prompt_compressed.npy", prompt_depth)

    def save_depth_vis(path: Path, arr: "np.ndarray") -> None:
        valid = np.isfinite(arr) & (arr > 0)
        vis = np.zeros(arr.shape, dtype=np.uint8)
        if valid.any():
            lo, hi = np.percentile(arr[valid], [2, 98])
            vis = (np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
            vis[~valid] = 0
        Image.fromarray(vis).save(path)

    save_depth_vis(out_dir / f"{name}_dense_raw_vis.png", raw_depth)
    save_depth_vis(out_dir / f"{name}_dense_prompt_compressed_vis.png", prompt_depth)
    print(
        "[Info] prompt depth compression:",
        f"enabled={prompt_compress_depth}",
        f"start={prompt_compress_start:.1f}",
        f"ratio={prompt_compress_ratio:.3f}",
        f"max={prompt_compress_max:.1f}",
        f"raw_minmax={float(np.nanmin(raw_depth)):.3f}/{float(np.nanmax(raw_depth)):.3f}",
        f"prompt_minmax={float(np.nanmin(prompt_depth)):.3f}/{float(np.nanmax(prompt_depth)):.3f}",
    )

    img = Image.open(img_path).convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
    rgb_u8 = np.asarray(img)
    rgb = rgb_u8.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    cv2 = None
    sky = np.zeros((h, w), dtype=bool)
    if paths.sky_ckpt.exists():
        try:
            import cv2 as _cv2
            import torch
            from InfiniDepth.utils.vis_utils import build_sky_model, run_skyseg

            cv2 = _cv2
            image_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
            sky_model = build_sky_model(str(paths.sky_ckpt))
            sky_raw = run_skyseg(sky_model, input_size=(320, 320), image=image_t)
            sky_raw = cv2.resize(sky_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            sky = sky_raw > 127

            mx = rgb.max(axis=2)
            mn = rgb.min(axis=2)
            sat = (mx - mn) / (mx + 1e-6)
            gx_detail = np.zeros_like(gray)
            gy_detail = np.zeros_like(gray)
            gx_detail[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
            gy_detail[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
            edge = gx_detail + gy_detail
            detail_rescue = (edge > np.percentile(edge, 98.5)) & (sat > 0.055)
            sky[detail_rescue] = False

            if sky_dilate > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sky_dilate + 1, 2 * sky_dilate + 1))
                sky = cv2.dilate(sky.astype("uint8"), kernel, iterations=1).astype(bool)
        except Exception as exc:
            print(f"[Warning] skyseg failed, continue without sky mask: {exc}")
            sky = np.zeros((h, w), dtype=bool)
    else:
        print(f"[Warning] sky checkpoint not found: {paths.sky_ckpt}; continue without sky mask")

    Image.fromarray((sky.astype(np.uint8) * 255)).save(out_dir / f"{name}_sky_mask.png")
    np.save(out_dir / f"{name}_sky_mask.npy", sky)

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

    relative_prompt_depth = prompt_depth.copy()
    far_prompt_source = prompt_depth
    relative_far_region = np.zeros_like(depth, dtype=bool)
    if far_relative_prompt:
        relative_far_region = np.isfinite(raw_depth) & (raw_depth >= far_min) & (raw_depth < far_cap) & (~sky)
        if int(relative_far_region.sum()) > 32:
            raw_vals = raw_depth[relative_far_region].astype(np.float32)
            raw_lo, raw_hi = np.percentile(raw_vals, [far_relative_q_low, far_relative_q_high])
            raw_lo = float(max(raw_lo, 1e-3))
            raw_hi = float(max(raw_hi, raw_lo + 1e-3))
            prompt_lo = float(min(far_relative_prompt_min, far_relative_prompt_max))
            prompt_hi = float(max(far_relative_prompt_min, far_relative_prompt_max))
            log_lo = float(np.log(raw_lo))
            log_hi = float(np.log(raw_hi))
            log_depth = np.log(np.clip(raw_depth, 1e-3, None))
            norm = np.clip((log_depth - log_lo) / (log_hi - log_lo + 1e-6), 0.0, 1.0)
            relative_prompt_depth[relative_far_region] = prompt_lo + norm[relative_far_region] * (prompt_hi - prompt_lo)
            far_prompt_source = relative_prompt_depth
            np.savez_compressed(
                out_dir / f"{name}_far_relative_mapping.npz",
                mask=relative_far_region.astype(np.uint8),
                raw_log_lo=np.float32(log_lo),
                raw_log_hi=np.float32(log_hi),
                prompt_min=np.float32(prompt_lo),
                prompt_max=np.float32(prompt_hi),
            )
            save_depth_vis(out_dir / f"{name}_far_relative_prompt_vis.png", relative_prompt_depth)
            print(
                "[Info] far relative prompt:",
                f"enabled={far_relative_prompt}",
                f"pixels={int(relative_far_region.sum())}",
                f"raw_range={raw_lo:.3f}-{raw_hi:.3f}",
                f"prompt_range={prompt_lo:.3f}-{prompt_hi:.3f}",
            )
        else:
            print(f"[Info] far relative prompt disabled: too few far pixels ({int(relative_far_region.sum())})")

    valid_nice = np.isfinite(depth) & (depth > 1.0) & (depth < nice_cap) & (~sky)
    if int(valid_nice.sum()) < 256:
        raise RuntimeError(f"Too few valid nice-depth pixels: {int(valid_nice.sum())}")

    stable = valid_nice.copy()
    stable &= depth_edge < np.quantile(depth_edge[valid_nice], 0.82)
    stable &= rgb_edge < np.quantile(rgb_edge[valid_nice], 0.92)

    def tile_limited_sample(mask: "np.ndarray", target: int, per_tile: int, tile: int, rng: "np.random.Generator") -> "np.ndarray":
        selected: List[int] = []
        if target <= 0 or per_tile <= 0:
            return np.array([], dtype=np.int64)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                block = mask[y0:y0 + tile, x0:x0 + tile]
                idx = np.flatnonzero(block.reshape(-1))
                if idx.size == 0:
                    continue
                take_n = min(per_tile, idx.size)
                take = rng.choice(idx, size=take_n, replace=False)
                yy = y0 + take // block.shape[1]
                xx = x0 + take % block.shape[1]
                selected.extend((yy * w + xx).tolist())
        if len(selected) > target:
            selected = rng.choice(np.array(selected, dtype=np.int64), size=target, replace=False).tolist()
        return np.array(selected, dtype=np.int64)

    def object_uniform_sample(mask: "np.ndarray", target: int, spacing: int, rng: "np.random.Generator") -> "np.ndarray":
        if target <= 0:
            return np.array([], dtype=np.int64)
        count = int(mask.sum())
        if count <= 0:
            return np.array([], dtype=np.int64)
        if spacing <= 0:
            spacing = max(2, int(np.ceil(np.sqrt(float(count) / float(max(1, target))))))

        selected: List[int] = []
        for y0 in range(0, h, spacing):
            for x0 in range(0, w, spacing):
                block = mask[y0:y0 + spacing, x0:x0 + spacing]
                idx = np.flatnonzero(block.reshape(-1))
                if idx.size == 0:
                    continue
                yy = y0 + idx // block.shape[1]
                xx = x0 + idx % block.shape[1]
                cy = y0 + 0.5 * (block.shape[0] - 1)
                cx = x0 + 0.5 * (block.shape[1] - 1)
                nearest = int(np.argmin((yy - cy) ** 2 + (xx - cx) ** 2))
                selected.append(int(yy[nearest] * w + xx[nearest]))

        selected_arr = np.array(selected, dtype=np.int64)
        if selected_arr.size > target:
            keep = np.linspace(0, selected_arr.size - 1, target).round().astype(np.int64)
            selected_arr = selected_arr[keep]
        elif selected_arr.size < target:
            fill_mask = mask.copy()
            fill_mask.reshape(-1)[selected_arr] = False
            fill = tile_limited_sample(
                fill_mask,
                target - int(selected_arr.size),
                max(1, mountain_rescue_per_tile // 2),
                max(2, mountain_rescue_tile),
                rng,
            )
            if fill.size:
                selected_arr = np.concatenate([selected_arr, fill])
        return selected_arr

    rng = np.random.default_rng(20260630)
    nice_sparse = np.zeros_like(depth, dtype=np.float32)
    bins = [
        (1, 10, 0.12),
        (10, 25, 0.18),
        (25, 55, 0.22),
        (55, 90, 0.22),
        (90, 120, 0.16),
        (120, nice_cap, 0.10),
    ]
    for lo, hi, frac in bins:
        n = max(1, int(round(nice_target * frac)))
        mask = stable & (depth >= lo) & (depth < hi)
        idx = tile_limited_sample(mask, n, nice_per_tile, nice_tile, rng)
        if idx.size:
            nice_sparse.reshape(-1)[idx] = prompt_depth.reshape(-1)[idx]

    cur = int((nice_sparse > 0).sum())
    if cur < nice_target:
        remain = valid_nice & (nice_sparse <= 0)
        idx = tile_limited_sample(remain, nice_target - cur, max(1, nice_per_tile // 2), nice_tile, rng)
        if idx.size:
            nice_sparse.reshape(-1)[idx] = prompt_depth.reshape(-1)[idx]

    far_valid = np.isfinite(depth) & (depth >= far_min) & (depth < far_cap) & (~sky) & (nice_sparse <= 0)
    far_sparse = np.zeros_like(depth, dtype=np.float32)
    if far_valid.any() and far_target > 0:
        edge_base = far_valid
        de_thr = np.quantile(depth_edge[edge_base], 0.50) if int(edge_base.sum()) > 16 else 0.0
        re_thr = np.quantile(rgb_edge[edge_base], 0.55) if int(edge_base.sum()) > 16 else 0.0
        far_candidate = far_valid & ((depth_edge >= de_thr) | (rgb_edge >= re_thr))
        if int(far_candidate.sum()) < min(1024, int(far_valid.sum()) // 4):
            far_candidate = far_valid

        selected: List[int] = []
        for y0 in range(0, h, far_tile):
            for x0 in range(0, w, far_tile):
                tile = far_candidate[y0:y0 + far_tile, x0:x0 + far_tile]
                idx = np.flatnonzero(tile.reshape(-1))
                if idx.size == 0:
                    continue
                take_n = min(far_per_tile, idx.size)
                take = rng.choice(idx, size=take_n, replace=False)
                yy = y0 + take // tile.shape[1]
                xx = x0 + take % tile.shape[1]
                selected.extend((yy * w + xx).tolist())

        if len(selected) > far_target:
            selected = rng.choice(np.array(selected, dtype=np.int64), size=far_target, replace=False).tolist()
        if selected:
            selected_arr = np.array(selected, dtype=np.int64)
            far_sparse.reshape(-1)[selected_arr] = far_prompt_source.reshape(-1)[selected_arr]

    fused_far_sparse = np.zeros_like(depth, dtype=np.float32)
    if fuse_two_stage_far and fuse_far_target > 0:
        fuse_source = far_prompt_source if far_relative_prompt and int(relative_far_region.sum()) > 32 else (raw_depth if fuse_far_use_raw else prompt_depth)
        fuse_valid = np.isfinite(depth) & (depth >= far_min) & (depth < far_cap) & (~sky) & (nice_sparse <= 0)
        if int(fuse_valid.sum()) > 0:
            fuse_stable = fuse_valid.copy()
            if int(fuse_valid.sum()) > 32:
                fuse_stable &= depth_edge < np.quantile(depth_edge[fuse_valid], 0.92)
                fuse_stable &= rgb_edge < np.quantile(rgb_edge[fuse_valid], 0.97)
            if int(fuse_stable.sum()) < min(1024, int(fuse_valid.sum()) // 3):
                fuse_stable = fuse_valid

            start = float(max(1.0, far_min))
            far_edges = [start, 220.0, 350.0, 500.0, 750.0, 1000.0, 1300.0, 1600.0, float(far_cap)]
            far_edges = sorted(set(edge for edge in far_edges if start <= edge <= far_cap))
            if far_edges[-1] < far_cap:
                far_edges.append(float(far_cap))
            far_ranges = [(far_edges[i], far_edges[i + 1]) for i in range(len(far_edges) - 1) if far_edges[i + 1] > far_edges[i]]
            if far_ranges:
                far_weights = np.linspace(1.0, 1.8, len(far_ranges), dtype=np.float32)
                far_counts = np.maximum(1, np.floor(far_weights / far_weights.sum() * fuse_far_target).astype(np.int64))
                far_counts[-1] += int(fuse_far_target - far_counts.sum())
                far_counts = np.maximum(1, far_counts)
                flat_fused = fused_far_sparse.reshape(-1)
                flat_source = fuse_source.reshape(-1)
                for (lo, hi), n in zip(far_ranges, far_counts):
                    mask = fuse_stable & (depth >= lo) & (depth < hi)
                    idx = np.flatnonzero(mask.reshape(-1))
                    if idx.size == 0:
                        continue
                    take = rng.choice(idx, size=min(int(n), idx.size), replace=False)
                    flat_fused[take] = flat_source[take]

            current_fused = int((fused_far_sparse > 0).sum())
            if current_fused < fuse_far_target:
                remain = fuse_valid & (fused_far_sparse <= 0)
                idx = np.flatnonzero(remain.reshape(-1))
                if idx.size > 0:
                    take = rng.choice(idx, size=min(fuse_far_target - current_fused, idx.size), replace=False)
                    fused_far_sparse.reshape(-1)[take] = fuse_source.reshape(-1)[take]
        print(
            "[Info] fused two-stage far sparse:",
            f"enabled={fuse_two_stage_far}",
            f"use_raw={fuse_far_use_raw}",
            f"overwrite={fuse_far_overwrite}",
            f"valid={int(fuse_valid.sum()) if 'fuse_valid' in locals() else 0}",
            f"selected={int((fused_far_sparse > 0).sum())}",
            f"target={fuse_far_target}",
        )

    final_sparse = nice_sparse.copy()
    mountain_sparse = np.zeros_like(depth, dtype=np.float32)
    explicit_far_rescue = mountain_rescue.lower() in ("1", "true", "yes", "on")
    auto_far_rescue = mountain_rescue.lower() == "auto"
    enable_mountain = explicit_far_rescue or auto_far_rescue
    if enable_mountain and mountain_rescue_target > 0:
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2
        y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        y_band = (y_norm >= mountain_y_min) & (y_norm <= mountain_y_max)
        far_or_saturated = np.isfinite(depth) & (depth >= min(far_min, nice_cap * 0.95)) & (depth <= far_cap + 1e-3)
        mountain_candidate = far_or_saturated & y_band & (~sky) & (nice_sparse <= 0)
        if mountain_protect_nice_px > 0 and int((nice_sparse > 0).sum()) > 0:
            k = 2 * mountain_protect_nice_px + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            nice_protect = cv2.dilate((nice_sparse > 0).astype(np.uint8), kernel, iterations=1) > 0
            mountain_candidate &= ~nice_protect
        mountain_candidate &= ~(y_norm > far_rescue_foreground_y_max)
        candidate_pixels = int(mountain_candidate.sum())
        if auto_far_rescue:
            min_pixels = max(2500, int(0.004 * h * w))
            if candidate_pixels < min_pixels:
                enable_mountain = False
                mountain_candidate[:] = False
                candidate_pixels = 0
                print("[Info] far rescue auto disabled:", f"candidate_pixels={candidate_pixels}", f"threshold={min_pixels}")
        if not enable_mountain:
            idx = np.array([], dtype=np.int64)
        elif mountain_object_uniform:
            idx = object_uniform_sample(mountain_candidate, mountain_rescue_target, mountain_uniform_spacing, rng)
        else:
            idx = tile_limited_sample(mountain_candidate, mountain_rescue_target, mountain_rescue_per_tile, mountain_rescue_tile, rng)
        if idx.size:
            mountain_values = far_prompt_source.reshape(-1)[idx].astype(np.float32).copy()
            if mountain_relief and int(mountain_candidate.sum()) > 32:
                tex = rgb_edge.astype(np.float32)
                tex_lo, tex_hi = np.percentile(tex[mountain_candidate], [10, 95])
                tex_norm = np.clip((tex - tex_lo) / (tex_hi - tex_lo + 1e-6), 0.0, 1.0)
                gray_lo, gray_hi = np.percentile(gray[mountain_candidate], [5, 95])
                gray_norm = np.clip((gray - gray_lo) / (gray_hi - gray_lo + 1e-6), 0.0, 1.0)
                yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
                yy = np.broadcast_to(yy, (h, w))
                high_valid = np.isfinite(depth) & (depth > 1.0) & (depth <= far_cap + 1e-3) & (~sky)
                d_hi = float(np.percentile(depth[high_valid], 99.9)) if int(high_valid.sum()) else far_cap
                d_hi = min(float(far_cap), max(float(nice_cap + 1.0), d_hi))
                relief = 0.60 * tex_norm + 0.25 * (1.0 - gray_norm) + 0.15 * (1.0 - yy)
                pseudo_depth = d_hi - mountain_relief_meters * relief
                pseudo_depth = np.clip(pseudo_depth, max(1.0, d_hi - mountain_relief_meters), d_hi - 0.5).astype(np.float32)
                high_selected = mountain_values >= min(far_min, d_hi - 0.5)
                if high_selected.any():
                    pseudo_values = pseudo_depth.reshape(-1)[idx]
                    mountain_values[high_selected] = pseudo_values[high_selected]
                    print(
                        "[Info] mountain relief applied:",
                        int(high_selected.sum()),
                        f"d_hi={d_hi:.3f}",
                        f"meters={mountain_relief_meters:.1f}",
                        f"range={float(mountain_values.min()):.3f}-{float(mountain_values.max()):.3f}",
                    )
            mountain_sparse.reshape(-1)[idx] = mountain_values
        print(
            "[Info] mountain object-uniform sampling:",
            f"enabled={mountain_object_uniform}",
            f"candidate_pixels={int(mountain_candidate.sum())}",
            f"selected={int(idx.size)}",
            f"spacing={mountain_uniform_spacing if mountain_uniform_spacing > 0 else 'auto'}",
        )
        np.save(out_dir / f"{name}_mountain_rescue_candidate_mask.npy", mountain_candidate)
        Image.fromarray((mountain_candidate.astype(np.uint8) * 255)).save(out_dir / f"{name}_mountain_rescue_candidate_mask.png")

    if allow_far_prompt:
        final_sparse[(final_sparse <= 0) & (far_sparse > 0)] = far_sparse[(final_sparse <= 0) & (far_sparse > 0)]
        print("[Warning] ALLOW_FAR_PROMPT=1: far diagnostics are merged into prompt.")
    else:
        print("[Info] official-safe: far diagnostics are NOT merged into prompt.")

    if enable_mountain and mountain_prompt:
        final_sparse[(final_sparse <= 0) & (mountain_sparse > 0)] = mountain_sparse[(final_sparse <= 0) & (mountain_sparse > 0)]
        print(f"[Info] mountain rescue enabled: added {int((mountain_sparse > 0).sum())} prompt points")
    elif enable_mountain:
        print(f"[Info] mountain rescue mask enabled for GS relief only: {int((mountain_sparse > 0).sum())} candidate points, prompt unchanged")
    else:
        print("[Info] mountain rescue disabled")

    if fuse_two_stage_far:
        fuse_mask = fused_far_sparse > 0
        if fuse_far_overwrite:
            fuse_mask &= depth >= far_min
            final_sparse[fuse_mask] = fused_far_sparse[fuse_mask]
            print(f"[Info] fused two-stage far overwritten into prompt: {int(fuse_mask.sum())} points")
        else:
            fuse_mask &= final_sparse <= 0
            final_sparse[fuse_mask] = fused_far_sparse[fuse_mask]
            print(f"[Info] fused two-stage far filled into prompt: {int(fuse_mask.sum())} points")

    saturated_far = np.isfinite(depth) & (depth >= far_min) & (~sky)
    np.save(out_dir / f"{name}_rejected_far_or_saturated_mask.npy", saturated_far)
    Image.fromarray((saturated_far.astype(np.uint8) * 255)).save(out_dir / f"{name}_rejected_far_or_saturated_mask.png")

    np.save(out_dir / f"{name}_nice_sparse.npy", nice_sparse)
    np.save(out_dir / f"{name}_far_fill_only.npy", far_sparse)
    np.save(out_dir / f"{name}_fused_two_stage_far_sparse.npy", fused_far_sparse)
    np.save(out_dir / f"{name}_mountain_rescue_sparse.npy", mountain_sparse)
    np.save(out_dir / f"{name}_nice_prompt_compress_sparse.npy", final_sparse)

    def save_preview(path: Path, sparse: "np.ndarray", color: Iterable[int]) -> None:
        valid = np.isfinite(depth) & (depth > 0)
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
        vis = np.repeat((norm[..., None] * 255).astype(np.uint8), 3, axis=2)
        vis[sky] = [0, 0, 255]
        vis[sparse > 0] = list(color)
        Image.fromarray(vis).save(path)

    save_preview(out_dir / f"{name}_nice_sparse_preview.png", nice_sparse, [255, 0, 0])
    save_preview(out_dir / f"{name}_far_fill_only_preview.png", far_sparse, [0, 255, 0])
    save_preview(out_dir / f"{name}_fused_two_stage_far_preview.png", fused_far_sparse, [0, 255, 255])
    save_preview(out_dir / f"{name}_mountain_rescue_preview.png", mountain_sparse, [255, 255, 0])
    save_preview(out_dir / f"{name}_final_sparse_preview.png", final_sparse, [255, 0, 0])

    print("dense shape:", depth.shape)
    print("sky pixels:", int(sky.sum()))
    print("nice sparse:", int((nice_sparse > 0).sum()))
    print("far fill only:", int((far_sparse > 0).sum()))
    print("fused two-stage far:", int((fused_far_sparse > 0).sum()))
    print("mountain rescue:", int((mountain_sparse > 0).sum()))
    print("final sparse:", int((final_sparse > 0).sum()))
    print("allow far prompt:", allow_far_prompt)
    print("mountain prompt:", mountain_prompt)
    print("mountain protect nice px:", mountain_protect_nice_px)
    print("mountain rescue mode:", mountain_rescue, "enabled:", enable_mountain)
    print("saved final sparse:", out_dir / f"{name}_nice_prompt_compress_sparse.npy")


def run_second_pass_gs(image_path: str, paths: Paths, cfg: Dict[str, str], name: str) -> None:
    print("[4/4] Second-pass DepthSensor GS")
    command = [
        sys.executable,
        "inference_gs.py",
        "--input-image-path",
        image_path,
        "--input-depth-path",
        str(paths.self_out / f"{name}_nice_prompt_compress_sparse.npy"),
        "--model-type=InfiniDepth_DepthSensor",
        "--depth-model-path",
        str(paths.depthsensor_ckpt),
        "--gs-model-path",
        str(paths.depthsensor_gs_ckpt),
        "--moge2-pretrained",
        str(paths.moge_ckpt),
        "--output-ply-dir",
        str(paths.gs_out),
        "--output-ply-name",
        f"{name}_nice_prompt_compress.ply",
        "--sample-point-num",
        cfg["GS_SAMPLE_POINTS"],
        "--max-prompt-depth",
        os.environ["INFINIDEPTH_MAX_PROMPT_DEPTH"],
        "--gs-max-sample-depth",
        os.environ["INFINIDEPTH_MAX_DEPTH"],
        "--gs-sample-filter-mode=max_depth",
        "--gs-far-scale-min-depth=80",
        "--gs-far-scale-multiplier",
        cfg["GS_FAR_SCALE_MULTIPLIER"],
        "--sample-sky-mask-dilate-px",
        cfg["SKY_DILATE"],
    ]
    if cfg["RENDER_VIDEO"] == "1":
        command.extend(
            [
                "--render-novel-video",
                f"--novel-video-path={paths.gs_out / f'{name}_nice_prompt_compress_novel.mp4'}",
                "--novel-trajectory=orbit",
                "--novel-num-frames=120",
                "--novel-video-fps=30",
                "--render-size",
                "720",
                "1280",
            ]
        )
    else:
        command.append("--no-render-novel-video")

    subprocess.run(command, cwd=str(paths.repo), env=os.environ.copy(), check=True)


def print_outputs(paths: Paths, name: str, download_host: str, render_video: bool) -> None:
    print("")
    print("Done.")
    print("Outputs:")
    output_paths = [
        paths.self_out / f"{name}_dense_vis.png",
        paths.self_out / f"{name}_dense_prompt_compressed_vis.png",
        paths.self_out / f"{name}_far_relative_prompt_vis.png",
        paths.self_out / f"{name}_nice_sparse_preview.png",
        paths.self_out / f"{name}_far_fill_only_preview.png",
        paths.self_out / f"{name}_fused_two_stage_far_preview.png",
        paths.self_out / f"{name}_mountain_rescue_preview.png",
        paths.self_out / f"{name}_rejected_far_or_saturated_mask.png",
        paths.self_out / f"{name}_final_sparse_preview.png",
        paths.trace_out / f"{name}_stage2_refined_dense_depth.png",
        paths.gs_out / f"{name}_nice_prompt_compress.ply",
    ]
    if render_video:
        output_paths.append(paths.gs_out / f"{name}_nice_prompt_compress_novel.mp4")

    for path in output_paths:
        print(f"  {path}")

    print("")
    print("Download commands from Windows PowerShell:")
    for path in output_paths:
        local_path = f"$env:USERPROFILE\\Desktop\\{path.name}"
        print(f'scp {download_host}:{path} "{local_path}"')


def print_settings(cfg: Dict[str, str], paths: Paths, image_path: str, name: str) -> None:
    print("[run_final.py] final settings:")
    print(f"  RUN_PROFILE={cfg.get('RUN_PROFILE', 'final')}")
    for key in ["SMART_PROBE_FAR_CAP", "SMART_STAGE", "AUTO_SCENE_LABEL"]:
        if key in cfg:
            print(f"  {key}={cfg[key]}")
    for key in sorted(DEFAULT_ENV):
        print(f"  {key}={cfg[key]}")
    print("[run_final.py] paths:")
    print(f"  image={image_path}")
    print(f"  name={name}")
    print(f"  repo={paths.repo}")
    print(f"  nice={paths.nice}")
    print(f"  self_out={paths.self_out}")
    print(f"  gs_out={paths.gs_out}")


def main() -> int:
    args = parse_args()
    image_path = args.image
    name = args.name or Path(image_path).stem
    cfg = make_config(args)
    paths = resolve_paths(args, name)

    if args.dry_run:
        print_settings(cfg, paths, image_path, name)
        return 0

    maybe_reexec_in_venv(args, paths)
    prepare_runtime(paths, cfg, name)
    print_settings(cfg, paths, image_path, name)

    if not args.no_restore:
        restore_snapshot(paths)
    compile_files(paths)
    if cfg.get("RUN_PROFILE") == "smart":
        print(f"[smart] probe first-pass depth cap={cfg.get('SMART_PROBE_FAR_CAP', cfg['FAR_CAP'])}")
    run_dense_depth(image_path, paths, name)
    apply_adaptive_policy(paths, cfg, name)

    if cfg.get("RUN_PROFILE") == "smart" and cfg.get("SMART_STAGE") == "auto_scene":
        print(
            "[smart] selected auto_scene; rerun first-pass dense depth with "
            f"FAR_CAP={cfg['FAR_CAP']} to match the auto profile."
        )
        refresh_official_env(paths, cfg, name)
        run_dense_depth(image_path, paths, name)

    refresh_official_env(paths, cfg, name)
    print(
        "[adaptive] active "
        f"NICE_CAP={cfg['NICE_CAP']}, "
        f"FAR_CAP={cfg['FAR_CAP']}, "
        f"PROMPT_CAP={compute_prompt_cap(cfg)}, "
        f"GS_SAMPLE_POINTS={cfg['GS_SAMPLE_POINTS']}, "
        f"GS_FAR_SCALE_MULTIPLIER={cfg['GS_FAR_SCALE_MULTIPLIER']}"
    )
    compile_files(paths)
    build_sparse_prompt(image_path, paths, cfg, name)
    run_second_pass_gs(image_path, paths, cfg, name)
    print_outputs(paths, name, args.download_host, cfg["RENDER_VIDEO"] == "1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
