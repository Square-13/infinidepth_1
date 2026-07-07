import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import tyro

from InfiniDepth.gs import GSPixelAlignPredictor, export_ply
from InfiniDepth.utils.inference_utils import (
    build_camera_matrices,
    filter_gaussians_by_statistical_outlier,
    prepare_metric_depth_inputs,
    resolve_camera_intrinsics_for_inference,
    resolve_ply_output_path,
    run_optional_sampling_sky_mask,
    unpack_gaussians_for_export,
)
from InfiniDepth.utils.gs_utils import (
    _build_sparse_uniform_gaussians,
    _render_novel_video,
    _resolve_video_render_size,
    _scale_intrinsics_for_render,
)
from InfiniDepth.utils.io_utils import load_image, depth_to_disparity
from InfiniDepth.utils.model_utils import build_model


@dataclass
class GSInferenceArgs:
    # Inputs
    input_image_path: str
    input_depth_path: Optional[str] = None

    # Outputs
    output_ply_dir: Optional[str] = None
    output_ply_name: Optional[str] = None

    # Model
    model_type: str = "InfiniDepth"  # [InfiniDepth, InfiniDepth_DepthSensor]
    depth_model_path: str = "checkpoints/depth/infinidepth.ckpt"
    gs_model_path: str = "checkpoints/gs/infinidepth_gs.ckpt"
    moge2_pretrained: str = "checkpoints/moge-2-vitl-normal/model.pt"  # Metric depth via MoGe-2 (used when input_depth_path is None)
    
    # Camera intrinsics
    fx_org: Optional[float] = None
    fy_org: Optional[float] = None
    cx_org: Optional[float] = None
    cy_org: Optional[float] = None

    # Resolution / sampling
    input_size: tuple[int, int] = (768, 1024)
    sample_point_num: int = 2000000
    max_prompt_depth: float = 500.0
    gs_max_sample_depth: Optional[float] = None
    gs_sample_filter_mode: Literal["none", "max_depth", "sky_mask"] = "max_depth"
    gs_far_detail_min_depth: float = 45.0
    gs_far_detail_edge_quantile: float = 0.65
    gs_far_detail_area_boost: float = 8.0
    gs_far_scale_min_depth: float = 80.0
    gs_far_scale_multiplier: float = 1.0
    coord_deterministic_sampling: bool = True
    enable_skyseg_model: bool = True
    sky_model_ckpt_path: str = "checkpoints/sky/skyseg.onnx"
    sample_sky_mask_dilate_px: int = 0

    # Optional novel-view rendering
    render_novel_video: bool = True
    novel_video_path: Optional[str] = None
    novel_trajectory: str = "orbit"  # orbit | swing
    novel_num_frames: int = 120
    novel_video_fps: int = 30
    novel_radius: float = 0.5
    novel_vertical: float = 0.15
    novel_forward: float = 0.6
    render_size: Optional[tuple[int, int]] = None
    novel_bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)


@torch.no_grad()
def main(args: GSInferenceArgs) -> None:
    if not os.path.exists(args.gs_model_path):
        raise FileNotFoundError(f"GS checkpoint not found: {args.gs_model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GS inference in this script.")

    device = torch.device("cuda")
    model = build_model(args.model_type, model_path=args.depth_model_path).to(device)
    model.eval()
    print(f"Loaded depth model: {model.__class__.__name__}")

    org_img, image, (org_h, org_w) = load_image(args.input_image_path, args.input_size)
    del org_img
    image = image.to(device)
    b, _, h, w = image.shape

    if args.model_type == "InfiniDepth_DepthSensor":
        assert args.input_depth_path is not None and os.path.exists(args.input_depth_path), "InfiniDepth_DepthSensor requires a valid input depth map for depth completion. Please provide --input_depth_path."

    gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=args.input_depth_path,
        input_size=args.input_size,
        image=image,
        device=device,
        moge2_pretrained=args.moge2_pretrained,
        depth_load_kwargs={
            "preserve_sparse": True,
            "max_prompt": args.max_prompt_depth,
            "num_samples": 100000000,
        },
    )
    if use_gt_depth and args.input_depth_path is not None:
        print(f"metric depth from `{args.input_depth_path}`")
    else:
        print(f"MoGe-2 prompt depth generated from `{args.moge2_pretrained}`")

    fx_org, fy_org, cx_org, cy_org, intrinsics_source = resolve_camera_intrinsics_for_inference(
        fx_org=args.fx_org,
        fy_org=args.fy_org,
        cx_org=args.cx_org,
        cy_org=args.cy_org,
        org_h=org_h,
        org_w=org_w,
        image=image,
        moge2_pretrained=args.moge2_pretrained,
        moge2_intrinsics=moge2_intrinsics,
    )
    if intrinsics_source == "moge2":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using MoGe-2 estimated intrinsics in original space: fx={fx_org:.2f}, fy={fy_org:.2f}, cx={cx_org:.2f}, cy={cy_org:.2f}"
        )
    elif intrinsics_source == "default":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using image-size defaults in original space: fx={fx_org:.2f}, fy={fy_org:.2f}, cx={cx_org:.2f}, cy={cy_org:.2f}"
        )

    gt = depth_to_disparity(gt_depth)
    prompt = depth_to_disparity(prompt_depth)

    _fx, _fy, _cx, _cy, intrinsics, extrinsics = build_camera_matrices(
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        batch=b,
        device=device,
    )

    sky_mask = run_optional_sampling_sky_mask(
        image=image,
        enable_skyseg_model=args.enable_skyseg_model,
        sky_model_ckpt_path=args.sky_model_ckpt_path,
        dilate_px=args.sample_sky_mask_dilate_px,
    )
    gs_max_sample_depth = args.gs_max_sample_depth
    if gs_max_sample_depth is None:
        gs_max_sample_depth = args.max_prompt_depth

    depthmap, dino_tokens, query_3d_uniform_coord, pred_depth_3d = model.inference_for_gs(
        image=image,
        intrinsics=intrinsics,
        gt_depth=gt,
        gt_depth_mask=gt_depth_mask,
        prompt_depth=prompt,
        prompt_mask=prompt>0,
        sky_mask=sky_mask,
        sample_point_num=args.sample_point_num,
        gs_max_sample_depth=gs_max_sample_depth,
        gs_sample_filter_mode=args.gs_sample_filter_mode,
        gs_far_detail_min_depth=args.gs_far_detail_min_depth,
        gs_far_detail_edge_quantile=args.gs_far_detail_edge_quantile,
        gs_far_detail_area_boost=args.gs_far_detail_area_boost,
        coord_deterministic_sampling=args.coord_deterministic_sampling,
    )
    # OFFICIAL_SAFE_TRACE_DEPTHMAP
    trace_out = os.environ.get("OFFICIAL_SAFE_TRACE_OUT")
    if trace_out:
        from pathlib import Path as _TracePath
        import numpy as _trace_np
        from PIL import Image as _TraceImage

        trace_name = os.environ.get("OFFICIAL_SAFE_TRACE_NAME", "trace")
        trace_dir = _TracePath(trace_out)
        trace_dir.mkdir(parents=True, exist_ok=True)
        arr = depthmap.detach().float().cpu().numpy().squeeze()
        _trace_np.save(trace_dir / f"{trace_name}_stage2_refined_dense_depth.npy", arr)
        valid = _trace_np.isfinite(arr) & (arr > 0)
        vis = _trace_np.zeros(arr.shape[-2:], dtype=_trace_np.uint8)
        if valid.any():
            lo, hi = _trace_np.percentile(arr[valid], [2, 98])
            vis = (_trace_np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(_trace_np.uint8)
            vis[~valid] = 0
        _TraceImage.fromarray(vis).save(trace_dir / f"{trace_name}_stage2_refined_dense_depth.png")
        q = query_3d_uniform_coord.detach().float().cpu().numpy()
        d = pred_depth_3d.detach().float().cpu().numpy()
        _trace_np.savez_compressed(
            trace_dir / f"{trace_name}_stage2_query_depth_samples.npz",
            query_coord=q,
            pred_depth_3d=d,
        )

    if query_3d_uniform_coord is None or pred_depth_3d is None:
        raise RuntimeError("inference_gs did not return 3d-uniform query outputs.")

    # OFFICIAL_SAFE_FAR_RELATIVE_UNCOMPRESS
    _far_relative_map = os.environ.get("OFFICIAL_SAFE_FAR_RELATIVE_MAP", "")
    if _far_relative_map and os.path.exists(_far_relative_map):
        try:
            import numpy as _fr_np
            import torch.nn.functional as _fr_F
            from PIL import Image as _FrImage
            from pathlib import Path as _FrPath

            _data = _fr_np.load(_far_relative_map)
            _mask_np = _data["mask"].astype("float32")
            if _mask_np.ndim > 2:
                _mask_np = _mask_np.squeeze()
            _mask_t = torch.from_numpy(_mask_np).to(device=device, dtype=depthmap.dtype).view(1, 1, *_mask_np.shape[-2:])
            _mask_t = _fr_F.interpolate(_mask_t, size=(h, w), mode="nearest")
            if b > 1:
                _mask_t = _mask_t.expand(b, -1, -1, -1)
            _alpha = _mask_t.clamp(0.0, 1.0)
            for _ in range(2):
                _alpha = _fr_F.avg_pool2d(_alpha, kernel_size=7, stride=1, padding=3)
            _alpha = torch.clamp(_alpha * 1.25, 0.0, 1.0)

            _raw_log_lo = float(_data["raw_log_lo"])
            _raw_log_hi = float(_data["raw_log_hi"])
            _prompt_min = float(_data["prompt_min"])
            _prompt_max = float(_data["prompt_max"])
            _prompt_span = max(_prompt_max - _prompt_min, 1e-6)
            _raw_log_span = max(_raw_log_hi - _raw_log_lo, 1e-6)
            _relative_gain = float(os.environ.get("OFFICIAL_SAFE_FAR_RELATIVE_GAIN", "1.0"))
            _relative_center = float(os.environ.get("OFFICIAL_SAFE_FAR_RELATIVE_CENTER", "0.5"))
            _relative_gain = max(0.05, min(_relative_gain, 8.0))
            _relative_center = max(0.0, min(_relative_center, 1.0))

            def _uncompress_relative(_z: torch.Tensor) -> torch.Tensor:
                _norm = torch.clamp((_z - _prompt_min) / _prompt_span, 0.0, 1.0)
                _norm = torch.clamp(_relative_center + (_norm - _relative_center) * _relative_gain, 0.0, 1.0)
                _raw_log = _raw_log_lo + _norm * _raw_log_span
                return torch.exp(_raw_log)

            _uncompressed_depthmap = _uncompress_relative(depthmap)
            depthmap = depthmap * (1.0 - _alpha) + _uncompressed_depthmap * _alpha

            _grid = query_3d_uniform_coord[..., [1, 0]].view(b, 1, -1, 2)
            _sample_alpha = _fr_F.grid_sample(_alpha, _grid, mode="bilinear", align_corners=False).view(b, -1, 1)
            _uncompressed_pred = _uncompress_relative(pred_depth_3d)
            pred_depth_3d = pred_depth_3d * (1.0 - _sample_alpha) + _uncompressed_pred * _sample_alpha

            print(
                "[Info] far relative prompt uncompressed:",
                f"pixels={int((_mask_t > 0.5).sum().item())}",
                f"query_points={int((_sample_alpha > 0.03).sum().item())}",
                f"prompt_range={_prompt_min:.2f}-{_prompt_max:.2f}",
                f"raw_range={float(_fr_np.exp(_raw_log_lo)):.2f}-{float(_fr_np.exp(_raw_log_hi)):.2f}",
                f"gain={_relative_gain:.2f}",
            )

            _trace_out = os.environ.get("OFFICIAL_SAFE_TRACE_OUT")
            if _trace_out:
                _trace_name = os.environ.get("OFFICIAL_SAFE_TRACE_NAME", "trace")
                _trace_dir = _FrPath(_trace_out)
                _trace_dir.mkdir(parents=True, exist_ok=True)
                _arr = depthmap.detach().float().cpu().numpy().squeeze()
                _fr_np.save(_trace_dir / f"{_trace_name}_stage2_far_relative_uncompressed_depth.npy", _arr)
                _mask_vis = (_mask_t[0, 0].detach().cpu().numpy() > 0.5).astype(_fr_np.uint8) * 255
                _FrImage.fromarray(_mask_vis).save(_trace_dir / f"{_trace_name}_stage2_far_relative_mask.png")
                _valid = _fr_np.isfinite(_arr) & (_arr > 0)
                _vis = _fr_np.zeros(_arr.shape[-2:], dtype=_fr_np.uint8)
                if _valid.any():
                    _lo, _hi = _fr_np.percentile(_arr[_valid], [2, 98])
                    _vis = (_fr_np.clip((_arr - _lo) / (_hi - _lo + 1e-6), 0, 1) * 255).astype(_fr_np.uint8)
                    _vis[~_valid] = 0
                _FrImage.fromarray(_vis).save(_trace_dir / f"{_trace_name}_stage2_far_relative_uncompressed_depth.png")
        except Exception as _fr_exc:
            print(f"[Warning] far relative prompt uncompress skipped: {_fr_exc}")

    # OFFICIAL_SAFE_FAR_COMPRESS
    _far_compress_enabled = os.environ.get("OFFICIAL_SAFE_FAR_COMPRESS_TO_GS", "0").lower() in ("1", "true", "yes", "on")
    _far_mask_path = os.environ.get("OFFICIAL_SAFE_RELIEF_MASK_PATH")
    if _far_compress_enabled and _far_mask_path and os.path.exists(_far_mask_path):
        try:
            import numpy as _fc_np
            import torch.nn.functional as _fc_F
            from PIL import Image as _FcImage
            from pathlib import Path as _FcPath

            _mask_np = _fc_np.load(_far_mask_path).astype("float32")
            if _mask_np.ndim > 2:
                _mask_np = _mask_np.squeeze()
            _mask_t = torch.from_numpy(_mask_np).to(device=device, dtype=depthmap.dtype).view(1, 1, *_mask_np.shape[-2:])
            _mask_t = _fc_F.interpolate(_mask_t, size=(h, w), mode="nearest") > 0.5
            if b > 1:
                _mask_t = _mask_t.expand(b, -1, -1, -1)

            _candidate_count = int(_mask_t.sum().item())
            if _candidate_count > 32:
                _start = float(os.environ.get("OFFICIAL_SAFE_FAR_COMPRESS_START", "120"))
                _ratio = float(os.environ.get("OFFICIAL_SAFE_FAR_COMPRESS_RATIO", "0.35"))
                _blend = float(os.environ.get("OFFICIAL_SAFE_FAR_COMPRESS_BLEND", "1.0"))
                _ratio = max(0.02, min(1.0, _ratio))
                _blend = max(0.0, min(1.0, _blend))
                _start_t = torch.tensor(_start, device=device, dtype=depthmap.dtype)

                def _compress_depth(_z):
                    _compressed = _start_t + torch.clamp(_z - _start_t, min=0.0) * _ratio
                    return torch.where(_z > _start_t, _compressed, _z)

                # Compress the whole non-sky far range, not only the rescued
                # far mask. Compressing only the far background can invert
                # depth ordering and pull mountains in front of nearer slopes.
                _non_sky = torch.ones_like(depthmap, dtype=torch.bool)
                if sky_mask is not None:
                    _sky_t = sky_mask.to(device=device, dtype=torch.bool)
                    if _sky_t.ndim == 3:
                        _sky_t = _sky_t[:, None]
                    elif _sky_t.ndim == 2:
                        _sky_t = _sky_t[None, None]
                    if _sky_t.shape[-2:] != (h, w):
                        _sky_t = _fc_F.interpolate(_sky_t.float(), size=(h, w), mode="nearest") > 0.5
                    _non_sky = ~_sky_t

                _compress_region = (depthmap > _start_t) & _non_sky
                _alpha = _compress_region.to(dtype=depthmap.dtype) * _blend
                _compressed_depthmap = _compress_depth(depthmap)
                depthmap = depthmap * (1.0 - _alpha) + _compressed_depthmap * _alpha

                _grid = query_3d_uniform_coord[..., [1, 0]].view(b, 1, -1, 2)
                _sample_alpha = _fc_F.grid_sample(_alpha, _grid, mode="nearest", align_corners=False).view(b, -1, 1)
                _compressed_pred = _compress_depth(pred_depth_3d)
                pred_depth_3d = pred_depth_3d * (1.0 - _sample_alpha) + _compressed_pred * _sample_alpha

                _shape_mode = os.environ.get("OFFICIAL_SAFE_FAR_SHAPE_PRIOR", "auto").lower()
                _shape_offset = torch.zeros_like(depthmap)
                _shape_applied = False
                _flat_range = torch.tensor(0.0, device=device, dtype=depthmap.dtype)
                if _shape_mode not in ("0", "false", "no", "off"):
                    _shape_region = _mask_t & _compress_region
                    if int(_shape_region.sum().item()) > 128:
                        _shape_vals = depthmap[_shape_region]
                        _p05 = torch.quantile(_shape_vals, 0.05)
                        _p95 = torch.quantile(_shape_vals, 0.95)
                        _flat_range = _p95 - _p05
                        _flat_thresh = float(os.environ.get("OFFICIAL_SAFE_FAR_SHAPE_FLAT_RANGE", "4"))
                        _auto_ok = (_shape_mode != "auto") or (float(_flat_range.item()) <= _flat_thresh)
                        if _auto_ok:
                            _img = image.detach().to(dtype=depthmap.dtype)
                            _gray = 0.299 * _img[:, 0:1] + 0.587 * _img[:, 1:2] + 0.114 * _img[:, 2:3]
                            _blur = _fc_F.avg_pool2d(_gray, kernel_size=31, stride=1, padding=15)
                            _contrast = torch.abs(_gray - _blur)
                            _gx = torch.zeros_like(_gray)
                            _gy = torch.zeros_like(_gray)
                            _gx[:, :, :, 1:] = torch.abs(_gray[:, :, :, 1:] - _gray[:, :, :, :-1])
                            _gy[:, :, 1:, :] = torch.abs(_gray[:, :, 1:, :] - _gray[:, :, :-1, :])
                            _edge = _gx + _gy

                            def _norm_in_region(_v):
                                _vals = _v[_shape_region]
                                _lo = torch.quantile(_vals, 0.05)
                                _hi = torch.quantile(_vals, 0.95)
                                return torch.clamp((_v - _lo) / (_hi - _lo + 1e-6), 0.0, 1.0)

                            _relief = 0.65 * _norm_in_region(_contrast) + 0.35 * _norm_in_region(_edge)
                            _relief = _fc_F.avg_pool2d(_relief, kernel_size=15, stride=1, padding=7)
                            _relief = torch.clamp(_relief, 0.0, 1.0)
                            _max_m = float(os.environ.get("OFFICIAL_SAFE_FAR_SHAPE_MAX_METERS", "10"))
                            # Back-only: positive offset means farther from the
                            # camera, so this cannot pull the far mountains in
                            # front of nearer slopes.
                            _shape_offset = _shape_region.to(dtype=depthmap.dtype) * _relief * _max_m
                            depthmap = depthmap + _shape_offset
                            _sample_offset = _fc_F.grid_sample(_shape_offset, _grid, mode="bilinear", align_corners=False).view(b, -1, 1)
                            pred_depth_3d = pred_depth_3d + _sample_offset
                            _shape_applied = True

                print(
                    "[Info] far depth compression applied:",
                    f"mask_pixels={_candidate_count}",
                    f"global_compress_pixels={int(_compress_region.sum().item())}",
                    f"query_points={int((_sample_alpha > 0).sum().item())}",
                    f"start={_start:.1f}",
                    f"ratio={_ratio:.3f}",
                    f"blend={_blend:.2f}",
                    f"shape_prior={_shape_mode}",
                    f"shape_applied={_shape_applied}",
                    f"shape_flat_range={float(_flat_range.item()):.3f}",
                )

                _trace_out = os.environ.get("OFFICIAL_SAFE_TRACE_OUT")
                if _trace_out:
                    _trace_name = os.environ.get("OFFICIAL_SAFE_TRACE_NAME", "trace")
                    _trace_dir = _FcPath(_trace_out)
                    _trace_dir.mkdir(parents=True, exist_ok=True)
                    _arr = depthmap.detach().float().cpu().numpy().squeeze()
                    _fc_np.save(_trace_dir / f"{_trace_name}_stage2_far_compress_depth.npy", _arr)
                    _mask_arr = _compress_region.detach().cpu().numpy().squeeze().astype(_fc_np.uint8) * 255
                    _FcImage.fromarray(_mask_arr).save(_trace_dir / f"{_trace_name}_stage2_far_compress_mask.png")
                    if _shape_applied:
                        _shape_arr = depthmap.detach().float().cpu().numpy().squeeze()
                        _fc_np.save(_trace_dir / f"{_trace_name}_stage2_far_shape_depth.npy", _shape_arr)
                        _shape_prior = _shape_offset.detach().float().cpu().numpy().squeeze()
                        _prior_vis = _shape_prior
                        if _prior_vis.max() > _prior_vis.min():
                            _prior_vis = (_fc_np.clip((_prior_vis - _prior_vis.min()) / (_prior_vis.max() - _prior_vis.min() + 1e-6), 0, 1) * 255).astype(_fc_np.uint8)
                        else:
                            _prior_vis = _fc_np.zeros_like(_prior_vis, dtype=_fc_np.uint8)
                        _FcImage.fromarray(_prior_vis).save(_trace_dir / f"{_trace_name}_stage2_far_shape_prior.png")

                        _valid_shape = _fc_np.isfinite(_shape_arr) & (_shape_arr > 0)
                        _shape_vis = _fc_np.zeros(_shape_arr.shape[-2:], dtype=_fc_np.uint8)
                        if _valid_shape.any():
                            _lo_s, _hi_s = _fc_np.percentile(_shape_arr[_valid_shape], [2, 98])
                            _shape_vis = (_fc_np.clip((_shape_arr - _lo_s) / (_hi_s - _lo_s + 1e-6), 0, 1) * 255).astype(_fc_np.uint8)
                            _shape_vis[~_valid_shape] = 0
                        _FcImage.fromarray(_shape_vis).save(_trace_dir / f"{_trace_name}_stage2_far_shape_depth.png")
                    _valid = _fc_np.isfinite(_arr) & (_arr > 0)
                    _vis = _fc_np.zeros(_arr.shape[-2:], dtype=_fc_np.uint8)
                    if _valid.any():
                        _lo, _hi = _fc_np.percentile(_arr[_valid], [2, 98])
                        _vis = (_fc_np.clip((_arr - _lo) / (_hi - _lo + 1e-6), 0, 1) * 255).astype(_fc_np.uint8)
                        _vis[~_valid] = 0
                    _FcImage.fromarray(_vis).save(_trace_dir / f"{_trace_name}_stage2_far_compress_depth.png")

                    _img_np = image.detach().float().cpu().numpy()[0].transpose(1, 2, 0)
                    _img_np = (_fc_np.clip(_img_np, 0, 1) * 255).astype(_fc_np.uint8)
                    _query_np = query_3d_uniform_coord.detach().float().cpu().numpy()[0]
                    _py = _fc_np.clip(((_query_np[:, 0] + 1.0) * (h / 2.0) - 0.5).round().astype(_fc_np.int32), 0, h - 1)
                    _px = _fc_np.clip(((_query_np[:, 1] + 1.0) * (w / 2.0) - 0.5).round().astype(_fc_np.int32), 0, w - 1)
                    _query_vis = _img_np.copy()
                    _query_vis[_py, _px] = _fc_np.array([255, 0, 0], dtype=_fc_np.uint8)
                    _FcImage.fromarray(_query_vis).save(_trace_dir / f"{_trace_name}_stage2_query_all_preview.png")

                    _far_sel = (_sample_alpha.detach().float().cpu().numpy()[0, :, 0] > 0)
                    _far_vis = _img_np.copy()
                    if _far_sel.any():
                        _far_vis[_py[_far_sel], _px[_far_sel]] = _fc_np.array([255, 0, 0], dtype=_fc_np.uint8)
                    _FcImage.fromarray(_far_vis).save(_trace_dir / f"{_trace_name}_stage2_query_far_compress_preview.png")
            else:
                print(f"[Info] far depth compression skipped: too few mask pixels ({_candidate_count})")
        except Exception as _fc_exc:
            print(f"[Warning] far depth compression skipped: {_fc_exc}")


    # OFFICIAL_SAFE_QUERY_RELIEF
    _relief_enabled = os.environ.get("OFFICIAL_SAFE_RELIEF_TO_GS", "0").lower() in ("1", "true", "yes", "on")
    _relief_mask_path = os.environ.get("OFFICIAL_SAFE_RELIEF_MASK_PATH")
    if _relief_enabled and _relief_mask_path and os.path.exists(_relief_mask_path):
        try:
            import numpy as _rel_np
            import torch.nn.functional as _rel_F
            from PIL import Image as _RelImage

            _mask_np = _rel_np.load(_relief_mask_path).astype("float32")
            if _mask_np.ndim > 2:
                _mask_np = _mask_np.squeeze()
            _mask_t = torch.from_numpy(_mask_np).to(device=device, dtype=depthmap.dtype).view(1, 1, *_mask_np.shape[-2:])
            _mask_t = _rel_F.interpolate(_mask_t, size=(h, w), mode="nearest") > 0.5
            if b > 1:
                _mask_t = _mask_t.expand(b, -1, -1, -1)

            _candidate_count = int(_mask_t.sum().item())
            if _candidate_count > 32:
                _alpha = _mask_t.to(dtype=depthmap.dtype)
                # Feather the relief boundary. A hard binary replacement makes
                # the mask boundary visible as a rectangular/stripe artifact in
                # the generated PLY.
                for _ in range(2):
                    _alpha = _rel_F.avg_pool2d(_alpha, kernel_size=9, stride=1, padding=4)
                _alpha = torch.clamp(_alpha * 1.35, 0.0, 1.0)

                _img = image.detach().to(dtype=depthmap.dtype)
                _gray = 0.299 * _img[:, 0:1] + 0.587 * _img[:, 1:2] + 0.114 * _img[:, 2:3]

                _gx = torch.zeros_like(_gray)
                _gy = torch.zeros_like(_gray)
                _gx[:, :, :, 1:] = torch.abs(_gray[:, :, :, 1:] - _gray[:, :, :, :-1])
                _gy[:, :, 1:, :] = torch.abs(_gray[:, :, 1:, :] - _gray[:, :, :-1, :])
                _edge = _gx + _gy

                _edge_vals = _edge[_mask_t]
                _gray_vals = _gray[_mask_t]
                _edge_lo = torch.quantile(_edge_vals, 0.10)
                _edge_hi = torch.quantile(_edge_vals, 0.95)
                _gray_lo = torch.quantile(_gray_vals, 0.05)
                _gray_hi = torch.quantile(_gray_vals, 0.95)

                _edge_norm = torch.clamp((_edge - _edge_lo) / (_edge_hi - _edge_lo + 1e-6), 0.0, 1.0)
                _gray_norm = torch.clamp((_gray - _gray_lo) / (_gray_hi - _gray_lo + 1e-6), 0.0, 1.0)
                _yy = torch.linspace(0.0, 1.0, h, device=device, dtype=depthmap.dtype).view(1, 1, h, 1).expand_as(depthmap)

                _far_cap = float(os.environ.get("OFFICIAL_SAFE_RELIEF_FAR_CAP", "220"))
                _meters = float(os.environ.get("OFFICIAL_SAFE_RELIEF_METERS", "22"))
                _blend = float(os.environ.get("OFFICIAL_SAFE_RELIEF_BLEND", "1.0"))
                _d_hi = torch.tensor(_far_cap, device=device, dtype=depthmap.dtype)
                _relief = 0.60 * _edge_norm + 0.25 * (1.0 - _gray_norm) + 0.15 * (1.0 - _yy)
                _pseudo = torch.clamp(_d_hi - _meters * _relief, min=_far_cap - _meters, max=_far_cap - 0.5)

                _alpha = _alpha * _blend
                depthmap = depthmap * (1.0 - _alpha) + _pseudo * _alpha

                _grid = query_3d_uniform_coord[..., [1, 0]].view(b, 1, -1, 2)
                _sample_alpha = _rel_F.grid_sample(_alpha, _grid, mode="bilinear", align_corners=False).view(b, -1, 1)
                _sample_mask = _sample_alpha > 0.03
                _sample_pseudo = _rel_F.grid_sample(_pseudo, _grid, mode="bilinear", align_corners=False).view(b, -1, 1)
                pred_depth_3d = pred_depth_3d * (1.0 - _sample_alpha) + _sample_pseudo * _sample_alpha

                print(
                    "[Info] GS relief applied:",
                    f"mask_pixels={_candidate_count}",
                    f"query_points={int(_sample_mask.sum().item())}",
                    f"far_cap={_far_cap:.1f}",
                    f"meters={_meters:.1f}",
                    f"blend={_blend:.2f}",
                )

                _trace_out = os.environ.get("OFFICIAL_SAFE_TRACE_OUT")
                if _trace_out:
                    from pathlib import Path as _TracePath
                    _trace_name = os.environ.get("OFFICIAL_SAFE_TRACE_NAME", "trace")
                    _trace_dir = _TracePath(_trace_out)
                    _trace_dir.mkdir(parents=True, exist_ok=True)
                    _arr = depthmap.detach().float().cpu().numpy().squeeze()
                    _rel_np.save(_trace_dir / f"{_trace_name}_stage2_gs_relief_depth.npy", _arr)
                    _valid = _rel_np.isfinite(_arr) & (_arr > 0)
                    _vis = _rel_np.zeros(_arr.shape[-2:], dtype=_rel_np.uint8)
                    if _valid.any():
                        _lo, _hi = _rel_np.percentile(_arr[_valid], [2, 98])
                        _vis = (_rel_np.clip((_arr - _lo) / (_hi - _lo + 1e-6), 0, 1) * 255).astype(_rel_np.uint8)
                        _vis[~_valid] = 0
                    _RelImage.fromarray(_vis).save(_trace_dir / f"{_trace_name}_stage2_gs_relief_depth.png")
            else:
                print(f"[Info] GS relief skipped: too few mask pixels ({_candidate_count})")
        except Exception as _rel_exc:
            print(f"[Warning] GS relief skipped: {_rel_exc}")

    # OFFICIAL_SAFE_FAR_RAW_GS_OVERRIDE
    _far_raw_depth_path = os.environ.get("OFFICIAL_SAFE_FAR_RAW_GS_DEPTH_PATH", "")
    if _far_raw_depth_path and os.path.exists(_far_raw_depth_path):
        try:
            import numpy as _raw_np
            import torch.nn.functional as _raw_F
            from PIL import Image as _RawImage
            from pathlib import Path as _RawPath

            _raw_arr = _raw_np.load(_far_raw_depth_path).astype("float32")
            _raw_arr = _raw_np.squeeze(_raw_arr)
            if _raw_arr.ndim != 2:
                raise ValueError(f"Expected 2D raw depth, got {_raw_arr.shape}")

            _raw_t = torch.from_numpy(_raw_arr).to(device=device, dtype=depthmap.dtype).view(1, 1, *_raw_arr.shape)
            if _raw_t.shape[-2:] != (h, w):
                _raw_t = _raw_F.interpolate(_raw_t, size=(h, w), mode="bilinear", align_corners=False)
            if b > 1:
                _raw_t = _raw_t.expand(b, -1, -1, -1)

            _raw_min = float(os.environ.get("OFFICIAL_SAFE_FAR_RAW_GS_MIN", "180"))
            _raw_max = float(os.environ.get("OFFICIAL_SAFE_FAR_RAW_GS_MAX", "2000"))
            _raw_blend = float(os.environ.get("OFFICIAL_SAFE_FAR_RAW_GS_BLEND", "1.0"))
            _apply_dense = os.environ.get("OFFICIAL_SAFE_FAR_RAW_GS_APPLY_DENSE", "0").lower() in ("1", "true", "yes", "on")
            _raw_blend = max(0.0, min(1.0, _raw_blend))

            _raw_mask = torch.isfinite(_raw_t) & (_raw_t >= _raw_min) & (_raw_t <= _raw_max)
            if sky_mask is not None:
                _sky_t = sky_mask.to(device=device, dtype=torch.bool)
                if _sky_t.ndim == 3:
                    _sky_t = _sky_t[:, None]
                elif _sky_t.ndim == 2:
                    _sky_t = _sky_t[None, None]
                if _sky_t.shape[-2:] != (h, w):
                    _sky_t = _raw_F.interpolate(_sky_t.float(), size=(h, w), mode="nearest") > 0.5
                _raw_mask &= ~_sky_t

            _alpha = _raw_mask.to(dtype=depthmap.dtype) * _raw_blend
            for _ in range(2):
                _alpha = _raw_F.avg_pool2d(_alpha, kernel_size=5, stride=1, padding=2)
            _alpha = torch.clamp(_alpha, 0.0, 1.0)

            _grid = query_3d_uniform_coord[..., [1, 0]].view(b, 1, -1, 2)
            _sample_alpha = _raw_F.grid_sample(_alpha, _grid, mode="bilinear", align_corners=False).view(b, -1, 1)
            _sample_raw = _raw_F.grid_sample(_raw_t, _grid, mode="bilinear", align_corners=False).view(b, -1, 1)
            pred_depth_3d = pred_depth_3d * (1.0 - _sample_alpha) + _sample_raw * _sample_alpha
            if _apply_dense:
                depthmap = depthmap * (1.0 - _alpha) + _raw_t * _alpha

            print(
                "[Info] far raw GS override:",
                f"pixels={int(_raw_mask.sum().item())}",
                f"query_points={int((_sample_alpha > 0.03).sum().item())}",
                f"range={_raw_min:.1f}-{_raw_max:.1f}",
                f"blend={_raw_blend:.2f}",
                f"apply_dense={_apply_dense}",
            )

            _trace_out = os.environ.get("OFFICIAL_SAFE_TRACE_OUT")
            if _trace_out:
                _trace_name = os.environ.get("OFFICIAL_SAFE_TRACE_NAME", "trace")
                _trace_dir = _RawPath(_trace_out)
                _trace_dir.mkdir(parents=True, exist_ok=True)
                _mask_vis = (_raw_mask[0, 0].detach().cpu().numpy().astype(_raw_np.uint8) * 255)
                _RawImage.fromarray(_mask_vis).save(_trace_dir / f"{_trace_name}_stage2_far_raw_gs_mask.png")
                _raw_np.savez_compressed(
                    _trace_dir / f"{_trace_name}_stage2_far_raw_gs_samples.npz",
                    alpha=_sample_alpha.detach().float().cpu().numpy(),
                    raw_depth=_sample_raw.detach().float().cpu().numpy(),
                    pred_depth_3d=pred_depth_3d.detach().float().cpu().numpy(),
                )
        except Exception as _raw_exc:
            print(f"[Warning] far raw GS override skipped: {_raw_exc}")

    
    print(
        f"Step1 depthmap: {tuple(depthmap.shape)}, "
        f"Step2 query: {tuple(query_3d_uniform_coord.shape)}, "
        f"Step2 depth: {tuple(pred_depth_3d.shape)}"
    )

    gs_predictor = GSPixelAlignPredictor(dino_feature_dim=dino_tokens.shape[-1]).to(device)
    gs_predictor.load_from_infinidepth_gs_checkpoint(args.gs_model_path)
    gs_predictor.eval()

    dense_gaussians = gs_predictor(
        image=image,
        depthmap=depthmap,
        dino_tokens=dino_tokens,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    pixel_gaussians = _build_sparse_uniform_gaussians(
        dense_gaussians=dense_gaussians,
        query_3d_uniform_coord=query_3d_uniform_coord,
        pred_depth_3d=pred_depth_3d,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        h=h,
        w=w,
        far_scale_min_depth=args.gs_far_scale_min_depth,
        far_scale_multiplier=args.gs_far_scale_multiplier,
    )

    pixel_gaussians = filter_gaussians_by_statistical_outlier(pixel_gaussians)
    means, harmonics, opacities, scales, rotations = unpack_gaussians_for_export(pixel_gaussians)

    output_ply_dir, output_ply_path = resolve_ply_output_path(
        input_image_path=args.input_image_path,
        model_type=args.model_type,
        output_ply_dir=args.output_ply_dir,
        output_ply_name=args.output_ply_name,
    )

    export_ply(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        path=output_ply_path,
        scales=scales,
        rotations=rotations,
        focal_length_px=(fx_org, fy_org),
        principal_point_px=(cx_org, cy_org),
        image_shape=(org_h, org_w),
        extrinsic_matrix=extrinsics[0],
    )
    print(f"Saved 3d-uniform gaussians: {means.shape[0]} points -> {output_ply_path}")

    if args.render_novel_video:
        if args.render_size is None:
            render_h, render_w = org_h, org_w
            print(f"Novel-view render size not provided. Using original input resolution: ({render_h}, {render_w})")
        else:
            render_h, render_w = args.render_size
            print(f"Using user-specified novel-view render size: ({render_h}, {render_w})")
        video_render_h, video_render_w = _resolve_video_render_size(render_h, render_w)
        if (video_render_h, video_render_w) != (render_h, render_w):
            print(
                "Adjusted novel-view render size for libx264/yuv420p compatibility: "
                f"({render_h}, {render_w}) -> ({video_render_h}, {video_render_w})"
            )
        render_h, render_w = video_render_h, video_render_w
        intrinsics_render = _scale_intrinsics_for_render(intrinsics[0], h, w, render_h, render_w)
        stem = os.path.splitext(os.path.basename(args.input_image_path))[0]

        novel_video_path = args.novel_video_path
        if novel_video_path is None:
            novel_video_path = os.path.join(
                output_ply_dir, f"{args.model_type}_{stem}_novel_{args.novel_trajectory}.mp4"
            )

        _render_novel_video(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            base_c2w=extrinsics[0],
            intrinsics=intrinsics_render,
            render_h=render_h,
            render_w=render_w,
            video_path=novel_video_path,
            trajectory=args.novel_trajectory,
            num_frames=args.novel_num_frames,
            fps=args.novel_video_fps,
            radius=args.novel_radius,
            vertical=args.novel_vertical,
            forward_amp=args.novel_forward,
            bg_color=args.novel_bg_color,
        )
        print(f"Saved novel-view video -> {novel_video_path}")


if __name__ == "__main__":
    main(tyro.cli(GSInferenceArgs))
