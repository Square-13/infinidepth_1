import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import numpy as np
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

def _snap_wrong_depth_islands(pred_depth_3d, query_coord, depthmap, intrinsics):
    if os.environ.get("GS_DEPTH_ISLAND_SNAP", "0") != "1":
        return pred_depth_3d

    if pred_depth_3d is None or query_coord is None or depthmap is None:
        return pred_depth_3d

    if pred_depth_3d.shape[0] != 1:
        print("[Warning] depth-island snap skipped: batch size must be 1")
        return pred_depth_3d

    voxel_size = float(os.environ.get("GS_DEPTH_ISLAND_VOXEL", "1.5"))
    min_points = int(os.environ.get("GS_DEPTH_ISLAND_MIN_POINTS", "50000"))
    keep_top = int(os.environ.get("GS_DEPTH_ISLAND_KEEP_TOP", "3"))
    max_fraction = float(os.environ.get("GS_DEPTH_ISLAND_MAX_FRACTION", "0.05"))
    min_depth = float(os.environ.get("GS_DEPTH_ISLAND_MIN_DEPTH", "35"))
    ratio_thr = float(os.environ.get("GS_DEPTH_ISLAND_RATIO", "1.12"))
    blend = float(os.environ.get("GS_DEPTH_ISLAND_BLEND", "0.85"))

    b, n, _ = pred_depth_3d.shape
    _, _, h, w = depthmap.shape

    grid = query_coord[..., [1, 0]].reshape(b, n, 1, 2).float()
    dense_at_query = F.grid_sample(
        depthmap.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(1).reshape(b, n, 1)

    q = query_coord[0].detach().float().cpu().numpy()
    z = pred_depth_3d[0, :, 0].detach().float().cpu().numpy()
    dense_z = dense_at_query[0, :, 0].detach().float().cpu().numpy()
    K = intrinsics[0].detach().float().cpu().numpy()

    px = (q[:, 1] + 1.0) * 0.5 * w - 0.5
    py = (q[:, 0] + 1.0) * 0.5 * h - 0.5

    x = (px - K[0, 2]) / max(K[0, 0], 1e-6) * z
    y = (py - K[1, 2]) / max(K[1, 1], 1e-6) * z
    xyz = np.stack([x, y, z], axis=1)

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(dense_z) & (z > min_depth) & (dense_z > 1.0)
    valid_idx = np.flatnonzero(valid)

    if valid_idx.size < 100:
        print("[Info] depth-island snap: too few valid far points")
        return pred_depth_3d

    xyz_v = xyz[valid_idx]
    vox = np.floor((xyz_v - xyz_v.min(axis=0)) / max(voxel_size, 1e-6)).astype(np.int32)
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)
    num_vox = len(uniq)

    parent = np.arange(num_vox, dtype=np.int32)
    rank = np.zeros(num_vox, dtype=np.uint8)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    index = {tuple(v): i for i, v in enumerate(uniq)}
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if (dx, dy, dz) > (0, 0, 0):
                    offsets.append((dx, dy, dz))

    for i, v in enumerate(uniq):
        base = (int(v[0]), int(v[1]), int(v[2]))
        for dx, dy, dz in offsets:
            j = index.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if j is not None:
                union(i, j)

    roots = np.array([find(i) for i in range(num_vox)], dtype=np.int32)
    _, comp_vox = np.unique(roots, return_inverse=True)
    comp_id = comp_vox[inv]
    comp_counts = np.bincount(comp_id)
    order = np.argsort(-comp_counts)

    keep_comp = np.zeros_like(comp_counts, dtype=bool)
    keep_comp[order[:max(keep_top, 1)]] = True
    keep_comp |= comp_counts >= min_points

    small_island_v = ~keep_comp[comp_id]

    rel = np.maximum(
        z[valid_idx] / np.clip(dense_z[valid_idx], 1e-3, None),
        dense_z[valid_idx] / np.clip(z[valid_idx], 1e-3, None),
    )
    wrong_depth_v = rel > ratio_thr

    snap_v = small_island_v & wrong_depth_v
    snap_idx = valid_idx[snap_v]
    snap_count = int(snap_idx.size)
    total = int(pred_depth_3d.shape[1])

    if snap_count == 0:
        print(
            "[Info] depth-island snap: nothing corrected",
            f"components={len(comp_counts)}",
            f"voxel={voxel_size}",
        )
        return pred_depth_3d

    if snap_count / max(1, total) > max_fraction:
        print(
            "[Warning] depth-island snap skipped:",
            f"would correct {snap_count}/{total}",
            f"fraction={snap_count / max(1, total):.4f}",
        )
        return pred_depth_3d

    corrected = pred_depth_3d.float().clone()
    snap_t = torch.from_numpy(snap_idx).to(corrected.device)

    corrected[0, snap_t, 0] = (
        corrected[0, snap_t, 0] * (1.0 - blend)
        + dense_at_query[0, snap_t, 0].float() * blend
    )

    print(
        "[Info] depth-island snap:",
        f"components={len(comp_counts)}",
        f"corrected={snap_count}/{total}",
        f"voxel={voxel_size}",
        f"min_points={min_points}",
        f"ratio>{ratio_thr}",
        f"blend={blend}",
    )

    return corrected.to(pred_depth_3d.dtype)


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
            "max_prompt": 220.0,
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

    depthmap, dino_tokens, query_3d_uniform_coord, pred_depth_3d = model.inference_for_gs(
        image=image,
        intrinsics=intrinsics,
        gt_depth=gt,
        gt_depth_mask=gt_depth_mask,
        prompt_depth=prompt,
        prompt_mask=prompt>0,
        sky_mask=sky_mask,
        sample_point_num=args.sample_point_num,
        coord_deterministic_sampling=args.coord_deterministic_sampling,
    )
    if query_3d_uniform_coord is None or pred_depth_3d is None:
        raise RuntimeError("inference_gs did not return 3d-uniform query outputs.")

    pred_depth_3d = _snap_wrong_depth_islands(
        pred_depth_3d=pred_depth_3d,
        query_coord=query_3d_uniform_coord,
        depthmap=depthmap,
        intrinsics=intrinsics,
    )
    
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
