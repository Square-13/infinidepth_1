# NICE / V29 Two-Stage Pipeline

This is the baseline workflow for all later InfiniDepth GS experiments in this project.

Important: here, "nice" means the V29 two-stage DepthSensor pipeline. It does not mean RGB-only GS.

## Definition

For any input image, the baseline is:

1. Restore the saved nice code snapshot.
2. Run first-pass `InfiniDepth` depth inference on the RGB image.
3. Convert the first-pass dense depth into a sparse `.npy` depth prompt.
4. Run second-pass `InfiniDepth_DepthSensor` GS inference using that sparse prompt.
5. Export `.ply` and novel-view `.mp4`.

The second pass must use:

- depth model: `InfiniDepth_DepthSensor`
- depth checkpoint: `/home/liuyanb/InfiniDepth/checkpoints/depth/infinidepth_depthsensor.ckpt`
- GS checkpoint: `/home/liuyanb/InfiniDepth/checkpoints/gs/infinidepth_depthsensor_gs.ckpt`
- image-specific sparse depth prompt: `/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_v29_sparse.npy`

Never use a sparse prompt generated from another image. For example, do not use a campus `.npy` for `ice.png`.

## Why This Is The Baseline

RGB-only GS uses only image features and MoGe metric depth. It is easier to run, but it is not the version we decided to preserve.

The V29/nice route uses the model's own first-pass depth as sparse guidance for the second pass. This gives the DepthSensor model extra geometric anchors, which improved campus reconstruction compared with plain RGB-only.

## Canonical Script

Use:

```bash
bash /home/liuyanb/InfiniDepth/run_nice_v29_two_stage.sh /home/liuyanb/InfiniDepth/example_data/image/ice.png ice
```

For another image:

```bash
bash /home/liuyanb/InfiniDepth/run_nice_v29_two_stage.sh /absolute/path/to/image.png output_name
```

The script writes:

```text
/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_dense.npy
/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_dense_vis.png
/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_v29_sparse.npy
/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_v29_sparse_preview.png
/tmp/liuyanb-infinidepth/outputs/gs_<name>_v29_nice/<name>_v29_nice.ply
/tmp/liuyanb-infinidepth/outputs/gs_<name>_v29_nice/<name>_v29_nice_novel.mp4
```

## Download Template

Replace `<name>` with the chosen output name:

```powershell
scp Point:/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_dense_vis.png "$env:USERPROFILE\Desktop\<name>_dense_vis.png"
scp Point:/tmp/liuyanb-infinidepth/outputs/self_prompt_<name>_v29/<name>_v29_sparse_preview.png "$env:USERPROFILE\Desktop\<name>_v29_sparse_preview.png"
scp Point:/tmp/liuyanb-infinidepth/outputs/gs_<name>_v29_nice/<name>_v29_nice.ply "$env:USERPROFILE\Desktop\<name>_v29_nice.ply"
scp Point:/tmp/liuyanb-infinidepth/outputs/gs_<name>_v29_nice/<name>_v29_nice_novel.mp4 "$env:USERPROFILE\Desktop\<name>_v29_nice_novel.mp4"
```

## Modification Rule

All later experiments should start from this baseline unless explicitly stated otherwise.

Allowed future changes should be layered on top of this route, for example:

- change sparse prompt generation
- change GS scale/opacity regularization
- add surface alignment
- add local masks
- change sampling distribution

But the default baseline should remain:

```text
RGB -> first-pass InfiniDepth dense depth -> V29 sparse prompt -> InfiniDepth_DepthSensor GS
```

