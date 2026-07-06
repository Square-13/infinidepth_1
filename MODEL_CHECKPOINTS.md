# Model Checkpoints

本 GitHub-ready 仓库不包含模型权重文件。原因是这些文件体积大，不适合直接提交到普通 GitHub 仓库。

## 需要准备的权重路径

服务器运行时通常需要：

```text
/tmp/liuyanb-infinidepth/checkpoints/infinidepth.ckpt
/tmp/liuyanb-infinidepth/checkpoints/moge/model.pt
/home/liuyanb/InfiniDepth/checkpoints/depth/infinidepth_depthsensor.ckpt
/home/liuyanb/InfiniDepth/checkpoints/gs/infinidepth_depthsensor_gs.ckpt
```

如果运行中提示 `checkpoint not found`，意思是模型权重路径不存在，需要先下载或复制对应权重。

## 英文术语

`checkpoint`：模型权重文件。

`Git LFS`：Git Large File Storage，用来管理大文件的 Git 扩展。

`model.pt`：PyTorch 权重文件。

`.ckpt`：checkpoint 文件，通常也是模型权重。

