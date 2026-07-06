# 当前代码算法思路与缺陷总结

本文总结当前保留的 `best` 版本，也就是 `run_nice_v29_prompt_compress.sh`，以及从最初源码到 V29/nice、far rescue、far compress、prompt compress 的演化思路。

## 1. 当前 best 版本整体目标

当前目标不是重新训练 InfiniDepth，而是在官方前馈单图 3DGS 流程基础上做工程复现和改进：

```text
单张 RGB
-> InfiniDepth 第一遍 dense depth
-> 对远景 depth 做 prompt 前压缩
-> 构造 sparse depth prompt
-> InfiniDepth_DepthSensor 第二遍补全/修正 depth
-> GS head 生成 Gaussian 属性
-> 3D-uniform sampling
-> 导出 PLY
```

核心改动是：

```text
在 DepthSensor 之前压缩远景深度，而不是在 GS 之后硬改点云。
```

这样做的原因是 DepthSensor 内部使用的是 disparity：

```text
disparity = 1 / depth
```

远景 depth 太大时，`1/depth` 差异很小，网络很难区分 180m、200m、300m 的远景层次。因此当前版本先把远景 depth 压到一个更容易被网络感知的范围。

## 2. 当前 best 版本关键流程

### 2.1 第一遍 InfiniDepth

脚本先调用普通 `InfiniDepth` 生成 dense depth：

```text
RGB image -> dense depth
```

输出包括：

```text
*_dense.npy
*_dense_vis.png
```

这一步提供完整深度图，但远景常常会出现：

```text
深度接近上限
远山 / 远楼变成大面积常数
天空和远景容易混淆
```

### 2.2 Prompt 前远景深度压缩

当前版本使用：

```text
PROMPT_COMPRESS_START=100
PROMPT_COMPRESS_RATIO=0.35
PROMPT_COMPRESS_MAX=160
```

公式：

```text
if depth <= 100:
    depth' = depth
else:
    depth' = 100 + (depth - 100) * 0.35
    depth' = min(depth', 160)
```

例子：

```text
80m  -> 80m
120m -> 107m
180m -> 128m
200m -> 135m
```

目的：

```text
近景保持 nice/V29 的稳定结果；
远景被压进更有效的 disparity 范围；
DepthSensor 第二遍能更容易读取远景 sparse prompt。
```

### 2.3 Sparse Prompt 构造

Sparse prompt 是一个 `.npy` 深度图：

```text
0      表示该像素没有深度提示
> 0    表示该像素有一个 depth prompt
```

当前版本仍按原始 dense depth 的几何和边缘信息选择采样位置，但写入的深度值使用压缩后的 `prompt_depth`。

采样策略：

```text
按深度范围分 bin
每个 bin 分配一定点数
tile-limited sampling 防止局部过密
避开强边缘和噪声点
```

这样做是为了避免：

```text
点全部集中在近处树木
远处建筑没有 prompt
强边缘处 prompt 误导 DepthSensor
```

### 2.4 Far Rescue

当前版本保留了远景 rescue 逻辑。

虽然变量名里仍有 `mountain`，但它不是只针对山。它表示：

```text
远景/饱和区域
非天空
远离已有 nice sparse 的区域
```

默认：

```text
FAR_RESCUE=auto
FAR_RESCUE_PROMPT=1
```

如果检测到远景缺失，就把远景 sparse 合并进最终 prompt。

### 2.5 DepthSensor 第二遍

第二遍运行：

```text
InfiniDepth_DepthSensor
```

它读取上一步生成的 sparse prompt。

关键点：

```text
不是把 .npy 当 RGB 第四通道；
而是把 sparse depth 转成 disparity；
再通过 prompt model 注入到 DINO feature 的第 3 stage；
最后由 ImplicitHead 重新解码 depth。
```

也就是说，DepthSensor 的作用是：

```text
用 sparse depth prompt 修正/约束 dense depth 预测。
```

### 2.6 GS Head 与 3D-uniform Sampling

DepthSensor 得到 refined dense depth 后，进入 GS 流程：

```text
RGB + refined depth + DINO tokens
-> GSPixelAlignPredictor
-> 每个像素预测 Gaussian 属性
```

Gaussian 属性包括：

```text
mean / position
opacity
scale
rotation
SH color / harmonics
```

随后使用 3D-uniform sampling：

```text
dense depth -> 反投影成 3D 顶点
相邻像素组成三角形
过滤天空 / 过远 / 深度突变三角形
按三角形面积采样 query 点
从 dense Gaussian map 采样属性
导出 sparse Gaussian PLY
```

这样比简单每像素一个点更适合 3DGS，因为可以让点更多分布在 3D 面积大的区域。

## 3. 之前各版本思路演化

### 3.1 官方源码 / 初始版本

官方 GS 流程大致是：

```text
RGB
-> InfiniDepth / DepthSensor
-> dense depth
-> 3D-uniform sampling
-> GS head
-> PLY
```

主要问题：

```text
默认剔除天空；
远景超过阈值会被过滤；
远处建筑 / 山体容易缺失；
远景 depth 常数化后会像纸板。
```

### 3.2 V1 思路

V1 基本是让 pipeline 能跑通：

```text
下载权重
修依赖
跑 inference_depth.py
跑 inference_gs.py
生成 PLY
```

问题：

```text
天空被剔除；
远景信息不足；
部分建筑粘连；
右侧建筑、吊车、远楼容易被当成天空或背景丢掉。
```

### 3.3 V2 / nice 思路

V2/nice 的方向是对的：

```text
先用第一遍 dense depth 生成 sparse prompt；
再用 DepthSensor 跑第二遍；
尽量保留官方稳定的近中景结构；
避免把过多噪声远景喂给 DepthSensor。
```

优点：

```text
近处建筑更稳定；
平面感更好；
粘连比初始版本少；
整体结构更干净。
```

缺点：

```text
太远的物体经常被过滤；
远山 / 远楼可能没有 sparse；
如果远景 depth 超过有效阈值，就直接消失。
```

### 3.4 V29 思路

V29 主要围绕：

```text
preserve_sparse=True
保持自己构造的 sparse prompt 不被官方 load_depth 二次随机采样
更平衡地采样远近区域
保留 nice 的稳定前景
```

优点：

```text
sparse prompt 更可控；
不会被官方随机采样破坏；
中近景建筑质量较好；
平面更规整。
```

缺点：

```text
远景仍受 depth 上限和天空 mask 影响；
远景没被采到时，GS 不可能恢复。
```

### 3.5 Delaunay / Component-aware 尝试

曾尝试过：

```text
Delaunay 三角化
component-aware 分割
禁止跨 component 连接
```

目标是解决：

```text
不同建筑之间桥接
天空和物体粘连
前后物体连在一起
```

结果：

```text
能部分减少桥接；
但会引入空洞；
分割质量不稳定；
对远处小建筑、吊车、树木效果不好。
```

原因：

```text
单图自动分割无法可靠区分所有建筑和背景；
几何粘连主要来自 depth 错误，不只是三角化错误。
```

### 3.6 Surface / Plane Snap 尝试

尝试过把局部平面压平：

```text
检测近似平面
对局部点做 plane snap
减少凸起和点漂移
```

优点：

```text
某些建筑表面更平；
部分孤岛点减少。
```

缺点：

```text
容易过度压平；
真实有起伏的地方会被改坏；
对“深度本身错了”的区域治标不治本。
```

### 3.7 GS 后处理 Far Compress

尝试过在 GS 阶段压缩远景：

```text
DepthSensor 输出 depth
-> 对 depth > 某阈值的区域压缩
-> 再反投影成 GS
```

优点：

```text
能让远景不要太远；
减少点被拉伸得过分。
```

缺点：

```text
如果只压远山，不压它前面的远景，会造成深度顺序反转；
山可能跑到前景前面；
如果原始远景 depth 是常数，压缩后仍然是纸板。
```

### 3.8 RGB Relief / Shape Prior 尝试

尝试过根据 RGB 纹理给远山增加伪深度起伏：

```text
纹理 / 边缘 / 亮度
-> 生成 pseudo relief
-> 改远景 depth
```

优点：

```text
可以打破纸板感。
```

缺点：

```text
不是模型真实 depth；
容易把纹理当几何；
山体、云、光照会产生假起伏；
不适合作为论文复现的可信结果。
```

所以当前 best 默认关闭：

```text
FAR_COMPRESS_TO_GS=0
FAR_SHAPE_PRIOR=off
```

## 4. 当前 best 版本的主要缺陷

### 4.1 远景常数化仍无法根治

如果第一遍 InfiniDepth 已经把远山输出为：

```text
200, 200, 200, 200...
```

那么 prompt compression 后只是：

```text
135, 135, 135, 135...
```

它仍然是纸板，只是距离被压近。

这不是 GS 导出问题，而是单目深度估计本身没有恢复远景几何。

### 4.2 远景压缩会改变真实尺度

当前压缩公式会让：

```text
真实 200m -> 135m
真实 300m -> 160m
```

这对视觉 GS 有帮助，但不再是严格 metric depth。

因此它适合：

```text
单图 3DGS 可视化
初步复现
工程展示
```

不适合直接作为：

```text
真实尺度测量
严格深度评估
```

### 4.3 天空和远景仍可能混淆

天空 mask 会影响：

```text
sparse prompt 生成
far rescue 区域
3D-uniform sampling
```

如果天空误分：

```text
吊车可能被删；
远楼边缘可能被当成天空；
天空也可能被误加入远景。
```

### 4.4 Sparse prompt 仍依赖第一遍 depth 质量

当前所有 prompt 都来自第一遍 dense depth。

如果第一遍 depth 对某个区域错了：

```text
第二遍 DepthSensor 可能被错误 prompt 误导；
建筑边缘会漂移；
前后建筑可能粘连。
```

### 4.5 3D-uniform sampling 不能修正 depth 错误

3D-uniform sampling 只能决定：

```text
点采在哪里
每个区域点多还是少
```

它不能决定：

```text
这个点真正应该有多深
```

所以如果 depth 本身错，采样再好也只是把错误 depth 变成更密的错误点云。

### 4.6 GS head 是前馈预测，不会做优化

当前流程不是传统 3DGS 训练优化。

它没有：

```text
多视角 photometric loss
迭代优化 Gaussian
相机位姿优化
真实多视角几何约束
```

因此它无法像多视角 3DGS 那样不断修正错误几何。

## 5. 当前代码适合什么

适合：

```text
论文初步复现
展示 InfiniDepth + GS head 的前馈单图流程
对比不同 depth prompt 策略
观察 sparse prompt 对 DepthSensor 的影响
生成可看的单图 PLY
```

不适合：

```text
高精度真实尺度重建
远山/天空/云层的真实几何恢复
复杂遮挡关系的严格 3D 分离
替代多视角 3DGS
```

## 6. 后续更合理的优化方向

### 6.1 更好的外部深度先验

可以尝试：

```text
Metric3D v2
Depth Pro
UniDepthV2
MoGe
真实 depth sensor / LiDAR / COLMAP
```

但要注意：外部深度必须和当前尺度、相机内参、图像 resize 对齐，否则会更差。

### 6.2 Prompt 压缩参数自适应

现在固定：

```text
start=100
ratio=0.35
max=160
```

后续可以根据每张图的 depth 分布自动决定：

```text
start = p75 或 p80
max = p98 压缩后范围
ratio = 根据远景饱和程度调整
```

### 6.3 更可靠的天空/远景分割

当前 skyseg 对吊车、远楼、云层不稳定。

后续可以：

```text
结合语义分割
保留高频细长结构
对天空边界做更保守处理
```

### 6.4 多图或视频信息

真正解决远景纸板，最可靠的是增加信息源：

```text
多视角
视频帧
相机运动
COLMAP
LiDAR / GPS / DEM
```

单张 RGB 本身无法稳定判断远山真实几何。

## 7. 最重要的结论

当前 best 版本的核心价值是：

```text
把远景处理提前到 DepthSensor prompt 阶段，
而不是在最终 GS 点云阶段硬改。
```

这比 GS 后处理更合理，因为它让网络在第二遍深度补全时就看到压缩后的远景提示。

但它仍然受限于：

```text
第一遍单目 depth 的质量；
远景是否已经被估成常数；
天空和远景是否被正确区分；
单图缺少真实几何约束。
```

所以现在的结果可以作为“前馈单图 3DGS 初步复现与改进版本”，但不能宣称已经解决了单图远景真实几何重建。
