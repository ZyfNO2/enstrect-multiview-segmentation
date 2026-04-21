# ENSTRECT 损伤分割技术详解

> 本文档详细解析 ENSTRECT 项目中结构表面损伤的分割实现原理，涵盖从 2D 图像分割到 3D 点云映射的完整技术流程。
>
> 项目地址: <https://github.com/ben-z-original/enstrect>

---

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [核心分割模型：nnU-Net-S2DS](#2-核心分割模型nnu-net-s2ds)
3. [损伤类别体系](#3-损伤类别体系)
4. [2D 图像分割流程](#4-2d-图像分割流程)
5. [2D 到 3D 的映射机制](#5-2d-到-3d-的映射机制)
6. [多视图概率融合策略](#6-多视图概率融合策略)
7. [损伤几何提取](#7-损伤几何提取)
8. [关键技术总结](#8-关键技术总结)

---

## 1. 系统架构概览

ENSTRECT 的损伤分割系统采用 **"2D 分割 + 3D 映射 + 几何提取"** 的三阶段架构：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   多视角图像     │ --> │  nnU-Net-S2DS   │ --> │  2D 分割概率图   │
│  (RGB Images)   │     │   (2D 分割模型)  │     │ (Softmax/Argmax)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                              ┌────────────────────────┘
                              ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  3D 几何提取     │ <-- │  Mapper+Fuser   │ <-- │  3D 点云投影     │
│(Centerlines/    │     │ (多视图概率融合)  │     │ (PyTorch3D HPR) │
│ Bounding Polys) │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲
        └─────────────────────────────────────────────┐
                          ┌─────────────────┐        │
                          │   3D 点云/Mesh   │────────┘
                          │  (Input Geometry)│
                          └─────────────────┘
```

**核心设计思想**：
- 利用成熟的 2D 深度学习分割模型获得高质量的像素级分类
- 通过多视图几何将 2D 分割结果投影并融合到 3D 空间
- 在 3D 空间中执行后处理和几何提取，获得结构化的损伤信息

---

## 2. 核心分割模型：nnU-Net-S2DS

### 2.1 模型概述

ENSTRECT 使用的核心分割模型是 **nnU-Net-S2DS**（nnU-Net for Structural Surface Damage Segmentation），基于 [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) 框架的 2D 语义分割模型。

### 2.2 网络架构

模型采用标准的 **PlainConvUNet** 架构：

| 参数 | 值 | 说明 |
|------|-----|------|
| 网络类型 | PlainConvUNet | 标准 U-Net 架构 |
| 基础特征数 | 32 | 初始卷积层输出通道 |
| 最大特征数 | 512 | 最深层特征通道数 |
| 下采样层数 | 7 层 | 2^7 = 128 倍下采样 |
| 卷积核尺寸 | 3×3 | 标准卷积核 |
| 池化核尺寸 | 2×2 | 标准池化核 |
| 输入尺寸 | 512×512 | 模型接受的图像分辨率 |
| 归一化方式 | Z-Score | 逐通道独立标准化 |
| 批次大小 | 12 | 训练时的 batch size |

### 2.3 模型初始化与推理配置

```python
class NNUNetS2DSModel(SegmenterInterface):
    def __init__(self, planpath=None, folds=(0,), allow_tqdm=True):
        # 实例化 nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,               # 滑动窗口步长为 patch 的 50%
            use_gaussian=True,                # 使用高斯加权融合重叠区域
            use_mirroring=True,               # TTA 镜像增强
            perform_everything_on_device=True, # 在 GPU 上执行所有操作
            device=torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=allow_tqdm,
        )

        # 自动下载预训练权重（Google Drive）
        if planpath is None:
            planpath = Path(__file__).parent / "checkpoints" / "nnUNetTrainer__nnUNetPlans__2d"

        zippath = Path(planpath).with_suffix(".zip")
        if not zippath.exists() and not Path(planpath).exists():
            url = "https://drive.google.com/uc?id=1UeXzpH76GYtZtyn2IjhDvD5Qu3u91YcC"
            gdown.download(url, str(zippath), quiet=False)

        # 加载训练好的模型
        self.predictor.initialize_from_trained_model_folder(
            str(planpath),
            use_folds=folds,
            checkpoint_name='checkpoint_final.pth',
        )
```

**推理配置说明**：

| 配置项 | 值 | 作用 |
|--------|-----|------|
| `tile_step_size=0.5` | 50% 重叠 | 滑动窗口推理时，相邻 patch 重叠 50%，提高边界区域分割质量 |
| `use_gaussian=True` | 高斯加权 | 对重叠区域使用高斯权重融合，中心区域权重高，边缘权重低 |
| `use_mirroring=True` | TTA 增强 | 测试时进行镜像翻转增强，提升分割鲁棒性 |
| `perform_everything_on_device=True` | GPU 全计算 | 所有操作在 GPU 上完成，减少 CPU-GPU 数据传输 |

---

## 3. 损伤类别体系

### 3.1 类别定义

模型支持 **7 个类别**，定义在 `dataset.json` 中：

| 类别标签 | ID | 类别权重 | 说明 |
|----------|-----|----------|------|
| background | 0 | 1 | 背景（无损伤区域） |
| crack | 1 | 10 | **裂缝**（最高权重，最难检测） |
| spalling | 2 | 4 | **剥落**（混凝土表层脱落） |
| corrosion | 3 | 1 | **腐蚀**（钢筋锈蚀） |
| efflorescence | 4 | 1 | **泛白/风化**（盐析现象） |
| vegetation | 5 | 1 | **植被**（植物生长） |
| control_point | 6 | 1 | **控制点**（测量标记） |

### 3.2 类别权重设计

```python
self.class_weight = torch.tensor([1, 10, 4, 1, 1, 1, 1], dtype=torch.float16)
```

**权重设计原理**：
- **crack（裂缝）权重 = 10**：裂缝在图像中通常表现为细长的低对比度线条，是最难检测的类别，因此在多视图融合时给予最高置信度
- **spalling（剥落）权重 = 4**：剥落区域通常比裂缝大，但仍需要较高的检测优先级
- **其他类别权重 = 1**：常规检测优先级

---

## 4. 2D 图像分割流程

### 4.1 分割前向传播

```python
def __call__(self, img: torch.Tensor):
    # 输入: (3, H, W) RGB 张量
    img = img.to(torch.float32)

    # 逐图像 Z-Score 标准化（不是全局标准化！）
    img = Normalize(img.mean((1, 2)), img.std((1, 2)))(img)

    # 添加 batch 维度 -> (B, C, H, W)
    img = img.unsqueeze(1)

    # 使用 nnU-Net 预测 logits
    logits = self.predictor.predict_logits_from_preprocessed_data(img).squeeze()

    # Softmax 概率 + Argmax 类别
    softmax = torch.nn.functional.softmax(logits.to(torch.float32), dim=0)
    argmax = torch.argmax(logits, dim=0).to(torch.uint8)

    return softmax, argmax
```

**输入输出规格**：

| 张量 | 形状 | 类型 | 说明 |
|------|------|------|------|
| 输入 `img` | `(3, H, W)` | `float32` | RGB 图像，值范围 [0, 255] 或 [0, 1] |
| 输出 `softmax` | `(7, H, W)` | `float32` | 每个像素的 7 类概率分布 |
| 输出 `argmax` | `(H, W)` | `uint8` | 每个像素的预测类别 ID |

### 4.2 概率膨胀（Dilation）—— 裂缝保护机制

针对裂缝在 3D 映射过程中容易丢失的问题，系统提供可选的概率膨胀后处理：

```python
if dilate:
    # 膨胀概率以确保狭窄的裂缝能够传播到（低分辨率）点云
    kernel_size = 3
    soft = F.max_pool2d(
        soft[None],
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    ).squeeze()
```

**技术说明**：
- 使用 **3×3 Max Pooling** 进行概率膨胀
- 作用：将裂缝区域的概率向周围 1 像素扩展
- 目的：确保狭窄的裂缝在映射到低分辨率点云时不会因为离散化而丢失
- 这是裂缝检测的关键后处理步骤

---

## 5. 2D 到 3D 的映射机制

### 5.1 映射流程概览

Mapper 类是 2D→3D 映射的核心，执行以下步骤：

```python
class Mapper:
    def __call__(self, pcd_pynt, dataset, store_probabilities=True, dilate=True):
        # 1. 将 PyntCloud 转换为 PyTorch3D 点云
        pcd_pyt3d = pynt_to_pyt3d(pcd_pynt, self.device)
        num_points = pcd_pyt3d.points_packed().shape[0]
        num_views = len(dataset)
        num_classes = len(self.model.classes)

        # 2. 初始化存储结构
        viewing_conditions = Dict({
            feat: np.zeros((num_points, num_views), dtype=np.float16)
            for feat in ["distances", "angles", "visible"]
        })
        probabilities = torch.zeros((num_points, num_views, num_classes), dtype=torch.float16)

        # 3. 遍历每个视角
        for i, sample in enumerate(dataset):
            # 3.1 将 3D 点投影到 2D 图像坐标
            coords = self.get_image_coordinates(pcd_pyt3d, sample["camera"])

            # 3.2 计算观测条件
            viewing_conditions.distances[:, i] = self.compute_distances(pcd_pyt3d, sample["camera"])
            viewing_conditions.angles[:, i] = self.compute_angular_view_deviation(...)
            viewing_conditions.visible[:, i] = self.compute_visibility(pcd_pyt3d, sample["camera"], coords, True)

            # 3.3 运行 2D 分割
            soft = self.model(sample["image"])[0]
            if dilate:
                soft = F.max_pool2d(soft[None], kernel_size=3, stride=1, padding=1).squeeze()

            # 3.4 将 2D 概率映射到 3D 点
            probabilities[:, i] = soft[:, coords[:, 1], coords[:, 0]].T
            probabilities[viewing_conditions.visible[:, i] != 1, i] = np.nan

        # 4. 多视图融合
        aggr, argmax = self.fuser(probabilities, viewing_conditions)

        # 5. 将结果写回点云
        pcd_pynt.points["argmax"] = np.ubyte(argmax.cpu().numpy())
        for i, cl in enumerate(self.model.classes):
            pcd_pynt.points[cl] = np.ubyte(np.array(pcd_pynt.points["argmax"]) == i)
```

### 5.2 3D 到 2D 投影

使用 PyTorch3D 的 `PerspectiveCameras` 将 3D 点投影到屏幕坐标：

```python
@staticmethod
def get_image_coordinates(pcd_pyt3d, camera):
    coords = camera.transform_points_screen(pcd_pyt3d.points_packed())
    coords = coords.to(torch.int)[:, :2].cpu().numpy()
    return coords
```

### 5.3 观测条件计算

#### 5.3.1 距离计算

```python
@staticmethod
def compute_distances(pcd_pyt3d, camera):
    distances = torch.linalg.norm(
        pcd_pyt3d.points_packed() - camera.get_camera_center(),
        dim=1
    ).cpu().numpy()
    return distances
```

计算每个 3D 点到相机中心的欧氏距离，用于后续的视角质量评估。

#### 5.3.2 视角偏差计算

```python
@staticmethod
def compute_angular_view_deviation(normals, camera):
    cam_view_direction = (camera.R @ torch.tensor([0, 0, 1.0], device=camera.device))
    cam_view_direction /= torch.linalg.norm(cam_view_direction)
    angles = torch.rad2deg(torch.arccos(normals @ cam_view_direction.T)).squeeze()
    return angles
```

计算点云法向量与相机视线方向的夹角，用于判断观测质量：
- 夹角接近 0°：正面观测，质量好
- 夹角接近 90°：切向观测，质量差

#### 5.3.3 可见性检测 —— HPR 算法

```python
@staticmethod
def compute_visibility(pcd_pyt3d, camera, coords, self_occlusion=True, diameter=1000):
    # 1. 基础可见性：点在图像范围内
    visibility = (
        (0 <= coords[:, 0]) * (coords[:, 0] < width) *
        (0 <= coords[:, 1]) * (coords[:, 1] < height)
    )

    if self_occlusion:
        # 2. 自遮挡检测：使用 Open3D 的 Hidden Point Removal (HPR)
        pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd_o3d.normals = o3d.utility.Vector3dVector(normals)

        idxs_visible = np.where(visibility)[0]
        pcd_o3d = pcd_o3d.select_by_index(idxs_visible)
        _, idxs = pcd_o3d.hidden_point_removal(camera_center, diameter)
        idxs_remove = list(set(range(len(idxs_visible))).difference(idxs))
        visibility[idxs_visible[idxs_remove]] = 0
```

**HPR（Hidden Point Removal）算法原理**：
1. 将点云沿相机中心进行球面翻转
2. 计算翻转后点云的凸包
3. 凸包上的点即为可见点，内部的点为被遮挡点

---

## 6. 多视图概率融合策略

### 6.1 Fuser 融合逻辑

```python
class Fuser:
    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def __call__(self, probabilities, viewing_conditions):
        # 1. 应用类别权重
        if self.class_weight is not None:
            probabilities = probabilities * self.class_weight

        # 2. 应用视角约束：只使用法向量与视线方向夹角在 50° 以内的观测
        # 即视角在 130°-230° 之间
        mask = (230 < viewing_conditions.angles) + (viewing_conditions.angles < 130)
        probabilities[mask, :] = torch.nan

        # 3. 跨视图平均概率（忽略 nan 值）
        aggr = torch.nanmean(probabilities, dim=1)

        # 4. 处理背景类
        aggr[:, 1:] = torch.nan_to_num(aggr[:, 1:], nan=0.0)
        aggr[:, 0] = torch.where(
            torch.isnan(aggr[:, 0]),
            1 - aggr[:, 1:].sum(dim=1),
            aggr[:, 0]
        )

        # 5. 取概率最大的类别
        argmax = torch.argmax(aggr, dim=1)
        return aggr, argmax
```

### 6.2 融合策略详解

| 步骤 | 操作 | 目的 |
|------|------|------|
| **类别加权** | crack ×10, spalling ×4 | 提升重要损伤类别的检测优先级 |
| **视角过滤** | 130° < angle < 230° | 只使用正面或近正面观测，过滤低质量侧视 |
| **NaN 均值** | `torch.nanmean` | 跨所有有效视图计算平均概率，自动处理缺失值 |
| **背景补全** | `1 - 前景概率之和` | 如果某点在所有视图中都是背景，正确计算背景概率 |
| **Argmax 决策** | `torch.argmax` | 选择概率最高的类别作为最终预测 |

---

## 7. 损伤几何提取

### 7.1 Exposed Rebar 后处理

Exposed Rebar（暴露钢筋）是一种复合损伤类型，通过后处理从 spalling 和 corrosion 推导：

```python
def prepare_exposed_rebar(pcd_pynt, eps_m=0.01):
    # exposed_rebar = spalling + corrosion 的交集
    pcd_pynt.points["exposed_rebar"] = pcd_pynt.points["spalling"] + pcd_pynt.points["corrosion"]
    idxs = np.nonzero(pcd_pynt.points["exposed_rebar"])[0]

    # 使用 DBSCAN 聚类
    cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(points)

    for i in np.unique(cluster):
        idxs_cluster = idxs[cluster == i]
        spalling_count = np.sum(pcd_pynt.points["spalling"][idxs_cluster])
        corrosion_count = np.sum(pcd_pynt.points["corrosion"][idxs_cluster])

        if spalling_count == len(idxs_cluster):
            # 纯 spalling -> 不是 exposed_rebar
            pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = 0
        elif corrosion_count == len(idxs_cluster):
            # 纯 corrosion -> 不是 exposed_rebar
            pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = 0
        else:
            # 混合区域 -> exposed_rebar
            pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = 1
            pcd_pynt.points.loc[idxs_cluster, "spalling"] = 0
```

**逻辑说明**：
- Exposed Rebar 定义为同时包含 **spalling（剥落）** 和 **corrosion（腐蚀）** 的空间区域
- 纯 spalling 区域：混凝土剥落但钢筋未腐蚀
- 纯 corrosion 区域：钢筋腐蚀但混凝土未剥落
- 混合区域：混凝土剥落且钢筋腐蚀 → **Exposed Rebar**

### 7.2 裂缝中心线提取

裂缝使用 **Laplacian-Based Contraction (LBC)** 算法提取中心线：

```python
def extract_centerlines(pcd_pynt, category="crack", eps_m=0.02, min_points=20,
                        min_samples_cluster=1, init_contraction=10.):
    # 1. DBSCAN 聚类分离不同的裂缝实例
    cluster = DBSCAN(eps=eps_m, min_samples=min_samples_cluster).fit_predict(points_np)

    for cluster_id, cluster_count in zip(cluster_ids, cluster_counts):
        # 2. Laplacian-Based Contraction (LBC) 收缩点云到中心线
        lbc = LBC(
            point_cloud=subcloud,
            down_sample=0.001,
            init_contraction=init_contraction,
            init_attraction=10.,
            max_attraction=2048,
            termination_ratio=0.003,
            max_iteration_steps=20
        )
        lbc.extract_skeleton()

        # 3. 构建完全连接图，计算最小生成树 (MST)
        G = nx.Graph()
        G.add_weighted_edges_from(entries)
        G = nx.minimum_spanning_tree(G)

        # 4. 处理分叉点，分离为多条简单路径
```

**技术栈**：

| 技术 | 用途 | 说明 |
|------|------|------|
| **DBSCAN** | 空间聚类 | 分离不同的裂缝实例 |
| **LBC** | 点云收缩 | 将裂缝点云收缩为骨架线 |
| **MST** | 图简化 | 从收缩后的点构建最小生成树 |
| **NetworkX** | 图处理 | 路径提取和分叉处理 |

### 7.3 边界多边形提取

剥落、腐蚀等面状损伤使用 **Alpha Shape** 提取边界多边形：

```python
def extract_bounding_polygons(pcd_pynt, category="corrosion", eps_m=0.01, min_points=100):
    # 1. DBSCAN 聚类
    cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(points_np)

    for cluster_id, cluster_count in zip(cluster_ids, cluster_counts):
        # 2. PCA 降维到 2D 平面
        points_mapped = pca.fit_transform(points_subcloud)

        # 3. Alpha Shape 提取边界
        alpha_shape = alphashape.alphashape(points_mapped, 100)

        # 4. 提取边界坐标并映射回 3D
        bound_coords = np.array(geom.boundary.coords)
        bound_points = points_subcloud[bound_idxs[:-1]]

        # 5. 构建闭合多边形图
        H.add_edges_from(edges)
```

**技术栈**：

| 技术 | 用途 | 说明 |
|------|------|------|
| **DBSCAN** | 空间聚类 | 分离不同的损伤实例 |
| **PCA** | 降维 | 将 3D 点云投影到最佳拟合平面 |
| **Alpha Shape** | 边界提取 | 从 2D 点集提取边界多边形 |
| **Shapely** | 几何操作 | 多边形处理和坐标提取 |

---

## 8. 关键技术总结

### 8.1 核心技术点

| 序号 | 技术 | 应用场景 | 重要性 |
|------|------|----------|--------|
| 1 | **nnU-Net v2** | 2D 图像分割骨干 | ⭐⭐⭐⭐⭐ |
| 2 | **PyTorch3D** | 3D-2D 投影和相机管理 | ⭐⭐⭐⭐⭐ |
| 3 | **HPR (Hidden Point Removal)** | 自遮挡检测 | ⭐⭐⭐⭐⭐ |
| 4 | **视角约束（130°-230°）** | 过滤低质量观测 | ⭐⭐⭐⭐ |
| 5 | **类别加权融合** | 提升裂缝检测精度 | ⭐⭐⭐⭐ |
| 6 | **概率膨胀** | 保护狭窄裂缝 | ⭐⭐⭐⭐ |
| 7 | **LBC 骨架提取** | 裂缝中心线提取 | ⭐⭐⭐⭐ |
| 8 | **Alpha Shape** | 损伤边界多边形 | ⭐⭐⭐⭐ |
| 9 | **DBSCAN 聚类** | 分离不同损伤实例 | ⭐⭐⭐ |

### 8.2 设计亮点

1. **多视图融合策略**：通过概率加权而非简单的投票机制，充分利用了每个视角的置信度信息
2. **裂缝保护机制**：概率膨胀确保细小裂缝在 3D 离散化过程中不丢失
3. **复合损伤推导**：Exposed Rebar 不是直接分割得到，而是通过 spalling 和 corrosion 的空间关系推导
4. **视角质量过滤**：通过法向量-视线夹角过滤，只使用高质量的正面观测
5. **结构化输出**：裂缝输出为中心线（图结构），面状损伤输出为多边形（边界），便于工程量化分析

### 8.3 潜在改进方向

- **时序一致性**：当前系统未利用视频时序信息，可引入时序一致性约束
- **不确定性量化**：可输出每个 3D 点的不确定性估计
- **增量更新**：支持增量式添加新视角并更新融合结果
- **边缘设备部署**：优化模型以实现移动端或嵌入式设备部署

---

> **文档版本**: v1.0
> **生成日期**: 2026-04-19
> **项目**: ENSTRECT - Engineering Structures Texture and Crack Tracker
> **作者**: Herta (AI Assistant)
