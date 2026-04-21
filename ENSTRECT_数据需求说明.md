# ENSTRECT 数据需求说明文档

> 详细说明运行 ENSTRECT 项目所需的输入数据格式、目录结构和准备方法

---

## 目录

1. [概述](#1-概述)
2. [2D图像分割（单帧/多帧）](#2-2d图像分割单帧多帧)
3. [完整3D流程（2.5D分割）](#3-完整3d流程25d分割)
4. [数据格式详解](#4-数据格式详解)
5. [自建数据指南](#5-自建数据指南)
6. [数据示例](#6-数据示例)

---

## 1. 概述

ENSTRECT 支持两种运行模式，对应不同的数据需求：

| 模式 | 输入数据 | 输出结果 | 用途 |
|------|----------|----------|------|
| **2D分割模式** | 单张/多张 RGB 图像 | 2D分割掩码 | 快速检测、批量处理 |
| **3D流程模式** | 3D模型 + 多视图图像 + 相机参数 | 3D点云分割 + 损伤几何 | 完整分析、量化测量 |

---

## 2. 2D图像分割（单帧/多帧）

### 2.1 数据需求清单

**必需数据**：
- [x] RGB 图像（`.jpg` 或 `.png`）

**可选数据**：
- [ ] 相机参数（用于3D投影，2D模式下不需要）

### 2.2 图像要求

| 属性 | 要求 | 说明 |
|------|------|------|
| **格式** | `.jpg` 或 `.png` | 标准图像格式 |
| **色彩** | RGB 彩色图像 | 不支持灰度图 |
| **分辨率** | 建议 ≥ 512×512 | 模型输入为512×512，自动缩放 |
| **内容** | 结构表面照片 | 混凝土、桥梁、建筑等 |

### 2.3 目录结构

```
images/                     # 图像目录（任意命名）
├── 0000.jpg               # 图像文件（任意命名）
├── 0001.jpg
├── 0002.jpg
└── ...
```

### 2.4 运行方式

```python
from run_multiview_segmentation import process_multiview

# 处理指定目录的所有图像
results = process_multiview(
    images_dir="path/to/your/images",
    output_dir="path/to/output",
    max_images=None  # None表示处理全部
)
```

### 2.5 输出结果

```
output/
├── segmentation_0000.png      # 每张图像的分割可视化
├── segmentation_0001.png
├── multiview_summary.png      # 汇总对比图
└── （终端输出统计信息）
```

---

## 3. 完整3D流程（2.5D分割）

### 3.1 数据需求清单

**必需数据**：
- [x] 3D模型文件（`.obj` 或 `.ply`）
- [x] 多视角 RGB 图像（`.jpg`）
- [x] 相机参数文件（`cameras.json`）

**可选数据**：
- [ ] 纹理贴图（`.png`）
- [ ] 真实标注（用于对比，`.obj`）

### 3.2 详细数据要求

#### 3.2.1 3D模型（mesh）

| 属性 | 要求 | 说明 |
|------|------|------|
| **格式** | `.obj`（推荐）或 `.ply` | Wavefront OBJ 或 Polygon File Format |
| **单位** | **米（m）** | ⚠️ 重要！影响后续参数计算 |
| **类型** | 三角网格 | 支持多边形网格 |
| **纹理** | 可选 `.mtl` + `.png` | 材质和纹理贴图 |

**文件结构**：
```
mesh/
├── mesh.obj          # 主模型文件
├── mesh.mtl          # 材质文件（可选）
└── mesh.png          # 纹理贴图（可选）
```

#### 3.2.2 多视角图像

| 属性 | 要求 | 说明 |
|------|------|------|
| **格式** | `.jpg`（推荐）或 `.png` | 有损压缩即可 |
| **数量** | 建议 10-30 张 | 覆盖目标结构的多视角 |
| **命名** | 与 cameras.json 键名对应 | 如 `0000.jpg` 对应键 `"0000"` |
| **分辨率** | 任意 | 程序会自动缩放处理 |
| **质量** | 清晰、曝光正常 | 影响分割精度 |

**拍摄建议**：
- 围绕目标结构从不同角度拍摄
- 相邻图像之间有 30-50% 重叠
- 避免过曝、欠曝、运动模糊

#### 3.2.3 相机参数（cameras.json）

**最重要的文件！** 定义每张图像的拍摄相机参数。

**格式**：JSON

**单位**：
- 焦距、主点：像素（px）
- 平移向量：米（m）
- 旋转矩阵：无单位

**结构示例**：
```json
{
  "0000": {
    "focal_length": [[ fx, fy ]],
    "principal_point": [[ cx, cy ]],
    "image_size": [[ height, width ]],
    "R": [
      [ [ r11, r12, r13 ] ],
      [ [ r21, r22, r23 ] ],
      [ [ r31, r32, r33 ] ]
    ],
    "T": [[ tx, ty, tz ]],
    "in_ndc": false
  },
  "0001": { ... },
  "0002": { ... }
}
```

**参数说明**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `focal_length` | `(1, 2)` | 焦距 `[fx, fy]`，单位：像素 |
| `principal_point` | `(1, 2)` | 主点 `[cx, cy]`，通常是图像中心 |
| `image_size` | `(1, 2)` | 图像尺寸 `[height, width]` |
| `R` | `(3, 3)` | 旋转矩阵（世界坐标系 → 相机坐标系）|
| `T` | `(1, 3)` | 平移向量，单位：米 |
| `in_ndc` | `bool` | 是否为归一化设备坐标，通常 `false` |

**坐标系说明**：
- 使用 **PyTorch3D 坐标系**
- X轴：向右
- Y轴：向上
- Z轴：从相机指向场景（视线方向）

---

## 4. 数据格式详解

### 4.1 完整目录结构（3D流程）

```
project/                          # 项目根目录
├── mesh/                         # 3D模型目录
│   ├── mesh.obj                 # 网格模型（必需）
│   ├── mesh.mtl                 # 材质文件（可选）
│   └── mesh.png                 # 纹理贴图（可选）
│
├── views/                        # 多视角图像目录（必需）
│   ├── 0000.jpg                 # 图像命名需与cameras.json对应
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
│
├── cameras.json                  # 相机参数（必需）
│
├── annotations/                  # 真实标注（可选，用于对比）
│   ├── crack.obj                # 裂缝标注
│   ├── spalling.obj             # 剥落标注
│   └── corrosion.obj            # 腐蚀标注
│
└── out/                          # 输出目录（自动生成）
    ├── pcd_100000_processed.ply  # 处理后的点云
    ├── crack.obj                 # 提取的裂缝中心线
    ├── spalling.obj              # 剥落实例
    ├── corrosion.obj             # 腐蚀实例
    └── exposed_rebar.obj         # 暴露钢筋实例
```

### 4.2 cameras.json 完整示例

```json
{
  "0000": {
    "focal_length": [[ 5758.29, 5758.29 ]],
    "principal_point": [[ 3000.0, 2000.0 ]],
    "image_size": [[ 4000, 6000 ]],
    "R": [
      [ [ 0.936, -0.016, 0.352 ] ],
      [ [ 0.023, 0.999, -0.026 ] ],
      [ [ -0.351, 0.033, 0.936 ] ]
    ],
    "T": [[ 1.234, 0.567, -3.456 ]],
    "in_ndc": false
  },
  "0001": {
    "focal_length": [[ 5758.29, 5758.29 ]],
    "principal_point": [[ 3000.0, 2000.0 ]],
    "image_size": [[ 4000, 6000 ]],
    "R": [
      [ [ 0.912, -0.023, 0.410 ] ],
      [ [ 0.031, 0.998, -0.052 ] ],
      [ [ -0.409, 0.061, 0.911 ] ]
    ],
    "T": [[ 1.456, 0.678, -3.234 ]],
    "in_ndc": false
  }
}
```

### 4.3 OBJ文件格式

**mesh.obj 示例**：
```obj
# OBJ file
mtllib mesh.mtl
v 1.0 0.0 0.0    # 顶点坐标 (x, y, z)
v 0.0 1.0 0.0
v 0.0 0.0 1.0
vn 0.0 0.0 1.0   # 顶点法线 (nx, ny, nz)
vt 0.0 0.0       # 纹理坐标 (u, v)
f 1/1/1 2/2/1 3/3/1  # 面定义 (v/vt/vn)
```

**单位要求**：必须是 **米（meters）**

---

## 5. 自建数据指南

### 5.1 数据准备流程

```
1. 3D重建
   └── 使用 photogrammetry 软件（如 Metashape、COLMAP）
   └── 输入：多视角照片
   └── 输出：mesh.obj + cameras.xml

2. 格式转换
   └── 将 cameras.xml 转换为 cameras.json（PyTorch3D格式）
   └── 确保单位是米

3. 数据组织
   └── 按标准目录结构存放
   └── mesh/ + views/ + cameras.json

4. 验证
   └── 运行 ENSTRECT 检查数据完整性
```

### 5.2 推荐的3D重建软件

| 软件 | 类型 | 导出格式 | 难度 |
|------|------|----------|------|
| **Agisoft Metashape** | 商业 | .obj + .xml | ⭐⭐ |
| **COLMAP** | 开源 | .ply + .txt | ⭐⭐⭐ |
| **Meshroom** | 开源 | .obj + .sfm | ⭐⭐⭐ |
| **RealityCapture** | 商业 | 多种格式 | ⭐ |

### 5.3 相机标定工具

如果已有3D模型但没有相机参数，可以使用：
- **PyTorch3D 相机拟合**
- **OpenCV PnP算法**
- **Blender 相机匹配**

### 5.4 注意事项

⚠️ **单位检查**：
- 3D模型必须是 **米（m）**
- 如果原始数据是毫米，需要除以 1000
- 错误的单位会导致投影失败或参数失效

⚠️ **坐标系对齐**：
- 确保 3D 模型和相机参数使用同一坐标系
- 检查模型的原点和朝向

⚠️ **命名对应**：
- `cameras.json` 中的键名必须与图像文件名对应
- 如 `"0000"` 对应 `0000.jpg`

---

## 6. 数据示例

### 6.1 官方测试数据

ENSTRECT 官方提供两组桥梁测试数据：

**Bridge B**（裂缝为主）：
- 2个结构段（segment_dev, segment_test）
- 每段 ~20 张图像
- 分辨率：~6000×4000

**Bridge G**（剥落+腐蚀）：
- 2个结构段
- 每段 ~20 张图像
- 多种损伤类型

**下载方式**：
```bash
cd enstrect_lib
python -m enstrect.datasets.download
```

### 6.2 最小数据集示例

最简单的可运行数据集只需3个文件：

```
minimal_example/
├── mesh/
│   └── mesh.obj          # 简单的立方体模型
├── views/
│   ├── 0000.jpg          # 至少1张图像
│   └── 0001.jpg
└── cameras.json          # 对应的相机参数
```

### 6.3 快速测试数据

如果只是测试2D分割功能，可以用任意建筑/结构照片：

```
test_images/
├── concrete_wall_01.jpg
├── bridge_crack_01.jpg
└── building_damage_01.jpg
```

---

## 7. 常见问题

### Q1: 没有相机参数可以运行吗？
- **2D分割模式**：✅ 可以，只需要图像
- **3D流程模式**：❌ 不可以，必须提供 cameras.json

### Q2: 可以用自己的3D模型吗？
✅ 可以，但需要：
1. 准备对应的 cameras.json
2. 确保单位为米
3. 格式为 .obj 或 .ply

### Q3: 图像和相机参数如何对应？
- `cameras.json` 中的键名（如 `"0000"`）必须与图像文件名（`0000.jpg`）对应
- 不区分大小写，但不要有扩展名

### Q4: 一个项目可以处理多个结构吗？
✅ 可以，每个结构放在独立的目录中，分别运行：
```bash
python -m enstrect.run --obj_or_ply_path structure_A/mesh/mesh.obj ...
python -m enstrect.run --obj_or_ply_path structure_B/mesh/mesh.obj ...
```

### Q5: 数据量很大怎么办？
- 使用 `--scale` 参数降低图像分辨率（如 `--scale 0.5`）
- 使用 `--num_points` 减少点云采样数量
- 使用 `--select_views` 只选择部分视角

---

## 8. 参考资源

- **ENSTRECT 官方仓库**: https://github.com/ben-z-original/enstrect
- **PyTorch3D 相机文档**: https://pytorch3d.org/docs/cameras
- **nnU-Net 文档**: https://github.com/MIC-DKFZ/nnUNet
- **COLMAP 教程**: https://colmap.github.io/

---

**文档版本**: v1.0  
**更新日期**: 2026-04-19  
**作者**: Herta (AI Assistant)
