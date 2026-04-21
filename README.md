# ENSTRECT 多视图结构损伤分割

> 基于 [ENSTRECT](https://github.com/ben-z-original/enstrect) 的多视图结构表面损伤检测演示项目

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 项目简介

本项目是基于 **ENSTRECT** (Engineering Structures Texture and Crack Tracker) 的多视图结构损伤分割演示。ENSTRECT 是一个用于结构表面损伤检测的 2.5D 深度学习框架，支持裂缝、剥落、腐蚀等多种损伤类型的检测。

### 核心特性

- 基于 **nnU-Net-S2DS** 的 2D 语义分割模型
- 支持 7 种损伤类别：裂缝、剥落、腐蚀、泛白、植被、控制点
- **🆕 图像质量检测与数据清洗** - 自动过滤运动模糊、失焦等低质量图像
- 多视图图像批量处理
- 自动统计损伤像素分布
- 可视化分割结果与叠加显示

---

## 损伤类别说明

| 类别 | 颜色 | 说明 | 权重 |
|------|------|------|------|
| 🔴 Crack | 红色 | 裂缝 | 10 |
| 🟢 Spalling | 绿色 | 混凝土剥落 | 4 |
| 🔵 Corrosion | 蓝色 | 钢筋腐蚀 | 1 |
| 🟡 Efflorescence | 黄色 | 泛白/风化 | 1 |
| 🟣 Vegetation | 紫色 | 植被 | 1 |
| 🔵 Control Point | 青色 | 控制点 | 1 |
| ⚫ Background | 黑色 | 背景 | 1 |

---

## 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 CUDA 12.4+)
- **显存**: 至少 8GB
- **内存**: 至少 16GB

### 软件环境
```
Python >= 3.10
PyTorch >= 2.5.0
CUDA >= 12.4
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/ZyfNO2/enstrect-multiview-segmentation.git
cd enstrect-multiview-segmentation
```

### 2. 安装依赖

```bash
# 创建conda环境
conda create -n enstrect python=3.10
conda activate enstrect

# 安装PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 下载ENSTRECT核心库

```bash
# 克隆ENSTRECT仓库到子目录
git clone https://github.com/ben-z-original/enstrect.git enstrect_lib
```

### 4. 下载测试数据

```bash
# 下载官方测试数据 (Bridge B & G)
cd enstrect_lib
python -m enstrect.datasets.download
```

数据将自动下载到 `enstrect_lib/src/enstrect/assets/` 目录

### 5. 运行分割

```bash
# 运行多视图分割
python run_multiview_segmentation.py
```

结果将保存到 `output/` 目录

---

## 项目结构

```
.
├── README.md                              # 项目说明
├── requirements.txt                       # Python依赖
├── run_multiview_segmentation.py          # 主运行脚本（基础版）
├── run_multiview_segmentation_with_filter.py  # 主运行脚本（带数据清洗）
├── image_quality_filter.py                # 图像质量检测模块
├── enstrect_lib/                          # ENSTRECT核心库
│   └── src/enstrect/
│       ├── assets/                        # 测试数据
│       │   ├── bridge_b/                  # Bridge B 数据集
│       │   └── bridge_g/                  # Bridge G 数据集
│       └── segmentation/                  # 分割模型
└── output/                                # 输出目录
    ├── segmentation_*.png                 # 分割结果
    ├── multiview_summary.png              # 汇总图
    └── quality_report.txt                 # 质量分析报告（带过滤版）
```

---

## 使用说明

### 基本用法

#### 1. 基础分割（无过滤）

```python
from run_multiview_segmentation import process_multiview

# 处理指定目录的多视图图像
results = process_multiview(
    images_dir="path/to/your/images",
    output_dir="path/to/output",
    max_images=10
)
```

#### 2. 带数据清洗的分割（推荐）⭐

```python
from run_multiview_segmentation_with_filter import process_multiview_with_filter

# 自动过滤模糊图像后进行分割
results = process_multiview_with_filter(
    images_dir="path/to/your/images",
    output_dir="path/to/output",
    max_images=10,
    enable_quality_filter=True,      # 启用质量过滤
    blur_threshold=100.0,            # 模糊检测阈值
    min_quality_score=0.3,           # 最低质量评分
    quality_method='laplacian',      # 检测方法
    max_blurry_ratio=0.5             # 最大允许模糊比例
)
```

或者直接运行：
```bash
# 带数据清洗的版本（推荐用于实际数据）
python run_multiview_segmentation_with_filter.py
```

### 自定义参数

在 `run_multiview_segmentation.py` 中修改以下参数：

```python
# 输入图像目录
images_dir = r"G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets\bridge_b\segment_test\views"

# 输出目录
output_dir = r"G:\Zed\ENSTRECTtest\output"

# 处理图像数量
max_images = 5  # 设置为None处理全部
```

---

## 🆕 图像质量检测与数据清洗

本项目提供了强大的图像质量检测功能，可在分割前自动过滤运动模糊、失焦等低质量图像，提高分割精度。

### 功能特点

- **多种检测算法**：拉普拉斯方差、Sobel梯度、FFT频域分析、综合评估
- **可配置阈值**：根据实际需求调整模糊判定标准
- **批量处理**：自动检测并分类大量图像
- **质量报告**：生成详细的图像质量分析报告

### 检测方法对比

| 方法 | 原理 | 适用场景 | 计算速度 |
|------|------|----------|----------|
| **Laplacian** | 拉普拉斯算子方差 | 通用运动模糊检测 | ⚡ 快 |
| **Sobel** | 边缘梯度强度 | 边缘清晰度评估 | ⚡ 快 |
| **FFT** | 频域高频能量 | 失焦/运动模糊 | 🐢 较慢 |
| **Combined** | 多方法综合 | 高精度要求 | 🐢 较慢 |

### 使用方法

#### 单独使用质量检测

```python
from image_quality_filter import filter_blurry_images

# 批量检测并分类图像
results = filter_blurry_images(
    images_dir="path/to/images",
    output_dir="path/to/output",      # 可选，自动分类保存
    blur_threshold=100.0,             # 模糊阈值
    method='laplacian',               # 检测方法
    visualize=True                    # 生成可视化报告
)

# 结果包含清晰和模糊图像列表
sharp_images = results['sharp']
blurry_images = results['blurry']
```

#### 集成到分割流程

```python
from run_multiview_segmentation_with_filter import process_multiview_with_filter

results = process_multiview_with_filter(
    images_dir="path/to/images",
    output_dir="path/to/output",
    enable_quality_filter=True,
    blur_threshold=100.0,             # 拉普拉斯方差阈值
    min_quality_score=0.3,            # 最低质量评分(0-1)
    quality_method='laplacian',       # 检测方法
    max_blurry_ratio=0.5              # 最大允许模糊比例
)
```

### 参数调优指南

#### blur_threshold 设置建议

| 场景 | 推荐阈值 | 说明 |
|------|----------|------|
| 严格过滤 | 150-200 | 只保留非常清晰的图像 |
| 平衡模式 | 80-120 | 默认推荐值 |
| 宽松模式 | 50-80 | 尽可能保留更多图像 |

#### 不同分辨率下的阈值参考

- **4K图像** (3840×2160)：阈值 150-250
- **1080p图像** (1920×1080)：阈值 80-150
- **512×512**：阈值 30-80

### 质量报告输出

运行带过滤的版本后，会生成以下额外输出：

```
output_filtered/
├── segmentation_*.png           # 分割结果
├── multiview_summary.png        # 汇总图
├── quality_analysis.png         # 质量分布可视化
└── quality_report.txt           # 详细质量报告
```

**quality_report.txt 示例**：
```
============================================================
图像质量分析报告
============================================================

总处理图像数: 8

质量评分统计:
  平均分: 0.623
  最高分: 0.891
  最低分: 0.234
  中位数: 0.612

每张图像的详细信息:

图像: 0000.jpg
  质量评分: 0.445
  拉普拉斯方差: 89.23
  损伤统计:
    crack: 7343 px (0.020%)
```

### 注意事项

1. **阈值调整**：不同相机/场景可能需要不同的阈值，建议先用小批量数据测试
2. **光照影响**：过曝/欠曝图像可能被误判为模糊，注意控制拍摄条件
3. **噪声干扰**：高ISO噪声可能影响检测结果，必要时先做降噪处理

---

## 示例输出

### 分割结果示例

程序会为每张输入图像生成三视图对比：
- **左图**: 原始输入图像
- **中图**: 分割掩码 (彩色编码)
- **右图**: 叠加可视化 (半透明融合)

### 统计报告

程序会自动输出每个类别的像素统计：

```
============================================================
分割结果统计
============================================================

图像: 0000.jpg
  background: 36144973 pixels (99.98%)
  crack: 7343 pixels (0.02%)
  vegetation: 4 pixels (0.00%)

图像: 0004.jpg
  background: 36115886 pixels (99.90%)
  crack: 27930 pixels (0.08%)
  spalling: 7946 pixels (0.02%)
  efflorescence: 555 pixels (0.00%)
```

---

## 技术细节

### 分割模型架构

- **骨干网络**: PlainConvUNet (nnU-Net v2)
- **输入尺寸**: 512×512
- **下采样层数**: 7层 (128倍下采样)
- **归一化**: Z-Score标准化
- **推理配置**:
  - 滑动窗口步长: 50% 重叠
  - TTA镜像增强
  - 高斯加权融合

### 关键技术点

1. **类别加权**: crack (10×) 和 spalling (4×) 有更高权重
2. **概率膨胀**: 使用Max Pooling保护狭窄裂缝
3. **批量处理**: 支持多图像自动批处理
4. **GPU加速**: 自动检测并使用CUDA

---

## 数据集

本项目使用 ENSTRECT 官方提供的桥梁检测数据集：

### Bridge B
- **场景**: 桥梁结构段
- **损伤类型**: 主要是裂缝 (crack)
- **图像数量**: 40+ 张多视角图像
- **分辨率**: ~6000×4000 像素

### Bridge G
- **场景**: 桥梁结构段
- **损伤类型**: 剥落 (spalling) + 腐蚀 (corrosion)
- **图像数量**: 40+ 张多视角图像
- **分辨率**: ~6000×4000 像素

---

## 注意事项

### 1. 模型权重下载

首次运行时会自动从 Google Drive 下载预训练模型 (~465MB)：
- 下载地址: https://drive.google.com/uc?id=1UeXzpH76GYtZtyn2IjhDvD5Qu3u91YcC
- 保存位置: `enstrect_lib/src/enstrect/segmentation/checkpoints/`

### 2. CUDA兼容性

- 当前配置: PyTorch 2.5.1 + CUDA 12.4
- 如遇CUDA版本不匹配，请根据实际环境调整PyTorch版本

### 3. 内存要求

- 高分辨率图像 (6000×4000) 需要较多显存
- 如显存不足，可在代码中增加图像缩放

---

## 进阶：完整3D流程

本项目目前只实现了 **2D图像分割** 部分。完整的 ENSTRECT 流程还包括：

1. **2D分割** → 本项目的核心功能 ✅
2. **3D点云投影** → 需要 PyTorch3D
3. **多视图融合** → 概率加权融合
4. **损伤几何提取** → 中心线/边界多边形

如需运行完整3D pipeline，请安装 PyTorch3D：

```bash
# Windows安装PyTorch3D (预编译wheel)
pip install https://miropsota.github.io/torch_packages_builder/pytorch3d/pytorch3d-0.7.8%2Bpt2.5.1-cp310-cp310-win_amd64.whl
```

然后运行：
```bash
python -m enstrect.run \
    --obj_or_ply_path path/to/mesh.obj \
    --images_dir path/to/views \
    --cameras_path path/to/cameras.json \
    --out_dir path/to/output
```

---

## 参考文献

```bibtex
@inproceedings{benz2024enstrect,
  title={ENSTRECT: 2.5D Instance Segmentation for Structured Damage Detection},
  author={Benz, Christian and Rodehorst, Volker},
  booktitle={European Conference on Computer Vision},
  year={2024}
}

@inproceedings{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  year={2021}
}
```

---

## 许可证

本项目基于 MIT 许可证开源。

原始 ENSTRECT 项目: https://github.com/ben-z-original/enstrect

---

## 致谢

- [ENSTRECT](https://github.com/ben-z-original/enstrect) - 原始项目作者 Christian Benz
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - 分割模型框架
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) - 3D数据处理

---

> 💡 **提示**: 本项目仅用于学习和研究目的。如需商业使用，请参考原始 ENSTRECT 项目的许可证要求。
