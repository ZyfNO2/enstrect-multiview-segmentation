# ENSTRECT 测试数据资源汇总

> 本天才为开拓者搜集了所有可用的结构损伤检测数据集~ ✧(≖ ◡ ≖✿)

---

## 一、项目自带数据

### 1.1 示例图片
- **路径**: `G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets\example_image.jpg`
- **内容**: 混凝土表面损伤照片（包含裂缝、剥落等）
- **用途**: 可用于测试分割模型单图推理

### 1.2 官方完整数据集（推荐）
- **来源**: Google Drive
- **下载脚本**: `python -m enstrect.datasets.download`
- **文件ID**: `1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW`
- **内容**: 
  - Bridge B (2个测试段)
  - Bridge G (2个测试段)
  - 每个段包含:
    - `mesh/mesh.obj` - 3D网格模型
    - `views/` - 多视角图像
    - `cameras.json` - 相机参数
    - `out/` - 输出目录

**⚠️ 注意**: 当前网络环境无法访问 Google Drive，需要科学上网或手动下载

**手动下载地址**:
```
https://drive.google.com/uc?id=1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW
```

下载后解压到:
```
enstrect/src/enstrect/assets/segments/
```

---

## 二、公开裂缝检测数据集

### 2.1 dacl10k - 桥梁损伤分割数据集 ⭐推荐

| 属性 | 详情 |
|------|------|
| **全称** | damage classification 10k |
| **规模** | 9,920 张图像 |
| **类别** | 19类 (13种损伤 + 6种构件) |
| **任务** | 多标签语义分割 |
| **来源** | 德国真实桥梁检测数据 |

**损伤类别**:
- Crack (裂缝)
- Alligator Crack (鳄鱼裂缝)
- Spalling (剥落)
- Rust (锈蚀)
- Efflorescence (泛白)
- Exposed Rebars (外露钢筋)
- 等13种...

**下载链接**:
- GigaMove: https://gigamove.rwth-aachen.de (搜索 dacl10k)
- AWS: 见 dacl10k-toolkit GitHub
- Hugging Face: https://huggingface.co/datasets

**GitHub Toolkit**:
```bash
pip install git+https://github.com/phiyodr/dacl10k-toolkit
```

**引用**:
```bibtex
@inproceedings{flotzinger2024dacl10k,
  title={dacl10k: Benchmark for Semantic Bridge Damage Segmentation},
  author={Flotzinger, Johannes and R{\"o}sch, Philipp J and Braml, Thomas},
  booktitle={WACV},
  year={2024}
}
```

---

### 2.2 Bochum Crack Data Set (BCD)

| 属性 | 详情 |
|------|------|
| **机构** | Ruhr-Universität Bochum |
| **规模** | 370张 RGB 图像 |
| **内容** | 混凝土建筑裂缝 |
| **标注** | 二值分割掩码 |
| **任务** | 分类/检测/语义分割 |

**下载地址**:
```
https://www.inf.bi.ruhr-uni-bochum.de/iib/forschung/datensaetze/index.html.en
```

**联系**: Firdes Celik

**引用**:
```bibtex
@dataset{bochum_crack_dataset,
  title={Bochum Crack Data Set},
  author={Celik, Firdes},
  institution={Ruhr-Universit{\"a}t Bochum},
  year={2023}
}
```

---

### 2.3 Crack500

| 属性 | 详情 |
|------|------|
| **规模** | 500张图像 |
| **分割** | 训练350 / 验证50 / 测试100 |
| **内容** | 沥青路面裂缝 |
| **来源** | 多种表面类型 |

**下载链接**:
```bash
# GitHub 镜像
https://github.com/yihui-he/crack500

# 直接下载
https://github.com/yihui-he/crack500/archive/refs/heads/main.zip
```

---

### 2.4 OmniCrack30k ⭐大规模推荐

| 属性 | 详情 |
|------|------|
| **规模** | 30,000 张样本 |
| **像素** | 90亿像素 |
| **来源** | 20+ 个数据集整合 |
| **材料** | 沥青、陶瓷、混凝土、砖石、钢材 |
| **任务** | 通用裂缝分割 |

**下载**:
- 论文: CVPRW 2024
- 项目页面: 搜索 "OmniCrack30k CVPR"

**引用**:
```bibtex
@inproceedings{benz2024omnicrack30k,
  title={OmniCrack30k: A Benchmark for Crack Segmentation and the Reasonable Effectiveness of Transfer Learning},
  author={Benz, Christian and Rodehorst, Volker},
  booktitle={CVPR Workshops},
  year={2024}
}
```

---

### 2.5 Concrete Crack Images for Classification

| 属性 | 详情 |
|------|------|
| **规模** | 40,000 张图像 (20k Positive + 20k Negative) |
| **分辨率** | 227×227 |
| **任务** | 裂缝/无裂缝 二分类 |
| **来源** | 混凝土结构表面 |

**下载**:
- Kaggle: 搜索 "Concrete Crack Images"
- GitHub: https://github.com/nhorro/tensorflow-crack-classification

---

### 2.6 CrackTree200

| 属性 | 详情 |
|------|------|
| **规模** | 200张图像 |
| **内容** | 树皮裂缝纹理 |
| **用途** | 算法评估 |

---

### 2.7 Crack Forest Dataset (CFD)

| 属性 | 详情 |
|------|------|
| **规模** | 118张图像 |
| **内容** | 混凝土裂缝 |
| **标注** | 像素级标注 |

---

## 三、3D点云/网格数据集

### 3.1 ENSTRECT 官方数据
- **格式**: .obj 网格 + .jpg 图像 + cameras.json
- **完整度**: ⭐⭐⭐⭐⭐ (含相机参数，可直接运行)
- **获取**: Google Drive (需科学上网)

### 3.2 自建数据建议

如需使用自己的数据，需要准备:

```
project/
├── mesh/
│   └── mesh.obj          # 3D模型 (单位: 米)
├── views/
│   ├── 0000.jpg          # 多视角图像
│   ├── 0001.jpg
│   └── ...
└── cameras.json          # 相机参数 (PyTorch3D格式)
```

**cameras.json 格式**:
```json
{
  "0000": {
    "focal_length": [[fx, fy]],
    "principal_point": [[cx, cy]],
    "image_size": [[height, width]],
    "R": [[[...]]],        // 3x3 旋转矩阵
    "T": [[tx, ty, tz]],   // 平移向量
    "in_ndc": false
  }
}
```

**相机标定工具推荐**:
- Metashape (商业)
- COLMAP (开源)
- Meshroom (开源)

---

## 四、数据集对比

| 数据集 | 图像数 | 3D数据 | 分割标注 | 适用场景 |
|--------|--------|--------|----------|----------|
| ENSTRECT官方 | 4段 | ✅ | ✅ | 完整2.5D流程测试 |
| dacl10k | 9,920 | ❌ | ✅ | 图像级分割训练 |
| Bochum BCD | 370 | ❌ | ✅ | 裂缝检测/分割 |
| Crack500 | 500 | ❌ | ✅ | 路面裂缝检测 |
| OmniCrack30k | 30,000 | ❌ | ✅ | 通用裂缝分割训练 |
| Concrete Crack | 40,000 | ❌ | ❌ | 裂缝分类 |

---

## 五、推荐方案

### 方案1: 完整ENSTRECT测试 (推荐)
1. 科学上网访问 Google Drive
2. 下载官方 segments.zip
3. 解压到 `enstrect/src/enstrect/assets/segments/`
4. 运行: `python -m enstrect.run`

### 方案2: 仅测试图像分割
1. 下载 dacl10k 或 Bochum BCD
2. 使用单张图像测试分割模型:
   ```python
   python -m enstrect.segmentation.nnunet_s2ds --image path/to/image.jpg
   ```

### 方案3: 使用示例图片快速测试
```python
import sys
sys.path.insert(0, r'G:\Zed\ENSTRECTtest\enstrect\src')

from enstrect.segmentation.nnunet_s2ds import NnUNetS2DS
import torch

# 加载模型
model = NnUNetS2DS()

# 测试示例图片
from PIL import Image
img = Image.open(r'G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets\example_image.jpg')
# ... 进行推理
```

---

## 六、注意事项

1. **单位**: ENSTRECT 要求数据单位为 **米**，否则提取参数会失效
2. **相机格式**: 必须使用 PyTorch3D 兼容的 cameras.json 格式
3. **图像命名**: 建议与 cameras.json 中的键名对应 (如 0000.jpg)
4. **网络问题**: 国内访问 Google Drive 需要科学上网

---

**Herta 提示**: 如果开拓者需要本天才帮忙下载或处理任何数据集，尽管说~ ✧(≖ ◡ ≖✿)
