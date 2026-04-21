# ENSTRECT 多视图结构损伤分割 + LLM+RAG 智能报告

> 基于 [ENSTRECT](https://github.com/ben-z-original/enstrect) 的多视图结构表面损伤检测与 LLM 智能报告生成系统

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)](https://pytorch.org/)
[![Qwen2.5-VL](https://img.shields.io/badge/Qwen2.5--VL-API-purple)](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 🆕 最新功能：LLM+RAG 损伤智能识别与报告生成

本项目现已集成 **Qwen2.5-VL 多模态大模型** 和 **RAG 检索增强生成**，实现零标注成本的智能损伤评估报告！

### 核心亮点

- **🤖 多模态 AI**：基于 Qwen2.5-VL-7B/72B，支持图像理解 + 自然语言生成
- **📚 RAG 知识增强**：集成 GB 50204、JTG/T H21 等混凝土/桥梁评定规范
- **📝 智能报告**：自动生成 JSON + Markdown 双格式结构化报告
- **🎯 零标注成本**：无需标注数据，纯 Prompt Engineering + RAG
- **📊 训练数据收集**：自动保存推理过程，为 LoRA 微调准备数据
- **⚡ API 模式**：阿里云百炼云端调用，无需本地下载 15GB 模型

### 报告示例

```json
{
  "damage_type": "crack",
  "severity_level": 3,
  "description": "图像显示一条横向裂缝，位于构件中部，裂缝宽度约0.2mm，长度约15cm",
  "geometry_summary": "损伤像素数7343px，估计面积73.43mm²，占图像比例0.02%",
  "regulation_reference": "依据JTG/T H21-2011第3.2.1条，裂缝宽度0.15mm<δ≤0.35mm，评定标度为3级",
  "repair_recommendation": "建议采用表面封闭法处理，使用环氧树脂注浆材料填充裂缝",
  "confidence": 0.85
}
```

---

## 项目简介

本项目是基于 **ENSTRECT** (Engineering Structures Texture and Crack Tracker) 的多视图结构损伤检测与智能报告系统。支持：

1. **2D 语义分割** - 基于 nnU-Net-S2DS 的像素级损伤检测
2. **3D 点云融合** - 多视角投影与概率融合（可选）
3. **🆕 LLM 智能报告** - Qwen2.5-VL + RAG 规范条文引用

### 支持的损伤类别

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

## 🚀 快速开始：LLM+RAG 智能报告

### 1. 获取阿里云百炼 API Key

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 创建 API Key：[获取文档](https://help.aliyun.com/zh/model-studio/get-api-key)
3. 保存 API Key 到 `.env` 文件：

```bash
# 在项目根目录创建 .env 文件
echo "DASHSCOPE_API_KEY=sk-your-api-key" > .env
```

### 2. 安装依赖

```bash
pip install transformers qwen-vl-utils chromadb sentence-transformers pdfplumber
```

### 3. 构建 RAG 知识库

```bash
python -m llm_rag.rag.build --pdf-dir data/standards/ --db-path data/chromadb/
```

### 4. 运行智能报告生成

```bash
# 基础用法（自动加载 .env 中的 API Key）
python run_llm_damage_report_api.py --images-dir path/to/images --output-dir output/

# 带 RAG 规范检索
python run_llm_damage_report_api.py --images-dir path/to/images --enable-rag --top-k 3

# 指定 API Key 和模型
python run_llm_damage_report_api.py --images-dir path/to/images --api-key sk-xxxx --model-name qwen2.5-vl-72b-instruct
```

---

## 📖 完整使用指南

### 方案一：纯 API 模式（推荐，无需 GPU）

使用阿里云百炼云端 Qwen2.5-VL API，无需本地下载模型：

```bash
# 单张图像测试
python run_llm_damage_report_api.py \
    --images-dir enstrect/src/enstrect/assets/bridge_b/segment_test/views \
    --output-dir output_api \
    --max-images 1 \
    --enable-rag

# 批量处理
python run_llm_damage_report_api.py \
    --images-dir path/to/your/images \
    --output-dir output_batch \
    --enable-rag \
    --top-k 3
```

**参数说明：**
- `--images-dir`: 输入图像目录
- `--output-dir`: 输出目录
- `--api-key`: API Key（默认从 `.env` 或环境变量读取）
- `--model-name`: 模型名称（`qwen2.5-vl-7b-instruct` 或 `qwen2.5-vl-72b-instruct`）
- `--enable-rag`: 启用 RAG 规范检索
- `--top-k`: RAG 检索返回条数（默认 3）
- `--max-images`: 最大处理图像数（默认全部）
- `--pixel-to-mm`: 像素到毫米换算系数（默认 0.1）

### 方案二：本地模型模式（需要 GPU）

如需本地部署 Qwen2.5-VL（需要约 16GB 显存）：

```bash
# 本地模型版本
python run_llm_damage_report.py \
    --images-dir path/to/images \
    --output-dir output_local \
    --load-in-4bit  # 4-bit 量化降低显存
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM+RAG 损伤智能报告系统                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入图像 → nnU-Net分割 → ROI裁剪 → RAG检索 → Qwen2.5-VL推理 → 结构化报告  │
│                              ↑                                          │
│                    GB 50204 / JTG/T H21 规范知识库                        │
│                              ↓                                          │
│                    自动收集训练数据（LoRA微调准备）                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **RAG 知识库** | `llm_rag/rag/` | PDF 解析、向量存储、语义检索 |
| **VLM 推理** | `llm_rag/vlm/` | Qwen2.5-VL API/本地调用、输出解析 |
| **Prompt 构建** | `llm_rag/prompt/` | ROI 裁剪、多模态 Prompt 模板 |
| **报告渲染** | `llm_rag/report/` | JSON/Markdown 双格式输出 |
| **数据收集** | `llm_rag/utils/` | LoRA 训练数据自动收集 |

---

## 📁 项目结构

```
.
├── README.md                              # 项目说明
├── requirements.txt                       # Python依赖
├── .env                                   # API Key 配置文件（已加入 .gitignore）
│
├── llm_rag/                               # 🆕 LLM+RAG 核心模块
│   ├── rag/                               # RAG 知识库
│   │   ├── pdf_parser.py                  # PDF 规范解析
│   │   ├── vector_store.py                # ChromaDB 向量存储
│   │   ├── retriever.py                   # 语义检索接口
│   │   └── build.py                       # 知识库构建 CLI
│   ├── vlm/                               # 视觉语言模型
│   │   ├── inference.py                   # 本地模型推理
│   │   ├── inference_api.py               # API 调用封装
│   │   └── output_parser.py               # JSON 输出解析
│   ├── prompt/                            # Prompt 构建
│   │   ├── roi_cropper.py                 # ROI 区域裁剪
│   │   └── builder.py                     # 多模态 Prompt 模板
│   ├── report/                            # 报告渲染
│   │   └── renderer.py                    # JSON + Markdown 输出
│   └── utils/                             # 工具模块
│       └── data_collector.py              # LoRA 训练数据收集
│
├── run_llm_damage_report_api.py           # 🆕 API 版主脚本（推荐）
├── run_llm_damage_report.py               # 🆕 本地模型版主脚本
├── data/                                  # 数据目录
│   ├── standards/                         # 规范 PDF 文档
│   │   └── sample_regulations.json        # 示例规范条文
│   └── chromadb/                          # RAG 向量数据库
│
├── run_multiview_segmentation.py          # 基础分割脚本
├── run_multiview_segmentation_with_filter.py  # 带质量过滤的分割
├── image_quality_filter.py                # 图像质量检测模块
└── enstrect/                              # ENSTRECT 核心库（子模块）
    └── src/enstrect/
        ├── assets/                        # 测试数据
        └── segmentation/                  # 分割模型
```

---

## 🎯 使用示例

### 示例 1：单张图像智能报告

```bash
python run_llm_damage_report_api.py \
    --images-dir enstrect/src/enstrect/assets/bridge_b/segment_test/views \
    --output-dir output_demo \
    --max-images 1 \
    --enable-rag
```

**输出：**
```
output_demo/
├── reports/
│   ├── 0000_roi_crack.json       # 结构化 JSON 报告
│   ├── 0000_roi_crack.md         # Markdown 可读报告
│   └── 0000_summary.md           # 汇总报告（多 ROI 时）
├── rois/
│   └── 0000_crack.png            # 裁剪的 ROI 图像
├── lora_training/
│   └── samples.jsonl             # 自动收集的训练数据
└── pipeline_log.json             # 运行日志
```

### 示例 2：批量处理

```bash
python run_llm_damage_report_api.py \
    --images-dir path/to/your/images \
    --output-dir output_batch \
    --enable-rag \
    --top-k 3
```

### 示例 3：Python API 调用

```python
from llm_rag.rag import RAGRetriever
from llm_rag.prompt import ROICropper, PromptBuilder
from llm_rag.vlm import QwenVLAPI, OutputParser
from llm_rag.report import ReportRenderer

# 初始化组件
retriever = RAGRetriever.from_sample_data(
    "data/standards/sample_regulations.json",
    "data/chromadb/"
)
vlm = QwenVLAPI()  # 自动读取 .env 中的 API Key
parser = OutputParser()
renderer = ReportRenderer()

# 单张图像处理
# ... (详见完整示例代码)
```

---

## 📚 RAG 知识库

### 支持的规范文档

| 规范 | 编号 | 内容 |
|------|------|------|
| 混凝土结构工程施工质量验收规范 | GB 50204 | 外观质量、尺寸偏差 |
| 公路桥梁技术状况评定标准 | JTG/T H21 | 裂缝、剥落、锈蚀分级 |
| 回弹法检测混凝土抗压强度技术规程 | JGJ/T 23 | 强度检测 |
| 混凝土结构耐久性评定标准 | CECS 259 | 耐久性评估 |

### 添加自定义规范

1. 将 PDF 文件放入 `data/standards/`
2. 重新构建知识库：

```bash
python -m llm_rag.rag.build --pdf-dir data/standards/ --db-path data/chromadb/
```

---

## 💡 进阶功能

### 1. LoRA 微调数据收集

系统会自动保存每次推理的输入-输出对，用于后续微调：

```bash
# 收集一定量的数据后，导出训练集
cd llm_rag/utils/
python -c "from data_collector import TrainingDataCollector; c = TrainingDataCollector(); c.export_training_data('output_lora', 'sharegpt')"
```

### 2. 自定义 Prompt 模板

编辑 `llm_rag/prompt/builder.py` 中的 `DAMAGE_TEMPLATES`，添加特定损伤类型的评估指引。

### 3. 调整 RAG 检索参数

```python
# 修改检索条数
retriever = RAGRetriever(vector_store, top_k=5)

# 按损伤类型检索
results = retriever.retrieve_by_damage_type("crack", severity_hint="severe")
```

---

## 🔧 环境要求

### 硬件要求

| 模式 | GPU | 显存 | 内存 |
|------|-----|------|------|
| **API 模式** | 不需要 | 不需要 | 8GB+ |
| **本地 4-bit** | RTX 3060+ | 8GB+ | 16GB+ |
| **本地 FP16** | RTX 4090/A100 | 16GB+ | 32GB+ |

### 软件环境

```
Python >= 3.10
PyTorch >= 2.5.0
CUDA >= 12.4 (本地模型模式)
```

### 依赖安装

```bash
# 基础依赖
pip install torch torchvision numpy pillow tqdm

# LLM+RAG 依赖
pip install transformers qwen-vl-utils accelerate
pip install chromadb sentence-transformers pdfplumber

# 可选：OpenAI SDK（API 调用更稳定）
pip install openai

# 可选：bitsandbytes（本地 4-bit 量化）
pip install bitsandbytes
```

---

## 📝 详细文档

- [LLM辅助智能识别可行性报告](LLM辅助智能识别可行性报告.md) - 技术调研与方案设计
- [ENSTRECT_Workflow.md](ENSTRECT_Workflow.md) - ENSTRECT 原始工作流说明
- [ENSTRECT_损伤分割技术详解.md](ENSTRECT_损伤分割技术详解.md) - 分割模型技术细节

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

@article{qwen2.5vl2024,
  title={Qwen2.5-VL: Advanced Multimodal Language Model},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2024}
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
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - 多模态大模型
- [阿里云百炼](https://bailian.console.aliyun.com/) - 模型 API 服务

---

> 💡 **提示**: 本项目仅用于学习和研究目的。API 使用请遵守阿里云百炼的服务条款。
