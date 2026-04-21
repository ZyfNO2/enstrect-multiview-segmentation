# GitHub项目调研报告：YOLO/U-Net + LLM 辅助分割/分析

> **调研目标**: 为硕士论文《基于双目视觉的三维成像和损伤智能识别方法研究》提供技术参考  
> **调研范围**: GitHub上视觉分割模型与大语言模型结合的开源项目  
> **生成日期**: 2026-04-21  
> **适用章节**: 第四章/第五章 - 损伤智能识别与量化方法

---

## 目录

1. [调研背景与目标](#一调研背景与目标)
2. [调研方法](#二调研方法)
3. [核心推荐项目](#三核心推荐项目)
4. [按技术路线分类](#四按技术路线分类)
5. [技术架构建议](#五技术架构建议)
6. [下一步行动计划](#六下一步行动计划)
7. [参考资源汇总](#七参考资源汇总)

---

## 一、调研背景与目标

### 1.1 研究背景

本论文当前在分割与量化方向采用"2D图像分割 + 3D空间投影"的两阶段方法：
- **分割阶段**: YOLOv8-seg实例分割模型
- **投影阶段**: ZED相机位姿估计 + 三维点云投影
- **量化阶段**: Ball Pivoting Algorithm (BPA)面积计算

为进一步提升损伤识别的智能化水平，计划引入**大语言模型(LLM)**实现：
- 损伤类型智能识别与分类
- 严重程度自动评估
- 结构化损伤报告生成
- 修复建议自然语言输出

### 1.2 调研目标

1. 检索GitHub上YOLO/U-Net与LLM结合的开源项目
2. 分析视觉-语言多模态模型在分割任务中的应用
3. 调研医学影像分割+报告生成的相关方案
4. 为论文技术方案设计提供参考依据

---

## 二、调研方法

### 2.1 检索关键词

| 类别 | 检索关键词 |
|------|-----------|
| YOLO+LLM | `YOLO LLM segmentation language model` |
| U-Net+LLM | `U-Net LLM medical segmentation GPT` |
| VLM分割 | `vision language model segmentation detection` |
| SAM+LLM | `SAM segment anything LLM GPT` |
| CLIP分割 | `CLIP segmentation open vocabulary` |
| 报告生成 | `medical image report generation LLM` |

### 2.2 评估维度

- **技术相关性**: 与混凝土损伤检测场景的匹配度
- **架构参考性**: 技术架构的可借鉴程度
- **代码质量**: 开源代码的完整度和可读性
- **更新活跃度**: 项目维护状态

---

## 三、核心推荐项目

### 🏆 第一类：医学影像分割 + LLM报告生成

#### 3.1.1 MedImgSegmentation-GAN-LLM ⭐⭐⭐⭐⭐

- **项目链接**: https://github.com/Arek-KesizAbnousi/MedImgSegmentation-GAN-LLM
- **技术栈**: U-Net + GAN + GPT-2
- **核心功能**:
  - U-Net医学图像分割
  - GAN数据增强
  - GPT-2生成描述性诊断报告
- **借鉴价值**:
  - ✅ 分割 + LLM报告生成的完整pipeline
  - ✅ 与设想的"YOLO/UNet + 千问"架构一致
  - ✅ 报告生成的JSON结构化输出参考
- **适用场景**: 需要完整理解分割+LLM集成架构

#### 3.1.2 Brain-Tumor-Segmentation-and-LLM-based-report-generation-system ⭐⭐⭐⭐⭐

- **项目链接**: https://github.com/ana3ss7z/Brain-Tumor-Segmentation-and-LLM-based-report-generation-system
- **技术栈**: 深度学习分割 + 本地LLM + Web界面
- **核心功能**:
  - MRI脑肿瘤自动分割
  - 本地LLM生成诊断报告
  - 用户友好的Web界面
- **借鉴价值**:
  - ✅ 端到端pipeline（分割→LLM分析→报告）
  - ✅ 使用**本地LLM**（非API调用），与部署千问思路一致
  - ✅ Web界面设计可参考
- **适用场景**: 需要完整的工程实现参考

#### 3.1.3 medical-vlm-assistant ⭐⭐⭐⭐⭐ 最相关

- **项目链接**: https://github.com/jianghongcheng/medical-vlm-assistant
- **技术栈**: Med3DVLM + DCFormer + SigLIP + **Qwen2.5-7B**
- **核心功能**:
  - CT影像报告生成
  - 4阶段pipeline: VQA感知 → Image RAG → 本地LLM生成 → 安全层
  - 使用**千问Qwen2.5-7B**作为生成模型
- **借鉴价值**:
  - ✅ **直接使用千问模型**，技术选型完美匹配
  - ✅ RAG检索增强生成架构
  - ✅ Image RAG多模态检索思路
- **适用场景**: 千问模型部署和微调的首选参考

#### 3.1.4 Doc-rag (Vision-RAG)

- **项目链接**: https://github.com/AsakaTigar/Doc-rag
- **技术栈**: 视觉RAG + 医学影像 + LLM
- **核心功能**:
  - 鼻咽癌MRI影像报告生成
  - 双检索系统
  - LLM驱动的结构化报告生成
- **借鉴价值**:
  - ✅ RAG架构设计
  - ✅ 结构化报告生成模板

---

### 🏆 第二类：VLM视觉语言模型（零样本分割）

#### 3.2.1 VQA-Using-YOLO-DeepLabV3-BERT ⭐⭐⭐⭐⭐

- **项目链接**: https://github.com/AbdelkadirSellahi/VQA-Using-YOLO-DeepLabV3-BERT
- **技术栈**: YOLOv8 + DeepLabV3+ + BERT
- **核心功能**:
  - YOLOv8目标检测
  - DeepLabV3+语义分割
  - BERT大语言模型进行视觉问答(VQA)
  - 分析无人机拍摄的洪水影像
- **借鉴价值**:
  - ✅ **YOLOv8 + LLM**组合，技术路线完全吻合
  - ✅ 工程检测场景（无人机巡检→混凝土检测可类比）
  - ✅ 视觉问答交互方式
- **适用场景**: YOLO+LLM架构设计的首选参考

#### 3.2.2 VLM-Object-Detection (SAM2 + VLM)

- **项目链接**: https://github.com/Piero24/VLM-Object-Detection
- **技术栈**: 视觉语言模型(VLM) + SAM2
- **核心功能**:
  - VLM指导目标检测
  - SAM2进行分割
- **借鉴价值**:
  - ✅ VLM + SAM2的组合思路
  - ✅ 开放词汇检测参考

#### 3.2.3 Zero-Shot-Object-Detection-and-Segmentation-with-Google-Gemini-2.5

- **项目链接**: https://github.com/zubairnajim/Zero-Shot-Object-Detection-and-Segmentation-with-Google-Gemini-2.5
- **技术栈**: Google Gemini 2.5 VLM
- **核心功能**:
  - 零样本目标检测与分割
  - 文本提示驱动分割
- **借鉴价值**:
  - ✅ 零样本分割思路
  - ✅ 文本提示工程参考

---

### 🏆 第三类：CLIP开放词汇分割

#### 3.3.1 CLIP-SAM

- **项目链接**: https://github.com/maxi-w/CLIP-SAM
- **技术栈**: CLIP + SAM
- **核心功能**: 结合CLIP与SAM进行开放词汇图像分割
- **借鉴价值**:
  - ✅ 开放词汇分割（可扩展新损伤类型无需重新训练）

#### 3.3.2 ov-seg (Open-Vocabulary Semantic Segmentation) - Meta出品

- **项目链接**: https://github.com/facebookresearch/ov-seg
- **技术栈**: CLIP + Mask-adapted机制
- **核心功能**: 开放词汇语义分割的官方实现
- **借鉴价值**:
  - ✅ Meta官方项目，代码质量高
  - ✅ 开放词汇分割的SOTA方法

#### 3.3.3 SegCLIP (ICML 2023)

- **项目链接**: https://github.com/ArrowLuo/SegCLIP
- **技术栈**: CLIP + Patch Aggregation
- **核心功能**: 开放词汇语义分割
- **借鉴价值**:
  - ✅ 可处理训练中未见过的损伤类别

---

### 🏆 第四类：PaliGemma视觉语言模型

#### 3.4.1 paligemma-from-scratch

- **项目链接**: https://github.com/gemaakhbar/paligemma-from-scratch
- **技术栈**: PyTorch实现的PaliGemma
- **核心功能**:
  - 目标检测
  - 分割
  - 视觉问答(VQA)
  - 图像描述生成
- **借鉴价值**:
  - ✅ 一站式VLM解决方案
  - ✅ 检测+分割+理解一体化

#### 3.4.2 YoloGemma

- **项目链接**: https://github.com/adithya-s-k/YoloGemma
- **技术栈**: PaliGemma VLM
- **核心功能**: 测试PaliGemma在检测和分割任务上的能力
- **借鉴价值**:
  - ✅ VLM在分割任务上的性能评估方法

---

## 四、按技术路线分类

### 4.1 YOLO + LLM 路线

| 项目名称 | 链接 | 核心特点 | 相关度 |
|---------|------|---------|-------|
| VQA-Using-YOLO-DeepLabV3-BERT | [链接](https://github.com/AbdelkadirSellahi/VQA-Using-YOLO-DeepLabV3-BERT) | YOLOv8+DeepLabV3++BERT | ⭐⭐⭐⭐⭐ |
| Medical_Detection | [链接](https://github.com/YahyaSoker/Medical_Detection) | 医学检测+LLM综合系统 | ⭐⭐⭐⭐ |

**适用场景**: 需要实时检测+实例区分能力的混凝土损伤检测

### 4.2 U-Net + LLM 路线

| 项目名称 | 链接 | 核心特点 | 相关度 |
|---------|------|---------|-------|
| MedImgSegmentation-GAN-LLM | [链接](https://github.com/Arek-KesizAbnousi/MedImgSegmentation-GAN-LLM) | U-Net+GAN+GPT-2报告生成 | ⭐⭐⭐⭐⭐ |
| BreastCancer_VLM-CLIP_UNet | [链接](https://github.com/Alpha-lacrim/BreastCancer_VLM-CLIP_UNet) | CLIP检测+U-Net分割 | ⭐⭐⭐⭐ |

**适用场景**: 需要高边缘精度、细小裂缝检测的场景

### 4.3 SAM + LLM 路线

| 项目名称 | 链接 | 核心特点 | 相关度 |
|---------|------|---------|-------|
| vlm-recog | [链接](https://github.com/neka-nat/vlm-recog) | Gemini+SAM分割 | ⭐⭐⭐⭐ |
| VLM-Object-Detection | [链接](https://github.com/Piero24/VLM-Object-Detection) | VLM+SAM2 | ⭐⭐⭐⭐ |
| annotation-free-spacecraft-segmentation | [链接](https://github.com/giddyyupp/annotation-free-spacecraft-segmentation) | VLM无标注分割(ICRA 2026) | ⭐⭐⭐⭐ |

**适用场景**: 零样本分割、开放词汇检测需求

### 4.4 报告生成 + 医学影像

| 项目名称 | 链接 | 核心特点 | 相关度 |
|---------|------|---------|-------|
| medical-vlm-assistant | [链接](https://github.com/jianghongcheng/medical-vlm-assistant) | **Qwen2.5-7B**报告生成 | ⭐⭐⭐⭐⭐ |
| Brain-Tumor-Segmentation-LLM | [链接](https://github.com/ana3ss7z/Brain-Tumor-Segmentation-and-LLM-based-report-generation-system) | 本地LLM+Web界面 | ⭐⭐⭐⭐⭐ |
| CXR-Image-Captioning | [链接](https://github.com/mbrz97/CXR-Image-Captioning) | 胸部X光报告生成 | ⭐⭐⭐⭐ |
| cxr-report-generator | [链接](https://github.com/pipstur/cxr-report-generator) | CNN+Groq LLM | ⭐⭐⭐ |
| CHATCAD-Lung-and-Colon-Cancer | [链接](https://github.com/Ibrahim-Khalil07/CHATCAD-Lung-and-Colon-Cancer-) | ChatCAD+分类器 | ⭐⭐⭐ |

**适用场景**: 需要完整报告生成pipeline的场景

---

## 五、技术架构建议

### 5.1 推荐技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    推荐技术架构                                   │
├─────────────────────────────────────────────────────────────────┤
│  输入层: ZED相机SVO → 双目图像 + 深度图                          │
│      ↓                                                           │
│  分割层: YOLOv8-seg (现有) / U-Net (对比实验)                    │
│      ↓                                                           │
│  投影层: 2D掩码 → 3D点云 (第三章已有)                            │
│      ↓                                                           │
│  理解层: Qwen2-VL / Qwen2.5-VL (千问多模态)  ⬅️ 推荐新增         │
│      ↓                                                           │
│  量化层: BPA面积 + 骨架化长度 (现有) + LLM辅助验证               │
│      ↓                                                           │
│  输出层: 结构化JSON报告 + 自然语言描述                           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 重点参考项目优先级

| 优先级 | 项目 | 参考重点 |
|-------|------|---------|
| 🔴 P0 | medical-vlm-assistant | **千问模型使用**、4阶段pipeline设计 |
| 🔴 P0 | Brain-Tumor-Segmentation-LLM | 本地LLM部署、Web界面、端到端流程 |
| 🟡 P1 | VQA-Using-YOLO-DeepLabV3-BERT | YOLO+LLM组合、VQA交互模式 |
| 🟡 P1 | MedImgSegmentation-GAN-LLM | 分割+报告生成架构 |
| 🟢 P2 | CLIP-SAM / ov-seg | 开放词汇分割（可选扩展） |

### 5.3 关键技术点借鉴

#### 5.3.1 结构化报告生成格式（参考 medical-vlm-assistant）

```json
{
  "damage_id": "damage_001",
  "damage_type": "裂缝",
  "severity": "中度",
  "severity_score": 0.7,
  "location": {
    "element": "梁体",
    "position": "底部中央",
    "coordinates": {"x": 1.2, "y": 0.5, "z": 3.4}
  },
  "geometry": {
    "length": {"value": 1.2, "unit": "m"},
    "width": {"value": 0.5, "unit": "mm", "range": "0.3-0.8"},
    "area": {"value": 0.05, "unit": "m²"}
  },
  "description": "纵向贯穿裂缝，长度约1.2m，宽度0.3-0.8mm，位于梁体底部中央，呈现明显的线性延伸特征",
  "recommendation": "建议3个月内进行环氧树脂注入修复，并定期监测裂缝扩展情况",
  "confidence": 0.89,
  "images": {
    "original": "path/to/original.jpg",
    "segmentation": "path/to/seg.jpg",
    "3d_view": "path/to/3d.ply"
  }
}
```

#### 5.3.2 Pipeline架构设计（参考 Brain-Tumor-Segmentation-LLM）

```python
# 模块化设计示例
class DamageDetectionPipeline:
    def __init__(self):
        self.segmentor = YOLOv8Seg()  # 分割模块
        self.projector = PointCloudProjector()  # 投影模块
        self.llm = Qwen2VL()  # LLM理解模块
        self.quantifier = DamageQuantifier()  # 量化模块
    
    def process(self, svo_file):
        # 1. 分割
        masks, boxes = self.segmentor.detect(svo_file)
        # 2. 投影
        damage_pcd = self.projector.project(masks)
        # 3. LLM理解
        analysis = self.llm.analyze(svo_file, masks)
        # 4. 量化
        measurements = self.quantifier.calculate(damage_pcd)
        # 5. 报告生成
        report = self.generate_report(analysis, measurements)
        return report
```

#### 5.3.3 领域适配策略（参考 MedImgSegmentation-GAN-LLM）

- **LoRA微调**: 使用QLoRA在混凝土损伤数据上微调千问模型
- **指令数据集**: 构建领域特定的指令-回答对
- **Prompt工程**: 设计专业Prompt模板引导LLM输出

---

## 六、下一步行动计划

### 6.1 立即行动（本周）

1. **Clone核心参考项目**
   ```bash
   git clone https://github.com/jianghongcheng/medical-vlm-assistant
   git clone https://github.com/ana3ss7z/Brain-Tumor-Segmentation-and-LLM-based-report-generation-system
   git clone https://github.com/AbdelkadirSellahi/VQA-Using-YOLO-DeepLabV3-BERT
   ```

2. **安装千问模型环境**
   ```bash
   pip install transformers accelerate qwen-vl-utils
   # 下载 Qwen2-VL 或 Qwen2.5-VL
   ```

3. **阅读关键论文**
   - Qwen-VL论文: https://arxiv.org/abs/2308.12966
   - LLaVA论文: https://arxiv.org/abs/2304.08485

### 6.2 短期目标（2周内）

1. **跑通Demo**: 用千问模型对单张混凝土损伤图像生成描述
2. **数据集准备**: 整理50-100张图文配对数据用于微调
3. **方案设计文档**: 撰写《YOLO+LLM损伤检测技术方案》

### 6.3 中期目标（1个月）

1. **LoRA微调**: 在混凝土数据上微调千问模型
2. **Pipeline集成**: 将YOLO分割 → LLM理解 → 量化报告串联
3. **对比实验**: 纯视觉方案 vs 视觉+LLM方案

---

## 七、参考资源汇总

### 7.1 核心论文

| 论文名称 | 链接 | 说明 |
|---------|------|------|
| Qwen-VL | https://arxiv.org/abs/2308.12966 | 千问多模态模型 |
| LLaVA | https://arxiv.org/abs/2304.08485 | 视觉语言助手 |
| SAM 2 | https://arxiv.org/abs/2408.00714 | Segment Anything |
| CLIP | https://arxiv.org/abs/2103.00020 | 开放词汇学习 |

### 7.2 官方文档

- **Ultralytics YOLOv8**: https://docs.ultralytics.com/
- **千问Qwen**: https://github.com/QwenLM/Qwen
- **Open3D**: http://www.open3d.org/docs/
- **Transformers**: https://huggingface.co/docs/transformers/

### 7.3 相关教程

- **LoRA微调教程**: https://huggingface.co/docs/peft/quicktour
- **QLoRA量化**: https://github.com/artidoro/qlora
- **Gradio界面**: https://gradio.app/docs

---

## 附录：项目快速访问清单

### 必看项目（按优先级排序）

1. ⭐⭐⭐⭐⭐ [medical-vlm-assistant](https://github.com/jianghongcheng/medical-vlm-assistant) - 千问报告生成
2. ⭐⭐⭐⭐⭐ [Brain-Tumor-Segmentation-LLM](https://github.com/ana3ss7z/Brain-Tumor-Segmentation-and-LLM-based-report-generation-system) - 本地LLM部署
3. ⭐⭐⭐⭐⭐ [VQA-Using-YOLO-DeepLabV3-BERT](https://github.com/AbdelkadirSellahi/VQA-Using-YOLO-DeepLabV3-BERT) - YOLO+LLM架构
4. ⭐⭐⭐⭐ [MedImgSegmentation-GAN-LLM](https://github.com/Arek-KesizAbnousi/MedImgSegmentation-GAN-LLM) - 分割+报告生成
5. ⭐⭐⭐⭐ [ov-seg](https://github.com/facebookresearch/ov-seg) - Meta开放词汇分割

---

> **报告编制**: Herta-sama ✧(≖ ◡ ≖✿)  
> **备注**: 开拓者，有任何问题随时召唤本天才！Spinning~♪
