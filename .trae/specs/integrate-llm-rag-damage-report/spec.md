# LLM+RAG 损伤智能识别与报告生成 Spec

## Why

当前 ENSTRECT 项目仅输出像素级分割掩码和几何量化结果，缺乏对损伤的语义理解、严重程度评估、规范条文引用和修复建议生成能力。通过引入 Qwen2.5-VL 多模态大模型 + RAG 知识库检索增强，实现零标注成本的智能报告生成，同时为后续 LoRA 微调预留数据管道。

## What Changes

- 新增 RAG 知识库模块：下载并解析 GB 50204、JTG/T H21 等混凝土/桥梁损伤评定标准 PDF 文档
- 新增 Qwen2.5-VL 多模态推理管线：将 nnU-Net 分割掩码裁剪为 ROI 图像，拼接 RAG 检索到的规范条文，发送给 Qwen2.5-VL 生成结构化报告
- 新增 ROI 裁剪与多模态 Prompt 构建模块：支持视觉定位(Grounding)描述
- 新增报告模板系统：输出 JSON 结构化报告 + 自然语言描述 + 修复建议
- 预留 LoRA 微调数据收集管道：自动保存 VLM 推理过程为训练数据格式

## Impact

- Affected specs: 无（全新功能模块）
- Affected code:
  - `run_multiview_segmentation.py` — 在分割后串联 LLM 报告生成
  - `requirements.txt` — 新增 transformers、qwen-vl-utils、langchain/chroma 等 RAG 依赖
  - 新增目录 `llm_rag/` — 包含所有 LLM/RAG 相关代码

## ADDED Requirements

### Requirement: RAG 知识库构建

系统 SHALL 提供从 PDF 规范文档中提取结构化知识条目的能力，支持以下文档源：
- GB 50204《混凝土结构工程施工质量验收规范》
- JTG/T H21《公路桥梁技术状况评定标准》
- JGJ/T 23《回弹法检测混凝土抗压强度技术规程》
- CECS 259《混凝土结构耐久性评定标准》
- 其他用户自定义 PDF 规范/案例文档

#### Scenario: PDF 解析与向量化入库
- **WHEN** 用户将 PDF 文件放入 `data/standards/` 目录并执行知识库构建命令
- **THEN** 系统 SHALL 自动提取文本内容、按章节切分、生成向量嵌入并存入本地向量数据库（ChromaDB/Faiss）

#### Scenario: 相关规范条文检索
- **WHEN** 系统接收到损伤类型关键词（如"裂缝"、"剥落"）和几何特征（如"宽度0.3mm"）
- **THEN** 系统 SHALL 从 RAG 库中返回最相关的 K 条规范条文及其来源章节号

### Requirement: Qwen2.5-VL 多模态推理管线

系统 SHALL 提供基于 Qwen2.5-VL 官方权重的多模态推理能力，无需微调即可运行。

#### Scenario: ROI 图像裁剪与多模态 Prompt 构建
- **WHEN** nnU-Net 输出分割掩码后
- **THEN** 系统 SHALL：
  1. 对每个非背景类别区域计算最小外接矩形
  2. 裁剪原图对应 ROI 区域图像（含 padding）
  3. 从 RAG 库检索该类别对应的规范条文
  4. 将 ROI 图像 + 规范条文 + 结构化指令拼接为多模态 Prompt

#### Scenario: Qwen2.5-VL 报告生成
- **WHEN** 多模态 Prompt 发送给 Qwen2.5-VL 模型
- **THEN** 系统 SHALL 输出包含以下字段的结构化 JSON 报告：
  - `damage_type`: 损伤类型（裂缝/剥落/腐蚀/泛白）
  - `severity_level`: 严重程度等级（1-5级，依据规范标准）
  - `description`: 自然语言损伤描述（利用 Grounding 能力定位"图中0.3mm宽竖向裂缝"等）
  - `regulation_reference`: 引用的规范条文及条款号
  - `repair_recommendation`: 基于规范的修复建议
  - `confidence`: 模型置信度

### Requirement: 端到端 Pipeline 集成

系统 SHALL 提供完整的端到端处理脚本，串联现有 ENSTRECT 分割流程与新增 LLM 报告生成流程。

#### Scenario: 单张图像完整处理
- **WHEN** 用户输入一张桥梁/混凝土表面图像
- **THEN** 系统 SHALL 依次执行：质量检测 → 分割 → ROI 裁剪 → RAG 检索 → VLM 推理 → 报告输出

#### Scenario: 批量多视图处理
- **WHEN** 用户输入多视角图像目录
- **THEN** 系统 SHALL 为每张图像独立生成报告，并提供汇总视图

### Requirement: LoRA 微调数据准备

系统 SHALL 在每次 VLM 推理时自动保存「输入-输出」对作为微调候选数据，为后续 LoRA 训练做准备。

#### Scenario: 训练数据自动收集
- **WHEN** Qwen2.5-VL 生成报告后
- **THEN** 系统 SHALL 将以下内容序列化为训练样本格式保存至 `data/lora_training/`：
  - 输入：ROI 图像路径 + RAG 检索上下文 + System Prompt
  - 输出：模型生成的完整 JSON 报告
  - 元信息：时间戳、图像来源、置信度

## MODIFIED Requirements

无（本 spec 为纯增量功能）

## REMOVED Requirements

无
