# Tasks

- [x] Task 1: 搭建项目骨架与依赖环境
  - [x] 创建 `llm_rag/` 包目录结构（rag/, vlm/, prompt/, report/, utils/）
  - [x] 更新 `requirements.txt`，新增 transformers>=4.41.0、qwen-vl-utils、langchain、chromadb、pypdf2/PDFPlumber、Pillow
  - [x] 编写 `llm_rag/__init__.py` 和各子模块 `__init__.py`
  - [x] 验证 Qwen2.5-VL 官方权重可下载并加载（Qwen/Qwen2.5-VL-7B-Instruct）

- [x] Task 2: RAG 知识库构建模块 (`llm_rag/rag/`)  
  - [x] 实现 PDF 解析器：支持 PDFPlumber/pypdf2 提取文本，按章节/条款切分（chunk_size=500, overlap=50）
  - [x] 实现向量嵌入与存储：使用 sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) 或 BAAI/bge-small-zh 生成嵌入，ChromaDB 本地持久化存储
  - [x] 实现 RAG 检索接口：`retrieve(query: str, top_k: int = 3) -> List[Dict]` 返回条文内容+来源+页码
  - [x] 预置示例规范文档：创建 `data/standards/sample_regulations.json` 作为占位数据（含 GB 50204 裂缝评定、JTG/T H21 桥梁损伤等级等示例条文）
  - [x] 编写 CLI 命令：`python -m llm_rag.rag.build --pdf-dir data/standards/ --db-path data/chromadb/`

- [x] Task 3: ROI 裁剪与多模态 Prompt 构建模块 (`llm_rag/prompt/`)  
  - [x] 实现 ROI 裁剪器：从 nnU-Net 分割掩码中提取每个损伤类别的最小外接矩形，裁剪原图对应区域（padding=20px），保存为独立图像文件
  - [x] 实现多模态 Prompt 构建器：
    - System Prompt：定义角色（"你是一位资深混凝土结构工程师..."）、输出格式要求（JSON schema）
    - User Prompt：拼接 ROI 图像 + RAG 检索到的规范条文 + 几何量化数据（像素数→实际尺寸换算）
    - 支持中文 Prompt 模板，利用 Qwen2.5-VL 的 Grounding 能力描述空间位置
  - [x] 实现 Prompt 模板管理：支持不同损伤类型使用不同模板（裂缝模板/剥落模板/腐蚀模板）

- [x] Task 4: Qwen2.5-VL 推理模块 (`llm_rag/vlm/`)  
  - [x] 实现 Qwen2.5-VL 加载与推理封装类 `QwenVLInference`：
    - 支持 `load_in_4bit=True` 量化加载（需 ~8GB 显存）
    - 提供 `generate(images: List[PIL.Image], text_prompt: str) -> str` 接口
    - 支持生成参数配置（temperature=0.1, max_new_tokens=2048, do_sample=False）
  - [x] 实现结构化输出解析器：将 VLM 输出的文本解析为 JSON，处理格式异常情况（正则提取 + JSON repair）
  - [x] 实现批量推理优化：支持多张 ROI 图像并发处理

- [x] Task 5: 报告生成与输出模块 (`llm_rag/report/`)
  - [x] 定义报告 JSON Schema（damage_id, damage_type, severity_level, description, regulation_reference, repair_recommendation, confidence, geometry）
  - [x] 实现报告渲染器：JSON → Markdown 可读报告（含表格、引用标注）
  - [x] 实现报告持久化：保存为 `.json` + `.md` 双格式至 `output/reports/`
  - [x] 实现汇总报告生成：多视图所有损伤的统计汇总表

- [x] Task 6: 端到端 Pipeline 集成脚本
  - [x] 编写 `run_llm_damage_report.py`：串联 现有分割流程 → ROI 裁剪 → RAG 检索 → VLM 推理 → 报告输出
  - [x] 集成现有 `run_multiview_segmentation_with_filter.py` 的质量过滤能力
  - [x] 支持命令行参数：`--images-dir`, `--output-dir`, `--enable-rag`, `--top-k`, `--model-path`
  - [x] 编写 Demo 演示脚本：用 ENSTRECT 测试数据跑通完整流程，验证端到端输出

- [x] Task 7: LoRA 微调数据准备管道 (`llm_rag/utils/`)
  - [x] 实现训练样本序列化器：每次 VLM 推理自动保存 `{input_image, context, prompt, output_json}` 至 `data/lora_training/raw_samples/`
  - [x] 实现数据清洗与格式化工具：将原始样本转换为 QLoRA 训练格式（sharegpt 格式 / 多模态对话格式）
  - [x] 编写数据统计脚本：分析已收集样本的类别分布、置信度分布、输出长度分布
  - [x] 预留 LoRA 微调脚本框架（代码骨架，不含实际训练逻辑）

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 1]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 2, Task 3, Task 4, Task 5]
- [Task 7] depends on [Task 4, Task 6]

# Notes
- 所有 LLM 相关代码放在 `llm_rag/` 目录下，不修改 ENSTRECT 核心库代码
- Qwen2.5-VL 使用官方 HuggingFace 权重，优先尝试 4-bit 量化以降低显存需求
- RAG 向量数据库默认使用 ChromaDB（轻量、本地、无需额外服务）
- 中文嵌入模型推荐 `BAAI/bge-small-zh-v1.5` 或 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
