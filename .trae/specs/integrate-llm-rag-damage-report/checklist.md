# Checklist

## Task 1: 项目骨架与依赖环境
- [x] `llm_rag/` 包目录结构已创建，包含 rag/, vlm/, prompt/, report/, utils/ 子目录
- [x] `requirements.txt` 已更新，包含 transformers, qwen-vl-utils, langchain, chromadb, PDFPlumber 等依赖
- [x] 所有 `__init__.py` 文件已创建
- [ ] Qwen2.5-VL-7B-Instruct 官方权重可成功下载并加载（验证 `from transformers import Qwen2VLForConditionalGeneration` 可 import）

## Task 2: RAG 知识库构建模块
- [x] `llm_rag/rag/pdf_parser.py` 已实现：PDF 文本提取 + 按章节切分功能
- [x] `llm_rag/rag/vector_store.py` 已实现：向量嵌入生成 + ChromaDB 存储与检索
- [x] `llm_rag/rag/retriever.py` 已实现：`retrieve(query, top_k)` 接口，返回条文+来源+页码
- [x] `data/standards/sample_regulations.json` 占位数据文件已创建，包含 GB 50204 和 JTG/T H21 示例条文
- [x] `python -m llm_rag.rag.build` CLI 命令可执行

## Task 3: ROI 裁剪与 Prompt 构建
- [x] `llm_rag/prompt/roi_cropper.py` 已实现：从分割掩码裁剪 ROI 图像并保存
- [x] `llm_rag/prompt/builder.py` 已实现：System Prompt + User Prompt（图像+RAG上下文+几何数据）拼接
- [x] 至少 3 套 Prompt 模板已定义：裂缝、剥落、腐蚀各一套

## Task 4: Qwen2.5-VL 推理模块
- [x] `llm_rag/vlm/inference.py` 中 `QwenVLInference` 类已实现：支持 4-bit 量化加载 + generate 接口
- [x] `llm_rag/vlm/output_parser.py` 已实现：VLM 输出文本 → JSON 解析，含异常处理
- [ ] 单张 ROI 图像推理测试通过：输入图像+Prompt 能输出有效 JSON

## Task 5: 报告生成模块
- [x] 报告 JSON Schema 已定义并文档化
- [x] `llm_rag/report/renderer.py` 已实现：JSON → Markdown 渲染
- [x] 报告可保存为 .json 和 .md 双格式
- [x] 多视图汇总报告可正常生成

## Task 6: 端到端 Pipeline 集成
- [x] `run_llm_damage_report.py` 主脚本可运行
- [ ] 使用 ENSTRECT Bridge B 测试数据跑通完整流程：输入图像 → 分割 → ROI → RAG → VLM → 报告
- [x] 输出报告中包含 damage_type, severity_level, description, regulation_reference, repair_recommendation 字段
- [x] 命令行参数 --images-dir, --output-dir, --enable-rag 正常工作

## Task 7: LoRA 微调准备管道
- [x] 每次推理自动保存样本至 `data/lora_training/raw_samples/`
- [x] 样本序列化格式包含 input_image_path, context, prompt, output_json
- [x] 数据清洗/格式化工具脚本可用（TrainingDataCollector.export_training_data 支持 sharegpt/jsonl_raw/multimodal_conv 三种格式）
- [x] LoRA 微调脚本框架代码已预留（Pipeline 内置 TrainingDataCollector 集成）
