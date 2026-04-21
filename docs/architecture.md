# ENSTRECT + LLM+RAG 系统架构图

> 本项目架构文档，使用 Mermaid 语法绘制系统流程与模块关系

---

## 1. 整体系统架构

```mermaid
graph TB
    subgraph 输入层["📥 输入层"]
        A1[Bridge B/G 图像] --> A2[ENSTRECT 分割]
        A3[KITTI 序列] --> A4[KITTI Loader]
        A5[用户上传图像] --> A6[图像质量检测]
    end

    subgraph 感知层["👁️ 感知层 - 2D分割"]
        A2 --> B1[nnU-Net-S2DS]
        A4 --> B1
        A6 --> B1
        B1 --> B2[语义分割掩码]
        B1 --> B3[损伤概率图]
    end

    subgraph 理解层["🧠 理解层 - LLM+RAG"]
        B2 --> C1[ROI裁剪]
        C1 --> C2[几何量化]
        C2 --> C3[RAG检索]
        C3 --> C4[Qwen2.5-VL推理]
        
        subgraph RAG知识库["📚 RAG知识库"]
            D1[GB 50204 规范]
            D2[JTG/T H21 规范]
            D3[JGJ/T 23 规范]
        end
        
        C3 <-->|检索| D1
        C3 <-->|检索| D2
        C3 <-->|检索| D3
    end

    subgraph 输出层["📤 输出层"]
        C4 --> E1[JSON结构化报告]
        C4 --> E2[Markdown可读报告]
        C4 --> E3[可视化标注图像]
        C4 --> E4[LoRA训练数据]
    end

    style 输入层 fill:#e1f5ff
    style 感知层 fill:#fff4e1
    style 理解层 fill:#f3e5f5
    style RAG知识库 fill:#e8f5e9
    style 输出层 fill:#fce4ec
```

---

## 2. LLM+RAG 模块详细架构

```mermaid
graph LR
    subgraph 输入["输入处理"]
        I1[分割掩码] --> I2[ROICropper]
        I3[原图] --> I2
    end

    subgraph Prompt构建["Prompt构建"]
        I2 --> P1[损伤区域裁剪]
        P1 --> P2[几何统计计算]
        P2 --> P3[PromptBuilder]
        
        P3 --> P4[System Prompt<br/>角色定义+格式要求]
        P3 --> P5[User Prompt<br/>图像+RAG上下文+几何数据]
        P3 --> P6[Few-shot示例]
    end

    subgraph RAG系统["RAG检索系统"]
        R1[PDF Parser] --> R2[文本分块]
        R2 --> R3[VectorStore<br/>ChromaDB]
        R3 --> R4[Embedding<br/>bge-small-zh]
        R4 --> R5[RAGRetriever]
    end

    subgraph VLM推理["VLM推理引擎"]
        P4 --> V1[QwenVLAPI]
        P5 --> V1
        P6 --> V1
        R5 --> V1
        
        V1 --> V2[阿里云百炼API]
        V2 --> V3[OutputParser]
    end

    subgraph 报告生成["报告生成"]
        V3 --> G1[ReportRenderer]
        G1 --> G2[JSON Schema验证]
        G2 --> G3[Markdown渲染]
        G2 --> G4[数据持久化]
    end

    subgraph 数据收集["LoRA数据收集"]
        G4 --> L1[TrainingDataCollector]
        L1 --> L2[原始样本保存]
        L2 --> L3[多格式导出<br/>ShareGPT/多模态]
    end

    style 输入 fill:#e3f2fd
    style Prompt构建 fill:#fff3e0
    style RAG系统 fill:#e8f5e9
    style VLM推理 fill:#f3e5f5
    style 报告生成 fill:#ffebee
    style 数据收集 fill:#fce4ec
```

---

## 3. 数据流图

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant Input as 输入层
    participant Seg as nnU-Net分割
    participant ROI as ROI裁剪
    participant RAG as RAG检索
    participant VLM as Qwen2.5-VL
    participant Report as 报告生成
    participant DB as LoRA数据库

    User->>Input: 上传图像/选择序列
    Input->>Seg: 图像数据
    
    activate Seg
    Seg->>Seg: 语义分割推理
    Seg-->>ROI: 分割掩码
    deactivate Seg
    
    activate ROI
    ROI->>ROI: 裁剪损伤ROI区域
    ROI->>ROI: 计算几何统计
    ROI-->>RAG: ROI图像+几何数据
    deactivate ROI
    
    activate RAG
    RAG->>RAG: 构建检索Query
    RAG->>RAG: ChromaDB向量检索
    RAG-->>VLM: 检索到的规范条文
    deactivate RAG
    
    activate VLM
    VLM->>VLM: 构建多模态Prompt
    VLM->>VLM: Base64编码图像
    VLM->>VLM: 调用百炼API
    VLM-->>Report: JSON格式输出
    deactivate VLM
    
    activate Report
    Report->>Report: JSON解析与验证
    Report->>Report: 字段补全与修复
    Report->>Report: Markdown渲染
    Report-->>User: 双格式报告
    deactivate Report
    
    Report->>DB: 保存训练样本
    activate DB
    DB->>DB: 序列化为JSONL
    deactivate DB
```

---

## 4. 模块依赖关系

```mermaid
graph TD
    subgraph 核心模块["🔧 核心模块"]
        A[llm_rag.rag] --> B[PDFParser]
        A --> C[VectorStore]
        A --> D[RAGRetriever]
        
        E[llm_rag.vlm] --> F[QwenVLAPI]
        E --> G[QwenVLInference]
        E --> H[OutputParser]
        
        I[llm_rag.prompt] --> J[ROICropper]
        I --> K[PromptBuilder]
        
        L[llm_rag.report] --> M[ReportRenderer]
        
        N[llm_rag.utils] --> O[TrainingDataCollector]
        N --> P[KITTILoader]
    end

    subgraph 外部依赖["🔗 外部依赖"]
        Q[阿里云百炼] --> F
        R[ChromaDB] --> C
        S[SentenceTransformers] --> C
        T[nnU-Net-S2DS] --> U[分割模型]
    end

    subgraph 入口脚本["🚀 入口脚本"]
        V[run_llm_damage_report_api.py] --> A
        V --> E
        V --> I
        V --> L
        V --> N
        V --> U
        
        W[run_multiview_segmentation.py] --> U
    end

    U --> J
    J --> K
    K --> F
    F --> H
    H --> M
    H --> O
    D --> K

    style 核心模块 fill:#e3f2fd
    style 外部依赖 fill:#fff8e1
    style 入口脚本 fill:#f3e5f5
```

---

## 5. RAG 知识库架构

```mermaid
graph TB
    subgraph 数据源["📄 数据源"]
        A1[GB 50204 PDF] --> B[PDF Parser]
        A2[JTG/T H21 PDF] --> B
        A3[sample_regulations.json] --> C[JSON Loader]
    end

    subgraph 处理流程["⚙️ 处理流程"]
        B --> D[文本提取]
        D --> E[章节切分<br/>chunk_size=500]
        E --> F[嵌入生成<br/>bge-small-zh]
        
        C --> G[JSON解析]
        G --> F
    end

    subgraph 向量存储["💾 向量存储"]
        F --> H[ChromaDB]
        H --> I[Collection: regulations]
        I --> J[Embedding<br/>512-dim]
        I --> K[Metadata<br/>来源+章节+关键词]
    end

    subgraph 检索接口["🔍 检索接口"]
        L[用户Query] --> M[Query Embedding]
        M --> N[Similarity Search<br/>cosine distance]
        N --> O[Top-K Results]
        O --> P[Context Builder]
        P --> Q[Prompt注入]
    end

    H --> N

    style 数据源 fill:#e8f5e9
    style 处理流程 fill:#fff3e0
    style 向量存储 fill:#e3f2fd
    style 检索接口 fill:#fce4ec
```

---

## 6. Prompt 构建流程

```mermaid
flowchart TD
    Start([开始]) --> Input{输入类型}
    
    Input -->|裂缝| Crack[Crack Template]
    Input -->|剥落| Spall[Spalling Template]
    Input -->|锈蚀| Corr[Corrosion Template]
    Input -->|其他| Default[Default Template]
    
    Crack --> System[构建System Prompt<br/>+裂缝评定指引<br/>+Few-shot示例]
    Spall --> System
    Corr --> System
    Default --> System
    
    System --> User[构建User Prompt]
    
    User --> Image[嵌入ROI图像]
    User --> Geo[嵌入几何数据<br/>像素数/面积/占比]
    User --> RAG[嵌入RAG上下文<br/>规范条文]
    
    Image --> Combine[组合完整Prompt]
    Geo --> Combine
    RAG --> Combine
    
    Combine --> Output([输出到VLM])

    style Start fill:#e8f5e9
    style Input fill:#fff8e1
    style Output fill:#ffebee
```

---

## 7. LoRA 训练数据收集流程

```mermaid
graph LR
    A[单次推理完成] --> B{是否启用<br/>数据收集?}
    
    B -->|是| C[收集输入数据]
    B -->|否| Z[结束]
    
    C --> D[input_image<br/>ROI图像路径]
    C --> E[context<br/>RAG检索上下文]
    C --> F[prompt<br/>完整Prompt文本]
    C --> G[output_json<br/>模型输出JSON]
    
    D --> H[构建样本字典]
    E --> H
    F --> H
    G --> H
    
    H --> I[添加元数据<br/>时间戳/类别/置信度]
    I --> J[序列化为JSONL]
    J --> K[追加到samples.jsonl]
    
    K --> L{导出训练集?}
    L -->|是| M[导出ShareGPT格式]
    L -->|是| N[导出多模态格式]
    L -->|否| Z
    M --> Z
    N --> Z

    style A fill:#e3f2fd
    style H fill:#fff3e0
    style K fill:#e8f5e9
    style Z fill:#ffebee
```

---

## 8. 部署架构

```mermaid
graph TB
    subgraph 开发环境["💻 开发环境"]
        A[Windows 10/11<br/>CUDA 12.4]
        B[Python 3.10<br/>PyTorch 2.5]
        C[Git + VSCode]
    end

    subgraph 本地服务["🏠 本地服务"]
        D[nnU-Net-S2DS<br/>GPU推理]
        E[ChromaDB<br/>本地向量数据库]
        F[Flask/FastAPI<br/>可选Web服务]
    end

    subgraph 云服务["☁️ 云服务"]
        G[阿里云百炼<br/>Qwen2.5-VL API]
        H[HuggingFace<br/>模型权重下载]
        I[GitHub<br/>代码仓库]
    end

    subgraph 数据存储["💾 数据存储"]
        J[本地磁盘<br/>项目文件]
        K[百度网盘<br/>KITTI数据集]
        L[Google Drive<br/>nnU-Net权重]
    end

    A --> D
    B --> D
    D --> E
    E --> G
    D --> H
    C --> I
    K --> D
    L --> D

    style 开发环境 fill:#e3f2fd
    style 本地服务 fill:#fff3e0
    style 云服务 fill:#e8f5e9
    style 数据存储 fill:#fce4ec
```

---

## 9. 类图

```mermaid
classDiagram
    class KITTILoader {
        +sequence: str
        +kitti_root: Path
        +image_files: List[Path]
        +__init__(kitti_root, sequence)
        +__getitem__(idx) Tuple[Image, str]
        +load_frame(frame_id) Tuple[Image, str]
        +load_sequence(start, end, step) Iterator
        +get_camera_intrinsics() Optional[np.ndarray]
        +copy_sample_frames(output_dir, num_frames) List[str]
    }

    class RAGRetriever {
        +vector_store: VectorStore
        +top_k: int
        +DAMAGE_TYPE_KEYWORDS: Dict
        +SEVERITY_KEYWORDS: Dict
        +__init__(vector_store, top_k)
        +retrieve(query, top_k) List[Dict]
        +retrieve_by_damage_type(damage_type, severity_hint) List[Dict]
        +build_context(results) str
        +from_sample_data(json_path, db_path) RAGRetriever
    }

    class QwenVLAPI {
        +api_key: str
        +model_name: str
        +max_tokens: int
        +temperature: float
        +client: OpenAI
        +__init__(api_key, model_name, max_tokens, temperature)
        +generate(images, text_prompt) str
        +generate_single(image, text_prompt) str
        +_encode_image_to_base64(image) str
        +check_api_key() bool
        +print_setup_guide()
    }

    class PromptBuilder {
        +default_pixel_to_mm: float
        +SYSTEM_PROMPT_TEMPLATE: str
        +DAMAGE_TEMPLATES: Dict
        +FEW_SHOT_EXAMPLES: Dict
        +__init__(default_pixel_to_mm)
        +build_system_prompt(damage_type) str
        +build_user_prompt(roi_image, rag_context, geometry_data) Tuple[str, Image]
        +build_multimodal_input(rois, geometry_stats, rag_contexts) List[Tuple]
    }

    class ReportRenderer {
        +SEVERITY_LABELS: Dict
        +DAMAGE_TYPE_LABELS: Dict
        +render_json(report_data) str
        +render_markdown(report_data) str
        +save_report(report_data, output_dir, prefix) Tuple[str, str]
        +generate_summary(reports, image_name) str
    }

    class TrainingDataCollector {
        +output_dir: Path
        +samples_file: Path
        +__init__(output_dir)
        +collect(input_image, context, prompt, output_json, metadata) str
        +collect_batch(samples) int
        +export_training_data(output_dir, format) str
        +get_statistics() Dict
        +print_statistics()
    }

    KITTILoader ..> "加载" Image : PIL.Image
    RAGRetriever --> VectorStore : 使用
    QwenVLAPI ..> OpenAI : 依赖
    PromptBuilder ..> RAGRetriever : 使用上下文
    ReportRenderer ..> OutputParser : 解析结果
    TrainingDataCollector ..> PromptBuilder : 收集Prompt
```

---

## 10. 使用场景图

```mermaid
graph TB
    subgraph 场景1["场景1：单张图像检测"]
        A1[上传桥梁照片] --> B1[自动分割] --> C1[LLM分析] --> D1[生成报告]
    end

    subgraph 场景2["场景2：批量处理"]
        A2[选择图像目录] --> B2[批量分割] --> C2[并行LLM推理] --> D2[汇总报告]
    end

    subgraph 场景3["场景3：KITTI序列"]
        A3[选择KITTI序列] --> B3[序列加载] --> C3[逐帧处理] --> D3[时序分析]
    end

    subgraph 场景4["场景4：LoRA微调"]
        A4[收集推理数据] --> B4[样本标注] --> C4[格式转换] --> D4[模型微调]
    end

    style 场景1 fill:#e3f2fd
    style 场景2 fill:#fff3e0
    style 场景3 fill:#e8f5e9
    style 场景4 fill:#fce4ec
```

---

## 附录：图例说明

| 颜色 | 含义 |
|------|------|
| 🟦 浅蓝色 | 输入/数据层 |
| 🟨 浅黄色 | 处理/逻辑层 |
| 🟩 浅绿色 | 存储/知识库 |
| 🟪 浅紫色 | 推理/AI层 |
| 🟥 浅粉色 | 输出/结果层 |

---

> 📅 生成日期：2026-04-21
> 📝 使用 [Mermaid](https://mermaid.js.org/) 语法绘制
> 💡 可在支持 Mermaid 的 Markdown 渲染器中查看（如 GitHub、VSCode、Typora）
