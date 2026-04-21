"""
ENSTRECT LLM+RAG 损伤智能识别与报告生成 - API 版本
使用阿里云百炼 API 调用 Qwen2.5-VL，无需本地下载模型

用法:
    # 自动加载 .env 文件中的 API Key
    python run_llm_damage_report_api.py --images-dir path/to/images
    
    # 或者直接在命令行指定
    python run_llm_damage_report_api.py --images-dir path/to/images --api-key sk-xxxx
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ============================================================
# 自动加载本地环境变量
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

if os.path.exists(ENV_FILE):
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value

# ============================================================
# 路径配置
# ============================================================
ENSTRECT_PATH = os.path.join(PROJECT_ROOT, "enstrect", "src")
if os.path.exists(ENSTRECT_PATH) and ENSTRECT_PATH not in sys.path:
    sys.path.insert(0, ENSTRECT_PATH)
LLM_RAG_PATH = PROJECT_ROOT
if LLM_RAG_PATH not in sys.path:
    sys.path.insert(0, LLM_RAG_PATH)

# ============================================================
# 导入
# ============================================================
import numpy as np
from PIL import Image
from tqdm import tqdm

# ENSTRECT 分割模型
try:
    from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel
    HAS_ENSTRECT = True
except ImportError:
    HAS_ENSTRECT = False

# LLM/RAG 模块 - 使用 API 版本
try:
    from llm_rag.rag import RAGRetriever
    from llm_rag.prompt import ROICropper, PromptBuilder
    from llm_rag.vlm import QwenVLAPI, OutputParser  # 使用 API 版本
    from llm_rag.report import ReportRenderer
    from llm_rag.utils import TrainingDataCollector
    HAS_LLM_RAG = True
except ImportError as e:
    HAS_LLM_RAG = False
    IMPORT_ERROR = str(e)

# 图像质量过滤（可选）
try:
    from image_quality_filter import filter_blurry_images, evaluate_image_quality
    HAS_QUALITY_FILTER = True
except ImportError:
    HAS_QUALITY_FILTER = False

import torch


# ============================================================
# 类别映射
# ============================================================
CLASS_NAMES = {
    0: "background", 1: "crack", 2: "spalling", 3: "corrosion",
    4: "efflorescence", 5: "vegetation", 6: "control_point"
}

CLASS_NAMES_CN = {
    0: "背景", 1: "裂缝", 2: "剥落", 3: "锈蚀",
    4: "泛白/风化", 5: "植被", 6: "控制点"
}


# ============================================================
# Pipeline 类
# ============================================================
class DamageReportPipelineAPI:
    """
    端到端损伤检测报告生成 Pipeline - API 版本
    
    流程: 图像输入 → 分割 → ROI裁剪 → RAG检索 → API调用VLM → 报告输出
    """

    def __init__(self,
                 images_dir: str,
                 output_dir: str,
                 api_key: Optional[str] = None,
                 model_name: str = "qwen2.5-vl-7b-instruct",
                 enable_rag: bool = True,
                 top_k: int = 3,
                 max_images: Optional[int] = None,
                 pixel_to_mm: float = 0.1,
                 enable_quality_filter: bool = False,
                 rag_db_path: str = "data/chromadb/",
                 rag_json_path: str = "data/standards/sample_regulations.json"):
        
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.model_name = model_name
        self.enable_rag = enable_rag
        self.top_k = top_k
        self.max_images = max_images
        self.pixel_to_mm = pixel_to_mm
        self.enable_quality_filter = enable_quality_filter
        self.rag_db_path = rag_db_path
        self.rag_json_path = rag_json_path
        
        # 创建输出子目录
        self.reports_dir = self.output_dir / "reports"
        self.rois_dir = self.output_dir / "rois"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.rois_dir.mkdir(parents=True, exist_ok=True)
        
        # 组件实例（延迟初始化）
        self.segmentor = None
        self.device = None
        self.cropper = None
        self.prompt_builder = None
        self.vlm = None
        self.parser = None
        self.renderer = None
        self.retriever = None
        self.data_collector = None
        
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        print("\n" + "=" * 60)
        print("初始化 Pipeline 组件 (API 模式)...")
        print("=" * 60)
        
        # 1. 设备（仅用于分割）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[设备] {self.device}")
        
        # 2. 分割模型
        if HAS_ENSTRECT:
            print("[分割模型] 加载 nnU-Net-S2DS...")
            self.segmentor = NNUNetS2DSModel(allow_tqdm=False)
            self.segmentor.predictor.device = self.device
            self.segmentor.predictor.perform_everything_on_device = True
            print("[分割模型] ✓ 加载完成")
        else:
            print("[分割模型] ⚠️ ENSTRECT 未安装，跳过")
        
        # 3. ROI裁剪器
        self.cropper = ROICropper(padding=20)
        print("[ROI裁剪器] ✓ 初始化完成")
        
        # 4. Prompt构建器
        self.prompt_builder = PromptBuilder(default_pixel_to_mm=self.pixel_to_mm)
        print("[Prompt构建器] ✓ 初始化完成")
        
        # 5. VLM API 客户端
        if HAS_LLM_RAG:
            print(f"[VLM] 初始化 API 客户端 ({self.model_name})...")
            try:
                self.vlm = QwenVLAPI(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    max_tokens=2048,
                    temperature=0.1
                )
                print("[VLM] ✓ API 客户端就绪")
            except ValueError as e:
                print(f"[VLM] ✗ 初始化失败: {e}")
                QwenVLAPI.print_setup_guide()
                raise
            
            # 6. 输出解析器
            self.parser = OutputParser()
            print("[输出解析器] ✓ 初始化完成")
            
            # 7. 报告渲染器
            self.renderer = ReportRenderer()
            print("[报告渲染器] ✓ 初始化完成")
            
            # 8. RAG检索器
            if self.enable_rag:
                print(f"[RAG] 构建知识库 ({self.rag_json_path})...")
                json_path = str(Path(PROJECT_ROOT) / self.rag_json_path)
                db_path = str(Path(PROJECT_ROOT) / self.rag_db_path)
                if Path(json_path).exists():
                    self.retriever = RAGRetriever.from_sample_data(
                        json_path, db_path, top_k=self.top_k
                    )
                    count = self.retriever.vector_store.collection_count()
                    print(f"[RAG] ✓ 知识库就绪 ({count} 条规范)")
                else:
                    print(f"[RAG] ⚠️ 规范JSON不存在: {json_path}, RAG禁用")
                    self.enable_rag = False
            
            # 9. 训练数据收集器
            lora_dir = self.output_dir / "lora_training"
            self.data_collector = TrainingDataCollector(output_dir=str(lora_dir))
            print("[LoRA数据收集] ✓ 就绪")
        
        print("=" * 60)

    def _segment_image(self, image_path: str):
        """对单张图像执行分割"""
        img = Image.open(image_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        
        with torch.no_grad():
            softmax, argmax = self.segmentor(img_tensor)
        
        mask = argmax.cpu().numpy()
        return img, mask, softmax.cpu().numpy()

    def _process_single_image(self, image_path: Path) -> Dict:
        """处理单张图像的完整pipeline"""
        result = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "reports": [],
            "error": None,
        }
        
        try:
            start_time = time.time()
            
            # Step 1: 分割
            print(f"\n  [1/5] 分割: {image_path.name}")
            original_img, mask, softmax = self._segment_image(str(image_path))
            
            # Step 2: ROI裁剪
            print(f"  [2/5] ROI裁剪...")
            rois = self.cropper.crop_largest_per_class(original_img, mask)
            
            if not rois:
                result["success"] = True
                result["message"] = "未检测到损伤区域"
                return result
            
            saved_roi_paths = self.cropper.save_rois(
                rois, str(self.rois_dir), prefix=image_path.stem
            )
            
            # Step 3: 几何统计 + RAG检索
            print(f"  [3/5] 几何计算 + RAG检索...")
            geometry_stats = {}
            rag_contexts = {}
            total_pixels = mask.shape[0] * mask.shape[1]
            geometry_stats["_total_pixels"] = total_pixels
            
            for cls_id in rois.keys():
                geo = self.cropper.compute_geometry_stats(mask, cls_id, self.pixel_to_mm)
                geo["image_ratio"] = geo.get("pixel_count", 0) / max(total_pixels, 1)
                geometry_stats[cls_id] = geo
                
                if self.enable_rag and self.retriever:
                    dmg_name = CLASS_NAMES.get(cls_id, "unknown")
                    hits = self.retriever.retrieve_by_damage_type(dmg_name)
                    rag_contexts[cls_id] = self.retriever.build_context(hits)
                else:
                    rag_contexts[cls_id] = ""
            
            # Step 4: VLM API 调用
            print(f"  [4/5] VLM API 调用 ({len(rois)} 个ROI区域)...")
            multimodal_inputs = self.prompt_builder.build_multimodal_input(
                rois, geometry_stats, rag_contexts
            )
            
            reports_for_image = []
            for i, (full_prompt, roi_img, system_prompt) in enumerate(multimodal_inputs):
                cls_id = list(rois.keys())[i] if i < len(rois) else 0
                
                # 使用完整的 prompt（system + user 已组合）
                raw_output = self.vlm.generate([roi_img], full_prompt)
                parsed = self.parser.parse_json(raw_output)
                
                # 补充元信息
                parsed["damage_id"] = f"{image_path.stem}_roi_{CLASS_NAMES.get(cls_id, cls_id)}"
                parsed["source_image"] = image_path.name
                parsed["roi_image"] = saved_roi_paths.get(cls_id, "")
                
                reports_for_image.append(parsed)
                
                # LoRA 数据收集
                if self.data_collector and saved_roi_paths.get(cls_id):
                    self.data_collector.collect(
                        input_image=saved_roi_paths[cls_id],
                        context=rag_contexts.get(cls_id, ""),
                        prompt=full_prompt,
                        output_json=json.dumps(parsed, ensure_ascii=False),
                        metadata={
                            "image_source": image_path.name,
                            "class_id": int(cls_id),
                            "class_name": str(CLASS_NAMES.get(cls_id, "")),
                            "confidence": float(parsed.get("confidence", 0)),
                        }
                    )
            
            # Step 5: 报告保存
            print(f"  [5/5] 保存报告...")
            for rpt in reports_for_image:
                json_path, md_path = self.renderer.save_report(
                    rpt, str(self.reports_dir), prefix=rpt.get("damage_id", f"{image_path.stem}")
                )
                rpt["_saved_json"] = json_path
                rpt["_saved_md"] = md_path
            
            # 汇总报告
            if len(reports_for_image) > 1:
                summary_md = self.renderer.save_summary(
                    reports_for_image, str(self.reports_dir), image_name=image_path.name
                )
                result["summary_path"] = summary_md
            
            elapsed = time.time() - start_time
            result["success"] = True
            result["reports"] = reports_for_image
            result["elapsed_seconds"] = round(elapsed, 2)
            result["num_detections"] = len(reports_for_image)
            
            print(f"  ✓ 完成! 检出 {len(reports_for_image)} 个损伤区域, 耗时 {elapsed:.1f}s")
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            print(f"  ✗ 错误: {e}")
        
        return result

    def run(self) -> List[Dict]:
        """运行完整的批量处理流程"""
        print("\n" + "=" * 60)
        print("ENSTRECT LLM+RAG 损伤智能报告生成 Pipeline (API版)")
        print("=" * 60)
        print(f"输入目录: {self.images_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"API 模型: {self.model_name}")
        print(f"RAG: {'✓ 启用' if self.enable_rag else '✗ 禁用'}")
        print(f"最大图像数: {self.max_images or '全部'}")
        
        # 收集图像文件
        image_files = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        if not image_files:
            print(f"❌ 未找到图像文件: {self.images_dir}")
            return []
        
        if self.max_images:
            image_files = image_files[:self.max_images]
        
        print(f"发现 {len(image_files)} 张图像\n")
        
        # 批量处理
        all_results = []
        for img_path in tqdm(image_files, desc="处理进度"):
            r = self._process_single_image(img_path)
            all_results.append(r)
        
        # 最终汇总
        self._print_final_summary(all_results)
        
        # 保存运行日志
        log_path = self.output_dir / "pipeline_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "run_time": datetime.now().isoformat(),
                "config": {
                    "images_dir": str(self.images_dir),
                    "model_name": self.model_name,
                    "enable_rag": self.enable_rag,
                    "top_k": self.top_k,
                    "pixel_to_mm": self.pixel_to_mm,
                },
                "results": all_results,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📄 运行日志已保存: {log_path}")
        
        # 打印LoRA数据统计
        if self.data_collector:
            print("\n📊 LoRA训练数据:")
            self.data_collector.print_statistics()
        
        return all_results

    def _print_final_summary(self, results: List[Dict]):
        """打印最终汇总"""
        success = sum(1 for r in results if r.get("success"))
        total_detections = sum(r.get("num_detections", 0) for r in results)
        total_time = sum(r.get("elapsed_seconds", 0) for r in results)
        
        print("\n" + "=" * 60)
        print("🎉 Pipeline 运行完毕!")
        print("=" * 60)
        print(f"  处理图像: {len(results)} 张")
        print(f"  成功: {success} 张")
        print(f"  总检出损伤: {total_detections} 处")
        print(f"  总耗时: {total_time:.1f}s")
        if total_time > 0:
            print(f"  平均耗时: {total_time/max(len(results),1):.1f}s/张")
        print(f"  报告目录: {self.reports_dir}")
        print("=" * 60)


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="ENSTRECT LLM+RAG 损伤智能报告生成 Pipeline (API版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 方式1: 设置环境变量后运行
  $env:DASHSCOPE_API_KEY='sk-xxxx'
  python run_llm_damage_report_api.py --images-dir enstrect/src/enstrect/assets/bridge_b/segment_test/views --output-dir output/

  # 方式2: 直接在命令行指定 API Key
  python run_llm_damage_report_api.py --images-dir ./test_images --api-key sk-xxxx --max-images 3

  # 不使用 RAG（纯 VLM 推理）
  python run_llm_damage_report_api.py --images-dir ./images --no-rag

API Key 获取: https://help.aliyun.com/zh/model-studio/get-api-key
        """
    )
    
    parser.add_argument("--images-dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output-dir", type=str, default="output_llm_report_api", help="输出目录")
    parser.add_argument("--api-key", type=str, default=None, help="阿里云百炼 API Key（默认从环境变量 DASHSCOPE_API_KEY 读取）")
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-7b-instruct", 
                        help="模型名称 (默认: qwen2.5-vl-7b-instruct, 可选: qwen2.5-vl-72b-instruct)")
    parser.add_argument("--enable-rag", action="store_true", default=True, help="启用RAG规范检索(默认开启)")
    parser.add_argument("--no-rag", action="store_true", help="禁用RAG")
    parser.add_argument("--top-k", type=int, default=3, help="RAG检索返回条数(默认3)")
    parser.add_argument("--max-images", type=int, default=None, help="最大处理图像数(默认全部)")
    parser.add_argument("--pixel-to-mm", type=float, default=0.1, help="像素到毫米换算系数(默认0.1)")
    parser.add_argument("--quality-filter", action="store_true", help="启用图像质量过滤")
    
    args = parser.parse_args()
    
    # 参数校验
    if not Path(args.images_dir).exists():
        print(f"❌ 图像目录不存在: {args.images_dir}")
        sys.exit(1)
    
    if not HAS_LLM_RAG:
        print(f"❌ LLM/RAG 模块导入失败: {IMPORT_ERROR}")
        print("请确保已安装依赖: pip install -r requirements.txt")
        sys.exit(1)
    
    if not HAS_ENSTRECT:
        print("⚠️ ENSTRECT 分割模块未安装，Pipeline无法执行分割步骤")
        print("请先确认 enstrect/src 目录存在且包含 nnunet_s2ds.py")
        sys.exit(1)
    
    # 检查 API Key
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未提供 API Key!")
        QwenVLAPI.print_setup_guide()
        sys.exit(1)
    
    # 创建并运行 Pipeline
    pipeline = DamageReportPipelineAPI(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        model_name=args.model_name,
        enable_rag=(args.enable_rag and not args.no_rag),
        top_k=args.top_k,
        max_images=args.max_images,
        pixel_to_mm=args.pixel_to_mm,
        enable_quality_filter=args.quality_filter,
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
