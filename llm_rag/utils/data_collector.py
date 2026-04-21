"""
LoRA微调训练数据收集器
每次VLM推理时自动保存 input-output 对，为后续QLoRA训练准备数据
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class TrainingDataCollector:
    """
    训练样本收集器
    
    功能:
    - 自动记录每次 VLM 推理的输入输出对
    - 支持多种导出格式 (JSONL / ShareGPT / Multimodal Conversation)
    - 数据统计与分析工具
    """

    def __init__(self, output_dir: str = "data/lora_training/raw_samples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_file = self.output_dir / "samples.jsonl"

    def collect(self, input_image: str, context: str, prompt: str,
                output_json: str, metadata: Optional[Dict] = None) -> str:
        """
        收集一条训练样本
        
        Args:
            input_image: ROI图像路径
            context: RAG检索到的上下文文本
            prompt: 发送给VLM的用户提示词
            output_json: VLM输出的JSON报告字符串
            metadata: 可选元信息（图像来源、置信度等）
            
        Returns:
            样本的唯一ID
        """
        sample_id = f"samp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        sample = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "image_path": str(input_image),
                "rag_context": context,
                "user_prompt": prompt,
            },
            "output": output_json,
            "metadata": metadata or {},
        }
        with open(self.samples_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        return sample_id

    def collect_batch(self, samples: List[Dict]) -> int:
        """批量收集多条样本"""
        count = 0
        for s in samples:
            self.collect(
                input_image=s.get("image_path", ""),
                context=s.get("context", ""),
                prompt=s.get("prompt", ""),
                output_json=s.get("output_json", "{}"),
                metadata=s.get("metadata", {}),
            )
            count += 1
        return count

    def export_training_data(self, output_dir: str, format: str = "sharegpt") -> str:
        """
        将原始样本转换为指定训练格式并导出
        
        Args:
            output_dir: 输出目录
            format: 目标格式 ('sharegpt' | 'jsonl_raw' | 'multimodal_conv')
            
        Returns:
            导出文件路径
        """
        samples = self._load_all_samples()
        if not samples:
            return ""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        if format == "sharegpt":
            return self._export_sharegpt(samples, out)
        elif format == "jsonl_raw":
            return self._export_jsonl_raw(samples, out)
        elif format == "multimodal_conv":
            return self._export_multimodal_conv(samples, out)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _load_all_samples(self) -> List[Dict]:
        """加载所有已收集的样本"""
        if not self.samples_file.exists():
            return []
        samples = []
        with open(self.samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return samples

    def _export_sharegpt(self, samples: List[Dict], output_dir: Path) -> str:
        """导出为 ShareGPT 格式（适用于 QLoRA/SFT 训练）"""
        filepath = output_dir / "training_data_sharegpt.jsonl"
        converted = []
        for samp in samples:
            inp = samp.get("input", {})
            conv = [
                {"role": "system", "content": inp.get("rag_context", "")},
                {"role": "user", "content": inp.get("user_prompt", "")},
                {"role": "assistant", "content": samp.get("output", "")},
            ]
            converted.append({"conversations": conv})
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return str(filepath)

    def _export_jsonl_raw(self, samples: List[Dict], output_dir: Path) -> str:
        """导出为原始 JSONL 格式"""
        filepath = output_dir / "training_data_raw.jsonl"
        with open(filepath, 'w', encoding='utf-8') as f:
            for samp in samples:
                f.write(json.dumps(samp, ensure_ascii=False) + '\n')
        return str(filepath)

    def _export_multimodal_conv(self, samples: List[Dict], output_dir: Path) -> str:
        """导出为多模态对话格式（含图像路径引用）"""
        filepath = output_dir / "training_data_multimodal.jsonl"
        converted = []
        for samp in samples:
            inp = samp.get("input", {})
            img_path = inp.get("image_path", "")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": inp.get("user_prompt", "")}
                ]},
                {"role": "assistant", "content": samp.get("output", "")},
            ]
            converted.append({"messages": messages})
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return str(filepath)

    def get_statistics(self) -> Dict:
        """分析已收集样本的统计信息"""
        samples = self._load_all_samples()
        if not samples:
            return {"total": 0, "message": "暂无样本"}
        
        damage_types = {}
        severity_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        confidence_values = []
        output_lengths = []
        
        for samp in samples:
            try:
                output = json.loads(samp.get("output", "{}"))
                dt = output.get("damage_type", "unknown")
                damage_types[dt] = damage_types.get(dt, 0) + 1
                sv = output.get("severity_level", 1)
                if isinstance(sv, int) and 1 <= sv <= 5:
                    severity_dist[sv] += 1
                cf = output.get("confidence", 0)
                if isinstance(cf, (int, float)):
                    confidence_values.append(float(cf))
                output_lengths.append(len(str(output)))
            except (json.JSONDecodeError, TypeError):
                damage_types["parse_error"] = damage_types.get("parse_error", 0) + 1
                output_lengths.append(0)
        
        stats = {
            "total": len(samples),
            "damage_type_distribution": damage_types,
            "severity_distribution": severity_dist,
            "avg_output_length": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            "max_output_length": max(output_lengths) if output_lengths else 0,
            "min_output_length": min(output_lengths) if output_lengths else 0,
        }
        if confidence_values:
            stats.update({
                "avg_confidence": sum(confidence_values) / len(confidence_values),
                "min_confidence": min(confidence_values),
                "max_confidence": max(confidence_values),
            })
        return stats

    def print_statistics(self) -> None:
        """打印统计摘要到控制台"""
        stats = self.get_statistics()
        print("=" * 50)
        print("📊 LoRA 训练数据统计")
        print("=" * 50)
        print(f"总样本数: {stats['total']}")
        
        if stats['total'] == 0:
            print("暂无训练样本")
            print("=" * 50)
            return
        
        print(f"\n损伤类型分布:")
        for dt, cnt in stats.get("damage_type_distribution", {}).items():
            print(f"  {dt}: {cnt}")
        print(f"\n严重程度分布:")
        for sv, cnt in stats.get("severity_distribution", {}).items():
            print(f"  {sv}级: {cnt}")
        if "avg_confidence" in stats:
            print(f"\n平均置信度: {stats['avg_confidence']:.1%}")
        if "avg_output_length" in stats:
            print(f"\n输出长度:")
            print(f"  平均: {stats['avg_output_length']:.0f} 字符")
            print(f"  范围: {stats['min_output_length']} ~ {stats['max_output_length']}")
        print("=" * 50)
