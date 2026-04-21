"""
结构损伤评估报告渲染模块
将VLM输出的JSON数据渲染为可读的Markdown/JSON双格式报告
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ReportRenderer:
    """
    损伤评估报告渲染器
    
    功能:
    - JSON Schema 验证与标准化
    - Markdown 格式报告生成（含表格、引用标注）
    - 双格式持久化（.json + .md）
    - 多视图汇总报告
    """

    SEVERITY_LABELS = {
        1: ("完好", "🟢", "结构状态良好，无需处理"),
        2: ("轻微", "🔵", "建议定期监测"),
        3: ("中等", "🟡", "建议安排检修"),
        4: ("较严重", "🟠", "应尽快安排维修"),
        5: ("严重/危险", "🔴", "需立即采取安全措施"),
    }

    DAMAGE_TYPE_LABELS = {
        "crack": "裂缝",
        "spalling": "剥落",
        "corrosion": "锈蚀",
        "efflorescence": "泛白/风化",
        "vegetation": "植被侵蚀",
        "control_point": "控制点",
        "unknown": "未知",
    }

    def __init__(self):
        pass

    def render_json(self, report_data: Dict) -> str:
        """渲染为格式化的JSON字符串（ensure_ascii=False支持中文）"""
        standardized = self._standardize(report_data)
        return json.dumps(standardized, ensure_ascii=False, indent=2)

    def render_markdown(self, report_data: Dict) -> str:
        """渲染为Markdown格式的可读报告"""
        data = self._standardize(report_data)
        lines = []
        
        dmg_type_cn = self.DAMAGE_TYPE_LABELS.get(data.get("damage_type", ""), data.get("damage_type", ""))
        severity = data.get("severity_level", 1)
        sev_info = self.SEVERITY_LABELS.get(severity, ("未知", "⚪", ""))
        conf = data.get("confidence", 0)
        
        lines.extend([
            f"# 📋 结构损伤智能评估报告",
            "",
            f"> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> **置信度**: {conf:.1%}",
            "",
            "---",
            "",
            "## 基本信息",
            "",
            f"| 项目 | 内容 |",
            f"|------|------|",
            f"| 损伤类型 | **{dmg_type_cn}** |",
            f"| 严重程度 | {sev_info[0]} ({severity}/5级) {sev_info[1]} |",
            f"| 处置建议级别 | {sev_info[2]} |",
            "",
            "---",
            "",
            "## 📝 损伤描述",
            "",
            data.get("description", "(无描述信息)"),
            "",
            "---",
            "",
            "## 📐 几何量化信息",
            "",
            data.get("geometry_summary", "(无几何数据)"),
            "",
            "---",
            "",
            "## 📖 参考规范条文",
            "",
            data.get("regulation_reference", "(未引用规范)"),
            "",
            "---",
            "",
            "## 🔧 修复建议",
            "",
            data.get("repair_recommendation", "(无修复建议)"),
            "",
            "---",
            "",
            "*本报告由 ENSTRECT-LLM 智能检测系统自动生成，仅供参考。最终判定请以现场工程师为准。*",
        ])
        
        return "\n".join(lines)

    def save_report(self, report_data: Dict, output_dir: str,
                    prefix: str = "report") -> Tuple[str, str]:
        """
        保存双格式报告到磁盘
        
        Returns:
            (json_path, md_path) 元组
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        json_content = self.render_json(report_data)
        md_content = self.render_markdown(report_data)
        
        json_path = out / f"{prefix}.json"
        md_path = out / f"{prefix}.md"
        
        json_path.write_text(json_content, encoding='utf-8')
        md_path.write_text(md_content, encoding='utf-8')
        
        return str(json_path), str(md_path)

    def generate_summary(self, reports: List[Dict], image_name: Optional[str] = None) -> str:
        """
        从多张ROI的报告列表生成汇总
        
        Args:
            reports: 同一图像多个损伤区域的报告字典列表
            image_name: 原始图像文件名
            
        Returns:
            Markdown 格式的汇总文本
        """
        if not reports:
            return "# 无损伤检测结果"
        
        lines = [
            f"# 📊 图像损伤检测汇总报告",
            "",
        ]
        if image_name:
            lines.append(f"> **源图像**: `{image_name}`")
            lines.append("")
        
        lines.extend([
            f"**检测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**检出损伤区域数**: {len(reports)}",
            "",
            "---",
            "",
            "## 各区域详情",
            "",
        ])
        
        for i, rpt in enumerate(reports, 1):
            dmg_type = self.DAMAGE_TYPE_LABELS.get(rpt.get("damage_type", "?"), "?")
            sev = rpt.get("severity_level", 1)
            sev_label = self.SEVERITY_LABELS.get(sev, ("?", ""))[0]
            
            lines.extend([
                f"### 区域 {i}: {dmg_type}",
                "",
                f"- **严重程度**: {sev_label} ({sev}/5级)",
                f"- **描述**: {rpt.get('description', 'N/A')[:100]}...",
                f"- **几何**: {rpt.get('geometry_summary', 'N/A')}",
                f"- **规范依据**: {rpt.get('regulation_reference', 'N/A')[:80]}...",
                f"- **修复建议**: {rpt.get('repair_recommendation', 'N/A')[:100]}...",
                "",
                "---",
                "",
            ])
        
        lines.extend([
            "## 统计总览",
            "",
            "| 区域 | 类型 | 严重程度 | 置信度 |",
            "|------|------|----------|--------|",
        ])
        for i, rpt in enumerate(reports, 1):
            dt = self.DAMAGE_TYPE_LABELS.get(rpt.get("damage_type", "?"), "?")
            sv = rpt.get("severity_level", "?")
            cf = rpt.get("confidence", 0)
            lines.append(f"| {i} | {dt} | {sv}/5 | {cf:.0%} |")
        
        max_severity = max((r.get("severity_level", 0) for r in reports), default=0)
        avg_conf = sum(r.get("confidence", 0) for r in reports) / len(reports) if reports else 0
        
        lines.extend([
            "",
            "**综合评定**:",
            f"- 最高严重等级: **{max_severity}/5**",
            f"- 平均置信度: **{avg_conf:.1%}**",
            "",
            "---",
            "*由 ENSTRECT-LLM 自动生成*",
        ])
        
        return "\n".join(lines)

    def save_summary(self, reports: List[Dict], output_dir: str,
                     image_name: Optional[str] = None) -> str:
        """保存汇总报告至MD文件"""
        content = self.generate_summary(reports, image_name)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = Path(image_name).stem if image_name else "summary"
        path = out / f"{prefix}_summary.md"
        path.write_text(content, encoding='utf-8')
        return str(path)

    def _standardize(self, data: Dict) -> Dict:
        """标准化报告数据：确保所有字段存在且类型正确"""
        result = dict(data)
        defaults = {
            "damage_id": data.get("damage_id", f"dmg_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            "damage_type": str(data.get("damage_type", "unknown")),
            "severity_level": int(data.get("severity_level", 1)),
            "description": str(data.get("description", "")),
            "geometry_summary": str(data.get("geometry_summary", "")),
            "regulation_reference": str(data.get("regulation_reference", "")),
            "repair_recommendation": str(data.get("repair_recommendation", "")),
            "confidence": float(data.get("confidence", 0.0)),
        }
        for key, default_val in defaults.items():
            if key not in result or result[key] is None:
                result[key] = default_val
        sv = result["severity_level"]
        result["severity_level"] = max(1, min(5, int(sv)))
        cf = result["confidence"]
        result["confidence"] = max(0.0, min(1.0, float(cf)))
        result["generated_at"] = datetime.now().isoformat()
        return result
