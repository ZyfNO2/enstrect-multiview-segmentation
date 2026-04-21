"""
VLM输出解析模块
将Qwen2.5-VL生成的文本解析为结构化JSON，处理各种格式异常情况
"""

import re
import json
from typing import Dict, Optional, Any, List


class OutputParser:
    """
    VLM输出解析器
    
    处理LLM输出的常见问题:
    - JSON被包裹在markdown代码块中
    - JSON前后有多余文字
    - JSON格式不完整或包含语法错误
    - 缺少必要字段时的默认值填充
    """

    REPORT_SCHEMA = {
        "damage_type": str,
        "severity_level": int,
        "description": str,
        "geometry_summary": str,
        "regulation_reference": str,
        "repair_recommendation": str,
        "confidence": float,
    }

    REQUIRED_FIELDS = ["damage_type", "severity_level", "description"]

    VALID_DAMAGE_TYPES = {"crack", "spalling", "corrosion", "efflorescence", "vegetation"}

    def __init__(self, strict_mode: bool = False):
        """Initialize output parser.

        Args:
            strict_mode: If True, raise ValueError when JSON cannot
                be parsed or repaired. If False, return empty dict.
                Default False.
        """
        self.strict_mode = strict_mode

    def parse_json(self, text: str) -> Dict:
        """
        从VLM输出文本中提取并解析JSON
        
        Args:
            text: VLM原始输出文本
            
        Returns:
            解析后的字典，解析失败返回空字典但保留raw_output
        """
        cleaned = self._extract_json_string(text)
        if not cleaned:
            result = {"raw_output": text, "parse_error": "无法提取JSON"}
            if self.strict_mode:
                raise ValueError(result["parse_error"])
            return result
        
        try:
            data = json.loads(cleaned)
            data = self._validate_and_fix(data)
            return data
        except json.JSONDecodeError as e:
            repaired = self.repair_json(cleaned)
            if repaired:
                try:
                    data = json.loads(repaired)
                    data = self._validate_and_fix(data)
                    data["_repaired"] = True
                    return data
                except json.JSONDecodeError:
                    pass
            result = {"raw_output": text, "parse_error": str(e)}
            if self.strict_mode:
                raise ValueError(result["parse_error"])
            return result

    def _extract_json_string(self, text: str) -> Optional[str]:
        """从文本中提取JSON字符串（处理markdown包裹等情况）"""
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                if candidate.startswith('{') and candidate.endswith('}'):
                    return candidate
        if text.strip().startswith('{'):
            return text.strip()
        bracket_start = text.find('{')
        bracket_end = text.rfind('}')
        if bracket_start >= 0 and bracket_end > bracket_start:
            return text[bracket_start:bracket_end + 1]
        return None

    def repair_json(self, broken_json: str) -> Optional[str]:
        """尝试修复损坏的JSON字符串"""
        s = broken_json
        s = re.sub(r',\s*([}\]])', r'\1', s)
        s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
        open_braces = s.count('{') - s.count('}')
        close_braces = s.count('}') - s.count('{')
        if open_braces > 0:
            s += '}' * open_braces
        if close_braces > 0:
            s = '{' * close_braces + s
        open_brackets = s.count('[') - s.count(']')
        if open_brackets > 0:
            s += ']' * open_brackets
        quotes_unclosed = len(re.findall(r'(?<!\\)"([^"]*(?<!\\))$', s.split('\n')[-1]))
        if quotes_unclosed % 2 != 0:
            s = s.rstrip() + '"'
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            return None

    def _validate_and_fix(self, data: Dict) -> Dict:
        """验证并修复JSON数据，填充缺失字段的默认值"""
        if not isinstance(data, dict):
            return {"raw_output": str(data), "parse_error": "非JSON对象"}
        for field, expected_type in self.REPORT_SCHEMA.items():
            if field not in data:
                data[field] = self._default_value(field)
            elif not isinstance(data[field], expected_type):
                try:
                    if expected_type == int:
                        data[field] = int(float(str(data[field])))
                    elif expected_type == float:
                        data[field] = float(data[field])
                    elif expected_type == str:
                        data[field] = str(data[field])
                except (ValueError, TypeError):
                    data[field] = self._default_value(field)
        dt = data.get("damage_type", "").lower().strip()
        if dt and dt not in self.VALID_DAMAGE_TYPES:
            for valid in self.VALID_DAMAGE_TYPES:
                if valid in dt or dt in valid:
                    data["damage_type"] = valid
                    break
        sv = data.get("severity_level")
        if isinstance(sv, (int, float)):
            data["severity_level"] = max(1, min(5, int(sv)))
        conf = data.get("confidence")
        if isinstance(conf, (int, float)):
            data["confidence"] = max(0.0, min(1.0, float(conf)))
        return data

    def _default_value(self, field: str) -> Any:
        defaults = {
            "damage_type": "unknown",
            "severity_level": 1,
            "description": "",
            "geometry_summary": "",
            "regulation_reference": "",
            "repair_recommendation": "",
            "confidence": 0.0,
        }
        return defaults.get(field, "")

    def format_as_readable(self, report: Dict) -> str:
        """将JSON报告格式化为人类可读的中文文本"""
        lines = [
            "=" * 50,
            "📋 结构损伤智能评估报告",
            "=" * 50,
            f"损伤类型: {report.get('damage_type', 'N/A')}",
            f"严重程度: {'⭐' * report.get('severity_level', 1)} ({report.get('severity_level', '?')}/5级)",
            f"置信度: {report.get('confidence', 0):.1%}",
            "-" * 50,
            "📝 损伤描述:",
            report.get('description', '(无)'),
            "-" * 50,
            "📐 几何信息:",
            report.get('geometry_summary', '(无)'),
            "-" * 50,
            "📖 参考规范:",
            report.get('regulation_reference', '(无)'),
            "-" * 50,
            "🔧 修复建议:",
            report.get('repair_recommendation', '(无)'),
            "=" * 50,
        ]
        return "\n".join(lines)
