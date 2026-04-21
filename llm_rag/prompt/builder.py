"""
多模态Prompt构建模块（优化版）
构建System Prompt + User Prompt（图像+RAG上下文+几何数据），发送给Qwen2.5-VL
优化：添加Few-shot示例，强制JSON Schema格式
"""

from typing import Dict, List, Tuple, Optional, Any
from PIL import Image


class PromptBuilder:
    """多模态Prompt构建器：组装Qwen2.5-VL推理所需的结构化输入"""

    SYSTEM_PROMPT_TEMPLATE = """你是一位资深混凝土结构工程师与桥梁检测专家，拥有20年以上的结构健康监测经验。
你的任务是根据提供的损伤区域图像、几何测量数据和工程规范条文，生成专业的结构损伤评估报告。

## ⚠️ 重要：输出格式要求
你必须**只输出纯JSON格式**，不要添加任何markdown代码块标记（如 ```json），不要添加任何解释性文字。
输出必须是可以直接解析的JSON对象，格式如下：

{
  "damage_type": "crack",
  "severity_level": 3,
  "description": "图像显示一条横向裂缝，位于构件中部，裂缝宽度约0.2mm，长度约15cm，贯穿保护层。",
  "geometry_summary": "损伤像素数7343px，估计面积73.43mm²，占图像比例0.02%",
  "regulation_reference": "依据JTG/T H21-2011第3.2.1条，裂缝宽度0.15mm<δ≤0.35mm，评定标度为3级（较差）",
  "repair_recommendation": "建议采用表面封闭法处理，使用环氧树脂注浆材料填充裂缝，防止水分渗入。",
  "confidence": 0.85
}

## 字段说明
- damage_type: 损伤类型，必须是以下之一：crack(裂缝)/spalling(剥落)/corrosion(锈蚀)/efflorescence(泛白)/vegetation(植被)
- severity_level: 严重程度等级，整数1-5（1=完好, 2=轻微, 3=中等, 4=较严重, 5=严重）
- description: 自然语言描述损伤特征（位置、形态、尺寸、颜色等视觉特征）
- geometry_summary: 几何量化数据摘要
- regulation_reference: 引用的规范条文（如有）
- repair_recommendation: 具体的修复建议
- confidence: 置信度0.0-1.0

## 评估原则
1. 如果图像中无明显损伤或损伤难以识别，damage_type填"unknown"，severity_level填1
2. 必须基于图像实际内容进行客观描述，不要编造数据
3. 修复建议要具体可操作，不要泛泛而谈"请专业人员处理"
4. 如不确定，在description中说明不确定性，降低confidence值
"""

    DAMAGE_TEMPLATES = {
        "crack": {
            "system_addon": """
## 裂缝评定专项指引
根据JTG/T H21-2011《公路桥梁技术状况评定标准》：
- 裂缝宽度δ≤0.05mm：1级（完好）
- 0.05mm<δ≤0.15mm：2级（轻微）
- 0.15mm<δ≤0.35mm：3级（中等）
- 0.35mm<δ≤0.60mm：4级（较严重）
- δ>0.60mm：5级（危险）

注意区分：受力裂缝、收缩裂缝、温度裂缝的不同特征。
""",
            "user_template": """【任务】分析以下混凝土结构裂缝图像，输出JSON格式评估报告。

【损伤区域图像】（附图）

【几何测量数据】
- 损伤像素数: {pixel_count}px
- 估计面积: {area_mm2} mm²
- 占图像比例: {image_ratio:.4f}

【参考规范条文】
{rag_context}

【输出要求】
请只输出JSON格式的评估报告，不要添加任何额外文字或markdown标记。格式示例：
{{"damage_type":"crack","severity_level":3,"description":"...","geometry_summary":"...","regulation_reference":"...","repair_recommendation":"...","confidence":0.85}}
""",
        },
        "spalling": {
            "system_addon": """
## 剥落评定专项指引
根据JTG/T H21-2011：
- 剥落面积占比S≤1%：1级
- 1%<S≤3%：2级
- 3%<S≤6%：3级
- 6%<S≤15%：4级
- S>15%：5级

剥落深度超过保护层或伴有露筋，等级应提高一级。
""",
            "user_template": """【任务】分析以下混凝土剥落损伤图像，输出JSON格式评估报告。

【损伤区域图像】（附图）

【几何测量数据】
- 损伤像素数: {pixel_count}px
- 估计面积: {area_mm2} mm²
- 占图像比例: {image_ratio:.4f}

【参考规范条文】
{rag_context}

【输出要求】
请只输出JSON格式的评估报告，不要添加任何额外文字或markdown标记。格式示例：
{{"damage_type":"spalling","severity_level":3,"description":"...","geometry_summary":"...","regulation_reference":"...","repair_recommendation":"...","confidence":0.85}}
""",
        },
        "corrosion": {
            "system_addon": """
## 锈蚀评定专项指引
根据JTG/T H21-2011钢筋锈蚀电位检测：
- E>-200mV：1级（无锈蚀活动性）
- -200mV≥E>-300mV：2级
- -300mV≥E>-400mV：3级
- -400mV≥E>-500mV：4级
- E≤-500mV：5级（严重锈蚀）

观察锈蚀颜色：黄褐色（初期）→红褐色（中期）→黑色（严重）。
""",
            "user_template": """【任务】分析以下钢筋锈蚀损伤图像，输出JSON格式评估报告。

【损伤区域图像】（附图）

【几何测量数据】
- 损伤像素数: {pixel_count}px
- 估计面积: {area_mm2} mm²
- 占图像比例: {image_ratio:.4f}

【参考规范条文】
{rag_context}

【输出要求】
请只输出JSON格式的评估报告，不要添加任何额外文字或markdown标记。格式示例：
{{"damage_type":"corrosion","severity_level":3,"description":"...","geometry_summary":"...","regulation_reference":"...","repair_recommendation":"...","confidence":0.85}}
""",
        },
        "default": {
            "system_addon": "",
            "user_template": """【任务】分析以下结构损伤图像，输出JSON格式评估报告。

【损伤区域图像】（附图）

【几何测量数据】
- 损伤像素数: {pixel_count}px
- 估计面积: {area_mm2} mm²
- 占图像比例: {image_ratio:.4f}

【参考规范条文】
{rag_context}

【输出要求】
请只输出JSON格式的评估报告，不要添加任何额外文字或markdown标记。必须包含以下字段：damage_type, severity_level, description, geometry_summary, regulation_reference, repair_recommendation, confidence
""",
        },
    }

    # Few-shot 示例，帮助模型理解输出格式
    FEW_SHOT_EXAMPLES = {
        "crack": """
【示例】裂缝损伤评估输出：
{"damage_type":"crack","severity_level":3,"description":"图像显示构件表面有一条横向裂缝，位于中部偏左位置，裂缝走向基本水平，长度约20cm，宽度目测约0.25mm，裂缝边缘较整齐，未见明显剥落。","geometry_summary":"损伤像素数7343px，估计面积73.43mm²，占图像比例0.02%","regulation_reference":"依据JTG/T H21-2011第3.2.1条，裂缝宽度0.15mm<δ≤0.35mm，评定标度为3级（较差），建议定期监测。","repair_recommendation":"建议采用环氧树脂表面封闭法处理，清理裂缝后注入低粘度环氧浆液，防止水分和有害物质渗入。处理后每半年检查一次裂缝发展情况。","confidence":0.82}
""",
        "spalling": """
【示例】剥落损伤评估输出：
{"damage_type":"spalling","severity_level":4,"description":"图像显示混凝土表面有明显剥落区域，位于构件边缘，剥落深度较深，可见内部骨料，局部有露筋现象，钢筋表面有轻微锈蚀痕迹。","geometry_summary":"损伤像素数15230px，估计面积152.3mm²，占图像比例0.05%","regulation_reference":"依据JTG/T H21-2011第3.3.2条，剥落面积占比约5%，且伴有露筋，评定标度为4级（较严重）。","repair_recommendation":"1. 清除松动混凝土至坚实基层；2. 对锈蚀钢筋进行除锈处理并涂刷防锈漆；3. 采用聚合物改性水泥砂浆或环氧砂浆修补；4. 修补后加强养护，7天内保持湿润。","confidence":0.78}
""",
        "corrosion": """
【示例】锈蚀损伤评估输出：
{"damage_type":"corrosion","severity_level":3,"description":"图像显示钢筋表面有明显锈蚀，呈黄褐色至红褐色，锈蚀区域集中在保护层较薄处，局部混凝土有锈胀裂缝，裂缝宽度约0.1mm。","geometry_summary":"锈蚀区域像素数4520px，估计面积45.2mm²，占图像比例0.015%","regulation_reference":"依据JTG/T H21-2011第3.4.1条及视觉评估，钢筋锈蚀程度中等，伴有轻微锈胀裂缝，评定标度为3级。","repair_recommendation":"1. 凿除锈蚀区域松散混凝土；2. 彻底除锈并涂刷两道防锈漆；3. 采用补偿收缩混凝土或聚合物砂浆修复保护层；4. 建议进行钢筋锈蚀电位检测进一步评估。","confidence":0.75}
""",
    }

    def __init__(self, default_pixel_to_mm: float = 0.1):
        self.default_pixel_to_mm = default_pixel_to_mm

    def build_system_prompt(self, damage_type: Optional[str] = None) -> str:
        """构建系统提示词，包含角色定义和输出格式要求"""
        base = self.SYSTEM_PROMPT_TEMPLATE
        template = self.DAMAGE_TEMPLATES.get(damage_type, self.DAMAGE_TEMPLATES["default"])
        addon = template.get("system_addon", "")
        
        # 添加 few-shot 示例
        few_shot = self.FEW_SHOT_EXAMPLES.get(damage_type, "")
        
        if damage_type and addon:
            base += "\n" + addon
        if few_shot:
            base += "\n" + few_shot
            
        return base

    def build_user_prompt(self, roi_image: Image.Image, rag_context: str,
                          geometry_data: Dict, damage_type: str = "default") -> Tuple[str, Image.Image]:
        """
        构建用户提示词（含图像）
        
        Returns:
            (text_prompt, pil_image) 元组，可直接传给Qwen2.5-VL
        """
        template = self.DAMAGE_TEMPLATES.get(damage_type, self.DAMAGE_TEMPLATES["default"])
        user_text = template["user_template"].format(
            pixel_count=geometry_data.get("pixel_count", 0),
            area_mm2=geometry_data.get("area_mm2", 0),
            image_ratio=geometry_data.get("image_ratio", 0),
            rag_context=rag_context if rag_context else "暂无相关规范条文，请依据通用工程经验进行评估。"
        )
        return user_text, roi_image

    def get_template(self, damage_type: str) -> Dict:
        """获取指定损伤类型的模板配置"""
        return self.DAMAGE_TEMPLATES.get(damage_type, self.DAMAGE_TEMPLATES["default"])

    def build_multimodal_input(self, rois: Dict[int, Image.Image],
                                 geometry_stats: Dict[int, Dict],
                                 rag_contexts: Dict[int, str]) -> List[Tuple[str, Image.Image, str]]:
        """
        为多个ROI批量构建多模态输入
        
        Returns:
            [(text_prompt, roi_image, system_prompt), ...] 列表
        """
        inputs = []
        for cls_id, roi_img in rois.items():
            dmg_type = geometry_stats.get(cls_id, {}).get("class_name", "default")
            geo_data = geometry_stats.get(cls_id, {})
            total_pixels = geometry_stats.get("_total_pixels", geo_data.get("pixel_count", 1))
            geo_data["image_ratio"] = geo_data.get("pixel_count", 0) / max(total_pixels, 1)
            ctx = rag_contexts.get(cls_id, "")
            
            # 构建 system + user prompt
            system_prompt = self.build_system_prompt(dmg_type)
            user_text, img = self.build_user_prompt(roi_img, ctx, geo_data, dmg_type)
            
            # 组合完整的 prompt（system + user）
            full_prompt = f"{system_prompt}\n\n{user_text}"
            inputs.append((full_prompt, img, system_prompt))
        return inputs
