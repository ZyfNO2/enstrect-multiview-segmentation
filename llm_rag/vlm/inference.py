"""
Qwen2.5-VL 多模态大模型推理模块
支持官方权重加载、4-bit量化、多图输入、结构化输出
"""

import torch
from typing import List, Any, Optional, Dict
from PIL import Image
import warnings

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False
    warnings.warn("qwen-vl-utils未安装，请执行: pip install qwen-vl-utils")


class QwenVLInference:
    """
    Qwen2.5-VL 推理引擎封装
    
    支持功能:
    - 4-bit/8-bit 量化加载 (需 bitsandbytes)
    - 多张ROI图像并发推理
    - 可配置生成参数
    - 自动设备检测(CUDA/CPU)
    
    使用示例:
        vlm = QwenVLInference(load_in_4bit=True)
        result = vlm.generate([roi_image], "请分析这张裂缝图像")
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 load_in_4bit: bool = True,
                 device_map: str = "auto",
                 torch_dtype: str = "float16",
                 max_new_tokens: int = 2048,
                 temperature: float = 0.1,
                 do_sample: bool = False,
                 trust_remote_code: bool = True):
        if not HAS_QWEN_VL:
            raise ImportError("缺少依赖，请安装: pip install transformers accelerate qwen-vl-utils")
        
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False

    def load_model(self) -> None:
        """加载模型和处理器（懒加载，首次调用generate时自动触发）"""
        if self._loaded:
            return
        print(f"[QwenVL] 加载模型: {self.model_name}")
        print(f"[QwenVL] 量化: {'4-bit' if self.load_in_4bit else 'FP16'}")
        
        kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        
        if hasattr(self.model, 'device'):
            self.device = next(self.model.parameters()).device
        elif hasattr(self.model, 'hf_device_map'):
            self.device = list(self.model.hf_device_map.values())[0]
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._loaded = True
        print(f"[QwenVL] ✓ 模型加载完成, 设备: {self.device}")

    def generate(self, images: List[Image.Image], text_prompt: str,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 do_sample: Optional[bool] = None) -> str:
        """
        执行多模态推理
        
        Args:
            images: PIL图像列表（ROI裁剪区域）
            text_prompt: 用户提示词文本
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            do_sample: 是否采样
            
        Returns:
            模型生成的文本（通常为JSON字符串）
        """
        self.load_model()
        
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": img}
                        for img in images
                    ],
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages, return_video=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            do_sample=do_sample if do_sample is not None else self.do_sample,
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
        input_token_len = inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_token_len:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return output_text.strip()

    def generate_single(self, image: Image.Image, text_prompt: str, **kwargs) -> str:
        """单张图像推理的便捷方法"""
        return self.generate([image], text_prompt, **kwargs)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """释放模型显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        print("[QwenVL] 模型已卸载，显存已释放")
