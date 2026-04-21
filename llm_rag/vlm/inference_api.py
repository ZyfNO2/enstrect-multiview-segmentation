"""
Qwen2.5-VL API 调用模块
通过阿里云百炼 API 调用多模态大模型，无需本地下载模型
"""

import os
import base64
from typing import List, Optional
from pathlib import Path
from PIL import Image
import json


class QwenVLAPI:
    """
    Qwen2.5-VL API 调用封装
    
    使用阿里云百炼 (DashScope) 平台的 API 服务
    支持 Base64 图像编码上传
    """

    DEFAULT_MODEL = "qwen2.5-vl-7b-instruct"
    API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = DEFAULT_MODEL,
                 max_tokens: int = 2048,
                 temperature: float = 0.1):
        """
        初始化 API 客户端
        
        Args:
            api_key: 阿里云百炼 API Key，默认从环境变量 DASHSCOPE_API_KEY 读取
            model_name: 模型名称，默认 qwen2.5-vl-7b-instruct
            max_tokens: 最大生成 token 数
            temperature: 采样温度
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "请提供 API Key 或设置环境变量 DASHSCOPE_API_KEY\n"
                "获取地址: https://help.aliyun.com/zh/model-studio/get-api-key"
            )
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 尝试导入 OpenAI SDK
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.API_BASE_URL
            )
            self._has_openai = True
        except ImportError:
            self._has_openai = False
            print("[警告] openai 未安装，将使用 requests 进行 HTTP 调用")
            import requests
            self._requests = requests

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """将 PIL 图像编码为 Base64 Data URL"""
        import io
        buffered = io.BytesIO()
        # 转换为 RGB 模式（去除透明度）
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

    def generate(self, images: List[Image.Image], text_prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """
        调用 API 进行多模态推理
        
        Args:
            images: PIL 图像列表
            text_prompt: 用户提示词
            max_tokens: 最大生成长度（覆盖默认值）
            temperature: 采样温度（覆盖默认值）
            
        Returns:
            模型生成的文本
        """
        # 构建消息内容
        content = []
        
        # 添加图像
        for img in images:
            img_url = self._encode_image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
        
        # 添加文本
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        if self._has_openai:
            return self._generate_openai(messages, max_tokens, temperature)
        else:
            return self._generate_requests(messages, max_tokens, temperature)

    def _generate_openai(self, messages: List[dict], 
                         max_tokens: Optional[int],
                         temperature: Optional[float]) -> str:
        """使用 OpenAI SDK 调用"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API 调用失败: {e}")

    def _generate_requests(self, messages: List[dict],
                           max_tokens: Optional[int],
                           temperature: Optional[float]) -> str:
        """使用 requests 进行 HTTP 调用"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature
        }
        
        try:
            response = self._requests.post(
                f"{self.API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"API 调用失败: {e}")

    def generate_single(self, image: Image.Image, text_prompt: str, **kwargs) -> str:
        """单张图像推理的便捷方法"""
        return self.generate([image], text_prompt, **kwargs)

    @staticmethod
    def check_api_key() -> bool:
        """检查是否配置了 API Key"""
        return bool(os.getenv("DASHSCOPE_API_KEY"))

    @staticmethod
    def print_setup_guide():
        """打印 API 配置指南"""
        print("=" * 60)
        print("🔑 阿里云百炼 API 配置指南")
        print("=" * 60)
        print("1. 访问阿里云百炼控制台:")
        print("   https://bailian.console.aliyun.com/")
        print()
        print("2. 获取 API Key:")
        print("   https://help.aliyun.com/zh/model-studio/get-api-key")
        print()
        print("3. 设置环境变量:")
        print("   Windows PowerShell:")
        print("   $env:DASHSCOPE_API_KEY='your-api-key'")
        print()
        print("   Windows CMD:")
        print("   set DASHSCOPE_API_KEY=your-api-key")
        print()
        print("   或者永久设置到系统环境变量")
        print()
        print("4. 可选：安装 OpenAI SDK（推荐）:")
        print("   pip install openai")
        print("=" * 60)
