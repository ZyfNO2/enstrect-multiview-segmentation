"""
加载本地环境变量工具
自动读取 .env 文件并设置到环境变量
"""

import os
from pathlib import Path


def load_env(env_file: str = ".env") -> dict:
    """
    从 .env 文件加载环境变量
    
    Args:
        env_file: 环境变量文件路径
        
    Returns:
        加载的环境变量字典
    """
    env_path = Path(env_file)
    if not env_path.exists():
        return {}
    
    env_vars = {}
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            # 解析 KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # 去除引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
                env_vars[key] = value
    
    return env_vars


def check_api_key() -> bool:
    """检查是否配置了 DASHSCOPE_API_KEY"""
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def get_api_key() -> str:
    """获取 API Key，如果不存在则抛出异常"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 环境变量！\n"
            "请执行以下操作之一:\n"
            "1. 创建 .env 文件并添加: DASHSCOPE_API_KEY=your-key\n"
            "2. 设置环境变量: $env:DASHSCOPE_API_KEY='your-key'\n"
            "3. 在命令行使用 --api-key 参数指定"
        )
    return api_key


if __name__ == "__main__":
    # 测试加载
    loaded = load_env()
    print(f"已加载 {len(loaded)} 个环境变量")
    if "DASHSCOPE_API_KEY" in loaded:
        key = loaded["DASHSCOPE_API_KEY"]
        masked = key[:8] + "****" + key[-4:] if len(key) > 12 else "****"
        print(f"DASHSCOPE_API_KEY: {masked}")
    else:
        print("警告: 未找到 DASHSCOPE_API_KEY")
