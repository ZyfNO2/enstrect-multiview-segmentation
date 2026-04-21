#!/usr/bin/env python3
"""批量测试CRACK500数据集"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from enstrect.src.enstrect.run_low_confidence import main
import torch

if __name__ == "__main__":
    print("=" * 80)
    print("开始批量测试 CRACK500 数据集")
    print("=" * 80)
    
    # 清空命令行参数并设置
    sys.argv = [
        "run_low_confidence.py",
        "--images_dir", r"g:\Zed\testData\pavement crack datasets\CRACK500\testdata",
        "--out_dir", r"g:\Zed\testData\pavement crack datasets\CRACK500\output",
        "--scale", "0.5",
        "--prob_threshold", "0.2",
        "--bg_threshold", "0.6",
        "--min_points", "3"
    ]
    
    # 运行测试
    main()
