"""
KITTI 数据集加载器测试脚本

用法:
    python tools/test_kitti_loader.py
    
环境变量:
    KITTI_ROOT: KITTI 数据集根目录 (默认: I:/BaiduNetdiskDownload/KITTI/data_odometry_color)
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_rag.utils import KITTILoader


def main():
    print("=" * 60)
    print("KITTI Odometry 数据集加载器测试")
    print("=" * 60)
    
    # 尝试加载序列 00
    try:
        print("\n[1/4] 加载 KITTI 序列 00...")
        kitti = KITTILoader(sequence="00")
        print(f"✓ 成功加载，共 {len(kitti)} 帧")
        
        # 显示前3帧信息
        print("\n[2/4] 显示前3帧信息...")
        for i in range(min(3, len(kitti))):
            img, path = kitti[i]
            print(f"  帧 {i}: {os.path.basename(path)}, 尺寸: {img.size}, 模式: {img.mode}")
        
        # 获取相机内参
        print("\n[3/4] 获取相机内参...")
        K = kitti.get_camera_intrinsics()
        if K is not None:
            print(f"  内参矩阵 K:\n{K}")
        else:
            print("  未找到标定文件")
        
        # 复制样本帧
        print("\n[4/4] 复制样本帧到 data/kitti/samples/...")
        copied = kitti.copy_sample_frames(
            output_dir="data/kitti/samples",
            num_frames=5,
            step=10
        )
        print(f"✓ 已复制 {len(copied)} 帧")
        
        print("\n" + "=" * 60)
        print("测试完成！样本保存在: data/kitti/samples/")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ 错误: {e}")
        print("\n请确认 KITTI 数据集路径:")
        print("  默认路径: I:/BaiduNetdiskDownload/KITTI/data_odometry_color")
        print("  或设置环境变量: $env:KITTI_ROOT='你的路径'")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
