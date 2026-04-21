"""
KITTI Odometry 数据集加载器
支持从百度网盘下载的 KITTI 数据加载序列图像
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
from PIL import Image
import numpy as np


class KITTILoader:
    """
    KITTI Odometry 数据集加载器
    
    数据集结构:
    data_odometry_color/
        dataset/
            sequences/
                00/
                    image_2/  # 左相机彩色图像
                        000000.png
                        000001.png
                        ...
                    image_3/  # 右相机彩色图像
                        000000.png
                        000001.png
                        ...
                01/
                ...
    """
    
    # KITTI 数据路径配置
    DEFAULT_KITTI_PATHS = [
        "I:/BaiduNetdiskDownload/KITTI/data_odometry_color",  # 百度网盘下载路径
        "data/kitti/data_odometry_color",  # 项目内路径
        "/data/kitti/data_odometry_color",  # Linux 路径
    ]
    
    def __init__(self, kitti_root: Optional[str] = None, sequence: str = "00"):
        """
        初始化 KITTI 加载器
        
        Args:
            kitti_root: KITTI 数据集根目录，默认自动查找
            sequence: 序列号 (00-10)
        """
        self.sequence = sequence
        self.kitti_root = self._find_kitti_root(kitti_root)
        self.image_dir = self.kitti_root / "dataset" / "sequences" / sequence / "image_2"
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"KITTI 图像目录不存在: {self.image_dir}")
        
        self.image_files = sorted(self.image_dir.glob("*.png"))
        print(f"[KITTI] 加载序列 {sequence}: 找到 {len(self.image_files)} 张图像")
    
    def _find_kitti_root(self, kitti_root: Optional[str]) -> Path:
        """自动查找 KITTI 数据集根目录"""
        if kitti_root:
            path = Path(kitti_root)
            if path.exists():
                return path
        
        # 尝试默认路径
        for path_str in self.DEFAULT_KITTI_PATHS:
            path = Path(path_str)
            if path.exists():
                print(f"[KITTI] 自动找到数据集: {path}")
                return path
        
        raise FileNotFoundError(
            "未找到 KITTI 数据集！请设置正确的路径:\n"
            "1. 设置环境变量: $env:KITTI_ROOT='I:/BaiduNetdiskDownload/KITTI/data_odometry_color'\n"
            "2. 或在初始化时指定: KITTILoader(kitti_root='your/path')"
        )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """获取指定索引的图像"""
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        return img, str(img_path)
    
    def load_frame(self, frame_id: int) -> Tuple[Image.Image, str]:
        """
        加载指定帧
        
        Args:
            frame_id: 帧号 (如 0, 1, 2, ...)
            
        Returns:
            (PIL图像, 图像路径)
        """
        img_name = f"{frame_id:06d}.png"
        img_path = self.image_dir / img_name
        
        if not img_path.exists():
            raise FileNotFoundError(f"帧不存在: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        return img, str(img_path)
    
    def load_sequence(self, start: int = 0, end: Optional[int] = None, 
                     step: int = 1) -> Iterator[Tuple[Image.Image, str, int]]:
        """
        批量加载序列帧
        
        Args:
            start: 起始帧号
            end: 结束帧号 (None表示到最后)
            step: 采样步长
            
        Yields:
            (PIL图像, 图像路径, 帧号)
        """
        if end is None:
            end = len(self.image_files)
        
        for frame_id in range(start, end, step):
            if frame_id >= len(self.image_files):
                break
            
            img, path = self.load_frame(frame_id)
            yield img, path, frame_id
    
    def get_camera_intrinsics(self) -> Optional[np.ndarray]:
        """
        获取相机内参 (如果有 calib 文件)
        
        Returns:
            3x3 内参矩阵
        """
        calib_file = self.kitti_root / "dataset" / "sequences" / self.sequence / "calib.txt"
        
        if not calib_file.exists():
            print(f"[警告] 未找到标定文件: {calib_file}")
            return None
        
        # 解析 calib.txt
        with open(calib_file, 'r') as f:
            for line in f:
                if line.startswith("P2:"):
                    # P2 是左相机投影矩阵
                    values = line.split()[1:]
                    P = np.array(values, dtype=np.float32).reshape(3, 4)
                    # 提取内参 K
                    K = P[:, :3]
                    return K
        
        return None
    
    def copy_sample_frames(self, output_dir: str, num_frames: int = 10, 
                          step: int = 10) -> List[str]:
        """
        复制样本帧到输出目录（用于测试）
        
        Args:
            output_dir: 输出目录
            num_frames: 复制的帧数
            step: 采样步长
            
        Returns:
            复制的文件路径列表
        """
        import shutil
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        copied_files = []
        for i, (img, img_path, frame_id) in enumerate(
            self.load_sequence(start=0, step=step)
        ):
            if i >= num_frames:
                break
            
            dest_path = output_path / f"kitti_{self.sequence}_{frame_id:06d}.png"
            shutil.copy(img_path, dest_path)
            copied_files.append(str(dest_path))
            print(f"  复制: {dest_path.name}")
        
        print(f"[KITTI] 已复制 {len(copied_files)} 帧到 {output_dir}")
        return copied_files


def process_kitti_sequence_with_llm(sequence: str = "00", 
                                     max_frames: int = 5,
                                     output_dir: str = "output_kitti"):
    """
    使用 LLM+RAG 处理 KITTI 序列
    
    示例用法:
        process_kitti_sequence_with_llm(sequence="00", max_frames=5)
    """
    from ..rag import RAGRetriever
    from ..prompt import ROICropper, PromptBuilder
    from ..vlm import QwenVLAPI, OutputParser
    from ..report import ReportRenderer
    
    # 加载 KITTI
    kitti = KITTILoader(sequence=sequence)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化组件
    vlm = QwenVLAPI()
    parser = OutputParser()
    renderer = ReportRenderer()
    
    print(f"\n开始处理 KITTI 序列 {sequence} 的前 {max_frames} 帧...\n")
    
    for i, (img, img_path, frame_id) in enumerate(kitti.load_sequence()):
        if i >= max_frames:
            break
        
        print(f"处理帧 {frame_id:06d}...")
        
        # 这里可以添加分割、ROI 裁剪、LLM 推理等逻辑
        # 简化示例：直接保存原图
        img.save(output_path / f"frame_{frame_id:06d}.png")
    
    print(f"\n✓ 完成！输出目录: {output_dir}")


if __name__ == "__main__":
    # 测试加载器
    print("测试 KITTI 加载器...")
    
    try:
        kitti = KITTILoader(sequence="00")
        
        # 显示前3帧
        for i in range(min(3, len(kitti))):
            img, path = kitti[i]
            print(f"  帧 {i}: {path}, 尺寸: {img.size}")
        
        # 复制样本用于测试
        kitti.copy_sample_frames("data/kitti/samples", num_frames=5)
        
    except FileNotFoundError as e:
        print(f"[错误] {e}")
