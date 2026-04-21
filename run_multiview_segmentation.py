"""
多视图图像分割示例
不需要PyTorch3D，仅展示多张图像的2D分割结果
"""

import sys
import os

# 添加ENSTRECT库路径 (根据实际路径修改)
# 方式1: 如果enstrect_lib在当前目录
ENSTRECT_PATH = os.path.join(os.path.dirname(__file__), 'enstrect_lib', 'src')
# 方式2: 使用绝对路径 (修改为你的路径)
# ENSTRECT_PATH = r'G:\Zed\ENSTRECTtest\enstrect\src'

if os.path.exists(ENSTRECT_PATH):
    sys.path.insert(0, ENSTRECT_PATH)
else:
    print(f"警告: ENSTRECT路径不存在: {ENSTRECT_PATH}")
    print("请修改脚本中的ENSTRECT_PATH为你的实际路径")
    sys.exit(1)

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel

# 颜色映射
def get_color_map():
    """为每个类别定义颜色"""
    return {
        0: [0, 0, 0],        # background - 黑色
        1: [255, 0, 0],      # crack - 红色
        2: [0, 255, 0],      # spalling - 绿色
        3: [0, 0, 255],      # corrosion - 蓝色
        4: [255, 255, 0],    # efflorescence - 黄色
        5: [255, 0, 255],    # vegetation - 紫色
        6: [0, 255, 255],    # control_point - 青色
    }

def apply_color_mask(image, mask, alpha=0.5):
    """将分割掩码以半透明叠加到原图上"""
    color_map = get_color_map()
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_map.items():
        colored_mask[mask == label_id] = color

    # 混合原图和掩码
    overlay = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    return overlay

def visualize_segmentation(image_path, model, output_path=None, device='cuda'):
    """对单张图像进行分割并可视化"""
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    # 转换为tensor (3, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

    # 运行分割
    with torch.no_grad():
        softmax, argmax = model(img_tensor)

    # 转换为numpy
    mask = argmax.cpu().numpy()

    # 创建可视化结果
    overlay = apply_color_mask(img_np, mask, alpha=0.4)

    # 创建组合图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=6)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()

    return mask, softmax.cpu().numpy()

def process_multiview(images_dir, output_dir, max_images=5):
    """处理多视图图像"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 获取所有图像文件
    image_files = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))

    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Processing first {min(max_images, len(image_files))} images...")

    # 加载模型
    print("Loading segmentation model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = NNUNetS2DSModel(allow_tqdm=False)
    model.predictor.device = device
    model.predictor.perform_everything_on_device = True

    # 处理每张图像
    results = []
    for i, img_path in enumerate(tqdm(image_files[:max_images], desc="Processing images")):
        output_path = output_dir / f"segmentation_{img_path.stem}.png"
        mask, softmax = visualize_segmentation(img_path, model, output_path, device)

        # 统计每个类别的像素数
        unique, counts = np.unique(mask, return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))

        results.append({
            'image': img_path.name,
            'mask_stats': stats,
            'output': output_path
        })

    # 打印统计信息
    print("\n" + "="*60)
    print("分割结果统计")
    print("="*60)

    class_names = {
        0: "background",
        1: "crack",
        2: "spalling",
        3: "corrosion",
        4: "efflorescence",
        5: "vegetation",
        6: "control_point"
    }

    for result in results:
        print(f"\n图像: {result['image']}")
        for class_id, count in sorted(result['mask_stats'].items()):
            percentage = count / sum(result['mask_stats'].values()) * 100
            print(f"  {class_names.get(class_id, f'class_{class_id}')}: {count} pixels ({percentage:.2f}%)")

    print(f"\n所有结果已保存到: {output_dir}")

    # 创建汇总图
    create_summary_figure(results, output_dir, images_dir)

    return results

def create_summary_figure(results, output_dir, images_dir):
    """创建汇总可视化图"""
    n_images = len(results)

    fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
    if n_images == 1:
        axes = axes.reshape(2, 1)

    for i, result in enumerate(results):
        # 读取原图和分割结果
        img_path = Path(images_dir) / result['image']
        seg_path = result['output']

        if img_path.exists():
            img = Image.open(img_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Input: {result['image']}")
            axes[0, i].axis('off')

        if seg_path.exists():
            seg_img = Image.open(seg_path)
            axes[1, i].imshow(seg_img)
            axes[1, i].set_title("Segmentation")
            axes[1, i].axis('off')

    plt.tight_layout()
    summary_path = output_dir / "multiview_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n汇总图已保存: {summary_path}")
    plt.close()

if __name__ == "__main__":
    # 配置路径 (根据你的实际路径修改)
    # 方式1: 使用相对路径
    base_dir = os.path.join(os.path.dirname(__file__), 'enstrect_lib', 'src', 'enstrect', 'assets')
    # 方式2: 使用绝对路径
    # base_dir = r"G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets"

    # Bridge B segment_test 数据
    images_dir = os.path.join(base_dir, 'bridge_b', 'segment_test', 'views')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    # 检查路径是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        print("请先下载测试数据: cd enstrect_lib && python -m enstrect.datasets.download")
        sys.exit(1)

    # 处理前5张图像 (设置为None处理全部)
    results = process_multiview(images_dir, output_dir, max_images=5)

    print("\n" + "="*60)
    print("✓ 多视图分割完成！")
    print("="*60)
