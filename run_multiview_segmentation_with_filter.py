"""
多视图图像分割示例（带数据清洗）
支持运动模糊检测和过滤
不需要PyTorch3D，仅展示多张图像的2D分割结果
"""

import sys
import os

# 添加ENSTRECT库路径 (根据实际路径修改)
ENSTRECT_PATH = os.path.join(os.path.dirname(__file__), 'enstrect_lib', 'src')
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

# 导入图像质量过滤模块
from image_quality_filter import (
    detect_blur_laplacian,
    filter_blurry_images,
    evaluate_image_quality
)


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


def visualize_segmentation(image_path, model, output_path=None, device='cuda', 
                          quality_info=None):
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

    # 创建组合图（增加质量信息）
    if quality_info:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 原图
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Original: {Path(image_path).name}')
        axes[0, 0].axis('off')
        
        # 质量信息
        quality_text = f"Image Quality:\n"
        quality_text += f"Score: {quality_info.get('score', 'N/A'):.3f}\n"
        if 'laplacian' in quality_info:
            quality_text += f"Laplacian: {quality_info['laplacian']['variance']:.1f}\n"
        if 'sobel' in quality_info:
            quality_text += f"Sobel: {quality_info['sobel']['magnitude']:.1f}"
        
        axes[0, 1].text(0.1, 0.5, quality_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis('off')
        
        # 分割掩码
        axes[1, 0].imshow(mask, cmap='tab10', vmin=0, vmax=6)
        axes[1, 0].set_title('Segmentation Mask')
        axes[1, 0].axis('off')
        
        # 叠加图
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay')
        axes[1, 1].axis('off')
    else:
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


def process_multiview_with_filter(images_dir, output_dir, 
                                  max_images=5,
                                  enable_quality_filter=True,
                                  blur_threshold=100.0,
                                  min_quality_score=0.3,
                                  quality_method='laplacian',
                                  max_blurry_ratio=0.5):
    """
    处理多视图图像（带质量过滤）
    
    Args:
        images_dir: 图像目录
        output_dir: 输出目录
        max_images: 最大处理图像数，None表示全部
        enable_quality_filter: 是否启用质量过滤
        blur_threshold: 模糊检测阈值
        min_quality_score: 最低质量评分（0-1）
        quality_method: 质量检测方法 'laplacian'|'sobel'|'fft'|'combined'
        max_blurry_ratio: 允许的最大模糊图像比例
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 获取所有图像文件
    image_files = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))

    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return

    print(f"\n{'='*60}")
    print("ENSTRECT 多视图分割（带数据清洗）")
    print(f"{'='*60}")
    print(f"发现 {len(image_files)} 张图像")

    # ========== 步骤1: 图像质量检测与过滤 ==========
    if enable_quality_filter:
        print(f"\n{'='*60}")
        print("步骤1: 图像质量检测与数据清洗")
        print(f"{'='*60}")
        print(f"检测方法: {quality_method}")
        print(f"模糊阈值: {blur_threshold}")
        print(f"最低质量评分: {min_quality_score}")
        
        sharp_images = []
        blurry_images = []
        quality_data = {}
        
        print(f"\n正在检测图像质量...")
        for img_path in tqdm(image_files, desc="质量检测"):
            # 综合质量评估
            if quality_method == 'combined':
                quality = evaluate_image_quality(img_path)
                score = quality['overall_score']
                is_blurry = quality['is_blurry']
            else:
                if quality_method == 'laplacian':
                    var, is_blurry, score = detect_blur_laplacian(img_path, blur_threshold)
                elif quality_method == 'sobel':
                    mag, is_blurry, score = detect_blur_sobel(img_path, blur_threshold)
                elif quality_method == 'fft':
                    ratio, is_blurry, score = detect_blur_fft(img_path, blur_threshold)
                else:
                    raise ValueError(f"未知方法: {quality_method}")
                
                quality = {'score': score, 'is_blurry': is_blurry}
            
            quality_data[img_path.name] = quality
            
            # 检查最低质量要求
            if score < min_quality_score or is_blurry:
                blurry_images.append(img_path)
            else:
                sharp_images.append(img_path)
        
        # 检查模糊图像比例
        blurry_ratio = len(blurry_images) / len(image_files)
        
        print(f"\n{'='*60}")
        print("质量检测完成")
        print(f"{'='*60}")
        print(f"清晰图像: {len(sharp_images)}/{len(image_files)} ({(1-blurry_ratio)*100:.1f}%)")
        print(f"模糊图像: {len(blurry_images)}/{len(image_files)} ({blurry_ratio*100:.1f}%)")
        
        # 如果模糊图像过多，发出警告
        if blurry_ratio > max_blurry_ratio:
            print(f"\n⚠️ 警告: 模糊图像比例({blurry_ratio*100:.1f}%)超过阈值({max_blurry_ratio*100:.1f}%)")
            print("建议检查数据源或调整检测参数")
        
        # 输出模糊图像列表
        if blurry_images:
            print(f"\n过滤掉的模糊图像:")
            for img_path in blurry_images[:10]:  # 只显示前10个
                q = quality_data[img_path.name]
                print(f"  ❌ {img_path.name}: 评分={q.get('score', 0):.3f}")
            if len(blurry_images) > 10:
                print(f"  ... 还有 {len(blurry_images)-10} 张")
        
        # 使用清晰图像继续处理
        image_files = sharp_images
        
        if len(image_files) == 0:
            print("\n❌ 错误: 没有通过质量检测的图像，请调整阈值或检查数据")
            return
        
        print(f"\n✅ 将使用 {len(image_files)} 张清晰图像进行分割")
    else:
        quality_data = {}
        print("\n⚠️ 质量过滤已禁用，将处理所有图像")

    # 限制处理数量
    if max_images is not None:
        image_files = image_files[:max_images]
        print(f"将处理前 {len(image_files)} 张图像")

    # ========== 步骤2: 加载分割模型 ==========
    print(f"\n{'='*60}")
    print("步骤2: 加载分割模型")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = NNUNetS2DSModel(allow_tqdm=False)
    model.predictor.device = device
    model.predictor.perform_everything_on_device = True
    print("✅ 模型加载完成")

    # ========== 步骤3: 分割处理 ==========
    print(f"\n{'='*60}")
    print("步骤3: 图像分割")
    print(f"{'='*60}")

    results = []
    for img_path in tqdm(image_files, desc="分割处理"):
        output_path = output_dir / f"segmentation_{img_path.stem}.png"
        
        # 获取质量信息（用于可视化）
        q_info = quality_data.get(img_path.name) if enable_quality_filter else None
        
        mask, softmax = visualize_segmentation(
            img_path, model, output_path, device, 
            quality_info=q_info
        )

        # 统计每个类别的像素数
        unique, counts = np.unique(mask, return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))

        results.append({
            'image': img_path.name,
            'mask_stats': stats,
            'output': output_path,
            'quality': q_info
        })

    # ========== 步骤4: 输出统计 ==========
    print(f"\n{'='*60}")
    print("分割结果统计")
    print(f"{'='*60}")

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
        if result['quality']:
            score = result['quality'].get('score', result['quality'].get('overall_score', 0))
            print(f"  质量评分: {score:.3f}")
        for class_id, count in sorted(result['mask_stats'].items()):
            percentage = count / sum(result['mask_stats'].values()) * 100
            print(f"  {class_names.get(class_id, f'class_{class_id}')}: {count} pixels ({percentage:.2f}%)")

    print(f"\n所有结果已保存到: {output_dir}")

    # 创建汇总图
    create_summary_figure(results, output_dir, images_dir, enable_quality_filter)

    # 生成质量报告
    if enable_quality_filter:
        _generate_quality_report(results, output_dir)

    return results


def create_summary_figure(results, output_dir, images_dir, show_quality=False):
    """创建汇总可视化图"""
    n_images = len(results)
    if n_images == 0:
        return

    # 如果显示质量信息，增加列数
    cols = 3 if show_quality else 2
    fig, axes = plt.subplots(cols, n_images, figsize=(5*n_images, 4*cols))
    
    if n_images == 1:
        axes = axes.reshape(cols, 1)

    for i, result in enumerate(results):
        # 读取原图和分割结果
        img_path = Path(images_dir) / result['image']
        seg_path = result['output']

        row = 0
        # 原图
        if img_path.exists():
            img = Image.open(img_path)
            axes[row, i].imshow(img)
            axes[row, i].set_title(f"Input: {result['image']}")
            axes[row, i].axis('off')
        row += 1

        # 分割结果
        if seg_path.exists():
            seg_img = Image.open(seg_path)
            axes[row, i].imshow(seg_img)
            axes[row, i].set_title("Segmentation")
            axes[row, i].axis('off')
        row += 1
        
        # 质量信息（可选）
        if show_quality and result['quality']:
            quality = result['quality']
            score = quality.get('score', quality.get('overall_score', 0))
            
            quality_text = f"Quality: {score:.3f}\n"
            if 'laplacian' in quality:
                quality_text += f"Lap: {quality['laplacian']['variance']:.1f}\n"
            if 'sobel' in quality:
                quality_text += f"Sobel: {quality['sobel']['magnitude']:.1f}"
            
            axes[row, i].text(0.5, 0.5, quality_text, 
                            ha='center', va='center', fontsize=10)
            axes[row, i].set_xlim(0, 1)
            axes[row, i].set_ylim(0, 1)
            axes[row, i].axis('off')

    plt.tight_layout()
    summary_path = output_dir / "multiview_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n汇总图已保存: {summary_path}")
    plt.close()


def _generate_quality_report(results, output_dir):
    """生成质量分析报告"""
    report_path = output_dir / "quality_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("图像质量分析报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"总处理图像数: {len(results)}\n")
        
        # 质量评分统计
        scores = []
        for r in results:
            if r['quality']:
                score = r['quality'].get('score', r['quality'].get('overall_score', 0))
                scores.append(score)
        
        if scores:
            f.write(f"\n质量评分统计:\n")
            f.write(f"  平均分: {np.mean(scores):.3f}\n")
            f.write(f"  最高分: {np.max(scores):.3f}\n")
            f.write(f"  最低分: {np.min(scores):.3f}\n")
            f.write(f"  中位数: {np.median(scores):.3f}\n")
        
        # 每张图像的详细信息
        f.write(f"\n{'='*60}\n")
        f.write("每张图像的详细信息\n")
        f.write(f"{'='*60}\n\n")
        
        for r in results:
            f.write(f"图像: {r['image']}\n")
            if r['quality']:
                score = r['quality'].get('score', r['quality'].get('overall_score', 0))
                f.write(f"  质量评分: {score:.3f}\n")
                
                if 'laplacian' in r['quality']:
                    f.write(f"  拉普拉斯方差: {r['quality']['laplacian']['variance']:.2f}\n")
                if 'sobel' in r['quality']:
                    f.write(f"  Sobel梯度: {r['quality']['sobel']['magnitude']:.2f}\n")
            
            # 损伤统计
            total_pixels = sum(r['mask_stats'].values())
            f.write(f"  损伤统计:\n")
            for class_id, count in sorted(r['mask_stats'].items()):
                if class_id != 0:  # 跳过背景
                    percentage = count / total_pixels * 100
                    class_names = {1: "crack", 2: "spalling", 3: "corrosion", 
                                  4: "efflorescence", 5: "vegetation", 6: "control_point"}
                    f.write(f"    {class_names.get(class_id, f'class_{class_id}')}: "
                           f"{count} px ({percentage:.3f}%)\n")
            f.write("\n")
    
    print(f"质量分析报告已保存: {report_path}")


if __name__ == "__main__":
    # 配置路径
    base_dir = os.path.join(os.path.dirname(__file__), 'enstrect_lib', 'src', 'enstrect', 'assets')
    # base_dir = r"G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets"  # 绝对路径

    # Bridge B segment_test 数据
    images_dir = os.path.join(base_dir, 'bridge_b', 'segment_test', 'views')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_filtered')

    # 检查路径
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        print("请先下载测试数据: cd enstrect_lib && python -m enstrect.datasets.download")
        sys.exit(1)

    # 运行带过滤的分割
    results = process_multiview_with_filter(
        images_dir=images_dir,
        output_dir=output_dir,
        max_images=10,                    # 处理前10张
        enable_quality_filter=True,       # 启用质量过滤
        blur_threshold=100.0,             # 模糊阈值
        min_quality_score=0.3,            # 最低质量评分
        quality_method='laplacian',       # 检测方法
        max_blurry_ratio=0.5              # 最大允许模糊比例
    )

    print("\n" + "="*60)
    print("✓ 多视图分割（带数据清洗）完成！")
    print("="*60)
