#!/usr/bin/env python3
"""批量测试CRACK500数据集 - 仅2D分割版本"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "enstrect" / "src"))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 80)
    print("开始批量测试 CRACK500 数据集 - 2D分割")
    print("=" * 80)
    
    # 导入必要的模块
    print("\n加载模型...")
    from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = NNUNetS2DSModel()
    
    print(f"✓ 模型已加载")
    
    # 定义路径
    images_dir = Path(r"g:\Zed\testData\pavement crack datasets\CRACK500\testdata")
    output_dir = Path(r"g:\Zed\testData\pavement crack datasets\CRACK500\output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 颜色映射 (禁用类别 4,5 -> 映射到background)
    color_map = {
        0: [0, 0, 0],      # background
        1: [255, 0, 0],    # crack
        2: [0, 255, 0],    # spalling
        3: [0, 0, 255],    # corrosion
        6: [255, 255, 0],  # control_point
    }
    
    label_names = {
        0: "background",
        1: "crack",
        2: "spalling", 
        3: "corrosion",
        6: "control_point"
    }
    
    # 获取所有图像(只取.jpg文件，不取mask)
    image_files = sorted([f for f in images_dir.glob("*.jpg") if "_mask" not in f.name])
    print(f"\n找到 {len(image_files)} 张图像")
    
    # 统计信息
    stats = defaultdict(lambda: {"count": 0, "total_pixels": 0})
    all_results = []
    
    # 处理每张图像
    seg_dir = output_dir / "per_frame_segmentation"
    seg_dir.mkdir(exist_ok=True)
    
    print("\n开始分割...")
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_files, desc="处理图像")):
            try:
                # 加载图像
                image = Image.open(img_path).convert("RGB")
                
                # 如果图像太大，进行缩放以节省内存
                max_size = 1280
                w, h = image.size
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = image.resize((new_w, new_h), Image.BILINEAR)
                    print(f"\n  缩放图像 {img_path.name}: {w}x{h} -> {new_w}x{new_h}")
                
                img_array = np.array(image)
                
                # 转换为PyTorch tensor
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(torch.float32)
                
                # 推理
                softmax, argmax = model(img_tensor)
                
                # 获取结果
                mask = argmax.cpu().numpy()
                prob = softmax.cpu().numpy()
                
                # 统计
                result = {
                    "filename": img_path.name,
                    "image_size": img_array.shape[:2],
                    "labels": {}
                }
                
                # 只统计有效类别 (0,1,2,3,6)，跳过禁用的 4,5
                for label_id in [0, 1, 2, 3, 6]:
                    pixel_count = np.sum(mask == label_id)
                    if pixel_count > 0:
                        stats[label_id]["count"] += 1
                        stats[label_id]["total_pixels"] += int(pixel_count)
                        result["labels"][label_id] = {
                            "pixels": int(pixel_count),
                            "percentage": float(pixel_count / mask.size * 100)
                        }
                
                all_results.append(result)
                
                # 保存可视化结果（每10张保存一次，节省空间）
                if i % 10 == 0 or i < 10:
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    for label_id, color in color_map.items():
                        colored_mask[mask == label_id] = color
                    
                    # 创建叠加图
                    overlay = (img_array * 0.6 + colored_mask * 0.4).astype(np.uint8)
                    
                    # 保存图像
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_array)
                    axes[0].set_title(f'Original: {img_path.name}')
                    axes[0].axis('off')
                    
                    axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=6)
                    axes[1].set_title('Segmentation Mask')
                    axes[1].axis('off')
                    
                    axes[2].imshow(overlay)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(seg_dir / f"seg_{img_path.stem}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                
            except Exception as e:
                print(f"\n处理 {img_path.name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 生成统计报告
    print("\n" + "=" * 80)
    print("生成统计报告...")
    print("=" * 80)
    
    # 创建统计摘要 (只统计有效类别 1,2,3,6)
    summary_data = []
    for label_id in [1, 2, 3, 6]:  # 跳过background和禁用类别
        if stats[label_id]["count"] > 0:
            summary_data.append({
                "Label": label_names[label_id],
                "Images Detected": stats[label_id]["count"],
                "Total Pixels": stats[label_id]["total_pixels"],
                "Avg Pixels/Image": int(stats[label_id]["total_pixels"] / stats[label_id]["count"])
            })
    
    # 打印统计
    print(f"\n处理完成: {len(all_results)}/{len(image_files)} 张图像")
    print("\n损伤统计:")
    for item in summary_data:
        print(f"  {item['Label']}: 在 {item['Images Detected']} 张图像中检测到")
        print(f"    - 总像素数: {item['Total Pixels']:,}")
        print(f"    - 平均每张图像: {item['Avg Pixels/Image']:,} 像素")
    
    # 保存详细报告
    report_path = output_dir / "detailed_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary_data,
            "detailed_results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    # 生成可视化汇总
    if len(list(seg_dir.glob("*.png"))) > 0:
        create_summary_grid(seg_dir, output_dir)
    
    print(f"\n✓ 详细报告已保存: {report_path}")
    print(f"✓ 分割结果已保存到: {seg_dir}")
    print("=" * 80)


def create_summary_grid(seg_dir, output_dir):
    """创建汇总网格图"""
    seg_files = sorted(seg_dir.glob("seg_*.png"))
    if len(seg_files) == 0:
        return
    
    n_images = min(9, len(seg_files))  # 最多显示9张
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_images):
        img = Image.open(seg_files[i])
        axes[i].imshow(img)
        axes[i].set_title(seg_files[i].name.replace("seg_", "").replace(".png", ""), fontsize=10)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    summary_path = output_dir / "segmentation_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"  汇总图: {summary_path}")
    plt.close()


if __name__ == "__main__":
    main()
