"""
图像质量过滤模块
用于检测和过滤运动模糊、失焦等低质量图像
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def detect_blur_laplacian(image_path, threshold=100.0):
    """
    使用拉普拉斯算子方差检测图像模糊程度
    
    原理: 模糊图像的拉普拉斯响应较低(边缘不明显)
    清晰图像有较多高频成分，拉普拉斯方差较大
    
    Args:
        image_path: 图像路径
        threshold: 模糊阈值，低于此值认为是模糊图像
                  默认100，可根据实际情况调整
                  
    Returns:
        variance: 拉普拉斯方差值
        is_blurry: 是否为模糊图像
        score: 清晰度评分(0-1)
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, True, 0.0
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算拉普拉斯算子
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 计算方差(标准差的平方)
    variance = laplacian.var()
    
    # 判断是否模糊
    is_blurry = variance < threshold
    
    # 归一化评分(假设阈值对应0.5分)
    score = min(variance / (threshold * 2), 1.0)
    
    return variance, is_blurry, score


def detect_blur_sobel(image_path, threshold=500.0):
    """
    使用Sobel算子检测边缘强度
    
    Args:
        image_path: 图像路径
        threshold: 边缘强度阈值
        
    Returns:
        magnitude: 梯度幅值
        is_blurry: 是否为模糊图像
        score: 清晰度评分
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, True, 0.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算Sobel梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
    
    is_blurry = magnitude < threshold
    score = min(magnitude / (threshold * 2), 1.0)
    
    return magnitude, is_blurry, score


def detect_blur_fft(image_path, threshold=0.1, radius=30):
    """
    使用FFT频域分析检测模糊
    
    原理: 清晰图像的高频成分较多
    
    Args:
        image_path: 图像路径
        threshold: 高频能量比例阈值
        radius: 中心低频区域半径
        
    Returns:
        ratio: 高频能量比例
        is_blurry: 是否为模糊图像
        score: 清晰度评分
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, True, 0.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # FFT变换
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # 创建圆形掩码
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # 计算高频能量比例
    low_freq_energy = magnitude[mask].sum()
    total_energy = magnitude.sum()
    
    if total_energy == 0:
        return 0, True, 0.0
    
    ratio = 1 - (low_freq_energy / total_energy)
    is_blurry = ratio < threshold
    score = min(ratio / (threshold * 2), 1.0)
    
    return ratio, is_blurry, score


def evaluate_image_quality(image_path, methods=None, weights=None):
    """
    综合评估图像质量
    
    Args:
        image_path: 图像路径
        methods: 使用的方法列表，默认['laplacian', 'sobel']
        weights: 各方法的权重，默认[0.6, 0.4]
        
    Returns:
        dict: 包含各种指标的字典
    """
    if methods is None:
        methods = ['laplacian', 'sobel']
    if weights is None:
        weights = [0.6, 0.4]
    
    results = {}
    scores = []
    
    # 拉普拉斯方法
    if 'laplacian' in methods:
        var, is_blur, score = detect_blur_laplacian(image_path)
        results['laplacian'] = {
            'variance': var,
            'is_blurry': is_blur,
            'score': score
        }
        scores.append(score)
    
    # Sobel方法
    if 'sobel' in methods:
        mag, is_blur, score = detect_blur_sobel(image_path)
        results['sobel'] = {
            'magnitude': mag,
            'is_blurry': is_blur,
            'score': score
        }
        scores.append(score)
    
    # FFT方法
    if 'fft' in methods:
        ratio, is_blur, score = detect_blur_fft(image_path)
        results['fft'] = {
            'high_freq_ratio': ratio,
            'is_blurry': is_blur,
            'score': score
        }
        scores.append(score)
    
    # 计算加权综合评分
    if scores:
        results['overall_score'] = np.average(scores, weights=weights[:len(scores)])
        results['is_blurry'] = results['overall_score'] < 0.5
    
    return results


def filter_blurry_images(images_dir, output_dir=None, 
                         blur_threshold=100.0, 
                         method='laplacian',
                         visualize=False):
    """
    批量过滤模糊图像
    
    Args:
        images_dir: 图像目录
        output_dir: 输出目录(可选)，用于保存可视化结果
        blur_threshold: 模糊阈值
        method: 检测方法 'laplacian'|'sobel'|'fft'|'combined'
        visualize: 是否生成可视化对比图
        
    Returns:
        dict: 包含清晰图像列表和模糊图像列表
    """
    images_dir = Path(images_dir)
    image_files = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"在 {images_dir} 中未找到图像")
        return {'sharp': [], 'blurry': []}
    
    print(f"\n开始质量检测，共 {len(image_files)} 张图像...")
    print(f"检测方法: {method}, 阈值: {blur_threshold}")
    
    sharp_images = []
    blurry_images = []
    quality_scores = []
    
    # 创建输出目录
    if output_dir:
        output_dir = Path(output_dir)
        sharp_dir = output_dir / 'sharp'
        blurry_dir = output_dir / 'blurry'
        sharp_dir.mkdir(parents=True, exist_ok=True)
        blurry_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_files, desc="检测图像质量"):
        # 检测模糊
        if method == 'combined':
            quality = evaluate_image_quality(img_path)
            score = quality['overall_score']
            is_blurry = quality['is_blurry']
            metric_value = score
        elif method == 'laplacian':
            var, is_blurry, score = detect_blur_laplacian(img_path, blur_threshold)
            metric_value = var
        elif method == 'sobel':
            mag, is_blurry, score = detect_blur_sobel(img_path, blur_threshold)
            metric_value = mag
        elif method == 'fft':
            ratio, is_blurry, score = detect_blur_fft(img_path, blur_threshold)
            metric_value = ratio
        else:
            raise ValueError(f"未知方法: {method}")
        
        quality_scores.append({
            'image': img_path.name,
            'score': score,
            'metric': metric_value,
            'is_blurry': is_blurry
        })
        
        if is_blurry:
            blurry_images.append(img_path)
            if output_dir:
                import shutil
                shutil.copy2(img_path, blurry_dir / img_path.name)
        else:
            sharp_images.append(img_path)
            if output_dir:
                import shutil
                shutil.copy2(img_path, sharp_dir / img_path.name)
    
    # 打印统计
    print(f"\n{'='*60}")
    print("图像质量检测完成")
    print(f"{'='*60}")
    print(f"总图像数: {len(image_files)}")
    print(f"清晰图像: {len(sharp_images)} ({len(sharp_images)/len(image_files)*100:.1f}%)")
    print(f"模糊图像: {len(blurry_images)} ({len(blurry_images)/len(image_files)*100:.1f}%)")
    
    if blurry_images:
        print(f"\n模糊图像列表:")
        for item in quality_scores:
            if item['is_blurry']:
                print(f"  ❌ {item['image']}: 评分={item['score']:.3f}, 指标={item['metric']:.2f}")
    
    print(f"\n清晰图像列表(前10):")
    sharp_items = [item for item in quality_scores if not item['is_blurry']]
    for item in sharp_items[:10]:
        print(f"  ✅ {item['image']}: 评分={item['score']:.3f}")
    
    # 生成可视化
    if visualize and len(quality_scores) > 0:
        _visualize_quality_scores(quality_scores, output_dir)
    
    if output_dir:
        print(f"\n分类结果已保存到:")
        print(f"  清晰图像: {sharp_dir}")
        print(f"  模糊图像: {blurry_dir}")
    
    return {
        'sharp': sharp_images,
        'blurry': blurry_images,
        'scores': quality_scores
    }


def _visualize_quality_scores(scores, output_dir):
    """可视化质量评分分布"""
    if output_dir is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 评分分布直方图
    all_scores = [s['score'] for s in scores]
    axes[0].hist(all_scores, bins=20, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0.5, color='red', linestyle='--', label='阈值=0.5')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Image Quality Distribution')
    axes[0].legend()
    
    # 评分排序图
    sorted_scores = sorted(scores, key=lambda x: x['score'])
    names = [s['image'][:15] for s in sorted_scores]
    values = [s['score'] for s in sorted_scores]
    colors = ['red' if s['is_blurry'] else 'green' for s in sorted_scores]
    
    axes[1].barh(range(len(names)), values, color=colors, alpha=0.7)
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel('Quality Score')
    axes[1].set_title('Image Quality Ranking')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n质量分析图已保存: {output_dir / 'quality_analysis.png'}")
    plt.close()


if __name__ == "__main__":
    # 测试示例
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python image_quality_filter.py <图像目录> [输出目录]")
        print("示例: python image_quality_filter.py ./images ./output")
        sys.exit(1)
    
    images_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 运行质量检测
    results = filter_blurry_images(
        images_dir=images_dir,
        output_dir=output_dir,
        blur_threshold=100.0,
        method='laplacian',
        visualize=True
    )
    
    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"{'='*60}")
