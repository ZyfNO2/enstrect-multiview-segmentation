"""
ROI区域裁剪模块
从nnU-Net分割掩码中提取各损伤类别的感兴趣区域(ROI)，裁剪原图对应部分
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2


class ROICropper:
    """ROI裁剪器：将语义分割掩码转换为各类别损伤的裁剪图像"""

    CLASS_NAMES = {
        0: "background",
        1: "crack",
        2: "spalling",
        3: "corrosion",
        4: "efflorescence",
        5: "vegetation",
        6: "control_point"
    }

    def __init__(self, padding: int = 20, min_size: int = 10):
        self.padding = padding
        self.min_size = min_size

    def crop_roi(self, image: Image.Image, mask: np.ndarray, class_id: int,
                 padding: Optional[int] = None) -> Optional[Image.Image]:
        """
        裁剪指定类别的单个ROI区域
        
        Args:
            image: 原始PIL图像(RGB)
            mask: 分割掩码(H,W), 像素值为类别ID
            class_id: 要裁剪的目标类别ID
            padding: 外接矩形扩展像素数
            
        Returns:
            裁剪后的PIL图像，若无该类别则返回None
        """
        pad = padding or self.padding
        binary_mask = (mask == class_id).astype(np.uint8)
        if binary_mask.sum() < self.min_size:
            return None
        coords = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = coords[0] if len(coords) > 1 else []
        if not contours:
            return None
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        img_array = np.array(image)
        H, W = img_array.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        roi = img_array[y1:y2, x1:x2]
        return Image.fromarray(roi)

    def crop_all_rois(self, image: Image.Image, mask: np.ndarray,
                       padding: Optional[int] = None) -> Dict[int, Image.Image]:
        """
        裁剪所有非背景类别的ROI
        
        Returns:
            {class_id: PIL_Image} 字典，仅包含有内容的类别
        """
        rois = {}
        unique_classes = np.unique(mask)
        for cls_id in unique_classes:
            if cls_id == 0:
                continue
            roi = self.crop_roi(image, mask, cls_id, padding=padding)
            if roi is not None:
                rois[cls_id] = roi
        return rois

    def crop_largest_per_class(self, image: Image.Image, mask: np.ndarray,
                                padding: Optional[int] = None) -> Dict[int, Image.Image]:
        """每个类别只保留面积最大的ROI区域"""
        pad = padding or self.padding
        rois = {}
        unique_classes = np.unique(mask)
        for cls_id in unique_classes:
            if cls_id == 0:
                continue
            binary_mask = (mask == cls_id).astype(np.uint8)
            if binary_mask.sum() < self.min_size:
                continue
            contours_info = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) > 1 else []
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            img_array = np.array(image)
            H, W = img_array.shape[:2]
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
            rois[cls_id] = Image.fromarray(img_array[y1:y2, x1:x2])
        return rois

    def save_rois(self, rois: Dict[int, Image.Image], output_dir: str,
                  prefix: str = "roi") -> Dict[int, str]:
        """保存ROI图像到磁盘，返回 {class_id: 保存路径} 映射"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        for cls_id, roi_img in rois.items():
            filename = f"{prefix}_{self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')}.png"
            filepath = output_path / filename
            roi_img.save(filepath)
            saved_paths[cls_id] = str(filepath)
        return saved_paths

    def compute_geometry_stats(self, mask: np.ndarray, class_id: int,
                                pixel_to_mm: float = 0.1) -> Dict:
        """
        计算某类别损伤区域的几何统计信息
        
        Args:
            mask: 分割掩码
            class_id: 目标类别
            pixel_to_mm: 像素到毫米的换算系数
            
        Returns:
            包含 area_px, area_mm2, pixel_count, bbox 等信息的字典
        """
        binary_mask = (mask == class_id).astype(np.uint8)
        pixel_count = int(binary_mask.sum())
        if pixel_count < self.min_size:
            return {"pixel_count": 0, "area_px": 0, "area_mm2": 0}
        area_px = pixel_count
        area_mm2 = area_px * (pixel_to_mm ** 2)
        coords = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = coords[0] if len(coords) > 1 else []
        if contours:
            all_pts = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            bbox = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        else:
            bbox = {}
        return {
            "class_id": class_id,
            "class_name": self.CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "pixel_count": pixel_count,
            "area_px": area_px,
            "area_mm2": round(area_mm2, 2),
            "bbox": bbox,
            "pixel_to_mm": pixel_to_mm
        }
