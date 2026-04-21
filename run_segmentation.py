import sys
sys.path.insert(0, r'G:\Zed\ENSTRECTtest\enstrect\src')

import torch
from pathlib import Path
from torchvision.io import read_image, ImageReadMode
from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("Loading example image...")
img_path = Path(r'G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets\example_image.jpg')
img_rgb_pyt = read_image(str(img_path), mode=ImageReadMode.RGB).to(torch.float32)

print("Loading segmentation model...")
segmenter = NNUNetS2DSModel()

print("Running inference...")
probs, argmax = segmenter(img_rgb_pyt)

print("Saving results...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原图
axes[0].imshow(img_rgb_pyt.moveaxis(0, -1).to(torch.uint8))
axes[0].set_title("Input Image")
axes[0].axis('off')

# 裂缝概率图
crack_prob = probs[1].cpu().numpy()
axes[1].imshow(crack_prob, cmap='hot')
axes[1].set_title("Crack Probability")
axes[1].axis('off')

# 分割结果
axes[2].imshow(argmax.cpu().numpy(), cmap='tab10')
axes[2].set_title("Segmentation Result")
axes[2].axis('off')

plt.tight_layout()
output_path = r'G:\Zed\ENSTRECTtest\segmentation_result.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")

# 统计各类别像素数
print("\nSegmentation statistics:")
classes = ["background", "crack", "spalling", "corrosion", "efflorescence", "vegetation", "control_point"]
unique, counts = np.unique(argmax.cpu().numpy(), return_counts=True)
for u, c in zip(unique, counts):
    pct = c / c.sum() * 100
    print(f"  {classes[u]}: {c} pixels ({pct:.2f}%)")

print("\nDone!")
