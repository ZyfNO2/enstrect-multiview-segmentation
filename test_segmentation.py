import sys
sys.path.insert(0, r'G:\Zed\ENSTRECTtest\enstrect\src')

from PIL import Image
import numpy as np

print('Loading example image...')
img_path = r'G:\Zed\ENSTRECTtest\enstrect\src\enstrect\assets\example_image.jpg'
img = Image.open(img_path)
print(f'Image loaded: {img.size}')

print('\nTesting segmentation model...')
from enstrect.segmentation.nnunet_s2ds import NnUNetS2DS

print('Creating model...')
model = NnUNetS2DS()
print('Model created successfully!')

print('\nRunning inference...')
# Convert PIL to numpy
img_array = np.array(img)
print(f'Input shape: {img_array.shape}')

# Run inference
import torch
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(f'Tensor shape: {img_tensor.shape}')

print('\nSegmentation test completed!')
