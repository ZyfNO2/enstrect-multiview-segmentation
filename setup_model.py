import shutil
import zipfile
from pathlib import Path

# 源文件
src_zip = Path(r'C:\Users\ZYF\Downloads\nnUNetTrainer__nnUNetPlans__2d.zip')
# 目标目录
dst_dir = Path(r'G:\Zed\ENSTRECTtest\enstrect\src\enstrect\segmentation\checkpoints')

print(f'Copying model from {src_zip} to {dst_dir}...')
shutil.copy(src_zip, dst_dir)
print('Copy successful!')

# 解压
zip_path = dst_dir / 'nnUNetTrainer__nnUNetPlans__2d.zip'
extract_dir = dst_dir / 'nnUNetTrainer__nnUNetPlans__2d'

print(f'Extracting {zip_path}...')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f'Extracted to {extract_dir}')

# 列出解压后的内容
print('\nExtracted files:')
for item in extract_dir.rglob('*'):
    print(f'  {item.relative_to(extract_dir)}')

print('\nModel setup complete!')
