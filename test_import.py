import sys
sys.path.insert(0, r'G:\Zed\ENSTRECTtest\enstrect\src')

print('Testing ENSTRECT import...')
import enstrect
print('ENSTRECT imported successfully!')

print('\nTesting core modules...')
from enstrect.segmentation.base import SegmenterInterface
print('SegmenterInterface imported successfully!')

from enstrect.mapping.fuser import Fuser
print('Fuser imported successfully!')

from enstrect.extraction.boundary import extract_bounding_polygons
print('extract_bounding_polygons imported successfully!')

print('\nAll core imports successful!')
