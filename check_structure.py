"""
分析结构检测的准确性
"""
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

from modules.utils import load_config
from modules.data_loaders import PubTabNetLoader
from modules.structure import TableStructureRecognizer

config = load_config('configs/config.yaml')
loader = PubTabNetLoader(config)
annotations = loader.load_annotations(split='val', num_samples=10)

model = TableStructureRecognizer(config)

print("结构检测准确性分析")
print("=" * 60)

for ann in annotations:
    image, annotation = loader.load_sample(ann, split='val')
    gt_html = loader.get_html_structure(annotation)
    
    # Count GT rows
    gt_rows = gt_html.count('<tr>')
    
    structure = model.recognize(image)
    pred_rows = len(structure['rows'])
    pred_cols = len(structure['columns'])
    
    filename = annotation['filename']
    row_diff = abs(gt_rows - pred_rows)
    status = "✓" if row_diff <= 1 else "✗"
    
    print(f"{status} {filename}: GT={gt_rows}行, Pred={pred_rows}行 {pred_cols}列")
