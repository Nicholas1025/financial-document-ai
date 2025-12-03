"""
TEDS 分析脚本 - 对比不同对齐算法的效果
"""
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

from modules.utils import load_config
from modules.data_loaders import PubTabNetLoader
from modules.structure import TableStructureRecognizer
from modules.ocr import TableOCR
from modules.metrics import calculate_teds

def main():
    config = load_config('configs/config.yaml')
    loader = PubTabNetLoader(config)
    annotations = loader.load_annotations(split='val', num_samples=1)
    
    # Load first sample
    image, ann = loader.load_sample(annotations[0], split='val')
    print('=' * 60)
    print(f'样本: {ann["filename"]}')
    print('=' * 60)
    
    # 1. Ground Truth HTML
    gt_html = loader.get_html_structure(ann)
    print('\n【1. Ground Truth HTML (真实标注)】')
    print(gt_html[:800] + ('...' if len(gt_html) > 800 else ''))
    
    # 2. Run Structure Recognition
    model = TableStructureRecognizer(config)
    structure = model.recognize(image)
    print(f'\n【2. 检测到的结构】')
    print(f'行数: {len(structure["rows"])}')
    print(f'列数: {len(structure["columns"])}')
    
    # Print row bboxes
    print('\n行边界框:')
    for i, row in enumerate(structure['rows']):
        print(f'  Row {i}: {row["bbox"]}')
    
    print('\n列边界框:')
    for j, col in enumerate(structure['columns']):
        print(f'  Col {j}: {col["bbox"]}')
    
    # 3. No OCR - empty cells
    pred_html_no_ocr = model.structure_to_html(structure, cell_texts=None)
    
    # 4. Run OCR
    ocr = TableOCR(lang='en')
    ocr_results = ocr.extract_text(image)
    print(f'\n【3. OCR 提取结果】')
    print(f'提取了 {len(ocr_results)} 个文本区域:')
    for i, r in enumerate(ocr_results):
        print(f'  {i+1}. "{r["text"]}" bbox={[int(x) for x in r["bbox"]]} (置信度: {r["confidence"]:.3f})')
    
    # 5. Compare alignment methods
    print('\n【4. 对齐算法对比】')
    
    methods = ['center', 'iou', 'hybrid']
    results = {}
    
    for method in methods:
        grid_texts = ocr.align_text_to_grid(
            ocr_results, 
            structure['rows'], 
            structure['columns'],
            method=method
        )
        
        # Count filled cells
        filled_cells = sum(1 for row in grid_texts for cell in row if cell.strip())
        
        # Generate HTML
        pred_html = model.structure_to_html(structure, cell_texts=grid_texts)
        
        # Calculate TEDS
        teds_score = calculate_teds(pred_html, gt_html, structure_only=False)
        
        results[method] = {
            'grid': grid_texts,
            'filled_cells': filled_cells,
            'teds': teds_score
        }
        
        print(f'\n  方法: {method.upper()}')
        print(f'  填充单元格数: {filled_cells}')
        print(f'  TEDS: {teds_score:.4f} ({teds_score*100:.2f}%)')
        print(f'  网格内容:')
        for i, row in enumerate(grid_texts):
            print(f'    Row {i}: {row}')
    
    # 6. Summary
    print('\n' + '=' * 60)
    print('【5. 总结】')
    print('=' * 60)
    
    # Structure TEDS
    teds_structure = calculate_teds(pred_html_no_ocr, gt_html, structure_only=True)
    print(f'\n结构 TEDS (无内容): {teds_structure:.4f} ({teds_structure*100:.2f}%)')
    
    print(f'\n各对齐方法的内容 TEDS:')
    for method in methods:
        teds = results[method]['teds']
        filled = results[method]['filled_cells']
        print(f'  {method:8s}: TEDS = {teds:.4f} ({teds*100:.2f}%), 填充单元格 = {filled}')
    
    # Best method
    best_method = max(methods, key=lambda m: results[m]['teds'])
    improvement = results[best_method]['teds'] - results['center']['teds']
    print(f'\n最佳方法: {best_method.upper()}')
    print(f'相比 CENTER 方法提升: {improvement:.4f} ({improvement*100:.2f}%)')

if __name__ == '__main__':
    main()
