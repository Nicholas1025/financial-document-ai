"""
对比不同对齐算法在多个样本上的效果
"""
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np

from modules.utils import load_config
from modules.data_loaders import PubTabNetLoader
from modules.structure import TableStructureRecognizer
from modules.ocr import TableOCR
from modules.metrics import calculate_teds


def main():
    config = load_config('configs/config.yaml')
    loader = PubTabNetLoader(config)
    annotations = loader.load_annotations(split='val', num_samples=20)
    
    print("Loading models...")
    model = TableStructureRecognizer(config)
    ocr = TableOCR(lang='en')
    
    # Warm up
    from PIL import Image
    dummy_img = Image.new('RGB', (100, 100), 'white')
    ocr.extract_text(dummy_img)
    
    methods = ['center', 'iou', 'hybrid']
    results = {m: {'teds': [], 'filled': []} for m in methods}
    
    print(f"\n对比 {len(annotations)} 个样本上的对齐算法...\n")
    
    for ann in tqdm(annotations, desc="Processing"):
        try:
            image, annotation = loader.load_sample(ann, split='val')
            gt_html = loader.get_html_structure(annotation)
            
            structure = model.recognize(image)
            ocr_results = ocr.extract_text(image)
            
            for method in methods:
                grid_texts = ocr.align_text_to_grid(
                    ocr_results,
                    structure['rows'],
                    structure['columns'],
                    method=method
                )
                
                pred_html = model.structure_to_html(structure, cell_texts=grid_texts)
                teds_score = calculate_teds(pred_html, gt_html, structure_only=False)
                
                filled_cells = sum(1 for row in grid_texts for cell in row if cell.strip())
                
                results[method]['teds'].append(teds_score)
                results[method]['filled'].append(filled_cells)
                
        except Exception as e:
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("对齐算法对比结果")
    print("=" * 60)
    
    print(f"\n{'方法':<10} {'平均TEDS':<12} {'标准差':<10} {'平均填充单元格':<15}")
    print("-" * 50)
    
    for method in methods:
        teds_scores = results[method]['teds']
        filled_counts = results[method]['filled']
        
        mean_teds = np.mean(teds_scores)
        std_teds = np.std(teds_scores)
        mean_filled = np.mean(filled_counts)
        
        print(f"{method:<10} {mean_teds:.4f}       {std_teds:.4f}     {mean_filled:.1f}")
    
    # Improvement
    center_mean = np.mean(results['center']['teds'])
    hybrid_mean = np.mean(results['hybrid']['teds'])
    improvement = hybrid_mean - center_mean
    
    print(f"\nHYBRID 相比 CENTER 提升: {improvement:.4f} ({improvement*100:.2f}%)")
    
    # Per-sample comparison
    print("\n样本级别对比:")
    improvements = []
    for i in range(len(results['center']['teds'])):
        diff = results['hybrid']['teds'][i] - results['center']['teds'][i]
        improvements.append(diff)
        if diff > 0.05:
            print(f"  样本 {i+1}: CENTER={results['center']['teds'][i]:.3f} → HYBRID={results['hybrid']['teds'][i]:.3f} (+{diff:.3f})")
    
    print(f"\n有提升的样本: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
    print(f"平均提升: {np.mean(improvements):.4f}")


if __name__ == '__main__':
    main()
