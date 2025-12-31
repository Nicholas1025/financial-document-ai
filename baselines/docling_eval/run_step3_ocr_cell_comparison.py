# -*- coding: utf-8 -*-
"""
Step 3 OCR Comparison: Cell-by-Cell Evaluation
==============================================
使用 IoU 匹配方式比较 EasyOCR 和 PaddleOCR
每个 GT word/cell 单独匹配对应的 OCR bbox，更精确的评估

输出：
- step3_cell_comparison_report.md - 对比报告
- step3_cell_comparison_histogram.png - CER 分布对比图
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
FINTABNET_PATH = Path('D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure')


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bboxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def calculate_cer(gt_text: str, pred_text: str) -> float:
    """Calculate Character Error Rate"""
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    return levenshtein_distance(gt_text, pred_text) / len(gt_text)


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return ' '.join(text.lower().split())


def match_ocr_to_gt(gt_words: List[Dict], ocr_results: List[Dict], 
                    iou_threshold: float = 0.3) -> List[Tuple[Dict, Optional[Dict], float]]:
    """Match OCR results to GT words based on bbox IoU"""
    matches = []
    used_ocr = set()
    
    for gt_word in gt_words:
        gt_bbox = gt_word['bbox']  # [x1, y1, x2, y2]
        best_match = None
        best_iou = 0.0
        best_idx = -1
        
        for idx, ocr_result in enumerate(ocr_results):
            if idx in used_ocr:
                continue
            ocr_bbox = ocr_result['bbox']
            iou = calculate_iou(gt_bbox, ocr_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = ocr_result
                best_idx = idx
        
        if best_iou >= iou_threshold and best_idx >= 0:
            used_ocr.add(best_idx)
            matches.append((gt_word, best_match, best_iou))
        else:
            matches.append((gt_word, None, 0.0))
    
    return matches


def run_easyocr_on_image(reader, img_path: str) -> List[Dict]:
    """Run EasyOCR and return results with bbox"""
    results = reader.readtext(img_path)
    ocr_words = []
    for bbox, text, conf in results:
        # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        ocr_words.append({
            'text': text,
            'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
            'confidence': conf
        })
    return ocr_words


def run_paddleocr_on_image(ocr, img_path: str) -> List[Dict]:
    """Run PaddleOCR and return results with bbox"""
    result = ocr.ocr(img_path, cls=True)
    ocr_words = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                bbox_points = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text_info = line[1]
                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                conf = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0
                
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                ocr_words.append({
                    'text': text,
                    'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                    'confidence': conf
                })
    return ocr_words


def evaluate_single_image(gt_words: List[Dict], ocr_words: List[Dict]) -> Dict:
    """Evaluate OCR on single image using cell-by-cell matching"""
    matches = match_ocr_to_gt(gt_words, ocr_words, iou_threshold=0.3)
    
    total_cells = len(gt_words)
    matched_cells = 0
    exact_matches = 0
    cer_sum = 0.0
    
    cell_results = []
    for gt_word, ocr_match, iou in matches:
        gt_text = normalize_text(gt_word['text'])
        
        if ocr_match is None:
            pred_text = ""
            cer = 1.0
        else:
            pred_text = normalize_text(ocr_match['text'])
            cer = calculate_cer(gt_text, pred_text)
            matched_cells += 1
            if gt_text == pred_text:
                exact_matches += 1
        
        cer_sum += cer
        cell_results.append({
            'gt': gt_text,
            'pred': pred_text,
            'cer': cer,
            'matched': ocr_match is not None,
            'iou': iou
        })
    
    return {
        'total_cells': total_cells,
        'matched_cells': matched_cells,
        'exact_matches': exact_matches,
        'avg_cer': cer_sum / total_cells if total_cells > 0 else 0.0,
        'exact_match_rate': exact_matches / total_cells if total_cells > 0 else 0.0,
        'cell_results': cell_results
    }


def main():
    parser = argparse.ArgumentParser(description='OCR Cell-by-Cell Comparison')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='./outputs/thesis_figures/step3_ocr_cell', help='Output directory')
    parser.add_argument('--only-easyocr', action='store_true', help='Only run EasyOCR')
    parser.add_argument('--only-paddleocr', action='store_true', help='Only run PaddleOCR')
    parser.add_argument('--generate-report', action='store_true', help='Generate report from cached results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache files
    easyocr_cache = output_dir / 'easyocr_cell_results.json'
    paddleocr_cache = output_dir / 'paddleocr_cell_results.json'
    
    print("\n" + "="*60)
    print("STEP 3: OCR CELL-BY-CELL COMPARISON")
    print("="*60)
    print(f"Dataset: FinTabNet_c")
    print(f"Samples: {args.num_samples}")
    print(f"Matching: IoU-based (threshold=0.3)")
    
    # Get files
    images_dir = FINTABNET_PATH / 'images'
    words_dir = FINTABNET_PATH / 'words'
    word_files = sorted(list(words_dir.glob('*.json')))[:args.num_samples]
    print(f"Found {len(word_files)} samples")
    
    easyocr_results = []
    paddleocr_results = []
    
    # Run EasyOCR
    if args.only_easyocr or (not args.only_paddleocr and not args.generate_report):
        if easyocr_cache.exists() and not args.only_easyocr:
            print(f"\nLoading EasyOCR cache: {easyocr_cache}")
            with open(easyocr_cache, 'r', encoding='utf-8') as f:
                easyocr_results = json.load(f)
        else:
            print("\n[EasyOCR] Loading...")
            import easyocr
            easyocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            
            print("[EasyOCR] Running evaluation...")
            for i, word_file in enumerate(word_files):
                base_name = word_file.stem.replace('_words', '')
                img_path = images_dir / f'{base_name}.jpg'
                
                if not img_path.exists():
                    continue
                
                with open(word_file, 'r', encoding='utf-8') as f:
                    gt_words = json.load(f)
                
                easyocr_words = run_easyocr_on_image(easyocr_reader, str(img_path))
                easyocr_eval = evaluate_single_image(gt_words, easyocr_words)
                easyocr_eval['sample_id'] = base_name
                easyocr_results.append(easyocr_eval)
                
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(word_files)}")
            
            # Save cache
            with open(easyocr_cache, 'w', encoding='utf-8') as f:
                json.dump(easyocr_results, f, indent=2)
            print(f"Saved cache: {easyocr_cache}")
    
    # Run PaddleOCR
    if args.only_paddleocr or (not args.only_easyocr and not args.generate_report):
        if paddleocr_cache.exists() and not args.only_paddleocr:
            print(f"\nLoading PaddleOCR cache: {paddleocr_cache}")
            with open(paddleocr_cache, 'r', encoding='utf-8') as f:
                paddleocr_results = json.load(f)
        else:
            print("\n[PaddleOCR] Loading (CPU mode)...")
            from paddleocr import PaddleOCR
            paddleocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
            
            print("[PaddleOCR] Running evaluation...")
            for i, word_file in enumerate(word_files):
                base_name = word_file.stem.replace('_words', '')
                img_path = images_dir / f'{base_name}.jpg'
                
                if not img_path.exists():
                    continue
                
                with open(word_file, 'r', encoding='utf-8') as f:
                    gt_words = json.load(f)
                
                paddleocr_words = run_paddleocr_on_image(paddleocr_engine, str(img_path))
                paddleocr_eval = evaluate_single_image(gt_words, paddleocr_words)
                paddleocr_eval['sample_id'] = base_name
                paddleocr_results.append(paddleocr_eval)
                
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(word_files)}")
            
            # Save cache
            with open(paddleocr_cache, 'w', encoding='utf-8') as f:
                json.dump(paddleocr_results, f, indent=2)
            print(f"Saved cache: {paddleocr_cache}")
    
    # Load from cache for report generation
    if args.generate_report:
        if easyocr_cache.exists():
            with open(easyocr_cache, 'r', encoding='utf-8') as f:
                easyocr_results = json.load(f)
            print(f"Loaded EasyOCR cache: {len(easyocr_results)} results")
        if paddleocr_cache.exists():
            with open(paddleocr_cache, 'r', encoding='utf-8') as f:
                paddleocr_results = json.load(f)
            print(f"Loaded PaddleOCR cache: {len(paddleocr_results)} results")
    
    # Check if we have both results
    if not easyocr_results or not paddleocr_results:
        if args.only_easyocr:
            print("\nEasyOCR evaluation complete. Run with --only-paddleocr next.")
        elif args.only_paddleocr:
            print("\nPaddleOCR evaluation complete. Run with --generate-report to create comparison.")
        else:
            print("\nMissing results. Run --only-easyocr then --only-paddleocr first.")
        return
    
    # Calculate overall metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    def calc_metrics(results):
        total_cells = sum(r['total_cells'] for r in results)
        total_matched = sum(r['matched_cells'] for r in results)
        total_exact = sum(r['exact_matches'] for r in results)
        
        # Weighted average CER (by number of cells)
        cer_weighted = sum(r['avg_cer'] * r['total_cells'] for r in results) / total_cells
        
        return {
            'total_cells': total_cells,
            'matched_cells': total_matched,
            'exact_matches': total_exact,
            'match_rate': total_matched / total_cells,
            'exact_match_rate': total_exact / total_cells,
            'avg_cer': cer_weighted,
            'char_accuracy': 1 - cer_weighted,
        }
    
    easyocr_metrics = calc_metrics(easyocr_results)
    paddleocr_metrics = calc_metrics(paddleocr_results)
    
    print(f"\nTotal cells evaluated: {easyocr_metrics['total_cells']}")
    
    print(f"\n{'Metric':<30} {'EasyOCR':>15} {'PaddleOCR':>15} {'Winner':>12}")
    print("-" * 72)
    
    metrics_to_show = [
        ('Exact Match Rate', 'exact_match_rate', True),
        ('Character Accuracy (1-CER)', 'char_accuracy', True),
        ('Average CER', 'avg_cer', False),
        ('Cell Match Rate', 'match_rate', True),
    ]
    
    for name, key, higher_better in metrics_to_show:
        e_val = easyocr_metrics[key]
        p_val = paddleocr_metrics[key]
        if higher_better:
            winner = 'EasyOCR' if e_val > p_val else 'PaddleOCR'
        else:
            winner = 'EasyOCR' if e_val < p_val else 'PaddleOCR'
        print(f"{name:<30} {e_val*100:>14.2f}% {p_val*100:>14.2f}% {winner:>12}")
    
    # Generate report
    report = f"""# Step 3: OCR Cell-by-Cell Evaluation Report
## EasyOCR vs PaddleOCR on FinTabNet_c

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Evaluation Method

This evaluation uses **Cell-by-Cell matching** with IoU (Intersection over Union):
- Each GT word is matched to the OCR result with highest bbox overlap
- IoU threshold: 0.3
- Only matched cells are compared for text accuracy

---

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | FinTabNet_c |
| Total Images | {len(easyocr_results)} |
| Total Cells | {easyocr_metrics['total_cells']:,} |
| Domain | Financial Tables |

---

## Performance Comparison

| Metric | EasyOCR | PaddleOCR | Winner |
|--------|---------|-----------|--------|
| Exact Match Rate | {easyocr_metrics['exact_match_rate']*100:.2f}% | {paddleocr_metrics['exact_match_rate']*100:.2f}% | {'EasyOCR' if easyocr_metrics['exact_match_rate'] > paddleocr_metrics['exact_match_rate'] else 'PaddleOCR'} |
| Character Accuracy (1-CER) | {easyocr_metrics['char_accuracy']*100:.2f}% | {paddleocr_metrics['char_accuracy']*100:.2f}% | {'EasyOCR' if easyocr_metrics['char_accuracy'] > paddleocr_metrics['char_accuracy'] else 'PaddleOCR'} |
| Average CER | {easyocr_metrics['avg_cer']*100:.2f}% | {paddleocr_metrics['avg_cer']*100:.2f}% | {'EasyOCR' if easyocr_metrics['avg_cer'] < paddleocr_metrics['avg_cer'] else 'PaddleOCR'} |
| Cell Match Rate | {easyocr_metrics['match_rate']*100:.2f}% | {paddleocr_metrics['match_rate']*100:.2f}% | {'EasyOCR' if easyocr_metrics['match_rate'] > paddleocr_metrics['match_rate'] else 'PaddleOCR'} |

---

## Comparison with Previous Results

### Previous (PubTabNet, Whole-Image):
| Metric | PaddleOCR |
|--------|-----------|
| Exact Match Rate | 70.47% |
| Character Accuracy | 84.36% |
| Average CER | 15.64% |

### Current (FinTabNet_c, Cell-by-Cell):
| Metric | EasyOCR | PaddleOCR |
|--------|---------|-----------|
| Exact Match Rate | {easyocr_metrics['exact_match_rate']*100:.2f}% | {paddleocr_metrics['exact_match_rate']*100:.2f}% |
| Character Accuracy | {easyocr_metrics['char_accuracy']*100:.2f}% | {paddleocr_metrics['char_accuracy']*100:.2f}% |
| Average CER | {easyocr_metrics['avg_cer']*100:.2f}% | {paddleocr_metrics['avg_cer']*100:.2f}% |

---

## Conclusion

**Winner: {'EasyOCR' if easyocr_metrics['char_accuracy'] > paddleocr_metrics['char_accuracy'] else 'PaddleOCR'}**

{'EasyOCR' if easyocr_metrics['char_accuracy'] > paddleocr_metrics['char_accuracy'] else 'PaddleOCR'} achieves higher character-level accuracy ({max(easyocr_metrics['char_accuracy'], paddleocr_metrics['char_accuracy'])*100:.2f}%) compared to {'PaddleOCR' if easyocr_metrics['char_accuracy'] > paddleocr_metrics['char_accuracy'] else 'EasyOCR'} ({min(easyocr_metrics['char_accuracy'], paddleocr_metrics['char_accuracy'])*100:.2f}%).
"""
    
    report_path = output_dir / 'step3_cell_comparison_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved report: {report_path}")
    
    # Generate histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    easyocr_cer = [r['avg_cer'] for r in easyocr_results]
    paddleocr_cer = [r['avg_cer'] for r in paddleocr_results]
    
    axes[0].hist(easyocr_cer, bins=30, alpha=0.6, label=f'EasyOCR (Mean: {np.mean(easyocr_cer):.4f})', color='steelblue')
    axes[0].hist(paddleocr_cer, bins=30, alpha=0.6, label=f'PaddleOCR (Mean: {np.mean(paddleocr_cer):.4f})', color='coral')
    axes[0].set_xlabel('Average CER per Image', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('CER Distribution (Cell-by-Cell)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart
    metrics = ['Exact Match\nRate', 'Character\nAccuracy', 'Cell Match\nRate']
    easyocr_vals = [easyocr_metrics['exact_match_rate'], easyocr_metrics['char_accuracy'], easyocr_metrics['match_rate']]
    paddleocr_vals = [paddleocr_metrics['exact_match_rate'], paddleocr_metrics['char_accuracy'], paddleocr_metrics['match_rate']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, [v*100 for v in easyocr_vals], width, label='EasyOCR', color='steelblue')
    bars2 = axes[1].bar(x + width/2, [v*100 for v in paddleocr_vals], width, label='PaddleOCR', color='coral')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylabel('Percentage (%)', fontsize=11)
    axes[1].set_title('OCR Performance Metrics', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 100)
    
    for bar in bars1:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    hist_path = output_dir / 'step3_cell_comparison_histogram.png'
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram: {hist_path}")
    
    # Save JSON summary
    summary = {
        'evaluation_method': 'cell-by-cell with IoU matching',
        'iou_threshold': 0.3,
        'num_images': len(easyocr_results),
        'total_cells': easyocr_metrics['total_cells'],
        'easyocr': {
            'exact_match_rate': easyocr_metrics['exact_match_rate'],
            'char_accuracy': easyocr_metrics['char_accuracy'],
            'avg_cer': easyocr_metrics['avg_cer'],
            'match_rate': easyocr_metrics['match_rate'],
        },
        'paddleocr': {
            'exact_match_rate': paddleocr_metrics['exact_match_rate'],
            'char_accuracy': paddleocr_metrics['char_accuracy'],
            'avg_cer': paddleocr_metrics['avg_cer'],
            'match_rate': paddleocr_metrics['match_rate'],
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    json_path = output_dir / 'step3_cell_comparison_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_path}")
    
    # Open output folder
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
