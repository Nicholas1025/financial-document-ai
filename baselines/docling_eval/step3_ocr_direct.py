"""
Step 3 OCR Evaluation - Simplified Direct Version
Uses PaddleOCR directly without subprocess wrapper
"""

import os
import sys
import json
import csv
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_academic_plotting():
    """Configure matplotlib for academic-style figures."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('ggplot')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

setup_academic_plotting()
ACADEMIC_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(gt: str, pred: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein_distance(gt, pred) / len(gt)


def word_error_rate(gt_words: List[str], pred_words: List[str]) -> float:
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein_distance(gt_words, pred_words) / len(gt_words)


def merge_ocr_boxes_in_cell(gt_bbox: List, ocr_results: List[Dict]) -> str:
    """Merge all OCR boxes that overlap with GT cell bbox."""
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
    overlapping_texts = []
    
    for ocr in ocr_results:
        ocr_bbox = ocr['bbox']
        ox1, oy1, ox2, oy2 = ocr_bbox
        
        overlap_x = max(0, min(gt_x2, ox2) - max(gt_x1, ox1))
        overlap_y = max(0, min(gt_y2, oy2) - max(gt_y1, oy1))
        overlap_area = overlap_x * overlap_y
        
        ocr_area = (ox2 - ox1) * (oy2 - oy1)
        if ocr_area > 0 and overlap_area / ocr_area > 0.3:
            x_center = (ox1 + ox2) / 2
            overlapping_texts.append((x_center, ocr['text']))
    
    overlapping_texts.sort(key=lambda x: x[0])
    merged_text = ' '.join([t[1] for t in overlapping_texts])
    return merged_text.strip()


def normalize_text(text: str) -> str:
    return ' '.join(text.lower().split())


class PubTabNetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.jsonl_path = self.data_dir / "PubTabNet_2.0.0.jsonl"
        self.annotations = {}
        
    def load_annotations(self, split: str = 'val', max_samples: Optional[int] = None):
        print(f"Loading PubTabNet annotations for split: {split}")
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                data = json.loads(line)
                if data.get('split') == split:
                    self.annotations[data['filename']] = data
                    count += 1
                    if max_samples and count >= max_samples:
                        break
        print(f"Loaded {len(self.annotations)} annotations")
        return self.annotations
    
    def get_image_path(self, filename: str, split: str = 'val') -> Path:
        return self.data_dir / split / filename
    
    def extract_cell_texts(self, annotation: Dict) -> List[Dict]:
        cells = []
        html_data = annotation.get('html', {})
        
        for cell in html_data.get('cells', []):
            tokens = cell.get('tokens', [])
            bbox = cell.get('bbox')
            
            if not tokens or not bbox:
                continue
            
            text_parts = []
            for token in tokens:
                if not token.startswith('<') and not token.startswith('</'):
                    text_parts.append(token)
            
            text = ''.join(text_parts).strip()
            if text:
                cells.append({'text': text, 'bbox': bbox})
        
        return cells


def run_paddleocr_direct(image_path: str, paddle_ocr) -> List[Dict]:
    """Run PaddleOCR directly without subprocess."""
    result = paddle_ocr.ocr(image_path, cls=True)
    
    ocr_results = []
    if result and result[0]:
        for line in result[0]:
            bbox_points = line[0]
            text = line[1][0]
            conf = line[1][1]
            
            # Convert 4-point polygon to [x1, y1, x2, y2]
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            
            ocr_results.append({
                'text': text,
                'bbox': bbox,
                'confidence': conf
            })
    
    return ocr_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Paths
    data_dir = "D:/datasets/PubTabNet/pubtabnet/pubtabnet"
    output_dir = Path(__file__).parent / "benchmark_results"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 3: OCR EVALUATION (Direct PaddleOCR)")
    print("=" * 70)
    print(f"Dataset: PubTabNet")
    print(f"Samples: {args.num_samples}")
    
    # Load dataset
    loader = PubTabNetLoader(data_dir)
    loader.load_annotations(split='val', max_samples=args.num_samples * 2)
    
    all_filenames = list(loader.annotations.keys())
    if len(all_filenames) > args.num_samples:
        sampled_filenames = random.sample(all_filenames, args.num_samples)
    else:
        sampled_filenames = all_filenames
    
    # Initialize PaddleOCR directly
    print("\nInitializing PaddleOCR...")
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=False,
        det_db_box_thresh=0.3
    )
    
    # Results
    all_cer = []
    all_wer = []
    total_cells = 0
    exact_matches = 0
    matched_cells = 0
    per_sample = []
    
    print(f"\nEvaluating on {len(sampled_filenames)} samples...")
    
    for filename in tqdm(sampled_filenames, desc="PaddleOCR"):
        annotation = loader.annotations[filename]
        image_path = loader.get_image_path(filename, 'val')
        
        if not image_path.exists():
            continue
        
        gt_cells = loader.extract_cell_texts(annotation)
        if not gt_cells:
            continue
        
        try:
            ocr_results = run_paddleocr_direct(str(image_path), paddle_ocr)
        except Exception as e:
            print(f"OCR error on {filename}: {e}")
            continue
        
        sample_cer = []
        sample_exact = 0
        
        for gt_cell in gt_cells:
            gt_text = gt_cell['text']
            gt_bbox = gt_cell['bbox']
            
            pred_text = merge_ocr_boxes_in_cell(gt_bbox, ocr_results)
            
            gt_norm = normalize_text(gt_text)
            pred_norm = normalize_text(pred_text)
            
            cer = character_error_rate(gt_norm, pred_norm)
            wer = word_error_rate(gt_norm.split(), pred_norm.split())
            
            all_cer.append(cer)
            all_wer.append(wer)
            sample_cer.append(cer)
            total_cells += 1
            
            if pred_text:
                matched_cells += 1
            
            if gt_norm == pred_norm:
                exact_matches += 1
                sample_exact += 1
        
        per_sample.append({
            'filename': filename,
            'gt_cells': len(gt_cells),
            'avg_cer': np.mean(sample_cer) if sample_cer else 1.0,
            'exact_matches': sample_exact,
        })
    
    # Calculate metrics
    results = {
        'total_cells': total_cells,
        'matched_cells': matched_cells,
        'exact_matches': exact_matches,
        'cell_detection_rate': matched_cells / total_cells if total_cells > 0 else 0,
        'exact_match_rate': exact_matches / total_cells if total_cells > 0 else 0,
        'avg_cer': np.mean(all_cer) if all_cer else 1.0,
        'avg_wer': np.mean(all_wer) if all_wer else 1.0,
        'std_cer': np.std(all_cer) if all_cer else 0,
        'accuracy': 1 - np.mean(all_cer) if all_cer else 0,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Cells: {total_cells}")
    print(f"Exact Match Rate: {results['exact_match_rate']*100:.2f}%")
    print(f"Avg CER: {results['avg_cer']*100:.2f}%")
    print(f"Avg WER: {results['avg_wer']*100:.2f}%")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    
    # Generate academic histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    cer_filtered = [c for c in all_cer if c <= 1.0]
    mean_val = results["avg_cer"]
    color = ACADEMIC_COLORS[0]
    
    ax.hist(cer_filtered, bins=30, alpha=0.6, color=color, edgecolor='white', linewidth=0.5,
            label=f'PaddleOCR (Mean CER: {mean_val:.3f})')
    
    # Add mean line
    ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Character Error Rate (CER)')
    ax.set_ylabel('Frequency (Number of Cells)')
    ax.set_title(f'Step 3: OCR Error Distribution\n(n={args.num_samples} images, {total_cells} cells)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = figures_dir / f'step3_ocr_cer_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\nHistogram saved: {fig_path}")
    
    # Save summary
    summary = {
        'step': 3,
        'task': 'OCR (Optical Character Recognition)',
        'dataset': 'PubTabNet',
        'split': 'val',
        'num_samples': len(sampled_filenames),
        'total_cells': total_cells,
        'ground_truth': 'Character-level cell text from PubTabNet JSONL annotations',
        'evaluation_metric': 'CER (Character Error Rate), WER (Word Error Rate), Exact Match Rate',
        'matching_method': 'Merge all OCR boxes overlapping >30% with GT cell bbox',
        'timestamp': datetime.now().isoformat(),
        'results': {
            'PaddleOCR': {
                'exact_match_rate': round(results['exact_match_rate'] * 100, 2),
                'avg_cer': round(results['avg_cer'] * 100, 2),
                'avg_wer': round(results['avg_wer'] * 100, 2),
                'accuracy': round(results['accuracy'] * 100, 2),
            }
        },
        'histogram_path': str(fig_path),
    }
    
    json_path = output_dir / f'step3_ocr_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {json_path}")
    
    # Save per-sample CSV
    csv_path = output_dir / f'step3_ocr_per_sample.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'gt_cells', 'avg_cer', 'exact_matches'])
        for r in per_sample:
            writer.writerow([r['filename'], r['gt_cells'], f"{r['avg_cer']:.4f}", r['exact_matches']])
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
