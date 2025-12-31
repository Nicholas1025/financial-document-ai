"""
Step 3 OCR Evaluation with FinTabNet
=====================================
运行 500 samples 的 OCR 评估，找到最好的样本并生成对比图

使用 PaddleOCR 对 FinTabNet 图片进行 OCR，与 GT words 对比计算 CER
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Dataset path
FINTABNET_PATH = Path('D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure')


def calculate_cer(gt_text: str, pred_text: str) -> float:
    """Calculate Character Error Rate using Levenshtein distance"""
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    
    # Levenshtein distance
    m, n = len(gt_text), len(pred_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_text[i-1] == pred_text[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, 1)


def run_ocr_evaluation(num_samples: int = 500, output_dir: Path = None):
    """
    运行 OCR 评估
    """
    print("\n" + "="*60)
    print("STEP 3: OCR EVALUATION (FinTabNet_c)")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    
    # Initialize EasyOCR
    print("\nLoading EasyOCR...")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)
    
    images_dir = FINTABNET_PATH / 'images'
    words_dir = FINTABNET_PATH / 'words'
    
    # Get word files
    word_files = sorted(list(words_dir.glob('*.json')))[:num_samples]
    print(f"Found {len(word_files)} samples")
    
    # Results storage
    results = []
    best_sample = None
    best_cer = float('inf')
    
    for i, word_file in enumerate(word_files):
        base_name = word_file.stem.replace('_words', '')
        img_path = images_dir / f'{base_name}.jpg'
        
        if not img_path.exists():
            continue
        
        # Load GT words
        with open(word_file, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        
        gt_words = [w['text'] for w in words_data]
        gt_text = ' '.join(gt_words)
        
        # Run OCR with EasyOCR
        try:
            ocr_result = reader.readtext(str(img_path))
            pred_words = [r[1] for r in ocr_result]
            pred_text = ' '.join(pred_words)
        except Exception as e:
            pred_words = []
            pred_text = ""
        
        # Calculate CER
        cer = calculate_cer(gt_text.lower(), pred_text.lower())
        
        # Store result
        result = {
            'sample_id': base_name,
            'img_path': str(img_path),
            'word_file': str(word_file),
            'gt_words': gt_words,
            'pred_words': pred_words,
            'gt_text': gt_text,
            'pred_text': pred_text,
            'cer': cer,
            'num_gt_words': len(gt_words),
            'num_pred_words': len(pred_words),
        }
        results.append(result)
        
        # Track best sample (lowest CER)
        if cer < best_cer:
            best_cer = cer
            best_sample = result
        
        # Progress
        if (i + 1) % 50 == 0:
            avg_cer = np.mean([r['cer'] for r in results])
            print(f"  Processed {i+1}/{len(word_files)}, Avg CER: {avg_cer:.4f}, Best CER: {best_cer:.4f}")
    
    # Final statistics
    all_cer = [r['cer'] for r in results]
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Average CER: {np.mean(all_cer):.4f}")
    print(f"Median CER: {np.median(all_cer):.4f}")
    print(f"Min CER: {np.min(all_cer):.4f}")
    print(f"Max CER: {np.max(all_cer):.4f}")
    print(f"Std CER: {np.std(all_cer):.4f}")
    print(f"\nBest Sample: {best_sample['sample_id']} (CER={best_cer:.4f})")
    
    return results, best_sample


def create_comparison_figure(sample: Dict, output_dir: Path):
    """
    为最好的样本创建对比图
    """
    print(f"\nCreating comparison figure for: {sample['sample_id']}")
    
    img_path = sample['img_path']
    word_file = sample['word_file']
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_width, img_height = img.size
    
    # Load word boxes
    with open(word_file, 'r', encoding='utf-8') as f:
        words_data = json.load(f)
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(20, 10))
    
    # Panel 1: Original image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title(f'Original Image\n({Path(img_path).name})', fontsize=12)
    ax1.axis('off')
    
    # Panel 2: Image with GT word boxes
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img)
    ax2.set_title(f'Ground Truth Words\n({len(words_data)} words)', fontsize=12)
    
    # Draw word boxes with different colors
    colors = plt.cm.Set3.colors
    for j, w in enumerate(words_data):
        bbox = w['bbox']
        color = colors[j % len(colors)]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.4
        )
        ax2.add_patch(rect)
        # Add text label for first few words
        if j < 10:
            ax2.text(bbox[0], bbox[1]-2, w['text'][:15], fontsize=6, color='black',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    ax2.axis('off')
    
    # Panel 3: Comparison text
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    ax3.set_title('OCR Comparison', fontsize=12)
    
    # Build comparison text
    comp_text = f"Sample: {sample['sample_id']}\n"
    comp_text += f"Image Size: {img_width}x{img_height}\n"
    comp_text += f"CER: {sample['cer']:.4f} ({sample['cer']*100:.2f}%)\n\n"
    comp_text += f"GT Words ({len(sample['gt_words'])}):\n"
    comp_text += "-" * 40 + "\n"
    
    gt_words_display = sample['gt_words'][:30]
    for i, w in enumerate(gt_words_display):
        comp_text += f"  {i+1}. \"{w}\"\n"
    if len(sample['gt_words']) > 30:
        comp_text += f"  ... (+{len(sample['gt_words'])-30} more)\n"
    
    comp_text += f"\nOCR Words ({len(sample['pred_words'])}):\n"
    comp_text += "-" * 40 + "\n"
    
    pred_words_display = sample['pred_words'][:30]
    for i, w in enumerate(pred_words_display):
        comp_text += f"  {i+1}. \"{w}\"\n"
    if len(sample['pred_words']) > 30:
        comp_text += f"  ... (+{len(sample['pred_words'])-30} more)\n"
    
    ax3.text(0.05, 0.95, comp_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    comp_path = output_dir / f'comparison_ocr_best_{sample["sample_id"]}.png'
    fig.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {comp_path}")
    
    return comp_path


def create_histogram(results: List[Dict], output_dir: Path):
    """
    创建 CER 分布直方图
    """
    all_cer = [r['cer'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_cer, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(all_cer), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_cer):.4f}')
    ax.axvline(np.median(all_cer), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_cer):.4f}')
    
    ax.set_xlabel('Character Error Rate (CER)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Step 3: OCR Evaluation on FinTabNet_c\n({len(results)} samples)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    hist_path = output_dir / 'step3_cer_histogram_fintabnet.png'
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    
    print(f"  Saved histogram: {hist_path}")
    
    return hist_path


def main():
    parser = argparse.ArgumentParser(description='Step 3 OCR Evaluation with FinTabNet')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='./outputs/thesis_figures/step3_ocr', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results, best_sample = run_ocr_evaluation(args.num_samples, output_dir)
    
    # Create comparison figure for best sample
    create_comparison_figure(best_sample, output_dir)
    
    # Create histogram
    create_histogram(results, output_dir)
    
    # Save results to CSV
    csv_path = output_dir / 'step3_fintabnet_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'cer', 'num_gt_words', 'num_pred_words'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'cer': r['cer'],
                'num_gt_words': r['num_gt_words'],
                'num_pred_words': r['num_pred_words'],
            })
    print(f"  Saved CSV: {csv_path}")
    
    # Save summary JSON
    summary = {
        'step': 3,
        'task': 'OCR Evaluation',
        'dataset': 'FinTabNet_c',
        'num_samples': len(results),
        'metrics': {
            'mean_cer': float(np.mean([r['cer'] for r in results])),
            'median_cer': float(np.median([r['cer'] for r in results])),
            'min_cer': float(np.min([r['cer'] for r in results])),
            'max_cer': float(np.max([r['cer'] for r in results])),
            'std_cer': float(np.std([r['cer'] for r in results])),
        },
        'best_sample': {
            'sample_id': best_sample['sample_id'],
            'cer': best_sample['cer'],
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    json_path = output_dir / 'step3_fintabnet_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {json_path}")
    
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
