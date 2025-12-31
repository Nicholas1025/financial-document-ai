"""
Step 3 OCR Comparison: EasyOCR vs PaddleOCR on FinTabNet
=========================================================
比较两种 OCR 引擎在 FinTabNet 金融表格数据集上的表现
生成对比报告和最佳样本的可视化

输出：
- comparison_ocr_best_*.png - 最佳样本对比图
- step3_ocr_comparison_report.md - 对比报告
- step3_ocr_comparison_summary.json - 汇总 JSON
- step3_cer_comparison_histogram.png - CER 分布对比图
"""

import os
import sys

# Fix protobuf issue for PaddleOCR
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

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


def run_easyocr_evaluation(word_files: List[Path], images_dir: Path) -> List[Dict]:
    """使用 EasyOCR 评估"""
    print("\n  Loading EasyOCR...")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    results = []
    for i, word_file in enumerate(word_files):
        base_name = word_file.stem.replace('_words', '')
        img_path = images_dir / f'{base_name}.jpg'
        
        if not img_path.exists():
            continue
        
        # Load GT
        with open(word_file, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        gt_words = [w['text'] for w in words_data]
        gt_text = ' '.join(gt_words)
        
        # Run OCR
        try:
            ocr_result = reader.readtext(str(img_path))
            pred_words = [r[1] for r in ocr_result]
            pred_text = ' '.join(pred_words)
        except Exception as e:
            pred_words = []
            pred_text = ""
        
        cer = calculate_cer(gt_text.lower(), pred_text.lower())
        
        results.append({
            'sample_id': base_name,
            'img_path': str(img_path),
            'word_file': str(word_file),
            'gt_words': gt_words,
            'pred_words': pred_words,
            'cer': cer,
        })
        
        if (i + 1) % 100 == 0:
            print(f"    EasyOCR: {i+1}/{len(word_files)}")
    
    return results


def run_paddleocr_single(args_tuple):
    """单个样本的 PaddleOCR 处理（用于子进程）"""
    import json
    from pathlib import Path
    
    word_file, images_dir_str = args_tuple
    images_dir = Path(images_dir_str)
    word_file = Path(word_file)
    
    base_name = word_file.stem.replace('_words', '')
    img_path = images_dir / f'{base_name}.jpg'
    
    if not img_path.exists():
        return None
    
    # Load GT
    with open(word_file, 'r', encoding='utf-8') as f:
        words_data = json.load(f)
    gt_words = [w['text'] for w in words_data]
    gt_text = ' '.join(gt_words)
    
    return {
        'sample_id': base_name,
        'img_path': str(img_path),
        'word_file': str(word_file),
        'gt_words': gt_words,
        'gt_text': gt_text,
    }


def run_paddleocr_evaluation(word_files: List[Path], images_dir: Path) -> List[Dict]:
    """使用 PaddleOCR 评估 - 通过子进程运行避免冲突"""
    import subprocess
    import tempfile
    
    print("\n  Running PaddleOCR via subprocess...")
    
    # 创建临时脚本 - ASCII only
    paddle_script = '''# -*- coding: utf-8 -*-
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import json
from pathlib import Path

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

results = []
for i, s in enumerate(samples):
    img_path = s["img_path"]
    try:
        ocr_result = ocr.ocr(img_path, cls=True)
        pred_words = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    pred_words.append(text)
        pred_text = " ".join(pred_words)
    except Exception as e:
        pred_words = []
        pred_text = ""
    
    results.append({
        "sample_id": s["sample_id"],
        "img_path": s["img_path"],
        "word_file": s["word_file"],
        "gt_words": s["gt_words"],
        "pred_words": pred_words,
        "pred_text": pred_text,
    })
    
    if (i + 1) % 50 == 0:
        print(f"PaddleOCR: {i+1}/{len(samples)}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
'''
    
    # 准备输入数据
    samples = []
    for word_file in word_files:
        base_name = word_file.stem.replace('_words', '')
        img_path = images_dir / f'{base_name}.jpg'
        
        if not img_path.exists():
            continue
        
        with open(word_file, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        gt_words = [w['text'] for w in words_data]
        
        samples.append({
            'sample_id': base_name,
            'img_path': str(img_path),
            'word_file': str(word_file),
            'gt_words': gt_words,
        })
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(paddle_script)
        script_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(samples, f)
        input_path = f.name
    
    output_path = tempfile.mktemp(suffix='.json')
    
    try:
        # 在新进程中运行 PaddleOCR
        result = subprocess.run(
            [sys.executable, script_path, input_path, output_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        print(result.stdout)
        if result.stderr:
            # 只打印关键错误
            for line in result.stderr.split('\n'):
                if 'Error' in line or 'error' in line:
                    print(f"  {line}")
        
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                paddle_results = json.load(f)
            
            # 计算 CER
            results = []
            for r in paddle_results:
                gt_text = ' '.join(r['gt_words'])
                pred_text = r.get('pred_text', '')
                cer = calculate_cer(gt_text.lower(), pred_text.lower())
                
                results.append({
                    'sample_id': r['sample_id'],
                    'img_path': r['img_path'],
                    'word_file': r['word_file'],
                    'gt_words': r['gt_words'],
                    'pred_words': r.get('pred_words', []),
                    'cer': cer,
                })
            
            return results
        else:
            print("  PaddleOCR subprocess failed, returning empty results")
            return []
    
    except subprocess.TimeoutExpired:
        print("  PaddleOCR subprocess timed out")
        return []
    except Exception as e:
        print(f"  PaddleOCR subprocess error: {e}")
        return []
    finally:
        # 清理临时文件
        for p in [script_path, input_path]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(output_path):
            os.remove(output_path)


def create_comparison_figure(easyocr_best: Dict, paddleocr_best: Dict, output_dir: Path):
    """为两个引擎的最佳样本创建对比图"""
    
    for ocr_name, sample in [('EasyOCR', easyocr_best), ('PaddleOCR', paddleocr_best)]:
        img_path = sample['img_path']
        word_file = sample['word_file']
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Load word boxes
        with open(word_file, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        
        # Create figure
        fig = plt.figure(figsize=(18, 8))
        
        # Panel 1: Original image with GT boxes
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        
        colors = plt.cm.Set3.colors
        for j, w in enumerate(words_data):
            bbox = w['bbox']
            color = colors[j % len(colors)]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.3
            )
            ax1.add_patch(rect)
        
        ax1.set_title(f'{ocr_name} Best Sample\n{sample["sample_id"]}\nCER: {sample["cer"]:.4f} ({sample["cer"]*100:.2f}%)', fontsize=11)
        ax1.axis('off')
        
        # Panel 2: Text comparison
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        
        comp_text = f"Ground Truth ({len(sample['gt_words'])} words):\n"
        comp_text += "-" * 40 + "\n"
        for i, w in enumerate(sample['gt_words'][:20]):
            comp_text += f"  {i+1}. \"{w}\"\n"
        if len(sample['gt_words']) > 20:
            comp_text += f"  ... (+{len(sample['gt_words'])-20} more)\n"
        
        comp_text += f"\n{ocr_name} Output ({len(sample['pred_words'])} words):\n"
        comp_text += "-" * 40 + "\n"
        for i, w in enumerate(sample['pred_words'][:20]):
            comp_text += f"  {i+1}. \"{w}\"\n"
        if len(sample['pred_words']) > 20:
            comp_text += f"  ... (+{len(sample['pred_words'])-20} more)\n"
        
        ax2.text(0.05, 0.95, comp_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        comp_path = output_dir / f'comparison_ocr_best_{ocr_name.lower()}_{sample["sample_id"]}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {comp_path}")


def create_comparison_histogram(easyocr_results: List[Dict], paddleocr_results: List[Dict], output_dir: Path):
    """创建 CER 对比直方图"""
    
    easyocr_cer = [r['cer'] for r in easyocr_results]
    paddleocr_cer = [r['cer'] for r in paddleocr_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparison
    axes[0].hist(easyocr_cer, bins=30, alpha=0.6, label=f'EasyOCR (Mean: {np.mean(easyocr_cer):.4f})', color='steelblue')
    axes[0].hist(paddleocr_cer, bins=30, alpha=0.6, label=f'PaddleOCR (Mean: {np.mean(paddleocr_cer):.4f})', color='coral')
    axes[0].set_xlabel('Character Error Rate (CER)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('CER Distribution Comparison', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart comparison
    metrics = ['Mean CER', 'Median CER', 'Min CER', 'Std CER']
    easyocr_vals = [np.mean(easyocr_cer), np.median(easyocr_cer), np.min(easyocr_cer), np.std(easyocr_cer)]
    paddleocr_vals = [np.mean(paddleocr_cer), np.median(paddleocr_cer), np.min(paddleocr_cer), np.std(paddleocr_cer)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, easyocr_vals, width, label='EasyOCR', color='steelblue')
    axes[1].bar(x + width/2, paddleocr_vals, width, label='PaddleOCR', color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylabel('CER Value', fontsize=11)
    axes[1].set_title('OCR Performance Metrics', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (e, p) in enumerate(zip(easyocr_vals, paddleocr_vals)):
        axes[1].text(i - width/2, e + 0.01, f'{e:.3f}', ha='center', fontsize=9)
        axes[1].text(i + width/2, p + 0.01, f'{p:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    hist_path = output_dir / 'step3_cer_comparison_histogram.png'
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    
    print(f"  Saved: {hist_path}")


def generate_report(easyocr_results: List[Dict], paddleocr_results: List[Dict], output_dir: Path):
    """生成对比报告"""
    
    easyocr_cer = [r['cer'] for r in easyocr_results]
    paddleocr_cer = [r['cer'] for r in paddleocr_results]
    
    easyocr_best = min(easyocr_results, key=lambda x: x['cer'])
    paddleocr_best = min(paddleocr_results, key=lambda x: x['cer'])
    
    # Markdown Report
    report = f"""# Step 3: OCR Evaluation Report
## EasyOCR vs PaddleOCR on FinTabNet_c

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | FinTabNet_c |
| Split | Test |
| Total Samples | {len(easyocr_results)} |
| Domain | Financial Tables |

---

## Performance Comparison

### Overall Metrics

| Metric | EasyOCR | PaddleOCR | Winner |
|--------|---------|-----------|--------|
| Mean CER | {np.mean(easyocr_cer):.4f} | {np.mean(paddleocr_cer):.4f} | {'EasyOCR' if np.mean(easyocr_cer) < np.mean(paddleocr_cer) else 'PaddleOCR'} |
| Median CER | {np.median(easyocr_cer):.4f} | {np.median(paddleocr_cer):.4f} | {'EasyOCR' if np.median(easyocr_cer) < np.median(paddleocr_cer) else 'PaddleOCR'} |
| Min CER | {np.min(easyocr_cer):.4f} | {np.min(paddleocr_cer):.4f} | {'EasyOCR' if np.min(easyocr_cer) < np.min(paddleocr_cer) else 'PaddleOCR'} |
| Max CER | {np.max(easyocr_cer):.4f} | {np.max(paddleocr_cer):.4f} | {'EasyOCR' if np.max(easyocr_cer) < np.max(paddleocr_cer) else 'PaddleOCR'} |
| Std CER | {np.std(easyocr_cer):.4f} | {np.std(paddleocr_cer):.4f} | - |

### Best Samples

| OCR Engine | Best Sample | CER |
|------------|-------------|-----|
| EasyOCR | {easyocr_best['sample_id']} | {easyocr_best['cer']:.4f} |
| PaddleOCR | {paddleocr_best['sample_id']} | {paddleocr_best['cer']:.4f} |

---

## Analysis

### Character Error Rate (CER) Distribution

- **EasyOCR**: Mean CER = {np.mean(easyocr_cer):.4f} ({np.mean(easyocr_cer)*100:.2f}%)
- **PaddleOCR**: Mean CER = {np.mean(paddleocr_cer):.4f} ({np.mean(paddleocr_cer)*100:.2f}%)

### Winner: **{'EasyOCR' if np.mean(easyocr_cer) < np.mean(paddleocr_cer) else 'PaddleOCR'}**

The winning OCR engine achieves a **{abs(np.mean(easyocr_cer) - np.mean(paddleocr_cer))*100:.2f}%** lower CER on average.

---

## Output Files

1. `comparison_ocr_best_easyocr_*.png` - Best EasyOCR sample visualization
2. `comparison_ocr_best_paddleocr_*.png` - Best PaddleOCR sample visualization
3. `step3_cer_comparison_histogram.png` - CER distribution comparison
4. `step3_ocr_comparison_summary.json` - Summary JSON
5. `step3_ocr_comparison_results.csv` - Per-sample results

---

## Conclusion

{'EasyOCR outperforms PaddleOCR' if np.mean(easyocr_cer) < np.mean(paddleocr_cer) else 'PaddleOCR outperforms EasyOCR'} on the FinTabNet_c financial table dataset with a mean CER of {min(np.mean(easyocr_cer), np.mean(paddleocr_cer)):.4f} compared to {max(np.mean(easyocr_cer), np.mean(paddleocr_cer)):.4f}.
"""
    
    report_path = output_dir / 'step3_ocr_comparison_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  Saved report: {report_path}")
    
    return easyocr_best, paddleocr_best


def main():
    parser = argparse.ArgumentParser(description='OCR Comparison: EasyOCR vs PaddleOCR')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='./outputs/thesis_figures/step3_ocr', help='Output directory')
    parser.add_argument('--only-easyocr', action='store_true', help='Only run EasyOCR')
    parser.add_argument('--only-paddleocr', action='store_true', help='Only run PaddleOCR')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation, only generate report from cached results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache files
    easyocr_cache = output_dir / 'easyocr_results_cache.json'
    paddleocr_cache = output_dir / 'paddleocr_results_cache.json'
    
    print("\n" + "="*60)
    print("STEP 3: OCR COMPARISON - EasyOCR vs PaddleOCR")
    print("="*60)
    print(f"Dataset: FinTabNet_c")
    print(f"Samples: {args.num_samples}")
    
    # Get files
    images_dir = FINTABNET_PATH / 'images'
    words_dir = FINTABNET_PATH / 'words'
    word_files = sorted(list(words_dir.glob('*.json')))[:args.num_samples]
    print(f"Found {len(word_files)} samples")
    
    easyocr_results = []
    paddleocr_results = []
    
    if not args.skip_eval:
        # Run EasyOCR
        if not args.only_paddleocr:
            if easyocr_cache.exists():
                print(f"\n[EasyOCR] Loading from cache: {easyocr_cache}")
                with open(easyocr_cache, 'r') as f:
                    easyocr_results = json.load(f)
                print(f"  Loaded {len(easyocr_results)} cached results")
            else:
                print("\n[1/2] Running EasyOCR evaluation...")
                easyocr_results = run_easyocr_evaluation(word_files, images_dir)
                # Save cache
                with open(easyocr_cache, 'w') as f:
                    json.dump(easyocr_results, f)
                print(f"  Saved cache: {easyocr_cache}")
        
        # Run PaddleOCR
        if not args.only_easyocr:
            if paddleocr_cache.exists():
                print(f"\n[PaddleOCR] Loading from cache: {paddleocr_cache}")
                with open(paddleocr_cache, 'r') as f:
                    paddleocr_results = json.load(f)
                print(f"  Loaded {len(paddleocr_results)} cached results")
            else:
                print("\n[2/2] Running PaddleOCR evaluation...")
                paddleocr_results = run_paddleocr_evaluation(word_files, images_dir)
                if paddleocr_results:
                    # Save cache
                    with open(paddleocr_cache, 'w') as f:
                        json.dump(paddleocr_results, f)
                    print(f"  Saved cache: {paddleocr_cache}")
    else:
        # Load from cache
        if easyocr_cache.exists():
            with open(easyocr_cache, 'r', encoding='utf-8') as f:
                easyocr_results = json.load(f)
            print(f"Loaded EasyOCR cache: {len(easyocr_results)} results")
        if paddleocr_cache.exists():
            with open(paddleocr_cache, 'r', encoding='utf-8') as f:
                paddleocr_results = json.load(f)
            print(f"Loaded PaddleOCR cache: {len(paddleocr_results)} results")
    
    # Check if we have results to report
    if not easyocr_results and not paddleocr_results:
        print("\nNo results to report. Run evaluation first.")
        return
    
    # Generate outputs
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)
    
    if easyocr_results and paddleocr_results:
        # Report
        easyocr_best, paddleocr_best = generate_report(easyocr_results, paddleocr_results, output_dir)
        
        # Comparison figures
        create_comparison_figure(easyocr_best, paddleocr_best, output_dir)
        
        # Histogram
        create_comparison_histogram(easyocr_results, paddleocr_results, output_dir)
        
        # Save CSV
        csv_path = output_dir / 'step3_ocr_comparison_results.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'easyocr_cer', 'paddleocr_cer'])
            for e, p in zip(easyocr_results, paddleocr_results):
                writer.writerow([e['sample_id'], e['cer'], p['cer']])
        print(f"  Saved CSV: {csv_path}")
        
        # Save summary JSON
        summary = {
            'step': 3,
            'task': 'OCR Comparison',
            'dataset': 'FinTabNet_c',
            'num_samples': len(easyocr_results),
            'easyocr': {
                'mean_cer': float(np.mean([r['cer'] for r in easyocr_results])),
                'median_cer': float(np.median([r['cer'] for r in easyocr_results])),
                'min_cer': float(np.min([r['cer'] for r in easyocr_results])),
                'max_cer': float(np.max([r['cer'] for r in easyocr_results])),
                'best_sample': easyocr_best['sample_id'],
            },
            'paddleocr': {
                'mean_cer': float(np.mean([r['cer'] for r in paddleocr_results])),
                'median_cer': float(np.median([r['cer'] for r in paddleocr_results])),
                'min_cer': float(np.min([r['cer'] for r in paddleocr_results])),
                'max_cer': float(np.max([r['cer'] for r in paddleocr_results])),
                'best_sample': paddleocr_best['sample_id'],
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        json_path = output_dir / 'step3_ocr_comparison_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved JSON: {json_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"EasyOCR   Mean CER: {summary['easyocr']['mean_cer']:.4f}")
        print(f"PaddleOCR Mean CER: {summary['paddleocr']['mean_cer']:.4f}")
        winner = 'EasyOCR' if summary['easyocr']['mean_cer'] < summary['paddleocr']['mean_cer'] else 'PaddleOCR'
        print(f"\nWinner: {winner}")
        print(f"\nAll outputs saved to: {output_dir}")
    else:
        # Only one OCR available - generate single report
        if easyocr_results:
            print("Only EasyOCR results available")
            cer_values = [r['cer'] for r in easyocr_results]
            print(f"  Mean CER: {np.mean(cer_values):.4f}")
            print(f"  Min CER: {np.min(cer_values):.4f}")
            print(f"  Run with --only-paddleocr to add PaddleOCR results")
        elif paddleocr_results:
            print("Only PaddleOCR results available")
            cer_values = [r['cer'] for r in paddleocr_results]
            print(f"  Mean CER: {np.mean(cer_values):.4f}")
            print(f"  Min CER: {np.min(cer_values):.4f}")
            print(f"  Run with --only-easyocr to add EasyOCR results")


if __name__ == '__main__':
    main()
