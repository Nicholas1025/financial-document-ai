"""
OCR Backend Comparison on FinTabNet Dataset

Compares PaddleOCR vs Docling using Ground Truth text from words/ directory.
Calculates Character Error Rate (CER) for true OCR accuracy comparison.

Usage:
  python experiments/ocr_comparison_histogram.py --num_samples 2000
  python experiments/ocr_comparison_histogram.py --num_samples 100 --quick  # Quick test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher

from modules.utils import load_config, ensure_dir
from modules.data_loaders import FinTabNetLoader
from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total chars in GT
    
    Lower is better. 0 = perfect match.
    """
    if not ground_truth:
        return 1.0 if predicted else 0.0
    
    # Use SequenceMatcher for edit distance calculation
    matcher = SequenceMatcher(None, ground_truth.lower(), predicted.lower())
    
    # Calculate edit operations
    edits = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            edits += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            edits += i2 - i1
        elif tag == 'insert':
            edits += j2 - j1
    
    return edits / len(ground_truth)


def calculate_word_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate word-level accuracy.
    
    Returns ratio of GT words found in predicted text.
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    
    gt_words = set(ground_truth.lower().split())
    pred_words = set(predicted.lower().split())
    
    if not gt_words:
        return 1.0
    
    matches = len(gt_words & pred_words)
    return matches / len(gt_words)


def run_comparison(num_samples: int = 2000, config_path: str = "configs/config.yaml",
                   seed: int = 42, save_outputs: bool = True):
    """
    Run OCR comparison on FinTabNet samples with Ground Truth.
    
    Returns:
        Dict with results for both backends
    """
    print("=" * 70)
    print("OCR Backend Comparison: PaddleOCR vs Docling")
    print(f"Dataset: FinTabNet (val split) with Ground Truth Words")
    print(f"Samples: {num_samples}")
    print("=" * 70)
    
    # Load config and data
    config = load_config(config_path)
    loader = FinTabNetLoader(config)
    
    # Get random samples
    np.random.seed(seed)
    annotations = loader.load_annotations(split='val', num_samples=num_samples)
    
    # Prepare image paths and ground truth
    samples = []
    for ann in annotations:
        img_path = loader.get_image_path(ann['filename'])
        gt_text = loader.get_ground_truth_text(ann['filename'])
        if Path(img_path).exists() and gt_text:
            samples.append({
                'image_path': img_path,
                'filename': ann['filename'],
                'ground_truth': gt_text
            })
    
    print(f"Loaded {len(samples)} samples with Ground Truth text")
    
    # Results storage
    results = {
        'paddleocr': {
            'cer_scores': [],  # Character Error Rate (lower is better)
            'word_accuracy': [],  # Word accuracy (higher is better)
            'char_counts': [],  # Characters extracted
            'times': [], 
            'errors': 0, 
            'cell_fill_rates': [],
            'numeric_cell_rates': [],
            'grid_rows': [], 
            'grid_cols': [],
            'validation_rates': [],
        },
        'docling': {
            'cer_scores': [],
            'word_accuracy': [],
            'char_counts': [],
            'times': [], 
            'errors': 0,
            'cell_fill_rates': [],
            'numeric_cell_rates': [],
            'grid_rows': [],
            'grid_cols': [],
            'validation_rates': [],
        }
    }
    
    backends = ['paddleocr', 'docling']
    validator = TableValidator(tolerance=0.02)
    
    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Testing: {backend.upper()}")
        print("=" * 60)
        
        # Initialize pipeline for this backend
        try:
            pipeline = FinancialTablePipeline(config_path=config_path, ocr_backend=backend)
        except Exception as e:
            print(f"Failed to initialize {backend}: {e}")
            continue
        
        for i, sample in enumerate(tqdm(samples, desc=f"{backend}")):
            try:
                import time
                import re
                start_time = time.time()
                
                result = pipeline.process_image(sample['image_path'])
                
                elapsed = time.time() - start_time
                results[backend]['times'].append(elapsed)
                
                # Get extracted text from grid
                grid = result['grid']
                extracted_text = ' '.join(cell for row in grid for cell in row if cell.strip())
                
                # Calculate OCR accuracy metrics against Ground Truth
                gt_text = sample['ground_truth']
                cer = calculate_cer(extracted_text, gt_text)
                word_acc = calculate_word_accuracy(extracted_text, gt_text)
                
                results[backend]['cer_scores'].append(cer)
                results[backend]['word_accuracy'].append(word_acc)
                
                # Calculate grid quality metrics
                total_cells = sum(len(row) for row in grid) if grid else 0
                
                # Cell fill rate (non-empty cells)
                non_empty_cells = sum(1 for row in grid for cell in row if cell.strip())
                cell_fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0
                results[backend]['cell_fill_rates'].append(cell_fill_rate)
                
                # Numeric cell rate (cells containing numbers)
                numeric_pattern = re.compile(r'[-−]?\d+[,.\d]*')
                numeric_cells = sum(1 for row in grid for cell in row if numeric_pattern.search(cell))
                numeric_rate = numeric_cells / total_cells if total_cells > 0 else 0
                results[backend]['numeric_cell_rates'].append(numeric_rate)
                
                # Total characters extracted
                total_chars = sum(len(cell) for row in grid for cell in row)
                results[backend]['char_counts'].append(total_chars)
                
                # Run validation
                validations = validator.validate_grid(
                    result['grid'], result['labels'], result['normalized_grid']
                )
                
                passed = sum(1 for v in validations if v.get('passed'))
                total = len(validations)
                rate = passed / total if total > 0 else 0.0
                
                results[backend]['validation_rates'].append(rate)
                results[backend]['grid_rows'].append(len(result['grid']))
                results[backend]['grid_cols'].append(len(result['grid'][0]) if result['grid'] else 0)
                
            except Exception as e:
                results[backend]['errors'] += 1
                results[backend]['validation_rates'].append(0.0)
                results[backend]['grid_rows'].append(0)
                results[backend]['grid_cols'].append(0)
                results[backend]['cell_fill_rates'].append(0.0)
                results[backend]['numeric_cell_rates'].append(0.0)
                results[backend]['char_counts'].append(0)
                results[backend]['cer_scores'].append(1.0)  # Max error
                results[backend]['word_accuracy'].append(0.0)
                
            # Progress update every 100 samples
            if (i + 1) % 100 == 0:
                avg_cer = np.mean(results[backend]['cer_scores'])
                avg_word = np.mean(results[backend]['word_accuracy'])
                print(f"  [{i+1}/{len(samples)}] Avg CER: {avg_cer:.3f}, Word Acc: {avg_word:.1%}")
    
    # Calculate summary statistics
    summary = {}
    for backend in backends:
        rates = results[backend]['validation_rates']
        times = results[backend]['times']
        fill_rates = results[backend]['cell_fill_rates']
        numeric_rates = results[backend]['numeric_cell_rates']
        char_counts = results[backend]['char_counts']
        cer_scores = results[backend]['cer_scores']
        word_acc = results[backend]['word_accuracy']
        
        summary[backend] = {
            'num_samples': len(rates),
            'errors': results[backend]['errors'],
            # OCR Accuracy Metrics (main comparison)
            'avg_cer': float(np.mean(cer_scores)) if cer_scores else 1.0,
            'median_cer': float(np.median(cer_scores)) if cer_scores else 1.0,
            'std_cer': float(np.std(cer_scores)) if cer_scores else 0,
            'avg_word_accuracy': float(np.mean(word_acc)) if word_acc else 0,
            'median_word_accuracy': float(np.median(word_acc)) if word_acc else 0,
            # Other metrics
            'avg_validation_rate': float(np.mean(rates)) if rates else 0,
            'avg_time': float(np.mean(times)) if times else 0,
            'total_time': float(np.sum(times)) if times else 0,
            'avg_grid_rows': float(np.mean(results[backend]['grid_rows'])) if results[backend]['grid_rows'] else 0,
            'avg_grid_cols': float(np.mean(results[backend]['grid_cols'])) if results[backend]['grid_cols'] else 0,
            'avg_cell_fill_rate': float(np.mean(fill_rates)) if fill_rates else 0,
            'avg_numeric_rate': float(np.mean(numeric_rates)) if numeric_rates else 0,
            'avg_char_count': float(np.mean(char_counts)) if char_counts else 0,
        }
    
    results['summary'] = summary
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': num_samples,
        'seed': seed,
        'dataset': 'FinTabNet'
    }
    
    return results


def generate_histogram(results: dict, output_dir: str = "outputs/figures"):
    """Generate comparison histogram with OCR accuracy metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paddle_cer = results['paddleocr']['cer_scores']
    docling_cer = results['docling']['cer_scores']
    paddle_word = results['paddleocr']['word_accuracy']
    docling_word = results['docling']['word_accuracy']
    
    # Create figure with subplots (2x3 for OCR accuracy focus)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('OCR Backend Comparison: PaddleOCR vs Docling on FinTabNet', fontsize=16, fontweight='bold')
    
    # 1. Histogram - Character Error Rate (CER) - Lower is better
    ax1 = axes[0, 0]
    bins = np.linspace(0, 2, 41)  # CER can exceed 1.0
    
    ax1.hist(paddle_cer, bins=bins, alpha=0.7, label='PaddleOCR', color='steelblue', edgecolor='black')
    ax1.hist(docling_cer, bins=bins, alpha=0.7, label='Docling', color='coral', edgecolor='black')
    ax1.set_xlabel('Character Error Rate (CER)', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Distribution of CER\n(Lower is Better)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(x=np.mean(paddle_cer), color='steelblue', linestyle='--', alpha=0.8, label=f'PaddleOCR mean')
    ax1.axvline(x=np.mean(docling_cer), color='coral', linestyle='--', alpha=0.8, label=f'Docling mean')
    
    # 2. Histogram - Word Accuracy - Higher is better
    ax2 = axes[0, 1]
    bins_acc = np.linspace(0, 1, 21)
    
    ax2.hist(paddle_word, bins=bins_acc, alpha=0.7, label='PaddleOCR', color='steelblue', edgecolor='black')
    ax2.hist(docling_word, bins=bins_acc, alpha=0.7, label='Docling', color='coral', edgecolor='black')
    ax2.set_xlabel('Word Accuracy', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Distribution of Word Accuracy\n(Higher is Better)', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Box plot - CER comparison
    ax3 = axes[0, 2]
    bp = ax3.boxplot([paddle_cer, docling_cer], tick_labels=['PaddleOCR', 'Docling'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax3.set_ylabel('Character Error Rate (CER)', fontsize=12)
    ax3.set_title('CER Distribution\n(Lower is Better)', fontsize=13)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Bar chart - Summary OCR Accuracy Metrics
    ax4 = axes[1, 0]
    metrics = ['CER ↓', 'Word Acc ↑', 'Time (s) ↓']
    paddle_vals = [
        results['summary']['paddleocr']['avg_cer'],
        results['summary']['paddleocr']['avg_word_accuracy'],
        results['summary']['paddleocr']['avg_time']
    ]
    docling_vals = [
        results['summary']['docling']['avg_cer'],
        results['summary']['docling']['avg_word_accuracy'],
        results['summary']['docling']['avg_time']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, paddle_vals, width, label='PaddleOCR', color='steelblue')
    bars2 = ax4.bar(x + width/2, docling_vals, width, label='Docling', color='coral')
    
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('OCR Accuracy Summary\n(↓=lower better, ↑=higher better)', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # 5. Box plot - Word Accuracy comparison  
    ax5 = axes[1, 1]
    bp2 = ax5.boxplot([paddle_word, docling_word], tick_labels=['PaddleOCR', 'Docling'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('steelblue')
    bp2['boxes'][1].set_facecolor('coral')
    ax5.set_ylabel('Word Accuracy', fontsize=12)
    ax5.set_title('Word Accuracy Distribution\n(Higher is Better)', fontsize=13)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Processing time comparison
    ax6 = axes[1, 2]
    if results['paddleocr']['times'] and results['docling']['times']:
        time_data = [results['paddleocr']['times'], results['docling']['times']]
        bp3 = ax6.boxplot(time_data, tick_labels=['PaddleOCR', 'Docling'], patch_artist=True)
        bp3['boxes'][0].set_facecolor('steelblue')
        bp3['boxes'][1].set_facecolor('coral')
        ax6.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax6.set_title('Processing Time per Image\n(Lower is Better)', fontsize=13)
        ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"ocr_comparison_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    # Also save as PDF for report
    pdf_path = output_dir / f"ocr_comparison_{timestamp}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()
    
    return fig_path


def print_summary(results: dict):
    """Print formatted summary with OCR accuracy metrics."""
    print("\n" + "=" * 80)
    print("OCR BACKEND COMPARISON SUMMARY (with Ground Truth)")
    print("=" * 80)
    
    summary = results['summary']
    
    print(f"\n{'Metric':<30} {'PaddleOCR':>15} {'Docling':>15} {'Winner':>12}")
    print("-" * 80)
    
    # CER (lower is better) - MAIN METRIC
    paddle_cer = summary['paddleocr']['avg_cer']
    docling_cer = summary['docling']['avg_cer']
    winner = "PaddleOCR" if paddle_cer < docling_cer else "Docling" if docling_cer < paddle_cer else "Tie"
    print(f"{'Avg CER (↓ better)':<30} {paddle_cer:>14.3f} {docling_cer:>14.3f} {winner:>12}")
    
    # Word Accuracy (higher is better)
    paddle_word = summary['paddleocr']['avg_word_accuracy']
    docling_word = summary['docling']['avg_word_accuracy']
    winner = "PaddleOCR" if paddle_word > docling_word else "Docling" if docling_word > paddle_word else "Tie"
    print(f"{'Avg Word Accuracy (↑ better)':<30} {paddle_word:>14.1%} {docling_word:>14.1%} {winner:>12}")
    
    # Cell fill rate (higher is better)
    paddle_fill = summary['paddleocr']['avg_cell_fill_rate']
    docling_fill = summary['docling']['avg_cell_fill_rate']
    winner = "PaddleOCR" if paddle_fill > docling_fill else "Docling" if docling_fill > paddle_fill else "Tie"
    print(f"{'Avg Cell Fill Rate':<30} {paddle_fill:>14.1%} {docling_fill:>14.1%} {winner:>12}")
    
    # Numeric rate (higher is better for financial docs)
    paddle_num = summary['paddleocr']['avg_numeric_rate']
    docling_num = summary['docling']['avg_numeric_rate']
    winner = "PaddleOCR" if paddle_num > docling_num else "Docling" if docling_num > paddle_num else "Tie"
    print(f"{'Avg Numeric Cell Rate':<30} {paddle_num:>14.1%} {docling_num:>14.1%} {winner:>12}")
    
    # Avg chars (higher = more text extracted)
    paddle_chars = summary['paddleocr']['avg_char_count']
    docling_chars = summary['docling']['avg_char_count']
    winner = "PaddleOCR" if paddle_chars > docling_chars else "Docling" if docling_chars > paddle_chars else "Tie"
    print(f"{'Avg Characters Extracted':<30} {paddle_chars:>15.0f} {docling_chars:>15.0f} {winner:>12}")
    
    # Processing time (lower is better)
    paddle_time = summary['paddleocr']['avg_time']
    docling_time = summary['docling']['avg_time']
    winner = "PaddleOCR" if paddle_time < docling_time else "Docling" if docling_time < paddle_time else "Tie"
    print(f"{'Avg Time (seconds)':<30} {paddle_time:>14.2f}s {docling_time:>14.2f}s {winner:>12}")
    
    # Errors (lower is better)
    paddle_err = summary['paddleocr']['errors']
    docling_err = summary['docling']['errors']
    winner = "PaddleOCR" if paddle_err < docling_err else "Docling" if docling_err < paddle_err else "Tie"
    print(f"{'Error Count':<30} {paddle_err:>15} {docling_err:>15} {winner:>12}")
    
    print("-" * 80)
    
    # Calculate overall winner based on OCR accuracy (CER is most important)
    paddle_cer = summary['paddleocr']['avg_cer']
    docling_cer = summary['docling']['avg_cer']
    paddle_word = summary['paddleocr']['avg_word_accuracy']
    docling_word = summary['docling']['avg_word_accuracy']
    
    paddle_wins = sum([
        paddle_cer < docling_cer,     # CER (lower better) - main metric
        paddle_word > docling_word,    # Word accuracy
        paddle_fill > docling_fill,    # Cell fill
        paddle_time < docling_time,    # Speed
        paddle_err <= docling_err,     # Errors
    ])
    docling_wins = 5 - paddle_wins
    
    overall = "PaddleOCR" if paddle_wins > docling_wins else "Docling" if docling_wins > paddle_wins else "Tie"
    print(f"\n{'OVERALL WINNER:':<25} {overall:>15} (wins {max(paddle_wins, docling_wins)}/5 metrics)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="OCR Backend Comparison on FinTabNet")
    parser.add_argument("--num_samples", type=int, default=2000,
                       help="Number of samples to evaluate")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 50 samples")
    parser.add_argument("--output", type=str, default="outputs/benchmark",
                       help="Output directory for results")
    parser.add_argument("--no_histogram", action="store_true",
                       help="Skip histogram generation")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 50
        print("Quick mode: using 50 samples")
    
    # Run comparison
    results = run_comparison(
        num_samples=args.num_samples,
        config_path=args.config,
        seed=args.seed
    )
    
    # Print summary
    print_summary(results)
    
    # Generate histogram
    if not args.no_histogram:
        generate_histogram(results)
    
    # Save results JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"ocr_comparison_{args.num_samples}samples_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # Prepare serializable results
    save_results = {
        'metadata': results['metadata'],
        'summary': results['summary'],
        'paddleocr': {
            'validation_rates': [convert_numpy(x) for x in results['paddleocr']['validation_rates']],
            'errors': results['paddleocr']['errors']
        },
        'docling': {
            'validation_rates': [convert_numpy(x) for x in results['docling']['validation_rates']],
            'errors': results['docling']['errors']
        }
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
