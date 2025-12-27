"""
Unified Benchmark Runner for All Pipeline Steps (1-5)
======================================================

This script runs comprehensive evaluation for all pipeline steps:
- Step 1: Table Detection (DocLayNet) - Table Transformer vs Docling
- Step 2: Table Structure Recognition (FinTabNet) - TT v1.1 vs TableFormer  
- Step 3: OCR (PubTabNet) - PaddleOCR vs Docling/RapidOCR
- Step 4: Numeric Normalisation (SynFinTabs) - NumericNormalizer
- Step 5: Semantic Cell Classification (SynFinTabs) - Heuristic vs Position-only

Each step generates:
- JSON results with full metrics
- CSV per-sample breakdown
- Histogram visualization
- Summary comparison table

Usage:
    python run_all_benchmarks.py --num-samples 1000 --run-all
    python run_all_benchmarks.py --num-samples 1000 --step 3  # Run only Step 3
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import matplotlib for histograms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def setup_academic_plotting():
    """Configure matplotlib for academic-style figures."""
    # Try to use seaborn style if available, otherwise fallback
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

# Academic color palette (Colorblind friendly)
ACADEMIC_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"
FIGURES_DIR = OUTPUT_DIR / "figures"

def setup_output_dirs():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR, FIGURES_DIR


# ============================================================================
# Dataset Paths
# ============================================================================

DATASETS = {
    'doclaynet': 'D:/datasets/DocLayNet',
    'fintabnet': 'D:/datasets/FinTabNet_OTSL/data',
    'pubtabnet': 'D:/datasets/PubTabNet/pubtabnet/pubtabnet',
    'synfintabs': 'D:/datasets/synfintabs/data',
}


# ============================================================================
# Step 1: Detection Evaluation
# ============================================================================

def run_step1_detection(num_samples: int, output_dir: Path) -> Dict:
    """
    Run Table Detection evaluation on DocLayNet.
    
    Compares:
    - Table Transformer (microsoft/table-transformer-detection)
    - Docling LayoutPredictor
    
    Metrics: Precision, Recall, F1 @ IoU 0.5
    """
    print("\n" + "="*70)
    print("STEP 1: TABLE DETECTION EVALUATION")
    print("="*70)
    print(f"Dataset: DocLayNet")
    print(f"Samples: {num_samples}")
    print(f"Metrics: Precision, Recall, F1 @ IoU 0.5")
    
    from baselines.docling_eval.stage2_detection import (
        load_doclaynet_samples, 
        TableTransformerDetector, 
        DoclingLayoutDetector,
        evaluate_detector
    )
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load samples
    samples = load_doclaynet_samples(DATASETS['doclaynet'], 'val', num_samples)
    
    # Initialize detectors
    detectors = {
        'Table Transformer': TableTransformerDetector(device, threshold=0.5),
    }
    
    # Try to load Docling - but handle errors gracefully
    try:
        docling_det = DoclingLayoutDetector(device, threshold=0.5)
        if docling_det.available:
            detectors['Docling LayoutPredictor'] = docling_det
    except Exception as e:
        print(f"Warning: Could not load Docling: {e}")
        print("Continuing with Table Transformer only...")
    
    # Run evaluation
    results = {}
    f1_distributions = {}
    
    for name, detector in detectors.items():
        print(f"\nEvaluating {name}...")
        try:
            result = evaluate_detector(detector, name, samples, iou_threshold=0.5)
            results[name] = result
            
            # Extract per-sample F1 for histogram
            f1_scores = [r['f1'] for r in result['per_sample_results']]
            f1_distributions[name] = f1_scores
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    # Generate academic histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, f1_scores) in enumerate(f1_distributions.items()):
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        mean_val = np.mean(f1_scores)
        
        # Plot histogram
        ax.hist(f1_scores, bins=20, alpha=0.6, 
                label=f'{name} (Mean: {mean_val:.3f})',
                color=color, edgecolor='white', linewidth=0.5)
        
        # Add mean line
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('F1 Score per Image')
    ax.set_ylabel('Frequency (Number of Images)')
    ax.set_title(f'Step 1: Table Detection Performance Distribution\n(n={num_samples} samples)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / f'step1_detection_f1_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Histogram saved: {fig_path}")
    
    # Build summary
    summary = {
        'step': 1,
        'task': 'Table Detection',
        'dataset': 'DocLayNet',
        'split': 'val',
        'num_samples': len(samples),
        'ground_truth': 'COCO format annotations with table bounding boxes',
        'evaluation_metric': 'Precision, Recall, F1 @ IoU 0.5',
        'timestamp': datetime.now().isoformat(),
        'results': {},
        'histogram_path': str(fig_path),
    }
    
    for name, result in results.items():
        m = result['metrics']
        summary['results'][name] = {
            'precision': round(m['precision'], 4),
            'recall': round(m['recall'], 4),
            'f1': round(m['f1'], 4),
            'avg_iou': round(m['avg_iou'], 4),
            'tp': m['total_tp'],
            'fp': m['total_fp'],
            'fn': m['total_fn'],
            'avg_time_ms': round(result['timing']['avg_time_per_image'] * 1000, 1),
        }
    
    # Save detailed results
    json_path = output_dir / f'step1_detection_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sample CSV
    csv_path = output_dir / f'step1_detection_per_sample.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'detector', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'avg_iou'])
        for name, result in results.items():
            for r in result['per_sample_results']:
                writer.writerow([
                    r['filename'], name, r['gt_count'], r['pred_count'],
                    r['tp'], r['fp'], r['fn'],
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['avg_iou']:.4f}"
                ])
    
    print(f"\nResults saved: {json_path}")
    return summary


# ============================================================================
# Step 2: TSR Evaluation
# ============================================================================

def run_step2_tsr(num_samples: int, output_dir: Path) -> Dict:
    """
    Run Table Structure Recognition evaluation on FinTabNet.
    
    Compares:
    - Table Transformer v1.1
    - Docling TableFormer
    
    Metrics: TEDS (Tree Edit Distance Similarity) - structural
    """
    print("\n" + "="*70)
    print("STEP 2: TABLE STRUCTURE RECOGNITION EVALUATION")
    print("="*70)
    print(f"Dataset: FinTabNet OTSL")
    print(f"Samples: {num_samples}")
    print(f"Metrics: TEDS (Tree Edit Distance Similarity)")
    
    from baselines.docling_eval.tsr_benchmark_v2 import (
        load_fintabnet_samples,
        TableTransformerTSR,
        DoclingTableFormerDirect,
        TEDSCalculator
    )
    from tqdm import tqdm
    
    # Load samples
    samples = load_fintabnet_samples(DATASETS['fintabnet'], num_samples, split='test')
    
    # Initialize methods
    methods = {
        'Table Transformer v1.1': TableTransformerTSR(),
        'Docling TableFormer': DoclingTableFormerDirect(mode='accurate'),
    }
    
    teds_calc = TEDSCalculator(structure_only=True)
    
    # Run evaluation
    results = {}
    teds_distributions = {}
    
    for name, method in methods.items():
        print(f"\nEvaluating {name}...")
        teds_scores = []
        per_sample = []
        
        for sample in tqdm(samples, desc=name):
            try:
                pred_html = method.predict(sample['image_path'])
                gt_html = sample['gt_html']
                teds = teds_calc.compute(pred_html, gt_html)
                teds_scores.append(teds)
                per_sample.append({
                    'filename': sample['filename'],
                    'teds': teds,
                })
            except Exception as e:
                print(f"Error on {sample['filename']}: {e}")
                teds_scores.append(0.0)
                per_sample.append({
                    'filename': sample['filename'],
                    'teds': 0.0,
                    'error': str(e)
                })
        
        results[name] = {
            'mean_teds': np.mean(teds_scores),
            'std_teds': np.std(teds_scores),
            'median_teds': np.median(teds_scores),
            'min_teds': np.min(teds_scores),
            'max_teds': np.max(teds_scores),
            'per_sample': per_sample,
        }
        teds_distributions[name] = teds_scores
    
    # Generate academic histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, scores) in enumerate(teds_distributions.items()):
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        mean_val = np.mean(scores)
        
        # Plot histogram
        ax.hist(scores, bins=20, alpha=0.6, 
                label=f'{name} (Mean: {mean_val:.3f})',
                color=color, edgecolor='white', linewidth=0.5)
        
        # Add mean line
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('TEDS Score (Tree Edit Distance Similarity)')
    ax.set_ylabel('Frequency (Number of Tables)')
    ax.set_title(f'Step 2: Table Structure Recognition Performance\n(n={num_samples} samples)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / f'step2_tsr_teds_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Histogram saved: {fig_path}")
    
    # Build summary
    summary = {
        'step': 2,
        'task': 'Table Structure Recognition',
        'dataset': 'FinTabNet OTSL',
        'split': 'test',
        'num_samples': len(samples),
        'ground_truth': 'HTML table structure from FinTabNet annotations',
        'evaluation_metric': 'TEDS (Tree Edit Distance Similarity) - structural only',
        'timestamp': datetime.now().isoformat(),
        'results': {},
        'histogram_path': str(fig_path),
    }
    
    for name, result in results.items():
        summary['results'][name] = {
            'mean_teds': round(result['mean_teds'], 4),
            'std_teds': round(result['std_teds'], 4),
            'median_teds': round(result['median_teds'], 4),
            'min_teds': round(result['min_teds'], 4),
            'max_teds': round(result['max_teds'], 4),
        }
    
    # Save results
    json_path = output_dir / f'step2_tsr_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sample CSV
    csv_path = output_dir / f'step2_tsr_per_sample.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'method', 'teds'])
        for name, result in results.items():
            for r in result['per_sample']:
                writer.writerow([r['filename'], name, f"{r['teds']:.4f}"])
    
    print(f"\nResults saved: {json_path}")
    return summary


# ============================================================================
# Step 3: OCR Evaluation (Improved with bbox merging)
# ============================================================================

def run_step3_ocr(num_samples: int, output_dir: Path) -> Dict:
    """
    Run OCR evaluation on PubTabNet.
    
    Improved method: Merge overlapping OCR boxes within GT cell bbox
    
    Compares:
    - PaddleOCR
    - Docling (RapidOCR)
    
    Metrics: CER, WER, Exact Match Rate
    """
    print("\n" + "="*70)
    print("STEP 3: OCR EVALUATION")
    print("="*70)
    print(f"Dataset: PubTabNet")
    print(f"Samples: {num_samples}")
    print(f"Metrics: CER, WER, Exact Match Rate")
    print(f"Method: Merge overlapping OCR boxes within GT cell bbox")
    
    from baselines.docling_eval.stage2_step3_ocr import (
        PubTabNetLoader,
        levenshtein_distance,
        character_error_rate,
        word_error_rate,
    )
    from modules.ocr import TableOCR, get_ocr_backend
    from tqdm import tqdm
    import random
    
    random.seed(42)
    
    # Load dataset
    loader = PubTabNetLoader(DATASETS['pubtabnet'])
    loader.load_annotations(split='val', max_samples=num_samples * 2)
    
    all_filenames = list(loader.annotations.keys())
    if len(all_filenames) > num_samples:
        sampled_filenames = random.sample(all_filenames, num_samples)
    else:
        sampled_filenames = all_filenames
    
    def merge_ocr_boxes_in_cell(gt_bbox: List, ocr_results: List[Dict]) -> str:
        """
        Merge all OCR boxes that overlap with GT cell bbox.
        This is more accurate than single-box matching.
        """
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        # Find all OCR boxes that overlap with GT cell
        overlapping_texts = []
        
        for ocr in ocr_results:
            ocr_bbox = ocr['bbox']
            ox1, oy1, ox2, oy2 = ocr_bbox
            
            # Check overlap
            overlap_x = max(0, min(gt_x2, ox2) - max(gt_x1, ox1))
            overlap_y = max(0, min(gt_y2, oy2) - max(gt_y1, oy1))
            overlap_area = overlap_x * overlap_y
            
            # If significant overlap (>30% of OCR box)
            ocr_area = (ox2 - ox1) * (oy2 - oy1)
            if ocr_area > 0 and overlap_area / ocr_area > 0.3:
                # Add with x-coordinate for sorting
                x_center = (ox1 + ox2) / 2
                overlapping_texts.append((x_center, ocr['text']))
        
        # Sort by x-coordinate and concatenate
        overlapping_texts.sort(key=lambda x: x[0])
        merged_text = ' '.join([t[1] for t in overlapping_texts])
        
        return merged_text.strip()
    
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return ' '.join(text.lower().split())
    
    # Initialize OCR backends
    backends = {
        'PaddleOCR': TableOCR(lang='en', use_gpu=False),
    }
    
    try:
        backends['Docling (RapidOCR)'] = get_ocr_backend('docling')
    except Exception as e:
        print(f"Warning: Could not load Docling OCR: {e}")
    
    # Run evaluation
    results = {}
    cer_distributions = {}
    
    for backend_name, ocr in backends.items():
        print(f"\nEvaluating {backend_name}...")
        
        all_cer = []
        all_wer = []
        total_cells = 0
        exact_matches = 0
        matched_cells = 0
        per_sample = []
        
        for filename in tqdm(sampled_filenames, desc=backend_name):
            annotation = loader.annotations[filename]
            image_path = loader.get_image_path(filename, 'val')
            
            if not image_path.exists():
                continue
            
            # Get GT cells
            gt_cells = loader.extract_cell_texts(annotation)
            if not gt_cells:
                continue
            
            # Run OCR
            try:
                ocr_results = ocr.extract_text(str(image_path))
            except Exception as e:
                print(f"OCR error on {filename}: {e}")
                continue
            
            # Evaluate each GT cell
            sample_cer = []
            sample_exact = 0
            
            for gt_cell in gt_cells:
                gt_text = gt_cell['text']
                gt_bbox = gt_cell['bbox']
                
                # Merge overlapping OCR boxes
                pred_text = merge_ocr_boxes_in_cell(gt_bbox, ocr_results)
                
                # Normalize
                gt_norm = normalize_text(gt_text)
                pred_norm = normalize_text(pred_text)
                
                # Calculate metrics
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
        
        results[backend_name] = {
            'total_cells': total_cells,
            'matched_cells': matched_cells,
            'exact_matches': exact_matches,
            'cell_detection_rate': matched_cells / total_cells if total_cells > 0 else 0,
            'exact_match_rate': exact_matches / total_cells if total_cells > 0 else 0,
            'avg_cer': np.mean(all_cer) if all_cer else 1.0,
            'avg_wer': np.mean(all_wer) if all_wer else 1.0,
            'std_cer': np.std(all_cer) if all_cer else 0,
            'accuracy': 1 - np.mean(all_cer) if all_cer else 0,
            'per_sample': per_sample,
        }
        cer_distributions[backend_name] = all_cer
    
    # Generate academic histogram (CER distribution)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, cer_scores) in enumerate(cer_distributions.items()):
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        mean_val = np.mean(cer_scores)
        
        # Filter out extreme values for better visualization
        cer_filtered = [c for c in cer_scores if c <= 1.0]
        
        ax.hist(cer_filtered, bins=30, alpha=0.6,
                label=f'{name} (Mean CER: {mean_val:.3f})',
                color=color, edgecolor='white', linewidth=0.5)
        
        # Add mean line
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Character Error Rate (CER)')
    ax.set_ylabel('Frequency (Number of Cells)')
    ax.set_title(f'Step 3: OCR Error Distribution\n(n={num_samples} images)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / f'step3_ocr_cer_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Histogram saved: {fig_path}")
    
    # Build summary
    summary = {
        'step': 3,
        'task': 'OCR (Optical Character Recognition)',
        'dataset': 'PubTabNet',
        'split': 'val',
        'num_samples': len(sampled_filenames),
        'ground_truth': 'Character-level cell text from PubTabNet JSONL annotations',
        'evaluation_metric': 'CER (Character Error Rate), WER (Word Error Rate), Exact Match Rate',
        'matching_method': 'Merge all OCR boxes overlapping >30% with GT cell bbox',
        'timestamp': datetime.now().isoformat(),
        'results': {},
        'histogram_path': str(fig_path),
    }
    
    for name, result in results.items():
        summary['results'][name] = {
            'total_cells': result['total_cells'],
            'exact_match_rate': round(result['exact_match_rate'] * 100, 2),
            'avg_cer': round(result['avg_cer'] * 100, 2),
            'avg_wer': round(result['avg_wer'] * 100, 2),
            'accuracy': round(result['accuracy'] * 100, 2),
        }
    
    # Save results
    json_path = output_dir / f'step3_ocr_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sample CSV
    csv_path = output_dir / f'step3_ocr_per_sample.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'backend', 'gt_cells', 'avg_cer', 'exact_matches'])
        for name, result in results.items():
            for r in result['per_sample']:
                writer.writerow([r['filename'], name, r['gt_cells'], f"{r['avg_cer']:.4f}", r['exact_matches']])
    
    print(f"\nResults saved: {json_path}")
    return summary


# ============================================================================
# Step 4: Numeric Normalisation
# ============================================================================

def run_step4_numeric(num_samples: int, output_dir: Path) -> Dict:
    """
    Run Numeric Normalisation evaluation on SynFinTabs.
    
    Tests NumericNormalizer on converting various number formats.
    
    Metrics: Exact Match, Relative Error, Scale Accuracy, Sign Accuracy
    """
    print("\n" + "="*70)
    print("STEP 4: NUMERIC NORMALISATION EVALUATION")
    print("="*70)
    print(f"Dataset: SynFinTabs")
    print(f"Samples: {num_samples} tables")
    print(f"Metrics: Exact Match, Relative Error, Scale/Sign Accuracy")
    
    from baselines.docling_eval.stage2_step4_numeric import (
        NumericNormalizer,
        calculate_metrics,
    )
    from datasets import load_dataset
    from tqdm import tqdm
    import random
    
    random.seed(42)
    np.random.seed(42)
    
    # Load dataset
    print(f"Loading SynFinTabs from {DATASETS['synfintabs']}...")
    ds = load_dataset('parquet', data_dir=DATASETS['synfintabs'], split='test')
    
    # Sample tables
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    # Initialize normalizer
    normalizer = NumericNormalizer()
    
    # Collect all QA pairs
    all_results = []
    relative_errors = []
    
    for idx in tqdm(indices, desc="Processing tables"):
        item = ds[int(idx)]
        
        for q in item['questions']:
            raw_text = q['answer']
            
            # GT value
            gt_value = normalizer.normalize_gt(raw_text)
            if gt_value is None:
                continue
            
            # Pred value
            pred_value, error_msg = normalizer.normalize(raw_text)
            
            # Metrics
            metrics = calculate_metrics(gt_value, pred_value)
            
            all_results.append({
                'sample_id': f"{item['id']}_{q['id']}",
                'raw_text': raw_text,
                'gt_value': gt_value,
                'pred_value': pred_value,
                'exact_match': metrics['exact_match'],
                'relative_error': metrics['relative_error'],
                'scale_correct': metrics['scale_correct'],
                'sign_correct': metrics['sign_correct'],
                'error_type': metrics['error_type'],
            })
            
            if metrics['relative_error'] != float('inf'):
                relative_errors.append(metrics['relative_error'])
    
    # Aggregate metrics
    n_total = len(all_results)
    n_exact = sum(1 for r in all_results if r['exact_match'])
    n_scale = sum(1 for r in all_results if r['scale_correct'])
    n_sign = sum(1 for r in all_results if r['sign_correct'])
    
    error_counts = defaultdict(int)
    for r in all_results:
        error_counts[r['error_type']] += 1
    
    # Generate academic histogram (Relative Error distribution)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for visualization (exclude zeros for log scale)
    errors_nonzero = [e for e in relative_errors if e > 1e-10]
    
    if errors_nonzero:
        color = ACADEMIC_COLORS[1]
        ax.hist(errors_nonzero, bins=50, alpha=0.7, color=color, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Frequency (Number of Values)')
        ax.set_title(f'Step 4: Numeric Normalisation Error Distribution\n(n={n_total} values, {n_exact} exact matches)')
        
        # Use log scale if errors span many orders of magnitude
        if max(errors_nonzero) / (min(errors_nonzero) + 1e-10) > 100:
            ax.set_xscale('log')
        else:
            ax.set_xlim(0, min(1.0, np.percentile(errors_nonzero, 99)))
            
        ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'All {n_total} values are exact matches!', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'Step 4: Numeric Normalisation - Perfect Results')
    
    fig_path = FIGURES_DIR / f'step4_numeric_error_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Histogram saved: {fig_path}")
    
    # Build summary
    summary = {
        'step': 4,
        'task': 'Numeric Normalisation',
        'dataset': 'SynFinTabs',
        'split': 'test',
        'num_tables': len(indices),
        'num_values': n_total,
        'ground_truth': 'Numeric values from SynFinTabs QA pairs (formatted strings)',
        'evaluation_metric': 'Exact Match, Relative Error, Scale Accuracy, Sign Accuracy',
        'timestamp': datetime.now().isoformat(),
        'results': {
            'exact_match_rate': round(n_exact / n_total * 100, 2) if n_total > 0 else 0,
            'scale_accuracy': round(n_scale / n_total * 100, 2) if n_total > 0 else 0,
            'sign_accuracy': round(n_sign / n_total * 100, 2) if n_total > 0 else 0,
            'mean_relative_error': round(np.mean(relative_errors), 6) if relative_errors else 0,
            'median_relative_error': round(np.median(relative_errors), 6) if relative_errors else 0,
        },
        'error_breakdown': dict(error_counts),
        'histogram_path': str(fig_path),
    }
    
    # Save results
    json_path = output_dir / f'step4_numeric_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sample CSV
    csv_path = output_dir / f'step4_numeric_per_sample.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'raw_text', 'gt_value', 'pred_value',
            'exact_match', 'relative_error', 'scale_correct', 'sign_correct', 'error_type'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    
    print(f"\nResults saved: {json_path}")
    return summary


# ============================================================================
# Step 5: Semantic Cell Classification
# ============================================================================

def run_step5_semantic(num_samples: int, output_dir: Path) -> Dict:
    """
    Run Semantic Cell Classification evaluation on SynFinTabs.
    
    Compares:
    - Heuristic Classifier (rule-based)
    - Position-only Classifier (baseline)
    
    Metrics: Accuracy, Macro F1, Per-class P/R/F1
    """
    print("\n" + "="*70)
    print("STEP 5: SEMANTIC CELL CLASSIFICATION EVALUATION")
    print("="*70)
    print(f"Dataset: SynFinTabs")
    print(f"Samples: {num_samples} tables")
    print(f"Metrics: Accuracy, Macro F1, Per-class F1")
    
    from baselines.docling_eval.stage2_step5_semantic import (
        HeuristicCellClassifier,
        PositionBasedClassifier,
        CELL_TYPES,
    )
    from datasets import load_dataset
    from tqdm import tqdm
    import random
    from collections import Counter
    
    random.seed(42)
    np.random.seed(42)
    
    # Load dataset
    print(f"Loading SynFinTabs from {DATASETS['synfintabs']}...")
    ds = load_dataset('parquet', data_dir=DATASETS['synfintabs'], split='test')
    
    # Sample tables
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    # Initialize classifiers
    classifiers = {
        'Heuristic': HeuristicCellClassifier(),
        'Position-only': PositionBasedClassifier(),
    }
    
    # Collect results
    results = {name: [] for name in classifiers}
    
    for idx in tqdm(indices, desc="Processing tables"):
        item = ds[int(idx)]
        rows_data = item['rows']  # Each row is a dict with 'cells' key
        
        num_rows = len(rows_data)
        # Get max columns from cells in each row
        num_cols = max(len(row.get('cells', [])) for row in rows_data) if rows_data else 0
        
        for row_idx, row_dict in enumerate(rows_data):
            cells = row_dict.get('cells', [])
            
            for col_idx, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    continue
                
                text = cell.get('text', '')
                gt_label = cell.get('label', 'data')
                
                for clf_name, clf in classifiers.items():
                    pred_label = clf.classify(
                        text=text,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        row_cells=cells  # Pass cells list, not row dict
                    )
                    
                    results[clf_name].append({
                        'table_id': item['id'],
                        'row_idx': row_idx,
                        'col_idx': col_idx,
                        'text': text[:50],  # Truncate for storage
                        'gt_label': gt_label,
                        'pred_label': pred_label,
                        'correct': gt_label == pred_label,
                    })
    
    def calculate_classification_metrics(predictions: List[Dict]) -> Dict:
        """Calculate classification metrics."""
        y_true = [p['gt_label'] for p in predictions]
        y_pred = [p['pred_label'] for p in predictions]
        
        # Accuracy
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        # Per-class metrics
        class_metrics = {}
        for label in CELL_TYPES:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(1 for t in y_true if t == label)
            }
        
        # Macro F1
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class': class_metrics,
        }
    
    # Calculate metrics for each classifier
    all_metrics = {}
    for clf_name, predictions in results.items():
        all_metrics[clf_name] = calculate_classification_metrics(predictions)
    
    # Generate academic bar chart (Per-class F1 comparison)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CELL_TYPES))
    width = 0.35
    
    for i, (clf_name, metrics) in enumerate(all_metrics.items()):
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        f1_scores = [metrics['per_class'][label]['f1'] for label in CELL_TYPES]
        offset = width * (i - 0.5)
        
        bars = ax.bar(x + offset, f1_scores, width, 
                     label=f'{clf_name} (Macro F1: {metrics["macro_f1"]:.3f})',
                     color=color, edgecolor='white', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Step 5: Semantic Classification Performance by Class\n(n={num_samples} tables)')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ').title() for t in CELL_TYPES], rotation=0, ha='center')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1.1) # Extra space for labels
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / f'step5_semantic_f1_barchart.png'
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Bar chart saved: {fig_path}")
    
    # Build summary
    summary = {
        'step': 5,
        'task': 'Semantic Cell Classification',
        'dataset': 'SynFinTabs',
        'split': 'test',
        'num_tables': len(indices),
        'num_cells': len(results['Heuristic']),
        'cell_types': CELL_TYPES,
        'ground_truth': 'Cell labels from SynFinTabs (section_title, column_header, row_header, data, currency_unit)',
        'evaluation_metric': 'Accuracy, Macro F1, Per-class Precision/Recall/F1',
        'timestamp': datetime.now().isoformat(),
        'results': {},
        'histogram_path': str(fig_path),
    }
    
    for clf_name, metrics in all_metrics.items():
        summary['results'][clf_name] = {
            'accuracy': round(metrics['accuracy'] * 100, 2),
            'macro_f1': round(metrics['macro_f1'] * 100, 2),
            'per_class': {
                label: {
                    'precision': round(m['precision'] * 100, 2),
                    'recall': round(m['recall'] * 100, 2),
                    'f1': round(m['f1'] * 100, 2),
                    'support': m['support']
                }
                for label, m in metrics['per_class'].items()
            }
        }
    
    # Save results
    json_path = output_dir / f'step5_semantic_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-cell CSV
    csv_path = output_dir / f'step5_semantic_per_cell.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['table_id', 'row_idx', 'col_idx', 'text', 'gt_label', 'heuristic_pred', 'position_pred'])
        
        heur_results = results['Heuristic']
        pos_results = results['Position-only']
        
        for h, p in zip(heur_results, pos_results):
            writer.writerow([
                h['table_id'], h['row_idx'], h['col_idx'], h['text'],
                h['gt_label'], h['pred_label'], p['pred_label']
            ])
    
    print(f"\nResults saved: {json_path}")
    return summary


# ============================================================================
# Summary Report Generator
# ============================================================================

def generate_summary_report(all_summaries: Dict[int, Dict], output_dir: Path):
    """Generate comprehensive summary report."""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE SUMMARY REPORT")
    print("="*70)
    
    # Create summary markdown
    report_lines = [
        "# Financial Document AI Pipeline - Comprehensive Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report presents evaluation results for all pipeline stages:",
        "",
    ]
    
    for step, summary in sorted(all_summaries.items()):
        report_lines.append(f"- **Step {step}:** {summary['task']}")
    
    report_lines.extend(["", "---", ""])
    
    # Step-by-step results
    for step, summary in sorted(all_summaries.items()):
        report_lines.extend([
            f"## Step {step}: {summary['task']}",
            "",
            f"**Dataset:** {summary['dataset']} ({summary.get('split', 'N/A')} split)",
            f"**Samples:** {summary.get('num_samples', summary.get('num_tables', 'N/A'))}",
            f"**Ground Truth:** {summary.get('ground_truth', 'N/A')}",
            f"**Metrics:** {summary.get('evaluation_metric', 'N/A')}",
            "",
            "### Results",
            "",
        ])
        
        # Format results table
        results = summary.get('results', {})
        if results:
            # Determine table format based on step
            if step == 1:  # Detection
                report_lines.extend([
                    "| Method | Precision | Recall | F1 | Avg IoU |",
                    "|--------|-----------|--------|-----|---------|",
                ])
                for name, m in results.items():
                    report_lines.append(
                        f"| {name} | {m.get('precision', 'N/A')} | {m.get('recall', 'N/A')} | "
                        f"{m.get('f1', 'N/A')} | {m.get('avg_iou', 'N/A')} |"
                    )
            elif step == 2:  # TSR
                report_lines.extend([
                    "| Method | Mean TEDS | Std | Median | Min | Max |",
                    "|--------|-----------|-----|--------|-----|-----|",
                ])
                for name, m in results.items():
                    report_lines.append(
                        f"| {name} | {m.get('mean_teds', 'N/A')} | {m.get('std_teds', 'N/A')} | "
                        f"{m.get('median_teds', 'N/A')} | {m.get('min_teds', 'N/A')} | {m.get('max_teds', 'N/A')} |"
                    )
            elif step == 3:  # OCR
                report_lines.extend([
                    "| Method | Exact Match % | CER % | WER % | Accuracy % |",
                    "|--------|---------------|-------|-------|------------|",
                ])
                for name, m in results.items():
                    report_lines.append(
                        f"| {name} | {m.get('exact_match_rate', 'N/A')} | {m.get('avg_cer', 'N/A')} | "
                        f"{m.get('avg_wer', 'N/A')} | {m.get('accuracy', 'N/A')} |"
                    )
            elif step == 4:  # Numeric
                report_lines.extend([
                    "| Metric | Value |",
                    "|--------|-------|",
                ])
                for key, val in results.items():
                    report_lines.append(f"| {key} | {val} |")
            elif step == 5:  # Semantic
                report_lines.extend([
                    "| Classifier | Accuracy % | Macro F1 % |",
                    "|------------|------------|------------|",
                ])
                for name, m in results.items():
                    report_lines.append(
                        f"| {name} | {m.get('accuracy', 'N/A')} | {m.get('macro_f1', 'N/A')} |"
                    )
        
        report_lines.extend(["", f"**Histogram:** See `{summary.get('histogram_path', 'N/A')}`", "", "---", ""])
    
    # Save report
    report_path = output_dir / "EVALUATION_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved: {report_path}")
    
    # Save combined JSON
    combined_json = output_dir / "all_results_combined.json"
    with open(combined_json, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"Combined results: {combined_json}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run All Pipeline Benchmarks')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples per step')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                        help='Run only specific step')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all steps')
    parser.add_argument('--skip-steps', type=int, nargs='+', default=[],
                        help='Steps to skip')
    
    args = parser.parse_args()
    
    # Setup
    output_dir, figures_dir = setup_output_dirs()
    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {figures_dir}")
    
    all_summaries = {}
    
    # Determine which steps to run
    if args.step:
        steps_to_run = [args.step]
    elif args.run_all:
        steps_to_run = [1, 2, 3, 4, 5]
    else:
        print("Please specify --run-all or --step N")
        return
    
    steps_to_run = [s for s in steps_to_run if s not in args.skip_steps]
    
    print(f"\nWill run steps: {steps_to_run}")
    print(f"Samples per step: {args.num_samples}")
    
    # Run each step
    step_functions = {
        1: run_step1_detection,
        2: run_step2_tsr,
        3: run_step3_ocr,
        4: run_step4_numeric,
        5: run_step5_semantic,
    }
    
    for step in steps_to_run:
        try:
            summary = step_functions[step](args.num_samples, output_dir)
            all_summaries[step] = summary
        except Exception as e:
            print(f"\nError in Step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    if all_summaries:
        generate_summary_report(all_summaries, output_dir)
    
    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
