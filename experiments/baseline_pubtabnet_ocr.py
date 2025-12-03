"""
Baseline Experiment: PubTabNet TEDS Evaluation with OCR

Evaluates table structure recognition on PubTabNet dataset using TEDS metric.
Includes OCR for cell content extraction to measure full TEDS score.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from modules.utils import load_config, get_device, ensure_dir, save_results
from modules.data_loaders import PubTabNetLoader
from modules.structure import TableStructureRecognizer
from modules.metrics import TEDSEvaluator, calculate_teds
from modules.ocr import TableOCR


def run_pubtabnet_ocr_baseline(config_path: str = "configs/config.yaml", 
                                num_samples: int = 100,
                                save_outputs: bool = True):
    """
    Run baseline experiment on PubTabNet with OCR integration.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to evaluate
        save_outputs: Whether to save results
    """
    print("=" * 60)
    print("PubTabNet Baseline Experiment - TEDS with OCR")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Initialize data loader
    print("\nLoading PubTabNet dataset...")
    loader = PubTabNetLoader(config)
    annotations = loader.load_annotations(split='val', num_samples=num_samples)
    
    # Initialize models
    print("\nLoading Table Structure Recognition model...")
    model = TableStructureRecognizer(config, device=device)
    
    print("\nInitializing OCR module (PaddleOCR)...")
    ocr = TableOCR(lang='en')
    # Warm up OCR by running once
    from PIL import Image
    dummy_img = Image.new('RGB', (100, 100), 'white')
    ocr.extract_text(dummy_img)
    print("OCR module ready!")
    
    # Initialize evaluators
    teds_evaluator = TEDSEvaluator(structure_only=False)
    teds_structure_only = TEDSEvaluator(structure_only=True)
    
    # Also track no-OCR scores for comparison
    teds_no_ocr = TEDSEvaluator(structure_only=False)
    
    # Results storage
    all_results = []
    teds_scores_ocr = []
    teds_scores_no_ocr = []
    teds_structure_scores = []
    
    print(f"\nEvaluating {len(annotations)} samples...")
    
    for idx, ann in enumerate(tqdm(annotations, desc="Processing")):
        try:
            # Load image
            image, annotation = loader.load_sample(ann, split='val')
            
            # Get ground truth HTML
            gt_html = loader.get_html_structure(annotation)
            
            # Run structure recognition
            structure = model.recognize(image)
            
            # Generate HTML without OCR (for comparison)
            pred_html_no_ocr = model.structure_to_html(structure)
            
            # Run OCR and align text to cells using improved hybrid method
            ocr_results = ocr.extract_text(image)
            grid_texts = ocr.align_text_to_grid(
                ocr_results,
                structure['rows'],
                structure['columns'],
                method='hybrid'  # Use improved hybrid alignment
            )
            
            # Generate HTML with OCR content
            pred_html_ocr = model.structure_to_html(structure, cell_texts=grid_texts)
            
            # Calculate TEDS scores
            teds_score_ocr = calculate_teds(pred_html_ocr, gt_html, structure_only=False)
            teds_score_no_ocr = calculate_teds(pred_html_no_ocr, gt_html, structure_only=False)
            teds_struct_score = calculate_teds(pred_html_ocr, gt_html, structure_only=True)
            
            teds_scores_ocr.append(teds_score_ocr)
            teds_scores_no_ocr.append(teds_score_no_ocr)
            teds_structure_scores.append(teds_struct_score)
            
            teds_evaluator.update(pred_html_ocr, gt_html)
            teds_structure_only.update(pred_html_ocr, gt_html)
            teds_no_ocr.update(pred_html_no_ocr, gt_html)
            
            # Count OCR extractions
            num_ocr_texts = len(ocr_results)
            num_filled_cells = sum(1 for row in grid_texts for cell in row if cell)
            
            # Store detailed results
            result = {
                'filename': annotation['filename'],
                'teds_with_ocr': teds_score_ocr,
                'teds_no_ocr': teds_score_no_ocr,
                'teds_structure_only': teds_struct_score,
                'num_rows': len(structure['rows']),
                'num_cols': len(structure['columns']),
                'num_spanning_cells': len(structure['spanning_cells']),
                'num_ocr_texts': num_ocr_texts,
                'num_filled_cells': num_filled_cells
            }
            all_results.append(result)
            
            # Free GPU memory
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError processing {ann['filename']}: {e}")
            continue
    
    # Compute final metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    final_teds_ocr = teds_evaluator.compute()
    final_teds_no_ocr = teds_no_ocr.compute()
    final_teds_struct = teds_structure_only.compute()
    
    print(f"\nTEDS (with OCR content):")
    print(f"  Mean:   {final_teds_ocr['teds_mean']:.4f}")
    print(f"  Std:    {final_teds_ocr['teds_std']:.4f}")
    print(f"  Median: {final_teds_ocr['teds_median']:.4f}")
    print(f"  Min:    {final_teds_ocr['teds_min']:.4f}")
    print(f"  Max:    {final_teds_ocr['teds_max']:.4f}")
    
    print(f"\nTEDS (without OCR - baseline):")
    print(f"  Mean:   {final_teds_no_ocr['teds_mean']:.4f}")
    
    print(f"\nTEDS (structure only):")
    print(f"  Mean:   {final_teds_struct['teds_mean']:.4f}")
    print(f"  Std:    {final_teds_struct['teds_std']:.4f}")
    
    # Calculate improvement
    improvement = final_teds_ocr['teds_mean'] - final_teds_no_ocr['teds_mean']
    print(f"\n📈 OCR Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
    
    # Save results
    if save_outputs:
        output_dir = ensure_dir(config['outputs']['results'])
        figures_dir = ensure_dir(config['outputs']['figures'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"pubtabnet_ocr_baseline_{timestamp}.json")
        save_results({
            'experiment': 'pubtabnet_ocr_baseline',
            'timestamp': timestamp,
            'num_samples': len(all_results),
            'teds_with_ocr': final_teds_ocr,
            'teds_no_ocr': final_teds_no_ocr,
            'teds_structure_only': final_teds_struct,
            'improvement': float(improvement),
            'detailed_results': all_results
        }, results_file)
        
        # Generate comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # TEDS with OCR distribution
        axes[0].hist(teds_scores_ocr, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[0].axvline(final_teds_ocr['teds_mean'], color='red', linestyle='--', 
                       label=f'Mean: {final_teds_ocr["teds_mean"]:.3f}')
        axes[0].set_xlabel('TEDS Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('TEDS (with OCR content)')
        axes[0].legend()
        
        # Comparison: OCR vs No-OCR
        categories = ['With OCR', 'Without OCR', 'Structure Only']
        means = [final_teds_ocr['teds_mean'], final_teds_no_ocr['teds_mean'], final_teds_struct['teds_mean']]
        colors = ['green', 'red', 'blue']
        bars = axes[1].bar(categories, means, color=colors)
        axes[1].bar_label(bars, fmt='%.3f')
        axes[1].set_ylabel('TEDS Score')
        axes[1].set_title('TEDS Comparison')
        axes[1].set_ylim(0, 1.0)
        
        # Scatter: OCR vs No-OCR per sample
        axes[2].scatter(teds_scores_no_ocr, teds_scores_ocr, alpha=0.5, s=10)
        axes[2].plot([0, 1], [0, 1], 'r--', label='No change')
        axes[2].set_xlabel('TEDS (without OCR)')
        axes[2].set_ylabel('TEDS (with OCR)')
        axes[2].set_title('Per-sample TEDS Improvement')
        axes[2].legend()
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"pubtabnet_ocr_comparison_{timestamp}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        print(f"\nResults saved to: {results_file}")
        print(f"Figure saved to: {fig_path}")
    
    return final_teds_ocr, final_teds_no_ocr, final_teds_struct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubTabNet OCR Baseline Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save outputs")
    
    args = parser.parse_args()
    
    run_pubtabnet_ocr_baseline(
        config_path=args.config,
        num_samples=args.num_samples,
        save_outputs=not args.no_save
    )
