"""
Baseline Experiment: PubTabNet TEDS Evaluation

Evaluates table structure recognition on PubTabNet dataset using TEDS metric.
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

from modules.utils import load_config, get_device, ensure_dir, save_results
from modules.data_loaders import PubTabNetLoader
from modules.structure import TableStructureRecognizer
from modules.metrics import TEDSEvaluator, calculate_teds


def run_pubtabnet_baseline(config_path: str = "configs/config.yaml", 
                           num_samples: int = 100,
                           save_outputs: bool = True):
    """
    Run baseline experiment on PubTabNet.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to evaluate
        save_outputs: Whether to save results
    """
    print("=" * 60)
    print("PubTabNet Baseline Experiment - TEDS Evaluation")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Initialize data loader
    print("\nLoading PubTabNet dataset...")
    loader = PubTabNetLoader(config)
    annotations = loader.load_annotations(split='val', num_samples=num_samples)
    
    # Initialize model
    print("\nLoading Table Structure Recognition model...")
    model = TableStructureRecognizer(config, device=device)
    
    # Initialize evaluator
    teds_evaluator = TEDSEvaluator(structure_only=False)
    teds_structure_only = TEDSEvaluator(structure_only=True)
    
    # Results storage
    all_results = []
    teds_scores = []
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
            
            # Convert to HTML
            pred_html = model.structure_to_html(structure)
            
            # Calculate TEDS
            teds_score = calculate_teds(pred_html, gt_html, structure_only=False)
            teds_struct_score = calculate_teds(pred_html, gt_html, structure_only=True)
            
            teds_scores.append(teds_score)
            teds_structure_scores.append(teds_struct_score)
            
            teds_evaluator.update(pred_html, gt_html)
            teds_structure_only.update(pred_html, gt_html)
            
            # Store detailed results
            result = {
                'filename': annotation['filename'],
                'teds': teds_score,
                'teds_structure_only': teds_struct_score,
                'num_rows': len(structure['rows']),
                'num_cols': len(structure['columns']),
                'num_spanning_cells': len(structure['spanning_cells'])
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
    
    final_teds = teds_evaluator.compute()
    final_teds_struct = teds_structure_only.compute()
    
    print(f"\nTEDS (with content):")
    print(f"  Mean:   {final_teds['teds_mean']:.4f}")
    print(f"  Std:    {final_teds['teds_std']:.4f}")
    print(f"  Median: {final_teds['teds_median']:.4f}")
    print(f"  Min:    {final_teds['teds_min']:.4f}")
    print(f"  Max:    {final_teds['teds_max']:.4f}")
    
    print(f"\nTEDS (structure only):")
    print(f"  Mean:   {final_teds_struct['teds_mean']:.4f}")
    print(f"  Std:    {final_teds_struct['teds_std']:.4f}")
    print(f"  Median: {final_teds_struct['teds_median']:.4f}")
    
    # Save results
    if save_outputs:
        output_dir = ensure_dir(config['outputs']['results'])
        figures_dir = ensure_dir(config['outputs']['figures'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"pubtabnet_baseline_{timestamp}.json")
        save_results({
            'experiment': 'pubtabnet_baseline',
            'timestamp': timestamp,
            'num_samples': len(all_results),
            'teds_metrics': final_teds,
            'teds_structure_only_metrics': final_teds_struct,
            'detailed_results': all_results
        }, results_file)
        
        # Generate figures
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # TEDS distribution
        axes[0].hist(teds_scores, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(final_teds['teds_mean'], color='red', linestyle='--', 
                       label=f'Mean: {final_teds["teds_mean"]:.3f}')
        axes[0].set_xlabel('TEDS Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('TEDS Score Distribution (with content)')
        axes[0].legend()
        
        # TEDS structure only distribution
        axes[1].hist(teds_structure_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(final_teds_struct['teds_mean'], color='red', linestyle='--',
                       label=f'Mean: {final_teds_struct["teds_mean"]:.3f}')
        axes[1].set_xlabel('TEDS Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('TEDS Score Distribution (structure only)')
        axes[1].legend()
        
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"pubtabnet_teds_distribution_{timestamp}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        print(f"\nResults saved to: {results_file}")
        print(f"Figure saved to: {fig_path}")
    
    return final_teds, final_teds_struct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubTabNet Baseline Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save outputs")
    
    args = parser.parse_args()
    
    run_pubtabnet_baseline(
        config_path=args.config,
        num_samples=args.num_samples,
        save_outputs=not args.no_save
    )
