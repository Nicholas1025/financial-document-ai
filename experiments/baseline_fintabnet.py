"""
Baseline Experiment: FinTabNet Structure Recognition

Evaluates table structure recognition on FinTabNet (financial tables).
Computes boundary accuracy for rows, columns, and cells.
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
from modules.data_loaders import FinTabNetLoader
from modules.structure import TableStructureRecognizer, StructureEvaluator


def run_fintabnet_baseline(config_path: str = "configs/config.yaml",
                           num_samples: int = 100,
                           save_outputs: bool = True):
    """
    Run baseline experiment on FinTabNet.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to evaluate
        save_outputs: Whether to save results
    """
    print("=" * 60)
    print("FinTabNet Baseline Experiment - Structure Recognition")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Initialize data loader
    print("\nLoading FinTabNet dataset...")
    loader = FinTabNetLoader(config)
    annotations = loader.load_annotations(split='val', num_samples=num_samples)
    
    # Initialize model
    print("\nLoading Table Structure Recognition model...")
    model = TableStructureRecognizer(config, device=device, use_v1_1=True)  # Use v1.1 for complex tables
    
    # Initialize evaluator
    evaluator = StructureEvaluator(iou_threshold=0.5)
    
    # Results storage
    all_results = []
    row_f1_scores = []
    col_f1_scores = []
    
    print(f"\nEvaluating {len(annotations)} samples...")
    
    for idx, ann in enumerate(tqdm(annotations, desc="Processing")):
        try:
            # Load image
            image, annotation = loader.load_sample(ann)
            
            # Get ground truth structure
            gt_structure = loader.get_structure_elements(annotation)
            
            # Convert to evaluator format
            gt_for_eval = {
                'rows': [{'bbox': bbox} for bbox in gt_structure['rows']],
                'columns': [{'bbox': bbox} for bbox in gt_structure['columns']]
            }
            
            # Run structure recognition
            pred_structure = model.recognize(image)
            
            # Update evaluator
            evaluator.update(pred_structure, gt_for_eval)
            
            # Store detailed results
            result = {
                'filename': annotation['filename'],
                'gt_rows': len(gt_structure['rows']),
                'gt_cols': len(gt_structure['columns']),
                'pred_rows': len(pred_structure['rows']),
                'pred_cols': len(pred_structure['columns']),
                'gt_spanning_cells': len(gt_structure['spanning_cells']),
                'pred_spanning_cells': len(pred_structure['spanning_cells'])
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
    
    metrics = evaluator.compute()
    
    print(f"\nRow Detection:")
    print(f"  Precision: {metrics['rows']['precision']:.4f}")
    print(f"  Recall:    {metrics['rows']['recall']:.4f}")
    print(f"  F1:        {metrics['rows']['f1']:.4f}")
    
    print(f"\nColumn Detection:")
    print(f"  Precision: {metrics['columns']['precision']:.4f}")
    print(f"  Recall:    {metrics['columns']['recall']:.4f}")
    print(f"  F1:        {metrics['columns']['f1']:.4f}")
    
    # Calculate statistics
    gt_rows = [r['gt_rows'] for r in all_results]
    pred_rows = [r['pred_rows'] for r in all_results]
    gt_cols = [r['gt_cols'] for r in all_results]
    pred_cols = [r['pred_cols'] for r in all_results]
    
    print(f"\nRow Count Statistics:")
    print(f"  GT Mean:   {np.mean(gt_rows):.2f}")
    print(f"  Pred Mean: {np.mean(pred_rows):.2f}")
    
    print(f"\nColumn Count Statistics:")
    print(f"  GT Mean:   {np.mean(gt_cols):.2f}")
    print(f"  Pred Mean: {np.mean(pred_cols):.2f}")
    
    # Save results
    if save_outputs:
        output_dir = ensure_dir(config['outputs']['results'])
        figures_dir = ensure_dir(config['outputs']['figures'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"fintabnet_baseline_{timestamp}.json")
        save_results({
            'experiment': 'fintabnet_baseline',
            'timestamp': timestamp,
            'num_samples': len(all_results),
            'metrics': metrics,
            'statistics': {
                'gt_rows_mean': float(np.mean(gt_rows)),
                'pred_rows_mean': float(np.mean(pred_rows)),
                'gt_cols_mean': float(np.mean(gt_cols)),
                'pred_cols_mean': float(np.mean(pred_cols))
            },
            'detailed_results': all_results
        }, results_file)
        
        # Generate figures
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Row count comparison
        axes[0, 0].scatter(gt_rows, pred_rows, alpha=0.5)
        max_val = max(max(gt_rows), max(pred_rows)) + 1
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[0, 0].set_xlabel('Ground Truth Row Count')
        axes[0, 0].set_ylabel('Predicted Row Count')
        axes[0, 0].set_title('Row Count: GT vs Predicted')
        axes[0, 0].legend()
        
        # Column count comparison
        axes[0, 1].scatter(gt_cols, pred_cols, alpha=0.5, color='green')
        max_val = max(max(gt_cols), max(pred_cols)) + 1
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[0, 1].set_xlabel('Ground Truth Column Count')
        axes[0, 1].set_ylabel('Predicted Column Count')
        axes[0, 1].set_title('Column Count: GT vs Predicted')
        axes[0, 1].legend()
        
        # Row count distribution
        axes[1, 0].hist(gt_rows, bins=15, alpha=0.5, label='Ground Truth')
        axes[1, 0].hist(pred_rows, bins=15, alpha=0.5, label='Predicted')
        axes[1, 0].set_xlabel('Row Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Row Count Distribution')
        axes[1, 0].legend()
        
        # Column count distribution
        axes[1, 1].hist(gt_cols, bins=15, alpha=0.5, label='Ground Truth')
        axes[1, 1].hist(pred_cols, bins=15, alpha=0.5, label='Predicted')
        axes[1, 1].set_xlabel('Column Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Column Count Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"fintabnet_structure_analysis_{timestamp}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        # Generate metrics bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(3)
        width = 0.35
        
        row_vals = [metrics['rows']['precision'], metrics['rows']['recall'], metrics['rows']['f1']]
        col_vals = [metrics['columns']['precision'], metrics['columns']['recall'], metrics['columns']['f1']]
        
        # Professional colors
        color_rows = '#4A90E2'  # Soft Blue
        color_cols = '#50E3C2'  # Teal/Green
        
        bars1 = ax.bar(x - width/2, row_vals, width, label='Rows', color=color_rows, edgecolor='white', linewidth=1, zorder=3)
        bars2 = ax.bar(x + width/2, col_vals, width, label='Columns', color=color_cols, edgecolor='white', linewidth=1, zorder=3)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('FinTabNet Structure Recognition Performance', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(['Precision', 'Recall', 'F1'], fontsize=12)
        
        # Clean up chart
        ax.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=True)
        ax.set_ylim(0, 1.15)  # More space for labels
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        metrics_fig_path = os.path.join(figures_dir, f"fintabnet_metrics_{timestamp}.png")
        plt.savefig(metrics_fig_path, dpi=300)
        plt.close()
        
        print(f"\nResults saved to: {results_file}")
        print(f"Figures saved to: {figures_dir}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinTabNet Baseline Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save outputs")
    
    args = parser.parse_args()
    
    run_fintabnet_baseline(
        config_path=args.config,
        num_samples=args.num_samples,
        save_outputs=not args.no_save
    )
