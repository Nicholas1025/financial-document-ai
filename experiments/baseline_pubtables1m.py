"""
Baseline Experiment: PubTables-1M Structure Recognition

Evaluates table structure recognition on PubTables-1M dataset (Microsoft).
This is different from PubTabNet (IBM) - uses XML annotations with row/column bboxes.
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
from modules.data_loaders import PubTables1MLoader
from modules.structure import TableStructureRecognizer, StructureEvaluator


def run_pubtables1m_baseline(config_path: str = "configs/config.yaml",
                              num_samples: int = 100,
                              save_outputs: bool = True):
    """
    Run baseline experiment on PubTables-1M.
    
    PubTables-1M (Microsoft) provides:
    - XML annotations with bounding boxes for rows, columns, cells
    - More robust structure annotations than PubTabNet
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to evaluate
        save_outputs: Whether to save results
    """
    print("=" * 60)
    print("PubTables-1M Baseline Experiment - Structure Recognition")
    print("(Microsoft Research Dataset)")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Initialize data loader
    print("\nLoading PubTables-1M dataset...")
    loader = PubTables1MLoader(config)
    annotations = loader.load_annotations(split='test', num_samples=num_samples)
    
    if len(annotations) == 0:
        print("ERROR: No annotations loaded. Check dataset path in config.yaml")
        return None
    
    # Initialize model
    print("\nLoading Table Structure Recognition model...")
    model = TableStructureRecognizer(config, device=device, use_v1_1=True)
    
    # Initialize evaluator
    evaluator = StructureEvaluator(iou_threshold=0.5)
    evaluator_strict = StructureEvaluator(iou_threshold=0.75)
    
    # Results storage
    all_results = []
    
    print(f"\nEvaluating {len(annotations)} samples...")
    
    for idx, ann in enumerate(tqdm(annotations, desc="Processing")):
        try:
            # Load image
            image, annotation = loader.load_sample(ann)
            
            # Get ground truth structure from XML
            gt_objects = annotation['objects']
            gt_structure = {
                'rows': [{'bbox': obj['bbox']} for obj in gt_objects if obj['name'] == 'table row'],
                'columns': [{'bbox': obj['bbox']} for obj in gt_objects if obj['name'] == 'table column'],
                'spanning_cells': [{'bbox': obj['bbox']} for obj in gt_objects if obj['name'] == 'table spanning cell']
            }
            
            # Run structure recognition
            pred_structure = model.recognize(image)
            
            # Update evaluators
            evaluator.update(pred_structure, gt_structure)
            evaluator_strict.update(pred_structure, gt_structure)
            
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
    
    metrics_50 = evaluator.compute()
    metrics_75 = evaluator_strict.compute()
    
    print(f"\n[IoU Threshold = 0.5]")
    print(f"Row Detection:")
    print(f"  Precision: {metrics_50['rows']['precision']:.4f}")
    print(f"  Recall:    {metrics_50['rows']['recall']:.4f}")
    print(f"  F1:        {metrics_50['rows']['f1']:.4f}")
    
    print(f"\nColumn Detection:")
    print(f"  Precision: {metrics_50['columns']['precision']:.4f}")
    print(f"  Recall:    {metrics_50['columns']['recall']:.4f}")
    print(f"  F1:        {metrics_50['columns']['f1']:.4f}")
    
    print(f"\n[IoU Threshold = 0.75 (Strict)]")
    print(f"Row Detection:")
    print(f"  Precision: {metrics_75['rows']['precision']:.4f}")
    print(f"  Recall:    {metrics_75['rows']['recall']:.4f}")
    print(f"  F1:        {metrics_75['rows']['f1']:.4f}")
    
    print(f"\nColumn Detection:")
    print(f"  Precision: {metrics_75['columns']['precision']:.4f}")
    print(f"  Recall:    {metrics_75['columns']['recall']:.4f}")
    print(f"  F1:        {metrics_75['columns']['f1']:.4f}")
    
    # Statistics
    if all_results:
        gt_rows = [r['gt_rows'] for r in all_results]
        pred_rows = [r['pred_rows'] for r in all_results]
        gt_cols = [r['gt_cols'] for r in all_results]
        pred_cols = [r['pred_cols'] for r in all_results]
        
        print(f"\nRow Count Statistics:")
        print(f"  GT Mean:   {np.mean(gt_rows):.2f} ± {np.std(gt_rows):.2f}")
        print(f"  Pred Mean: {np.mean(pred_rows):.2f} ± {np.std(pred_rows):.2f}")
        
        print(f"\nColumn Count Statistics:")
        print(f"  GT Mean:   {np.mean(gt_cols):.2f} ± {np.std(gt_cols):.2f}")
        print(f"  Pred Mean: {np.mean(pred_cols):.2f} ± {np.std(pred_cols):.2f}")
    
    # Save results
    if save_outputs and all_results:
        output_dir = ensure_dir(config['outputs']['results'])
        figures_dir = ensure_dir(config['outputs']['figures'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"pubtables1m_baseline_{timestamp}.json")
        save_results({
            'experiment': 'pubtables1m_baseline',
            'dataset': 'PubTables-1M (Microsoft)',
            'timestamp': timestamp,
            'num_samples': len(all_results),
            'metrics_iou50': metrics_50,
            'metrics_iou75': metrics_75,
            'statistics': {
                'gt_rows_mean': float(np.mean(gt_rows)),
                'gt_rows_std': float(np.std(gt_rows)),
                'pred_rows_mean': float(np.mean(pred_rows)),
                'pred_rows_std': float(np.std(pred_rows)),
                'gt_cols_mean': float(np.mean(gt_cols)),
                'gt_cols_std': float(np.std(gt_cols)),
                'pred_cols_mean': float(np.mean(pred_cols)),
                'pred_cols_std': float(np.std(pred_cols))
            },
            'detailed_results': all_results
        }, results_file)
        
        # Generate figures
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Row count comparison
        axes[0, 0].scatter(gt_rows, pred_rows, alpha=0.5, c='steelblue')
        max_val = max(max(gt_rows), max(pred_rows)) + 1
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[0, 0].set_xlabel('Ground Truth Row Count')
        axes[0, 0].set_ylabel('Predicted Row Count')
        axes[0, 0].set_title('PubTables-1M: Row Count Comparison')
        axes[0, 0].legend()
        
        # Column count comparison
        axes[0, 1].scatter(gt_cols, pred_cols, alpha=0.5, c='forestgreen')
        max_val = max(max(gt_cols), max(pred_cols)) + 1
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[0, 1].set_xlabel('Ground Truth Column Count')
        axes[0, 1].set_ylabel('Predicted Column Count')
        axes[0, 1].set_title('PubTables-1M: Column Count Comparison')
        axes[0, 1].legend()
        
        # Metrics comparison at different IoU thresholds
        x = np.arange(3)
        width = 0.35
        
        row_50 = [metrics_50['rows']['precision'], metrics_50['rows']['recall'], metrics_50['rows']['f1']]
        row_75 = [metrics_75['rows']['precision'], metrics_75['rows']['recall'], metrics_75['rows']['f1']]
        
        axes[1, 0].bar(x - width/2, row_50, width, label='IoU=0.5', color='steelblue')
        axes[1, 0].bar(x + width/2, row_75, width, label='IoU=0.75', color='lightsteelblue')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Row Detection Performance')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Precision', 'Recall', 'F1'])
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1.0)
        
        col_50 = [metrics_50['columns']['precision'], metrics_50['columns']['recall'], metrics_50['columns']['f1']]
        col_75 = [metrics_75['columns']['precision'], metrics_75['columns']['recall'], metrics_75['columns']['f1']]
        
        axes[1, 1].bar(x - width/2, col_50, width, label='IoU=0.5', color='forestgreen')
        axes[1, 1].bar(x + width/2, col_75, width, label='IoU=0.75', color='lightgreen')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Column Detection Performance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Precision', 'Recall', 'F1'])
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1.0)
        
        plt.suptitle('PubTables-1M Baseline Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"pubtables1m_structure_analysis_{timestamp}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        print(f"\nResults saved to: {results_file}")
        print(f"Figure saved to: {fig_path}")
    
    return {'iou_50': metrics_50, 'iou_75': metrics_75}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubTables-1M Baseline Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save outputs")
    
    args = parser.parse_args()
    
    run_pubtables1m_baseline(
        config_path=args.config,
        num_samples=args.num_samples,
        save_outputs=not args.no_save
    )
