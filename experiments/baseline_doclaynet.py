"""
Baseline Experiment: DocLayNet Table Detection

Evaluates table detection on DocLayNet dataset.
Computes Precision, Recall, F1, and mAP.
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
from modules.data_loaders import DocLayNetLoader
from modules.detection import TableDetector, TableDetectionEvaluator


def run_doclaynet_baseline(config_path: str = "configs/config.yaml",
                           num_samples: int = 100,
                           save_outputs: bool = True):
    """
    Run baseline experiment on DocLayNet.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to evaluate
        save_outputs: Whether to save results
    """
    print("=" * 60)
    print("DocLayNet Baseline Experiment - Table Detection")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Initialize data loader
    print("\nLoading DocLayNet dataset...")
    loader = DocLayNetLoader(config)
    annotations = loader.get_table_annotations(split='val', num_samples=num_samples)
    
    # Initialize model
    print("\nLoading Table Detection model...")
    detector = TableDetector(config, device=device)
    
    # Initialize evaluator for multiple IoU thresholds
    iou_thresholds = [0.5, 0.75, 0.9]
    evaluators = {iou: TableDetectionEvaluator(iou_threshold=iou) for iou in iou_thresholds}
    
    # Results storage
    all_results = []
    all_scores = []
    
    print(f"\nEvaluating {len(annotations)} images with tables...")
    
    for idx, ann in enumerate(tqdm(annotations, desc="Processing")):
        try:
            # Load image
            image, annotation = loader.load_sample(ann)
            
            # Get ground truth boxes
            gt_boxes = [table['bbox'] for table in annotation['tables']]
            
            # Run detection
            detections = detector.detect(image)
            pred_boxes = detections['boxes'].tolist()
            scores = detections['scores'].tolist()
            
            all_scores.extend(scores)
            
            # Update evaluators
            for iou, evaluator in evaluators.items():
                evaluator.update(pred_boxes, gt_boxes)
            
            # Store detailed results
            result = {
                'filename': annotation['filename'],
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'pred_scores': scores
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
    
    all_metrics = {}
    for iou, evaluator in evaluators.items():
        metrics = evaluator.compute()
        all_metrics[f'iou_{iou}'] = metrics
        
        print(f"\nIoU Threshold: {iou}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Avg IoU:   {metrics['avg_iou']:.4f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    # Calculate AP@50
    main_metrics = all_metrics['iou_0.5']
    
    # Statistics
    gt_counts = [r['gt_count'] for r in all_results]
    pred_counts = [r['pred_count'] for r in all_results]
    
    print(f"\nDetection Statistics:")
    print(f"  Total GT tables:   {sum(gt_counts)}")
    print(f"  Total Pred tables: {sum(pred_counts)}")
    print(f"  Avg GT per image:  {np.mean(gt_counts):.2f}")
    print(f"  Avg Pred per image: {np.mean(pred_counts):.2f}")
    print(f"  Avg confidence:    {np.mean(all_scores):.4f}" if all_scores else "  No detections")
    
    # Save results
    if save_outputs:
        output_dir = ensure_dir(config['outputs']['results'])
        figures_dir = ensure_dir(config['outputs']['figures'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"doclaynet_baseline_{timestamp}.json")
        save_results({
            'experiment': 'doclaynet_baseline',
            'timestamp': timestamp,
            'num_samples': len(all_results),
            'metrics': all_metrics,
            'statistics': {
                'total_gt_tables': int(sum(gt_counts)),
                'total_pred_tables': int(sum(pred_counts)),
                'avg_gt_per_image': float(np.mean(gt_counts)),
                'avg_pred_per_image': float(np.mean(pred_counts)),
                'avg_confidence': float(np.mean(all_scores)) if all_scores else 0
            },
            'detailed_results': all_results
        }, results_file)
        
        # Generate figures
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Precision-Recall at different IoU thresholds
        ious = list(evaluators.keys())
        precisions = [all_metrics[f'iou_{iou}']['precision'] for iou in ious]
        recalls = [all_metrics[f'iou_{iou}']['recall'] for iou in ious]
        f1s = [all_metrics[f'iou_{iou}']['f1'] for iou in ious]
        
        x = np.arange(len(ious))
        width = 0.25
        
        axes[0, 0].bar(x - width, precisions, width, label='Precision')
        axes[0, 0].bar(x, recalls, width, label='Recall')
        axes[0, 0].bar(x + width, f1s, width, label='F1')
        axes[0, 0].set_xlabel('IoU Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Detection Performance at Different IoU Thresholds')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([str(iou) for iou in ious])
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.0)
        
        # Confidence score distribution
        if all_scores:
            axes[0, 1].hist(all_scores, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(np.mean(all_scores), color='red', linestyle='--',
                              label=f'Mean: {np.mean(all_scores):.3f}')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Detection Confidence Distribution')
            axes[0, 1].legend()
        
        # GT vs Predicted count
        axes[1, 0].scatter(gt_counts, pred_counts, alpha=0.5)
        max_val = max(max(gt_counts), max(pred_counts)) + 1
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[1, 0].set_xlabel('Ground Truth Table Count')
        axes[1, 0].set_ylabel('Predicted Table Count')
        axes[1, 0].set_title('Table Count: GT vs Predicted')
        axes[1, 0].legend()
        
        # Count distribution
        axes[1, 1].hist(gt_counts, bins=10, alpha=0.5, label='Ground Truth')
        axes[1, 1].hist(pred_counts, bins=10, alpha=0.5, label='Predicted')
        axes[1, 1].set_xlabel('Table Count per Image')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Table Count Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"doclaynet_detection_analysis_{timestamp}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        # Generate confusion matrix style summary
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics_50 = all_metrics['iou_0.5']
        labels = ['True Positive', 'False Positive', 'False Negative']
        values = [metrics_50['tp'], metrics_50['fp'], metrics_50['fn']]
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(labels, values, color=colors, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Detection Results Summary (IoU=0.5)')
        
        for bar, val in zip(bars, values):
            ax.annotate(str(val),
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        summary_path = os.path.join(figures_dir, f"doclaynet_detection_summary_{timestamp}.png")
        plt.savefig(summary_path, dpi=150)
        plt.close()
        
        print(f"\nResults saved to: {results_file}")
        print(f"Figures saved to: {figures_dir}")
    
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocLayNet Baseline Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save outputs")
    
    args = parser.parse_args()
    
    run_doclaynet_baseline(
        config_path=args.config,
        num_samples=args.num_samples,
        save_outputs=not args.no_save
    )
