"""
Run All Baseline Experiments

Executes all baseline experiments and generates a summary report.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import json

from modules.utils import load_config, ensure_dir, save_results

from baseline_pubtabnet import run_pubtabnet_baseline
from baseline_pubtables1m import run_pubtables1m_baseline
from baseline_fintabnet import run_fintabnet_baseline
from baseline_doclaynet import run_doclaynet_baseline


def run_all_baselines(config_path: str = "configs/config.yaml",
                      num_samples: int = 100):
    """
    Run all baseline experiments and generate summary.
    """
    print("=" * 70)
    print("FINANCIAL TABLE UNDERSTANDING - BASELINE EXPERIMENTS")
    print("=" * 70)
    
    config = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'timestamp': timestamp,
        'num_samples': num_samples,
        'experiments': {}
    }
    
    # 1. PubTabNet Baseline (IBM - HTML/TEDS)
    print("\n" + "=" * 70)
    print("[1/4] PUBTABNET BASELINE (IBM - TEDS)")
    print("=" * 70)
    try:
        teds, teds_struct = run_pubtabnet_baseline(config_path, num_samples, save_outputs=True)
        summary['experiments']['pubtabnet'] = {
            'status': 'success',
            'teds_mean': teds['teds_mean'],
            'teds_structure_only_mean': teds_struct['teds_mean']
        }
    except Exception as e:
        print(f"Error in PubTabNet baseline: {e}")
        summary['experiments']['pubtabnet'] = {'status': 'failed', 'error': str(e)}
    
    # 2. PubTables-1M Baseline (Microsoft - Structure)
    print("\n" + "=" * 70)
    print("[2/4] PUBTABLES-1M BASELINE (Microsoft - Structure)")
    print("=" * 70)
    try:
        metrics = run_pubtables1m_baseline(config_path, num_samples, save_outputs=True)
        if metrics:
            summary['experiments']['pubtables1m'] = {
                'status': 'success',
                'row_f1_iou50': metrics['iou_50']['rows']['f1'],
                'column_f1_iou50': metrics['iou_50']['columns']['f1'],
                'row_f1_iou75': metrics['iou_75']['rows']['f1'],
                'column_f1_iou75': metrics['iou_75']['columns']['f1']
            }
        else:
            summary['experiments']['pubtables1m'] = {'status': 'skipped', 'reason': 'No data loaded'}
    except Exception as e:
        print(f"Error in PubTables-1M baseline: {e}")
        summary['experiments']['pubtables1m'] = {'status': 'failed', 'error': str(e)}
    
    # 3. FinTabNet Baseline (Financial Tables)
    print("\n" + "=" * 70)
    print("[3/4] FINTABNET BASELINE (Financial Tables)")
    print("=" * 70)
    try:
        metrics = run_fintabnet_baseline(config_path, num_samples, save_outputs=True)
        summary['experiments']['fintabnet'] = {
            'status': 'success',
            'row_f1': metrics['rows']['f1'],
            'column_f1': metrics['columns']['f1']
        }
    except Exception as e:
        print(f"Error in FinTabNet baseline: {e}")
        summary['experiments']['fintabnet'] = {'status': 'failed', 'error': str(e)}
    
    # 4. DocLayNet Baseline (Document Layout - Table Detection)
    print("\n" + "=" * 70)
    print("[4/4] DOCLAYNET BASELINE (Table Detection)")
    print("=" * 70)
    try:
        metrics = run_doclaynet_baseline(config_path, num_samples, save_outputs=True)
        summary['experiments']['doclaynet'] = {
            'status': 'success',
            'precision_50': metrics['iou_0.5']['precision'],
            'recall_50': metrics['iou_0.5']['recall'],
            'f1_50': metrics['iou_0.5']['f1']
        }
    except Exception as e:
        print(f"Error in DocLayNet baseline: {e}")
        summary['experiments']['doclaynet'] = {'status': 'failed', 'error': str(e)}
    
    # Generate Summary Report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\nExperiments run at: {timestamp}")
    print(f"Samples per dataset: {num_samples}")
    
    print("\n" + "-" * 50)
    print("RESULTS:")
    print("-" * 50)
    
    if summary['experiments'].get('pubtabnet', {}).get('status') == 'success':
        print(f"\n[PubTabNet - TEDS (IBM)]")
        print(f"  TEDS (with content):   {summary['experiments']['pubtabnet']['teds_mean']:.4f}")
        print(f"  TEDS (structure only): {summary['experiments']['pubtabnet']['teds_structure_only_mean']:.4f}")
    
    if summary['experiments'].get('pubtables1m', {}).get('status') == 'success':
        print(f"\n[PubTables-1M - Structure Recognition (Microsoft)]")
        print(f"  Row F1 (IoU=0.5):    {summary['experiments']['pubtables1m']['row_f1_iou50']:.4f}")
        print(f"  Column F1 (IoU=0.5): {summary['experiments']['pubtables1m']['column_f1_iou50']:.4f}")
        print(f"  Row F1 (IoU=0.75):   {summary['experiments']['pubtables1m']['row_f1_iou75']:.4f}")
        print(f"  Column F1 (IoU=0.75):{summary['experiments']['pubtables1m']['column_f1_iou75']:.4f}")
    
    if summary['experiments'].get('fintabnet', {}).get('status') == 'success':
        print(f"\n[FinTabNet - Structure Recognition (Financial)]")
        print(f"  Row F1:    {summary['experiments']['fintabnet']['row_f1']:.4f}")
        print(f"  Column F1: {summary['experiments']['fintabnet']['column_f1']:.4f}")
    
    if summary['experiments'].get('doclaynet', {}).get('status') == 'success':
        print(f"\n[DocLayNet - Table Detection @ IoU=0.5]")
        print(f"  Precision: {summary['experiments']['doclaynet']['precision_50']:.4f}")
        print(f"  Recall:    {summary['experiments']['doclaynet']['recall_50']:.4f}")
        print(f"  F1:        {summary['experiments']['doclaynet']['f1_50']:.4f}")
    
    # Save summary
    output_dir = ensure_dir(config['outputs']['results'])
    summary_path = os.path.join(output_dir, f"baseline_summary_{timestamp}.json")
    save_results(summary, summary_path)
    
    print("\n" + "=" * 70)
    print(f"Summary saved to: {summary_path}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run All Baseline Experiments")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per dataset")
    
    args = parser.parse_args()
    
    run_all_baselines(
        config_path=args.config,
        num_samples=args.num_samples
    )
