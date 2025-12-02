"""
Generate Chapter 4 Figures for FYP Thesis

Creates comprehensive visualization of baseline experiment results across all datasets.
"""
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import load_config, ensure_dir


def load_latest_results(results_dir: str):
    """Load the latest results for each dataset."""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Determine dataset from filename
            if 'doclaynet' in filename:
                results['doclaynet'] = data
            elif 'pubtables1m' in filename:
                results['pubtables1m'] = data
            elif 'fintabnet' in filename:
                results['fintabnet'] = data
            elif 'pubtabnet' in filename:
                results['pubtabnet'] = data
    
    return results


def create_detection_comparison(results: dict, output_dir: str):
    """Create table detection comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # DocLayNet detection results at different IoU thresholds
    if 'doclaynet' in results:
        doclaynet = results['doclaynet']
        metrics_data = doclaynet.get('metrics', doclaynet)  # Handle both formats
        
        thresholds = ['IoU@0.5', 'IoU@0.75', 'IoU@0.9']
        metrics = ['precision', 'recall', 'f1']
        
        x = np.arange(len(thresholds))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = []
            for thresh_key in ['iou_0.5', 'iou_0.75', 'iou_0.9']:
                if thresh_key in metrics_data:
                    values.append(metrics_data[thresh_key].get(metric, 0))
                else:
                    values.append(0)
            
            bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
            ax.bar_label(bars, fmt='%.2f', fontsize=8)
        
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('Score')
        ax.set_title('DocLayNet: Table Detection Performance at Different IoU Thresholds')
        ax.set_xticks(x + width)
        ax.set_xticklabels(thresholds)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'chapter4_detection_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def create_structure_comparison(results: dict, output_dir: str):
    """Create structure recognition comparison across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = []
    row_f1 = []
    col_f1 = []
    
    # PubTables-1M - format: metrics_iou50.rows.f1
    if 'pubtables1m' in results:
        pt1m = results['pubtables1m']
        metrics = pt1m.get('metrics_iou50', {})
        datasets.append('PubTables-1M')
        row_f1.append(metrics.get('rows', {}).get('f1', 0))
        col_f1.append(metrics.get('columns', {}).get('f1', 0))
    
    # FinTabNet - format: metrics.rows.f1
    if 'fintabnet' in results:
        fin = results['fintabnet']
        metrics = fin.get('metrics', {})
        datasets.append('FinTabNet')
        row_f1.append(metrics.get('rows', {}).get('f1', 0))
        col_f1.append(metrics.get('columns', {}).get('f1', 0))
    
    if datasets:
        x = np.arange(len(datasets))
        width = 0.35
        
        # Row Detection
        bars1 = axes[0].bar(x - width/2, row_f1, width, label='Row F1', color='steelblue')
        bars2 = axes[0].bar(x + width/2, col_f1, width, label='Column F1', color='coral')
        
        axes[0].bar_label(bars1, fmt='%.3f', fontsize=10)
        axes[0].bar_label(bars2, fmt='%.3f', fontsize=10)
        
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('Structure Recognition: Row & Column Detection (IoU@0.5)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(datasets)
        axes[0].legend()
        axes[0].set_ylim(0, 1.15)
        axes[0].grid(axis='y', alpha=0.3)
    
    # TEDS scores for PubTabNet - format: teds_metrics.teds_mean, teds_structure_only_metrics.teds_mean
    if 'pubtabnet' in results:
        pt = results['pubtabnet']
        teds_data = pt.get('teds_metrics', {})
        teds_struct_data = pt.get('teds_structure_only_metrics', {})
        
        categories = ['TEDS\n(with content)', 'TEDS\n(structure only)']
        means = [teds_data.get('teds_mean', 0), teds_struct_data.get('teds_mean', 0)]
        stds = [teds_data.get('teds_std', 0), teds_struct_data.get('teds_std', 0)]
        
        bars = axes[1].bar(categories, means, yerr=stds, capsize=5, color=['#2ecc71', '#3498db'])
        axes[1].bar_label(bars, fmt='%.3f', fontsize=10)
        
        axes[1].set_ylabel('TEDS Score')
        axes[1].set_title('PubTabNet: Table Edit Distance Similarity (TEDS)')
        axes[1].set_ylim(0, 1.0)
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'chapter4_structure_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def create_overall_summary(results: dict, output_dir: str):
    """Create overall summary radar chart and table."""
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Summary table using matplotlib table
    ax1 = fig.add_subplot(121)
    ax1.axis('off')
    ax1.set_title('Baseline Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Prepare table data
    table_data = []
    
    if 'doclaynet' in results:
        doc = results['doclaynet']
        metrics = doc.get('metrics', {})
        f1 = metrics.get('iou_0.5', {}).get('f1', 0)
        table_data.append(['DocLayNet', 'Table Detection', 'F1@IoU0.5', f'{f1:.4f}'])
    
    if 'pubtables1m' in results:
        pt1m = results['pubtables1m']
        metrics = pt1m.get('metrics_iou50', {})
        row_f1 = metrics.get('rows', {}).get('f1', 0)
        col_f1 = metrics.get('columns', {}).get('f1', 0)
        table_data.append(['PubTables-1M', 'Structure Recognition', 'Row F1', f'{row_f1:.4f}'])
        table_data.append(['', '', 'Column F1', f'{col_f1:.4f}'])
    
    if 'fintabnet' in results:
        fin = results['fintabnet']
        metrics = fin.get('metrics', {})
        row_f1 = metrics.get('rows', {}).get('f1', 0)
        col_f1 = metrics.get('columns', {}).get('f1', 0)
        table_data.append(['FinTabNet', 'Structure Recognition', 'Row F1', f'{row_f1:.4f}'])
        table_data.append(['', '(Financial Tables)', 'Column F1', f'{col_f1:.4f}'])
    
    if 'pubtabnet' in results:
        pt = results['pubtabnet']
        teds = pt.get('teds_metrics', {}).get('teds_mean', 0)
        teds_struct = pt.get('teds_structure_only_metrics', {}).get('teds_mean', 0)
        table_data.append(['PubTabNet', 'End-to-End', 'TEDS', f'{teds:.4f}'])
        table_data.append(['', '', 'TEDS (struct)', f'{teds_struct:.4f}'])
    
    # Create table with proper alignment
    col_labels = ['Dataset', 'Task', 'Metric', 'Score']
    
    table = ax1.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.30, 0.25, 0.15]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    # Right: Bar chart comparison
    ax2 = fig.add_subplot(122)
    
    # Collect all F1 scores for comparison
    metrics_list = []
    scores = []
    colors = []
    
    if 'doclaynet' in results:
        doc = results['doclaynet']
        doc_metrics = doc.get('metrics', {})
        metrics_list.append('DocLayNet\nDetection F1')
        scores.append(doc_metrics.get('iou_0.5', {}).get('f1', 0))
        colors.append('#e74c3c')
    
    if 'pubtables1m' in results:
        pt1m = results['pubtables1m']
        pt1m_metrics = pt1m.get('metrics_iou50', {})
        metrics_list.append('PubTables-1M\nRow F1')
        scores.append(pt1m_metrics.get('rows', {}).get('f1', 0))
        colors.append('#3498db')
        
        metrics_list.append('PubTables-1M\nColumn F1')
        scores.append(pt1m_metrics.get('columns', {}).get('f1', 0))
        colors.append('#2980b9')
    
    if 'fintabnet' in results:
        fin = results['fintabnet']
        fin_metrics = fin.get('metrics', {})
        metrics_list.append('FinTabNet\nRow F1')
        scores.append(fin_metrics.get('rows', {}).get('f1', 0))
        colors.append('#27ae60')
        
        metrics_list.append('FinTabNet\nColumn F1')
        scores.append(fin_metrics.get('columns', {}).get('f1', 0))
        colors.append('#1e8449')
    
    if 'pubtabnet' in results:
        pt = results['pubtabnet']
        metrics_list.append('PubTabNet\nTEDS (struct)')
        scores.append(pt.get('teds_structure_only_metrics', {}).get('teds_mean', 0))
        colors.append('#9b59b6')
    
    bars = ax2.barh(metrics_list, scores, color=colors)
    ax2.bar_label(bars, fmt='%.3f', padding=3)
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel('Score')
    ax2.set_title('Baseline Performance Comparison Across Datasets')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'chapter4_overall_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def create_dataset_characteristics(output_dir: str):
    """Create dataset characteristics comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dataset sizes (approximate)
    datasets = ['PubTabNet', 'PubTables-1M', 'FinTabNet', 'DocLayNet']
    sizes = [568000, 1000000, 112887, 80863]  # Number of tables/images
    colors = ['#9b59b6', '#3498db', '#27ae60', '#e74c3c']
    
    bars = axes[0].bar(datasets, [s/1000 for s in sizes], color=colors)
    axes[0].bar_label(bars, fmt='%.0fK', fontsize=10)
    axes[0].set_ylabel('Number of Tables (thousands)')
    axes[0].set_title('Dataset Size Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Domain distribution
    domains = ['Scientific\n(PubMed)', 'Scientific\n(PubMed)', 'Financial\n(SEC Filings)', 'Mixed\n(Documents)']
    domain_counts = {'Scientific\n(PubMed)': 2, 'Financial\n(SEC Filings)': 1, 'Mixed\n(Documents)': 1}
    
    axes[1].pie(domain_counts.values(), labels=domain_counts.keys(), autopct='%1.0f%%',
                colors=['#3498db', '#27ae60', '#e74c3c'], startangle=90)
    axes[1].set_title('Dataset Domain Distribution')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'chapter4_dataset_characteristics.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Generating Chapter 4 Figures")
    print("=" * 60)
    
    # Load config
    config = load_config('configs/config.yaml')
    
    # Setup directories
    results_dir = config['outputs']['results']
    figures_dir = config['outputs']['figures']
    ensure_dir(figures_dir)
    
    # Load all results
    print("\nLoading experiment results...")
    results = load_latest_results(results_dir)
    print(f"Loaded results for: {list(results.keys())}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    create_detection_comparison(results, figures_dir)
    create_structure_comparison(results, figures_dir)
    create_overall_summary(results, figures_dir)
    create_dataset_characteristics(figures_dir)
    
    print("\n" + "=" * 60)
    print("Chapter 4 Figures Generated Successfully!")
    print("=" * 60)
    print(f"\nFigures saved to: {figures_dir}")
    
    # Print summary for thesis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY FOR THESIS")
    print("=" * 60)
    
    if 'doclaynet' in results:
        doc = results['doclaynet']
        metrics = doc.get('metrics', {})
        iou05 = metrics.get('iou_0.5', {})
        print(f"\n[DocLayNet - Table Detection]")
        print(f"  Precision@IoU0.5: {iou05.get('precision', 0):.4f}")
        print(f"  Recall@IoU0.5:    {iou05.get('recall', 0):.4f}")
        print(f"  F1@IoU0.5:        {iou05.get('f1', 0):.4f}")
    
    if 'pubtables1m' in results:
        pt1m = results['pubtables1m']
        metrics = pt1m.get('metrics_iou50', {})
        print(f"\n[PubTables-1M - Structure Recognition]")
        print(f"  Row F1:    {metrics.get('rows', {}).get('f1', 0):.4f}")
        print(f"  Column F1: {metrics.get('columns', {}).get('f1', 0):.4f}")
    
    if 'fintabnet' in results:
        fin = results['fintabnet']
        metrics = fin.get('metrics', {})
        print(f"\n[FinTabNet - Financial Table Structure]")
        print(f"  Row F1:    {metrics.get('rows', {}).get('f1', 0):.4f}")
        print(f"  Column F1: {metrics.get('columns', {}).get('f1', 0):.4f}")
    
    if 'pubtabnet' in results:
        pt = results['pubtabnet']
        teds = pt.get('teds_metrics', {})
        teds_struct = pt.get('teds_structure_only_metrics', {})
        print(f"\n[PubTabNet - End-to-End TEDS]")
        print(f"  TEDS (with content):    {teds.get('teds_mean', 0):.4f} ± {teds.get('teds_std', 0):.4f}")
        print(f"  TEDS (structure only):  {teds_struct.get('teds_mean', 0):.4f} ± {teds_struct.get('teds_std', 0):.4f}")


if __name__ == '__main__':
    main()
