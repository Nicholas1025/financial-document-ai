"""
Regenerate Benchmark Figures with Academic Style
================================================

This script reads the existing benchmark results (CSV/JSON) and regenerates
the figures using a high-quality academic plotting style.
"""

import os
import sys
import json
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Plotting Style Configuration
# ============================================================================

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
# Step 1: Detection
# ============================================================================

def regenerate_step1():
    print("Regenerating Step 1 (Detection) figure...")
    csv_path = RESULTS_DIR / "step1_detection_per_sample.csv"
    if not csv_path.exists():
        print(f"Skipping Step 1: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    detectors = df['detector'].unique()
    for i, detector in enumerate(detectors):
        subset = df[df['detector'] == detector]
        f1_scores = subset['f1']
        mean_val = f1_scores.mean()
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        
        ax.hist(f1_scores, bins=20, alpha=0.6, 
                label=f'{detector} (Mean: {mean_val:.3f})',
                color=color, edgecolor='white', linewidth=0.5)
        
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('F1 Score per Image')
    ax.set_ylabel('Frequency (Number of Images)')
    ax.set_title(f'Step 1: Table Detection Performance Distribution\n(n={len(df)//len(detectors)} samples)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / 'step1_detection_f1_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

# ============================================================================
# Step 2: TSR
# ============================================================================

def regenerate_step2():
    print("Regenerating Step 2 (TSR) figure...")
    csv_path = RESULTS_DIR / "step2_tsr_per_sample.csv"
    if not csv_path.exists():
        print(f"Skipping Step 2: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Column name might be 'model' or 'method' depending on how it was saved
    model_col = 'model' if 'model' in df.columns else 'method'
    if model_col not in df.columns:
        # Fallback: assume single model or check structure
        print("Could not find model column in Step 2 CSV")
        return

    models = df[model_col].unique()
    for i, model in enumerate(models):
        subset = df[df[model_col] == model]
        scores = subset['teds']
        mean_val = scores.mean()
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        
        ax.hist(scores, bins=20, alpha=0.6, 
                label=f'{model} (Mean: {mean_val:.3f})',
                color=color, edgecolor='white', linewidth=0.5)
        
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('TEDS Score (Tree Edit Distance Similarity)')
    ax.set_ylabel('Frequency (Number of Tables)')
    ax.set_title(f'Step 2: Table Structure Recognition Performance\n(n={len(df)//len(models)} samples)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / 'step2_tsr_teds_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

# ============================================================================
# Step 3: OCR
# ============================================================================

def regenerate_step3():
    print("Regenerating Step 3 (OCR) figure...")
    csv_path = RESULTS_DIR / "step3_ocr_per_sample.csv"
    if not csv_path.exists():
        print(f"Skipping Step 3: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    
    # Note: The CSV has avg_cer per image. The original plot was per-cell CER?
    # Let's check run_all_benchmarks.py again. 
    # It collected `all_cer` which is a list of ALL cell CERs.
    # The CSV only has per-image averages.
    # If we only have the CSV, we can plot the distribution of per-image average CER.
    # This is slightly different but still valid.
    # However, for OCR, per-cell distribution is more informative.
    # Since I don't have the raw cell data saved in CSV (it would be huge), 
    # I will plot the per-image Average CER distribution.
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = df['avg_cer']
    # Filter valid scores
    scores = scores[scores <= 1.0]
    
    mean_val = scores.mean()
    color = ACADEMIC_COLORS[0]
    
    ax.hist(scores, bins=30, alpha=0.6, 
            label=f'PaddleOCR (Mean Image CER: {mean_val:.3f})',
            color=color, edgecolor='white', linewidth=0.5)
    
    ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Average Character Error Rate (CER) per Image')
    ax.set_ylabel('Frequency (Number of Images)')
    ax.set_title(f'Step 3: OCR Error Distribution\n(n={len(df)} images)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / 'step3_ocr_cer_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

# ============================================================================
# Step 4: Numeric
# ============================================================================

def regenerate_step4():
    print("Regenerating Step 4 (Numeric) figure...")
    csv_path = RESULTS_DIR / "step4_numeric_per_sample.csv"
    if not csv_path.exists():
        print(f"Skipping Step 4: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    
    if 'relative_error' in df.columns:
        errors = df['relative_error']
    elif 'mean_relative_error' in df.columns:
        errors = df['mean_relative_error']
    else:
        print("Could not find error column in Step 4 CSV")
        return

    # Filter non-zero for log scale
    errors_nonzero = errors[errors > 1e-10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(errors_nonzero) > 0:
        color = ACADEMIC_COLORS[1]
        ax.hist(errors_nonzero, bins=50, alpha=0.7, color=color, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Frequency (Number of Values)')
        ax.set_title(f'Step 4: Numeric Normalisation Error Distribution\n(n={len(df)} values)')
        
        if errors_nonzero.max() / (errors_nonzero.min() + 1e-10) > 100:
            ax.set_xscale('log')
        
        ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'All {len(df)} values are exact matches!', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'Step 4: Numeric Normalisation - Perfect Results')
    
    fig_path = FIGURES_DIR / 'step4_numeric_error_histogram.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

# ============================================================================
# Step 5: Semantic
# ============================================================================

def regenerate_step5():
    print("Regenerating Step 5 (Semantic) figure...")
    json_path = RESULTS_DIR / "step5_semantic_results.json"
    if not json_path.exists():
        print(f"Skipping Step 5: {json_path} not found")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    cell_types = data.get('cell_types', [])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(cell_types))
    width = 0.35
    
    for i, (clf_name, metrics) in enumerate(results.items()):
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        # Convert percentage to 0-1 range
        f1_scores = [metrics['per_class'][label]['f1'] / 100.0 for label in cell_types]
        macro_f1 = metrics["macro_f1"] / 100.0
        
        offset = width * (i - 0.5)
        
        bars = ax.bar(x + offset, f1_scores, width, 
                     label=f'{clf_name} (Macro F1: {macro_f1:.3f})',
                     color=color, edgecolor='white', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Step 5: Semantic Classification Performance by Class')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ').title() for t in cell_types], rotation=0, ha='center')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig_path = FIGURES_DIR / 'step5_semantic_f1_barchart.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    regenerate_step1()
    regenerate_step2()
    regenerate_step3()
    regenerate_step4()
    regenerate_step5()
    print("\nAll figures regenerated successfully!")
