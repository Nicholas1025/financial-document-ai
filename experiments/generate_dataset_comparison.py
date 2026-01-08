"""
Generate Dataset Comparison Figures for Thesis Chapter 4

Figure 4.2: Comparison of dataset sizes and domain distribution
(a) Dataset Size Comparison (Grouped Bar Chart)
(b) Dataset Domain Distribution (Donut Chart)

Academic style with clean, professional appearance and HCI-compliant colors.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ============================================================================
# Academic Style Configuration
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Output directory
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dataset Statistics
# ============================================================================
DATASETS = {
    'PubTabNet': {
        'train': 500777,
        'val': 9115,
        'test': 9138,
        'domain': 'Scientific',
    },
    'PubTables-1M': {
        'train': 758849,
        'val': 94856,
        'test': 94994,
        'domain': 'Scientific',
    },
    'FinTabNet': {
        'train': 78067,
        'val': 10656,
        'test': 10656,
        'domain': 'Financial',
    },
    'SynFinTab': {
        'train': 10000,
        'val': 1000,
        'test': 1000,
        'domain': 'Financial',
    },
    'DocLayNet': {
        'train': 69375,
        'val': 6489,
        'test': 4994,
        'domain': 'Mixed',
    }
}

# ============================================================================
# HCI Compliant Color Palette (Tableau 10 / Colorblind Friendly)
# ============================================================================
# Distinct colors for domains
DOMAIN_COLORS = {
    'Scientific': '#4E79A7',  # Blue
    'Financial': '#F28E2B',   # Orange
    'Mixed': '#59A14F',       # Green
}

# Distinct colors for splits (Train/Val/Test)
# Using a sequential-like but distinct palette
SPLIT_COLORS = {
    'train': '#2C3E50',       # Dark Slate Blue/Grey
    'val': '#E15759',         # Red (High contrast against dark blue)
    'test': '#76B7B2',        # Teal (Distinct from others)
}

def format_number(n):
    """Format large numbers for labels (e.g., 1.2M, 500K)"""
    if n >= 1_000_000:
        return f'{n/1_000_000:.1f}M'
    elif n >= 1_000:
        return f'{n/1_000:.0f}K'
    else:
        return str(n)

def generate_figure_4_2():
    """
    Generate Figure 4.2: Dataset Comparison (Academic Style)
    
    (a) Dataset Size Comparison (Bar Chart)
    (b) Domain Distribution (Donut Chart) - Optimized for clarity
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # =========================================================================
    # (a) Dataset Size Comparison
    # =========================================================================
    ax1 = axes[0]
    
    datasets = list(DATASETS.keys())
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.25
    
    train_sizes = [DATASETS[d]['train'] for d in datasets]
    val_sizes = [DATASETS[d]['val'] for d in datasets]
    test_sizes = [DATASETS[d]['test'] for d in datasets]
    
    # Create grouped bars
    # Using zorder to put bars in front of grid
    ax1.bar(x - width, train_sizes, width, label='Training', 
            color=SPLIT_COLORS['train'], edgecolor='white', linewidth=0.7, zorder=3)
    ax1.bar(x, val_sizes, width, label='Validation',
            color=SPLIT_COLORS['val'], edgecolor='white', linewidth=0.7, zorder=3)
    ax1.bar(x + width, test_sizes, width, label='Test',
            color=SPLIT_COLORS['test'], edgecolor='white', linewidth=0.7, zorder=3)
    
    # Formatting
    ax1.set_ylabel('Number of Tables (Log Scale)', fontweight='bold')
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=0)
    
    # Legend
    ax1.legend(loc='upper right', frameon=True, fancybox=True, 
               edgecolor='#CCCCCC', framealpha=0.95, fontsize=10)
    
    # Log scale
    ax1.set_yscale('log')
    ax1.set_ylim(500, 5_000_000)
    
    # Grid
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, which='major', zorder=0)
    
    # Add total annotations
    for i, d in enumerate(datasets):
        total = DATASETS[d]['train'] + DATASETS[d]['val'] + DATASETS[d]['test']
        # Position text slightly above the highest bar
        max_height = max(train_sizes[i], val_sizes[i], test_sizes[i])
        ax1.annotate(f"Total:\n{format_number(total)}",
                    xy=(i, max_height),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#333333')
    
    # Subfigure label
    ax1.set_title('(a) Dataset Size Comparison', loc='left', fontsize=14, fontweight='bold', pad=15)
    
    # =========================================================================
    # (b) Domain Distribution (Donut Chart)
    # =========================================================================
    ax2 = axes[1]
    
    # Calculate domain totals
    domain_totals = {}
    for name, info in DATASETS.items():
        domain = info['domain']
        total = info['train'] + info['val'] + info['test']
        domain_totals[domain] = domain_totals.get(domain, 0) + total
    
    # Sort by size for better visual
    sorted_domains = sorted(domain_totals.items(), key=lambda x: x[1], reverse=True)
    domains = [d[0] for d in sorted_domains]
    sizes = [d[1] for d in sorted_domains]
    colors = [DOMAIN_COLORS[d] for d in domains]
    
    # Create Donut Chart
    wedges, texts, autotexts = ax2.pie(
        sizes, 
        labels=None, # We will add custom legend/labels
        colors=colors,
        autopct='%1.1f%%', 
        pctdistance=0.80, # Move % further out
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'}
    )
    
    # Fix label colors (if segment is light, use dark text, else white)
    # Here our colors are quite dark/saturated, so white is good.
    
    # Add center text
    total_tables = sum(sizes)
    ax2.text(0, 0, f'Total\n{format_number(total_tables)}', 
             ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
    
    # Custom Legend with counts
    legend_labels = [f'{d}: {format_number(s)}' for d, s in zip(domains, sizes)]
    ax2.legend(wedges, legend_labels,
              title="Domain Distribution",
              loc="center left",
              bbox_to_anchor=(0.9, 0, 0.5, 1),
              frameon=False,
              fontsize=11)
    
    # Subfigure label
    ax2.set_title('(b) Domain Distribution', loc='left', fontsize=14, fontweight='bold', pad=15)
    
    # =========================================================================
    # Final Layout
    # =========================================================================
    plt.tight_layout()
    
    # Save
    fig.savefig(OUTPUT_DIR / 'figure_4_2_dataset_comparison.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'figure_4_2_dataset_comparison.pdf')
    
    print(f"Saved: {OUTPUT_DIR / 'figure_4_2_dataset_comparison.png'}")
    plt.close()


def generate_figure_4_2_vertical():
    """
    Vertical layout version (Stacked)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # (a) Bar Chart
    ax1 = axes[0]
    datasets = list(DATASETS.keys())
    x = np.arange(len(datasets))
    width = 0.25
    
    train_sizes = [DATASETS[d]['train'] for d in datasets]
    val_sizes = [DATASETS[d]['val'] for d in datasets]
    test_sizes = [DATASETS[d]['test'] for d in datasets]
    
    ax1.bar(x - width, train_sizes, width, label='Training', color=SPLIT_COLORS['train'], edgecolor='white')
    ax1.bar(x, val_sizes, width, label='Validation', color=SPLIT_COLORS['val'], edgecolor='white')
    ax1.bar(x + width, test_sizes, width, label='Test', color=SPLIT_COLORS['test'], edgecolor='white')
    
    ax1.set_ylabel('Number of Tables (Log Scale)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_yscale('log')
    ax1.set_ylim(500, 5_000_000)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('(a) Dataset Size Comparison', loc='left', fontweight='bold')

    # (b) Donut Chart
    ax2 = axes[1]
    domain_totals = {}
    for name, info in DATASETS.items():
        domain = info['domain']
        domain_totals[domain] = domain_totals.get(domain, 0) + info['train'] + info['val'] + info['test']
    
    sorted_domains = sorted(domain_totals.items(), key=lambda x: x[1], reverse=True)
    domains = [d[0] for d in sorted_domains]
    sizes = [d[1] for d in sorted_domains]
    colors = [DOMAIN_COLORS[d] for d in domains]
    
    wedges, texts, autotexts = ax2.pie(
        sizes, colors=colors, autopct='%1.1f%%', pctdistance=0.85,
        startangle=90, counterclock=False,
        wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'}
    )
    
    ax2.text(0, 0, f'Total\n{format_number(sum(sizes))}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    legend_labels = [f'{d}: {format_number(s)}' for d, s in zip(domains, sizes)]
    ax2.legend(wedges, legend_labels, title="Domain", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False)
    ax2.set_title('(b) Domain Distribution', loc='left', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_4_2_vertical.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'figure_4_2_vertical.pdf')
    print(f"Saved: {OUTPUT_DIR / 'figure_4_2_vertical.png'}")
    plt.close()

if __name__ == "__main__":
    print("Generating optimized figures...")
    generate_figure_4_2()
    generate_figure_4_2_vertical()
