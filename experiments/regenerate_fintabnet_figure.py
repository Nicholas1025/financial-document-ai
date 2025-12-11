import json
import os
import matplotlib.pyplot as plt
import numpy as np

def regenerate_figure():
    # Load the specific result file
    json_path = 'outputs/results/fintabnet_baseline_20251202_212220.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    timestamp = data['timestamp']
    figures_dir = 'outputs/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate metrics bar chart
    # Use a clean style
    plt.rcParams.update({'font.size': 12})
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
    
    # Add value labels with background for readability
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
    print(f"Regenerated figure saved to: {metrics_fig_path}")
    plt.close()

if __name__ == '__main__':
    regenerate_figure()
