import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_pubtabnet_ocr_figure():
    # Load the specific result file
    json_path = 'outputs/results/pubtabnet_ocr_baseline_20251203_224805.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    teds_ocr = data['teds_with_ocr']
    teds_no_ocr = data['teds_no_ocr']
    teds_struct = data['teds_structure_only']
    timestamp = data['timestamp']
    
    figures_dir = 'outputs/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate comparison figure
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Structure Only', 'With OCR', 'Without OCR']
    means = [teds_struct['teds_mean'], teds_ocr['teds_mean'], teds_no_ocr['teds_mean']]
    
    # Professional colors
    colors = ['#4A90E2', '#50E3C2', '#E74C3C']  # Blue, Teal, Red
    
    bars = ax.bar(categories, means, color=colors, edgecolor='white', linewidth=1, width=0.6, zorder=3)
    
    ax.set_ylabel('TEDS Score', fontsize=12, fontweight='bold')
    ax.set_title('Impact of OCR on TEDS Score (PubTabNet)', fontsize=14, fontweight='bold', pad=20)
    
    # Clean up chart
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    # Coordinates for arrow from No OCR to With OCR
    # No OCR is index 2, With OCR is index 1
    x_no_ocr = 2
    y_no_ocr = means[2]
    x_ocr = 1
    y_ocr = means[1]
    
    # Draw arrow
    ax.annotate('', xy=(x_ocr, y_ocr + 0.02), xytext=(x_no_ocr, y_no_ocr + 0.02),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.2", color='#2C3E50', lw=1.5))
    
    # Add text for improvement
    improvement = means[1] - means[2]
    mid_x = (x_ocr + x_no_ocr) / 2
    mid_y = max(y_ocr, y_no_ocr) + 0.05
    ax.text(mid_x, mid_y, f'+{improvement:.3f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color='#27AE60')

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"pubtabnet_ocr_comparison_{timestamp}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to: {fig_path}")
    plt.close()

if __name__ == '__main__':
    generate_pubtabnet_ocr_figure()
