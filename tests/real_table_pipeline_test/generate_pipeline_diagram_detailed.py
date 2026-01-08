"""
Generate detailed pipeline diagram for thesis
Creates a professional-looking pipeline architecture visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram():
    """Create a detailed pipeline diagram for thesis"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(9, 13.5, 'Financial Table Extraction Pipeline', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(9, 13.0, 'End-to-End Architecture with Table Transformer', 
            fontsize=12, ha='center', va='center', style='italic', color='gray')
    
    # Colors
    colors = {
        'input': '#E3F2FD',
        'detection': '#FFECB3',
        'tsr': '#C8E6C9',
        'ocr': '#F8BBD0',
        'numeric': '#D1C4E9',
        'semantic': '#B2DFDB',
        'qa': '#FFCCBC',
        'output': '#E8F5E9'
    }
    
    edge_colors = {
        'input': '#1565C0',
        'detection': '#FF8F00',
        'tsr': '#2E7D32',
        'ocr': '#C2185B',
        'numeric': '#512DA8',
        'semantic': '#00695C',
        'qa': '#BF360C',
        'output': '#1B5E20'
    }
    
    # Box width and height
    box_width = 2.4
    box_height = 1.8
    
    # Positions (x, y) for each step
    positions = {
        'input': (0.3, 10),
        'step1': (3.2, 10),
        'step2': (6.1, 10),
        'step3': (9.0, 10),
        'step4': (11.9, 10),
        'step5': (14.8, 10),
        'step6': (14.8, 6),
        'output': (14.8, 2)
    }
    
    # Draw INPUT box
    input_box = FancyBboxPatch((positions['input'][0], positions['input'][1] - box_height/2),
                               box_width, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor=colors['input'],
                               edgecolor=edge_colors['input'],
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(positions['input'][0] + box_width/2, positions['input'][1] + 0.5,
            'INPUT', fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(positions['input'][0] + box_width/2, positions['input'][1] + 0.1,
            'Financial', fontsize=9, ha='center', va='center')
    ax.text(positions['input'][0] + box_width/2, positions['input'][1] - 0.2,
            'Document', fontsize=9, ha='center', va='center')
    ax.text(positions['input'][0] + box_width/2, positions['input'][1] - 0.6,
            '(PNG/PDF)', fontsize=8, ha='center', va='center', color='gray')
    
    # Step definitions
    steps = [
        ('step1', 'STEP 1', 'Table Detection', 'Table Transformer', 'microsoft/table-\ntransformer-detection',
         '15.4s', 'mAP 95%', colors['detection'], edge_colors['detection']),
        ('step2', 'STEP 2', 'Structure Recognition', 'Table Transformer', 'microsoft/table-\ntransformer-tsr',
         '1.9s', 'TEDS 89%', colors['tsr'], edge_colors['tsr']),
        ('step3', 'STEP 3', 'OCR Extraction', 'EasyOCR', 'GPU-accelerated',
         '7.2s', 'Conf 88%', colors['ocr'], edge_colors['ocr']),
        ('step4', 'STEP 4', 'Numeric Normal.', 'Rule-based', 'Pattern matching',
         '<1ms', '94.9%', colors['numeric'], edge_colors['numeric']),
        ('step5', 'STEP 5', 'Semantic Class.', 'Rule-based', 'Position + content',
         '<1ms', '~95%', colors['semantic'], edge_colors['semantic']),
    ]
    
    # Draw main pipeline steps (horizontal)
    for step_key, step_num, step_name, model, model_detail, time, metric, fcolor, ecolor in steps:
        x, y = positions[step_key]
        
        # Main box
        box = FancyBboxPatch((x, y - box_height/2),
                            box_width, box_height,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=fcolor,
                            edgecolor=ecolor,
                            linewidth=2)
        ax.add_patch(box)
        
        # Step number and name
        ax.text(x + box_width/2, y + 0.6, step_num, fontsize=9, fontweight='bold', 
                ha='center', va='center', color=ecolor)
        ax.text(x + box_width/2, y + 0.25, step_name, fontsize=9, fontweight='bold', 
                ha='center', va='center')
        
        # Model info
        ax.text(x + box_width/2, y - 0.15, model, fontsize=8, ha='center', va='center')
        ax.text(x + box_width/2, y - 0.5, model_detail, fontsize=7, ha='center', va='center', color='gray')
        
        # Time and metric boxes at bottom
        ax.text(x + 0.6, y - box_height/2 - 0.25, f'Time: {time}', fontsize=7, ha='center', va='center')
        ax.text(x + box_width - 0.6, y - box_height/2 - 0.25, f'{metric}', fontsize=7, ha='center', va='center')
    
    # Draw Step 6 (QA) - below Step 5
    x6, y6 = positions['step6']
    box6 = FancyBboxPatch((x6, y6 - box_height/2),
                          box_width, box_height,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['qa'],
                          edgecolor=edge_colors['qa'],
                          linewidth=2)
    ax.add_patch(box6)
    ax.text(x6 + box_width/2, y6 + 0.6, 'STEP 6', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=edge_colors['qa'])
    ax.text(x6 + box_width/2, y6 + 0.25, 'Table QA', fontsize=9, fontweight='bold', ha='center', va='center')
    ax.text(x6 + box_width/2, y6 - 0.15, 'Fuzzy Lookup', fontsize=8, ha='center', va='center')
    ax.text(x6 + box_width/2, y6 - 0.5, 'Row/Col matching', fontsize=7, ha='center', va='center', color='gray')
    ax.text(x6 + 0.6, y6 - box_height/2 - 0.25, f'Time: 14ms', fontsize=7, ha='center', va='center')
    ax.text(x6 + box_width - 0.6, y6 - box_height/2 - 0.25, f'30.0%', fontsize=7, ha='center', va='center')
    
    # Draw OUTPUT box
    x_out, y_out = positions['output']
    output_box = FancyBboxPatch((x_out, y_out - box_height/2),
                                box_width, box_height,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=colors['output'],
                                edgecolor=edge_colors['output'],
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(x_out + box_width/2, y_out + 0.5, 'OUTPUT', fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(x_out + box_width/2, y_out + 0.1, 'Answer', fontsize=9, ha='center', va='center')
    ax.text(x_out + box_width/2, y_out - 0.3, '"1,511,947"', fontsize=9, ha='center', va='center', 
            fontfamily='monospace', color='green')
    
    # Draw arrows
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=8"
    kw = dict(arrowstyle=arrow_style, color='#424242')
    
    # Horizontal arrows (input → step1 → step2 → step3 → step4 → step5)
    arrow_y = 10
    arrows_horiz = [
        (positions['input'][0] + box_width, positions['step1'][0]),
        (positions['step1'][0] + box_width, positions['step2'][0]),
        (positions['step2'][0] + box_width, positions['step3'][0]),
        (positions['step3'][0] + box_width, positions['step4'][0]),
        (positions['step4'][0] + box_width, positions['step5'][0]),
    ]
    
    for x1, x2 in arrows_horiz:
        arrow = FancyArrowPatch((x1 + 0.1, arrow_y), (x2 - 0.1, arrow_y), **kw)
        ax.add_patch(arrow)
    
    # Vertical arrow (step5 → step6)
    arrow = FancyArrowPatch((positions['step5'][0] + box_width/2, positions['step5'][1] - box_height/2 - 0.1),
                           (positions['step6'][0] + box_width/2, positions['step6'][1] + box_height/2 + 0.1), **kw)
    ax.add_patch(arrow)
    
    # Vertical arrow (step6 → output)
    arrow = FancyArrowPatch((positions['step6'][0] + box_width/2, positions['step6'][1] - box_height/2 - 0.1),
                           (positions['output'][0] + box_width/2, positions['output'][1] + box_height/2 + 0.1), **kw)
    ax.add_patch(arrow)
    
    # Add data flow labels on arrows
    labels = [
        (positions['input'][0] + box_width + 0.4, arrow_y + 0.4, 'Image'),
        (positions['step1'][0] + box_width + 0.4, arrow_y + 0.4, 'BBox'),
        (positions['step2'][0] + box_width + 0.4, arrow_y + 0.4, 'Grid'),
        (positions['step3'][0] + box_width + 0.4, arrow_y + 0.4, 'Cells'),
        (positions['step4'][0] + box_width + 0.4, arrow_y + 0.4, 'Values'),
        (positions['step5'][0] + box_width/2 + 0.5, 8, 'Types'),
        (positions['step6'][0] + box_width/2 + 0.5, 4, 'Answer'),
    ]
    
    for x, y, label in labels:
        ax.text(x, y, label, fontsize=7, ha='left', va='center', color='#616161', style='italic')
    
    # Add intermediate outputs visualization
    # Detection output
    det_box = patches.Rectangle((3.2, 7.2), 2.4, 1.5, fill=False, 
                                 edgecolor='#FF8F00', linestyle='--', linewidth=1)
    ax.add_patch(det_box)
    ax.text(4.4, 8.4, 'Detected Table', fontsize=7, ha='center', color='#FF8F00')
    ax.text(4.4, 7.5, 'BBox [x1,y1,x2,y2]', fontsize=7, ha='center', color='gray')
    ax.text(4.4, 7.0, 'conf: 99.99%', fontsize=6, ha='center', color='gray')
    
    # TSR output
    tsr_box = patches.Rectangle((6.1, 7.2), 2.4, 1.5, fill=False,
                                edgecolor='#2E7D32', linestyle='--', linewidth=1)
    ax.add_patch(tsr_box)
    ax.text(7.3, 8.4, 'Structure', fontsize=7, ha='center', color='#2E7D32')
    ax.text(7.3, 7.5, '36 rows x 7 cols', fontsize=7, ha='center', color='gray')
    ax.text(7.3, 7.0, 'cell bboxes', fontsize=6, ha='center', color='gray')
    
    # OCR output
    ocr_box = patches.Rectangle((9.0, 7.2), 2.4, 1.5, fill=False,
                                edgecolor='#C2185B', linestyle='--', linewidth=1)
    ax.add_patch(ocr_box)
    ax.text(10.2, 8.4, 'OCR Grid', fontsize=7, ha='center', color='#C2185B')
    ax.text(10.2, 7.5, '107 text cells', fontsize=7, ha='center', color='gray')
    ax.text(10.2, 7.0, 'grid[row][col]', fontsize=6, ha='center', color='gray')
    
    # Add legend box
    legend_x, legend_y = 0.3, 3.5
    legend_box = FancyBboxPatch((legend_x, legend_y),
                                4.5, 4.5,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor='white',
                                edgecolor='#BDBDBD',
                                linewidth=1)
    ax.add_patch(legend_box)
    ax.text(legend_x + 2.25, legend_y + 4.2, 'Pipeline Summary', fontsize=10, fontweight='bold', ha='center')
    
    summary_text = [
        ('Total Time:', '24.5 seconds'),
        ('Detection Model:', 'Table Transformer'),
        ('OCR Engine:', 'EasyOCR'),
        ('Final QA Accuracy:', '30.0%'),
        ('Test Type:', 'FAIR (no GT peek)'),
    ]
    
    for i, (label, value) in enumerate(summary_text):
        ax.text(legend_x + 0.3, legend_y + 3.5 - i*0.6, label, fontsize=8, ha='left', fontweight='bold')
        ax.text(legend_x + 2.3, legend_y + 3.5 - i*0.6, value, fontsize=8, ha='left', color='#424242')
    
    # Add error cascade visualization
    cascade_x, cascade_y = 5.5, 3.5
    cascade_box = FancyBboxPatch((cascade_x, cascade_y),
                                 4.5, 4.5,
                                 boxstyle="round,pad=0.1,rounding_size=0.2",
                                 facecolor='#FFEBEE',
                                 edgecolor='#EF5350',
                                 linewidth=1)
    ax.add_patch(cascade_box)
    ax.text(cascade_x + 2.25, cascade_y + 4.2, 'Error Cascade', fontsize=10, fontweight='bold', 
            ha='center', color='#C62828')
    
    cascade_steps = [
        ('Step 1: Detection', '99.99%', '#4CAF50'),
        ('Step 2: TSR', '~95%', '#4CAF50'),
        ('Step 3: OCR', '54% (CER)', '#F44336'),
        ('Step 4: Numeric', '94.9%', '#4CAF50'),
        ('Step 5: Semantic', '~95%', '#4CAF50'),
        ('Step 6: QA', '30.0%', '#F44336'),
    ]
    
    for i, (step, acc, color) in enumerate(cascade_steps):
        ax.text(cascade_x + 0.3, cascade_y + 3.5 - i*0.6, step, fontsize=7, ha='left')
        ax.text(cascade_x + 3.5, cascade_y + 3.5 - i*0.6, acc, fontsize=7, ha='right', 
                color=color, fontweight='bold')
    
    # Add bottleneck indicator
    ax.annotate('', xy=(cascade_x + 4.2, cascade_y + 2.3), 
                xytext=(cascade_x + 4.7, cascade_y + 2.3),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=2))
    ax.text(cascade_x + 4.9, cascade_y + 2.3, 'Bottleneck', fontsize=7, color='#F44336', va='center')
    
    # Add QA Question Example
    qa_x, qa_y = 10.7, 3.5
    qa_box = FancyBboxPatch((qa_x, qa_y),
                            4.5, 4.5,
                            boxstyle="round,pad=0.1,rounding_size=0.2",
                            facecolor='#FFF3E0',
                            edgecolor='#FF9800',
                            linewidth=1)
    ax.add_patch(qa_box)
    ax.text(qa_x + 2.25, qa_y + 4.2, 'QA Example', fontsize=10, fontweight='bold', 
            ha='center', color='#E65100')
    
    qa_example = [
        'Question:',
        '"Property, plant and equipment"',
        'for "Group 2025"',
        '',
        'Lookup: row=5, col=1',
        'Expected: 1,511,947',
        'Predicted: 1,511,947 [OK]',
    ]
    
    for i, line in enumerate(qa_example):
        style = 'italic' if i == 0 else 'normal'
        color = '#2E7D32' if '[OK]' in line else '#424242'
        fontsize = 8 if i > 0 else 9
        ax.text(qa_x + 0.3, qa_y + 3.5 - i*0.5, line, fontsize=fontsize, ha='left', 
                color=color, style=style)
    
    plt.tight_layout()
    plt.savefig('tests/real_table_pipeline_test/output_fair/pipeline_diagram_detailed.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('tests/real_table_pipeline_test/output_fair/pipeline_diagram_detailed.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Saved: pipeline_diagram_detailed.png and .pdf")
    
    plt.show()

if __name__ == "__main__":
    create_pipeline_diagram()
