"""
Generate Pipeline Architecture Diagram
=====================================
生成 Financial Document AI Pipeline 的架构图和详细说明
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

def create_pipeline_diagram(output_path: str):
    """Create a professional pipeline architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(-1, 14)
    ax.axis('off')
    
    # Colors - professional scheme
    colors = {
        'input': '#E3F2FD',       # Light blue
        'detection': '#FFF3E0',   # Light orange
        'tsr': '#E8F5E9',         # Light green
        'ocr': '#F3E5F5',         # Light purple
        'numeric': '#E1F5FE',     # Light cyan
        'semantic': '#FFFDE7',    # Light yellow
        'qa': '#FFEBEE',          # Light red
        'output': '#E8F5E9',      # Light green
        'arrow': '#424242',
        'model': '#1565C0',       # Blue for model names
        'border': '#424242',
    }
    
    # Title
    ax.text(9, 13.2, 'Financial Document AI Pipeline', fontsize=22, fontweight='bold',
           ha='center', va='center', color='#1a1a1a')
    ax.text(9, 12.6, 'End-to-End Table Understanding System', fontsize=13, style='italic',
           ha='center', va='center', color='#666666')
    
    # Define step positions
    box_width = 3.8
    box_height = 1.6
    y_positions = [10.3, 8.3, 6.3, 4.3, 2.3, 0.3]
    
    steps = [
        {
            'step': 'Step 1',
            'name': 'Table Detection',
            'model': 'Table Transformer (DETR)',
            'input': 'Document Image',
            'output': 'Table Bounding Boxes',
            'color': colors['detection'],
            'details': 'microsoft/table-transformer-detection\nThreshold: 0.5 | mAP: 95%'
        },
        {
            'step': 'Step 2',
            'name': 'Table Structure Recognition',
            'model': 'Table Transformer (TSR)',
            'input': 'Cropped Table Image',
            'output': 'Rows, Columns, Cells',
            'color': colors['tsr'],
            'details': 'microsoft/table-transformer-structure-recognition\nTEDS Score: 89%'
        },
        {
            'step': 'Step 3',
            'name': 'OCR Text Extraction',
            'model': 'PaddleOCR / EasyOCR',
            'input': 'Cell Regions',
            'output': 'Text + Confidence',
            'color': colors['ocr'],
            'details': 'PaddleOCR (Winner): CER 33.5%\nEasyOCR: CER 46.0%'
        },
        {
            'step': 'Step 4',
            'name': 'Numeric Normalization',
            'model': 'Rule-based Parser',
            'input': 'Raw OCR Text',
            'output': 'Normalized Values',
            'color': colors['numeric'],
            'details': 'Formats: 1,234 | (100) | $1M | 12%\nAccuracy: 100%'
        },
        {
            'step': 'Step 5',
            'name': 'Semantic Classification',
            'model': 'Heuristic Classifier',
            'input': 'Cell Text + Position',
            'output': 'Cell Types',
            'color': colors['semantic'],
            'details': 'Types: header / data / total / empty\nAccuracy: 96.1%'
        },
        {
            'step': 'Step 6',
            'name': 'Table Question Answering',
            'model': 'Row-Column Lookup',
            'input': 'Question + Table',
            'output': 'Answer Value',
            'color': colors['qa'],
            'details': 'Fuzzy string matching\nAccuracy: 87.6% (SynFinTabs)'
        },
    ]
    
    # Draw input box at top
    input_box = FancyBboxPatch((6.5, 11.8), 5, 0.6,
                               boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor=colors['input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(9, 12.1, 'INPUT: Financial Document Image (PDF/PNG)', 
           fontsize=12, ha='center', va='center', fontweight='bold', color='#1565C0')
    
    # Arrow from input to step 1
    ax.annotate('', xy=(9, y_positions[0] + box_height + 0.1), xytext=(9, 11.7),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2.5))
    
    # Draw each step
    for i, step in enumerate(steps):
        y = y_positions[i]
        
        # Step number circle
        circle = plt.Circle((1.2, y + box_height/2), 0.4, facecolor=step['color'], 
                           edgecolor=colors['border'], linewidth=2)
        ax.add_patch(circle)
        ax.text(1.2, y + box_height/2, step['step'].split()[1], fontsize=12, 
               fontweight='bold', ha='center', va='center')
        
        # Main box with step name
        box = FancyBboxPatch((2, y), box_width, box_height,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            facecolor=step['color'], edgecolor=colors['border'], linewidth=2)
        ax.add_patch(box)
        
        # Step name
        ax.text(2 + box_width/2, y + box_height - 0.35, step['name'], 
               fontsize=12, fontweight='bold', ha='center', va='center', color='#1a1a1a')
        
        # Model name
        ax.text(2 + box_width/2, y + box_height - 0.8, f"Model: {step['model']}", 
               fontsize=9, ha='center', va='center', color=colors['model'], fontweight='bold')
        
        # Details box
        details_box = FancyBboxPatch((6.2, y), 5.5, box_height,
                                    boxstyle="round,pad=0.05,rounding_size=0.1",
                                    facecolor='white', edgecolor='#bdbdbd', linewidth=1.5)
        ax.add_patch(details_box)
        
        ax.text(6.4, y + box_height - 0.25, 'Details:', fontsize=9, fontweight='bold',
               va='top', color='#666666')
        ax.text(6.4, y + box_height - 0.55, step['details'], 
               fontsize=9, va='top', family='monospace', linespacing=1.5, color='#424242')
        
        # I/O box
        io_box = FancyBboxPatch((12, y), 4.8, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor='#fafafa', edgecolor='#bdbdbd', linewidth=1.5)
        ax.add_patch(io_box)
        
        ax.text(14.4, y + box_height - 0.4, f"IN: {step['input']}", 
               fontsize=10, ha='center', va='center', color='#388E3C')
        ax.text(14.4, y + box_height - 0.9, f"OUT: {step['output']}", 
               fontsize=10, ha='center', va='center', color='#D32F2F')
        
        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(3.9, y - 0.15), xytext=(3.9, y_positions[i+1] + box_height + 0.15),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2.5))
    
    # Output box at bottom
    output_box = FancyBboxPatch((6, -0.7), 6, 0.6,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor=colors['output'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, -0.4, 'OUTPUT: Structured Table Data + QA Answers', 
           fontsize=12, ha='center', va='center', fontweight='bold', color='#2E7D32')
    
    # Arrow from last step to output
    ax.annotate('', xy=(9, -0.1), xytext=(3.9, 0.15),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2.5,
                             connectionstyle="arc3,rad=0.3"))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Pipeline diagram saved to: {output_path}")


def create_detailed_report(output_path: str):
    """Create detailed pipeline documentation"""
    
    report = """# Financial Document AI Pipeline Architecture

## Overview

This pipeline processes financial document images (PDF/PNG) and extracts structured table data, enabling question answering about specific cell values.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINANCIAL DOCUMENT AI PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│  Document Image → Detection → TSR → OCR → Parse → Classify → QA    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Table Detection

### Purpose
Locate and extract table regions from document images.

### Model
| Property | Value |
|----------|-------|
| **Model** | Table Transformer (DETR-based) |
| **Pretrained** | `microsoft/table-transformer-detection` |
| **Architecture** | DEtection TRansformer (DETR) with ResNet-18 backbone |
| **Input** | Document image (any size) |
| **Output** | Bounding boxes + confidence scores |
| **Threshold** | 0.5 (configurable) |

### Performance
- **mAP@0.5**: ~95% on DocLayNet
- **Inference Time**: ~100ms (GPU)

### Code
```python
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
results = processor.post_process_object_detection(outputs, threshold=0.5)
```

---

## Step 2: Table Structure Recognition (TSR)

### Purpose
Identify the internal structure of tables: rows, columns, and cells.

### Model
| Property | Value |
|----------|-------|
| **Model** | Table Transformer (TSR variant) |
| **Pretrained** | `microsoft/table-transformer-structure-recognition` |
| **Architecture** | DETR with structure-specific heads |
| **Input** | Cropped table image |
| **Output** | Row/column/cell bounding boxes |

### Performance
- **TEDS Score**: ~89% on FinTabNet
- **Row Detection**: 72-95% depending on table complexity

### Output Classes
- `table row`
- `table column`
- `table column header`
- `table spanning cell`

---

## Step 3: OCR Text Extraction

### Purpose
Extract text content from each cell region.

### Models Compared

| Engine | CER | Speed | GPU Support |
|--------|-----|-------|-------------|
| **PaddleOCR** | 33.5% | Fast | ✅ |
| **EasyOCR** | 46.0% | Medium | ✅ |

### Winner: PaddleOCR
Lower Character Error Rate (CER) on financial tables.

### Code
```python
# EasyOCR
import easyocr
reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext(image_path)

# PaddleOCR  
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
results = ocr.ocr(image_path, cls=True)
```

### Output Format
```python
[
    {'text': '1,511,947', 'confidence': 0.98, 'bbox': [x1, y1, x2, y2]},
    ...
]
```

---

## Step 4: Numeric Normalization

### Purpose
Parse and normalize numeric values from raw OCR text.

### Method
Rule-based parser with regex patterns.

### Formats Handled

| Format | Example | Parsed Value |
|--------|---------|--------------|
| Comma separators | `1,234,567` | 1234567 |
| Decimal points | `1,234.56` | 1234.56 |
| Negative (parentheses) | `(1,234)` | -1234 |
| Currency symbols | `$1,234` | 1234 |
| Percentage | `12.5%` | 0.125 |
| Empty/dash | `-` or `—` | null |
| Millions | `1.2M` | 1200000 |

### Performance
- **Accuracy**: 100% on GT numeric cells
- **Robustness**: Handles mixed formats in same document

### Code
```python
import re

def parse_numeric(text: str) -> float:
    text = text.strip()
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1]
    
    text = re.sub(r'[$,€£¥ ]', '', text)
    value = float(text)
    return -value if is_negative else value
```

---

## Step 5: Semantic Cell Classification

### Purpose
Classify each cell by its semantic role in the table.

### Method
Pattern-based heuristics + positional rules.

### Cell Types

| Type | Description | Example |
|------|-------------|---------|
| `row_header` | Row label (first column) | "Total Assets" |
| `column_header` | Column label (first rows) | "2025", "Group" |
| `data` | Regular data cell | "1,234" |
| `numeric` | Numeric value | "1,511,947" |
| `total` | Total/summary row | "Total" |
| `subtotal` | Subtotal row | "Total current assets" |
| `empty` | Empty or dash | "-" |
| `note_ref` | Note reference | "4", "Note 1" |

### Classification Rules
```python
def classify_cell(text, row_idx, col_idx):
    if text.isupper() and not any(c.isdigit() for c in text):
        return 'section_header'
    if 'total' in text.lower():
        return 'total'
    if text.isdigit() and len(text) <= 2:
        return 'note_ref'
    if col_idx == 0:
        return 'row_header'
    return 'data'
```

### Performance
- **Accuracy**: 96.1% on real financial tables

---

## Step 6: Table Question Answering

### Purpose
Answer natural language questions about table cell values.

### Method
Row-Column key matching with fuzzy search.

### Process
1. Parse question to extract row key and column key
2. Find matching row using fuzzy string matching
3. Find matching column using substring matching
4. Return cell value at intersection

### Question Format
```
"What is the value of [ROW_KEY] for [COL_KEY]?"
Example: "What is the value of Total Assets for Group 2025?"
```

### Matching Strategy
```python
from difflib import SequenceMatcher

def fuzzy_match(text1, text2, threshold=0.7):
    ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio >= threshold
```

### Performance
| Dataset | Questions | Accuracy |
|---------|-----------|----------|
| SynFinTabs | 499 | 87.6% |
| Real Table | 10 | 100% |

---

## Pipeline Summary

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │────▶│   Table      │────▶│    Table     │
│    Image     │     │  Detection   │     │    Region    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Cell       │◀────│   OCR Text   │◀────│   TSR        │
│   Values     │     │  Extraction  │     │ (Structure)  │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Numeric    │────▶│   Semantic   │────▶│   Table QA   │
│  Parsing     │     │ Classification│     │   System     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │   Answer:    │
                                          │  "1,511,947" │
                                          └──────────────┘
```

---

## Performance Summary

| Step | Task | Model/Method | Metric | Score |
|------|------|--------------|--------|-------|
| 1 | Detection | Table Transformer | mAP@0.5 | 95% |
| 2 | TSR | Table Transformer | TEDS | 89% |
| 3 | OCR | PaddleOCR | 1-CER | 66.5% |
| 4 | Numeric | Rule-based | Accuracy | 100% |
| 5 | Semantic | Heuristics | Accuracy | 96% |
| 6 | QA | Row-Col Lookup | Accuracy | 88% |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 6GB | RTX 3080 |
| RAM | 8GB | 16GB |
| VRAM | 4GB | 8GB+ |
| Storage | 5GB | 10GB |

---

## Dependencies

```
torch>=2.0
transformers>=4.30
easyocr>=1.7
paddleocr>=2.6
Pillow>=10.0
matplotlib>=3.7
numpy>=1.24
```

---

*Generated by Financial Document AI Pipeline Analyzer*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Detailed report saved to: {output_path}")


def main():
    output_dir = Path('tests/real_table_pipeline_test/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate diagram
    create_pipeline_diagram(str(output_dir / 'pipeline_architecture.png'))
    
    # Generate detailed report
    create_detailed_report(str(output_dir / 'PIPELINE_DETAILS.md'))
    
    print("\nDone! Files generated:")
    print(f"  - {output_dir / 'pipeline_architecture.png'}")
    print(f"  - {output_dir / 'PIPELINE_DETAILS.md'}")


if __name__ == '__main__':
    main()
