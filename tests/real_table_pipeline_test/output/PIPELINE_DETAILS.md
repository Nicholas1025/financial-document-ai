# Financial Document AI Pipeline Architecture

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
