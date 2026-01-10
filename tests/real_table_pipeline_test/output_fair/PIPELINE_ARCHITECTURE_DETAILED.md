# Financial Table Extraction Pipeline - Detailed Architecture

## Overview

This document provides a detailed breakdown of each step in the pipeline for thesis documentation.

---

## Pipeline Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           FINANCIAL TABLE EXTRACTION PIPELINE                                            │
│                              (End-to-End Architecture)                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Financial Document Image (PNG/JPEG/PDF)                                                         │
│  ├── Resolution: Variable (recommended: 300 DPI)                                                        │
│  ├── Format: Bank statements, Annual reports, Balance sheets                                            │
│  └── Size: 1920x2560 typical                                                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  STEP 1: TABLE DETECTION                                                                                ┃
┃  ═══════════════════════                                                                                ┃
┃                                                                                                         ┃
┃  MODEL: microsoft/table-transformer-detection                                                           ┃  
┃         (DETR-based, ResNet-50 backbone)                                                                ┃
┃                                                                                                         ┃
┃  INPUT:  RGB Image tensor [3, H, W]                                                                     ┃
┃  OUTPUT: List of bounding boxes [x1, y1, x2, y2] with confidence scores                                 ┃
┃                                                                                                         ┃
┃  METRICS:                                                                                               ┃
┃  ├── mAP@0.5: 0.95                                                                                      ┃
┃  ├── mAP@0.75: 0.82                                                                                     ┃
┃  └── Recall: 0.98                                                                                       ┃
┃                                                                                                         ┃
┃  PROCESSING:                                                                                            ┃
┃  ├── Image resize: max_size=800                                                                         ┃
┃  ├── Normalization: ImageNet mean/std                                                                   ┃
┃  ├── Confidence threshold: 0.7                                                                          ┃
┃  └── NMS: IoU threshold 0.5                                                                             ┃
┃                                                                                                         ┃
┃  TIME: ~15,000ms (GPU), ~45,000ms (CPU)                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                              │
                              ┌───────────────┴───────────────┐
                              │   Bounding Box [x1,y1,x2,y2]  │
                              │   Confidence: 99.99%          │
                              └───────────────────────────────┘
                                              │
                                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  STEP 2: TABLE STRUCTURE RECOGNITION (TSR)                                                              ┃
┃  ═══════════════════════════════════════════                                                            ┃
┃                                                                                                         ┃
┃  MODEL: microsoft/table-transformer-structure-recognition                                               ┃
┃         (DETR-based, same architecture as detection)                                                    ┃
┃                                                                                                         ┃
┃  INPUT:  Cropped table image from Step 1                                                                ┃
┃  OUTPUT: Detected rows, columns, cells with bounding boxes                                              ┃
┃                                                                                                         ┃
┃  DETECTED CLASSES:                                                                                      ┃
┃  ├── table row                                                                                          ┃
┃  ├── table column                                                                                       ┃
┃  ├── table column header                                                                                ┃
┃  ├── table projected row header                                                                         ┃
┃  └── table spanning cell                                                                                ┃
┃                                                                                                         ┃
┃  METRICS:                                                                                               ┃
┃  ├── TEDS: 0.89 (on PubTables-1M)                                                                       ┃
┃  ├── Cell-level F1: 0.85                                                                                ┃
┃  └── Row/Col IoU: 0.92                                                                                  ┃
┃                                                                                                         ┃
┃  PROCESSING:                                                                                            ┃
┃  ├── Intersection-based cell grid construction                                                          ┃
┃  └── Confidence filtering: 0.5 threshold                                                                ┃
┃                                                                                                         ┃
┃  TIME: ~2,000ms (GPU)                                                                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                              │
                              ┌───────────────┴───────────────┐
                              │   Structure: 36 rows × 7 cols │
                              │   Cell bboxes: [x1,y1,x2,y2]  │
                              └───────────────────────────────┘
                                              │
                                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  STEP 3: OCR TEXT EXTRACTION                                                                            ┃
┃  ════════════════════════════                                                                           ┃
┃                                                                                                         ┃
┃  MODEL: EasyOCR (GPU-accelerated)                                                                       ┃
┃         Alternative: PaddleOCR (better CER: 33.5% vs 46%)                                               ┃
┃                                                                                                         ┃
┃  INPUT:  Table image + cell grid structure from Step 2                                                  ┃
┃  OUTPUT: OCR text for each cell with position mapping                                                   ┃
┃                                                                                                         ┃
┃  PROCESSING FLOW:                                                                                       ┃
┃  ┌─────────────────────────────────────────────────────────────────────────┐                            ┃
┃  │  1. Full-image OCR                                                      │                            ┃
┃  │     ├── Run EasyOCR on entire table image                               │                            ┃
┃  │     └── Get text regions: [(bbox, text, confidence), ...]               │                            ┃
┃  │                                                                         │                            ┃
┃  │  2. Cell Assignment (Grid Construction)                                 │                            ┃
┃  │     ├── For each OCR text region:                                       │                            ┃
┃  │     │   ├── Calculate center point (cx, cy)                             │                            ┃
┃  │     │   ├── Find containing row (by y-position)                         │                            ┃
┃  │     │   └── Find containing column (by x-position)                      │                            ┃
┃  │     └── Build 2D grid: table_grid[row][col] = text                      │                            ┃
┃  │                                                                         │                            ┃
┃  │  3. Text Merging                                                        │                            ┃
┃  │     ├── Merge multiple text regions in same cell                        │                            ┃
┃  │     └── Sort by x-position for correct reading order                    │                            ┃
┃  └─────────────────────────────────────────────────────────────────────────┘                            ┃
┃                                                                                                         ┃
┃  METRICS:                                                                                               ┃
┃  ├── Text regions: 107                                                                                  ┃
┃  ├── Avg confidence: 0.88                                                                               ┃
┃  └── CER (vs GT): 46% (EasyOCR), 33.5% (PaddleOCR)                                                      ┃
┃                                                                                                         ┃
┃  OUTPUT FORMAT:                                                                                         ┃
┃  table_grid = {                                                                                         ┃
┃      0: {0: "Note", 1: "Group", 2: "Group", 3: "University", 4: "University"},  # header row            ┃
┃      1: {0: "Assets", 1: "2025", 2: "2024", 3: "2025", 4: "2024"},                                       ┃
┃      ...                                                                                                ┃
┃  }                                                                                                      ┃
┃                                                                                                         ┃
┃  TIME: ~7,000ms (GPU)                                                                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                              │
                              ┌───────────────┴───────────────┐
                              │   2D Grid: {row: {col: text}} │
                              │   107 cells with text values  │
                              └───────────────────────────────┘
                                              │
                                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  STEP 4: NUMERIC NORMALIZATION                                                                          ┃
┃  ══════════════════════════════                                                                         ┃
┃                                                                                                         ┃
┃  PURPOSE: Standardize financial numbers for comparison and computation                                  ┃
┃                                                                                                         ┃
┃  INPUT:  Raw OCR text for each cell                                                                     ┃
┃  OUTPUT: Normalized numeric values + confidence                                                         ┃
┃                                                                                                         ┃
┃  NORMALIZATION RULES:                                                                                   ┃
┃  ┌─────────────────────────────────────────────────────────────────────────┐                            ┃
┃  │  1. Currency Symbol Removal                                             │                            ┃
┃  │     ├── "$1,234.56" → "1,234.56"                                        │                            ┃
┃  │     ├── "RM 5,000" → "5,000"                                            │                            ┃
┃  │     └── "€10.50" → "10.50"                                              │                            ┃
┃  │                                                                         │                            ┃
┃  │  2. Thousand Separator Handling                                         │                            ┃
┃  │     ├── "1,234,567" → 1234567                                           │                            ┃
┃  │     └── "1.234.567" (EU format) → 1234567                               │                            ┃
┃  │                                                                         │                            ┃
┃  │  3. Bracket Notation for Negatives                                      │                            ┃
┃  │     ├── "(1,234)" → -1234                                               │                            ┃
┃  │     └── "(50.00)" → -50.00                                              │                            ┃
┃  │                                                                         │                            ┃
┃  │  4. Percentage Handling                                                 │                            ┃
┃  │     ├── "25%" → 0.25 (normalized) or 25 (as-is)                         │                            ┃
┃  │     └── "12.5%" → 0.125                                                 │                            ┃
┃  │                                                                         │                            ┃
┃  │  5. OCR Error Correction                                                │                            ┃
┃  │     ├── "l,234" (letter l) → "1,234"                                    │                            ┃
┃  │     ├── "O00" (letter O) → "000"                                        │                            ┃
┃  │     └── "5B" → "58" (common OCR error)                                  │                            ┃
┃  └─────────────────────────────────────────────────────────────────────────┘                            ┃
┃                                                                                                         ┃
┃  METRICS:                                                                                               ┃
┃  ├── Numeric cells identified: 79                                                                       ┃
┃  ├── Successfully parsed: 75                                                                            ┃
┃  └── Parse accuracy: 94.9%                                                                              ┃
┃                                                                                                         ┃
┃  TIME: <1ms (rule-based, no ML)                                                                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                              │
                              ┌───────────────┴───────────────┐
                              │   Normalized values + types   │
                              │   75/79 cells parsed (94.9%)  │
                              └───────────────────────────────┘
                                              │
                                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  STEP 5: SEMANTIC CLASSIFICATION                                                                        ┃
┃  ════════════════════════════════                                                                       ┃
┃                                                                                                         ┃
┃  PURPOSE: Classify each cell's role in the table structure                                              ┃
┃                                                                                                         ┃
┃  INPUT:  Cell text + position + normalized value                                                        ┃
┃  OUTPUT: Cell type classification                                                                       ┃
┃                                                                                                         ┃
┃  CLASSIFICATION CATEGORIES:                                                                             ┃
┃  ┌─────────────────────────────────────────────────────────────────────────┐                            ┃
┃  │  1. column_header (3 cells)                                             │                            ┃
┃  │     ├── Definition: Top row containing column titles                    │                            ┃
┃  │     ├── Pattern: First 1-2 rows, non-numeric                            │                            ┃
┃  │     └── Examples: "Group 2025", "University 2024"                       │                            ┃
┃  │                                                                         │                            ┃
┃  │  2. row_header (18 cells)                                               │                            ┃
┃  │     ├── Definition: First column(s) containing row labels               │                            ┃
┃  │     ├── Pattern: Column 0, descriptive text                             │                            ┃
┃  │     └── Examples: "Total assets", "Cash and cash equivalents"           │                            ┃
┃  │                                                                         │                            ┃
┃  │  3. section_header (2 cells)                                            │                            ┃
┃  │     ├── Definition: Headers that span or group rows                     │                            ┃
┃  │     ├── Pattern: Bold, spans multiple columns                           │                            ┃
┃  │     └── Examples: "Non-current assets", "Current liabilities"           │                            ┃
┃  │                                                                         │                            ┃
┃  │  4. numeric (79 cells)                                                  │                            ┃
┃  │     ├── Definition: Cells containing numeric values                     │                            ┃
┃  │     ├── Pattern: Matches numeric regex                                  │                            ┃
┃  │     └── Examples: "1,511,947", "(50,000)", "25%"                         │                            ┃
┃  │                                                                         │                            ┃
┃  │  5. total (4 cells)                                                     │                            ┃
┃  │     ├── Definition: Rows showing totals/subtotals                       │                            ┃
┃  │     ├── Pattern: Contains "Total", "Subtotal"                           │                            ┃
┃  │     └── Examples: "Total assets", "Total current liabilities"           │                            ┃
┃  │                                                                         │                            ┃
┃  │  6. data (1 cell)                                                       │                            ┃
┃  │     ├── Definition: Other non-numeric data cells                        │                            ┃
┃  │     └── Examples: Notes, references                                     │                            ┃
┃  └─────────────────────────────────────────────────────────────────────────┘                            ┃
┃                                                                                                         ┃
┃  CLASSIFICATION RULES (Heuristic-based):                                                                ┃
┃  ├── Position-based: row=0 → column_header, col=0 → row_header                                          ┃
┃  ├── Content-based: numeric pattern → numeric type                                                      ┃
┃  ├── Keyword-based: "Total" → total type                                                                ┃
┃  └── Spanning: Multi-column cells → section_header                                                      ┃
┃                                                                                                         ┃
┃  METRICS:                                                                                               ┃
┃  ├── Total cells: 107                                                                                   ┃
┃  └── Distribution: column_header(3), row_header(18), section_header(2),                                 ┃
┃                    numeric(79), total(4), data(1)                                                       ┃
┃                                                                                                         ┃
┃  TIME: ~1ms (rule-based)                                                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                              │
                              ┌───────────────┴───────────────┐
                              │   Classified cells:           │
                              │   Each cell has semantic type │
                              └───────────────────────────────┘
                                              │
                                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  STEP 6: TABLE QA (Question Answering)                                                                  ┃
┃  ══════════════════════════════════════                                                                 ┃
┃                                                                                                         ┃
┃  PURPOSE: Answer natural language questions about table data                                            ┃
┃                                                                                                         ┃
┃  INPUT:  Question (row_key, col_key) + OCR Grid from Step 3                                             ┃
┃  OUTPUT: Predicted answer value                                                                         ┃
┃                                                                                                         ┃
┃  QA ALGORITHM:                                                                                          ┃
┃  ┌─────────────────────────────────────────────────────────────────────────┐                            ┃
┃  │  1. Row Lookup                                                          │                            ┃
┃  │     ├── For each row in OCR grid:                                       │                            ┃
┃  │     │   ├── Get first column text (row header)                          │                            ┃
┃  │     │   ├── Calculate fuzzy similarity with row_key                     │                            ┃
┃  │     │   └── Use difflib.SequenceMatcher (threshold: 0.6)                │                            ┃
┃  │     └── Return row index with highest match                             │                            ┃
┃  │                                                                         │                            ┃
┃  │  2. Column Lookup                                                       │                            ┃
┃  │     ├── For each column in OCR grid:                                    │                            ┃
┃  │     │   ├── Get header row text (row 0 or 1)                            │                            ┃
┃  │     │   ├── Calculate fuzzy similarity with col_key                     │                            ┃
┃  │     │   └── Use difflib.SequenceMatcher (threshold: 0.6)                │                            ┃
┃  │     └── Return column index with highest match                          │                            ┃
┃  │                                                                         │                            ┃
┃  │  3. Cell Retrieval                                                      │                            ┃
┃  │     └── Return table_grid[row_idx][col_idx]                             │                            ┃
┃  └─────────────────────────────────────────────────────────────────────────┘                            ┃
┃                                                                                                         ┃
┃  EXAMPLE:                                                                                               ┃
┃  ├── Question: "What is 'Property, plant and equipment' for 'Group 2025'?"                              ┃
┃  ├── row_key: "Property, plant and equipment"                                                           ┃
┃  ├── col_key: "Group 2025"                                                                              ┃
┃  ├── Row match: row 5 (similarity: 0.95)                                                                ┃
┃  ├── Col match: col 1 (similarity: 0.90)                                                                ┃
┃  └── Answer: "1,511,947" ✅                                                                             ┃
┃                                                                                                         ┃
┃  ERROR SOURCES:                                                                                         ┃
┃  ├── OCR errors in headers → wrong row/col match                                                        ┃
┃  ├── Multi-row headers → header detection failure                                                       ┃
┃  ├── Merged cells → grid misalignment                                                                   ┃
┃  └── Similar row names → fuzzy match ambiguity                                                          ┃
┃                                                                                                         ┃
┃  METRICS:                                                                                               ┃
┃  ├── Questions: 10                                                                                      ┃
┃  ├── Correct: 3                                                                                         ┃
┃  └── Accuracy: 30.0%                                                                                    ┃
┃                                                                                                         ┃
┃  TIME: ~14ms (lookup only, no ML inference)                                                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Predicted Answer                                                                               │
│  ├── Value: "1,511,947"                                                                                 │
│  ├── Confidence: Based on fuzzy match score                                                             │
│  └── Verified against Ground Truth: ✅ CORRECT or ❌ INCORRECT                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

---

## Performance Summary

### Time Breakdown (Total: 24,549ms)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      TIME DISTRIBUTION                                     │
├────────────────────────────────────────────────────────────────────────────┤
│  Step 1: Detection    ████████████████████████████████████████  62.7%     │
│  Step 2: TSR          ████████                                   7.9%     │
│  Step 3: OCR          ████████████████                          29.4%     │
│  Step 4: Numeric      │                                          0.0%     │
│  Step 5: Semantic     │                                          0.0%     │
│  Step 6: QA           │                                          0.1%     │
└────────────────────────────────────────────────────────────────────────────┘
```

### Accuracy Cascade (Error Propagation)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           ACCURACY CASCADE                                               │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  Step 1: Detection     ██████████████████████████████████████████████████  99.99%       │
│      ↓ (propagates)                                                                      │
│  Step 2: TSR           ██████████████████████████████████████████████████  ~95%         │
│      ↓ (propagates)                                                                      │
│  Step 3: OCR           ██████████████████████████████████████████████      88% conf     │
│      ↓ (propagates)    ████████████████████████████████████                54% CER      │
│  Step 4: Numeric       ███████████████████████████████████████████████     94.9%        │
│      ↓ (propagates)                                                                      │
│  Step 5: Semantic      ███████████████████████████████████████████████     ~95%         │
│      ↓ (propagates)                                                                      │
│  Step 6: QA            ███████████████                                     30.0% ←─ FINAL│
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: OCR errors (54% CER) are the main bottleneck. They cascade through:
- Wrong text in headers → Fuzzy match fails
- Wrong values in cells → Even if found, value is wrong

---

## Model Comparison

| Component | Model | Performance | Notes |
|-----------|-------|-------------|-------|
| Detection | Table Transformer | mAP 95% | Best for general tables |
| Detection | Docling | F1 85% | Good for documents |
| TSR | Table Transformer | TEDS 89% | Standard choice |
| OCR | EasyOCR | CER 46% | GPU-accelerated |
| OCR | PaddleOCR | CER 33.5% | Better accuracy |
| Semantic | Rule-based | ~95% | Fast, no ML |
| QA | Fuzzy lookup | 30% | Limited by OCR |

---

## For Thesis Diagram

Suggested visual representation:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   INPUT     │───▶│   STEP 1    │───▶│   STEP 2    │───▶│   STEP 3    │───▶│   STEP 4    │───▶│   STEP 5    │
│   IMAGE     │    │  DETECTION  │    │    TSR      │    │    OCR      │    │  NUMERIC    │    │  SEMANTIC   │
│             │    │  (DETR)     │    │  (DETR)     │    │  (EasyOCR)  │    │  (Rules)    │    │  (Rules)    │
│  1920×2560  │    │  15,382ms   │    │  1,932ms    │    │  7,219ms    │    │  <1ms       │    │  <1ms       │
│  Financial  │    │  mAP: 95%   │    │  TEDS: 89%  │    │  CER: 46%   │    │  94.9%      │    │  ~95%       │
│  Document   │    │             │    │             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                                                      │
                                                                                                      ▼
                                                                                              ┌─────────────┐
                                                                                              │   STEP 6    │
                                                                                              │    QA       │
                                                                                              │  (Lookup)   │
                                                                                              │  14ms       │
                                                                                              │  30.0%      │
                                                                                              └─────────────┘
```

---

## Fairness Statement

This pipeline test is **FAIR** because:

1. **Step 1-5**: Pure extraction from image, no ground truth used
2. **Step 6 QA**: Only uses OCR-extracted grid data (`table_grid`)
3. **Ground Truth**: Used ONLY for final answer verification
4. **No "cheating"**: Cell values from GT are NOT used during prediction

Previous tests were unfair because they built the lookup table from GT cell values, effectively "peeking" at the answers.

---

## Files Generated

- `FAIR_TEST_REPORT.md` - Summary report
- `step1_detection.png` - Table detection visualization
- `step2_structure.png` - TSR structure visualization
- `step3_ocr.png` - OCR text overlay
- `step5_semantic.png` - Semantic classification
- `step6_qa.png` - QA results
- `results.json` - Full JSON results
