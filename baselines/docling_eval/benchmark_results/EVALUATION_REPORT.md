# Financial Document AI Pipeline - Comprehensive Evaluation Report

**Generated:** 2025-12-28 01:44:58

---

## Overview

This report presents evaluation results for all pipeline stages:

- **Step 2:** Table Structure Recognition

---

## Step 2: Table Structure Recognition

**Dataset:** FinTabNet OTSL (test split)
**Samples:** 1000
**Ground Truth:** HTML table structure from FinTabNet annotations
**Metrics:** TEDS (Tree Edit Distance Similarity) - structural only

### Results

| Method | Mean TEDS | Std | Median | Min | Max |
|--------|-----------|-----|--------|-----|-----|
| Table Transformer v1.1 | 0.8293 | 0.0904 | 0.8468 | 0.3625 | 0.9692 |
| Docling TableFormer | 0.8137 | 0.1104 | 0.8333 | 0.012 | 0.9716 |

**Histogram:** See `C:\Users\User\Documents\financial-document-ai\baselines\docling_eval\benchmark_results\figures\step2_tsr_teds_histogram.png`

---
End-to-End Demo Results
Step	Description	Results
Step 1: Detection	Docling LayoutPredictor	1 table detected @ 97.5% confidence
Step 2: TSR	Table Transformer v1.1	25 rows × 4 cols = 100 cells
Step 3: OCR	PaddleOCR (subprocess)	94 text regions detected
Step 4: Numeric	NumericNormalizer	59/100 cells parsed (59%)
Step 5: Semantic	Heuristic Classifier	6 types classified
Evaluation Against Ground Truth
Metric	Score
OCR Exact Match	71/92 (77.2%)
OCR Avg CER	12.2%
Numeric Match	58/65 (89.2%)
Semantic Match	90/100 (90.0%)
The pipeline successfully:

✅ Detected the table with high confidence
✅ Correctly identified 25 rows and 4 columns (matching GT)
✅ Extracted text from cells (77% exact match)
✅ Parsed numeric values (89% match)
✅ Classified semantic types (90% accuracy)
The output shows the table preview with proper content extraction. Some OCR errors like RM'OOO instead of RM'000 are typical OCR mistakes that could be fixed with post-processing rules.

Conclude overall phase 1, 