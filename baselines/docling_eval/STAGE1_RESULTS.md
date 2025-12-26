# Stage I Evaluation Results - FinTabNet 100 Samples
## FYP Report: Table Extraction Pipeline Comparison

**Date**: 2025-12-27  
**Dataset**: FinTabNet (OTSL format)  
**Samples**: n=100 (random seed=42, reproducible)  
**Fair Condition**: 公平条件：Docling 与 TT 均为 image-only；不使用 oracle GT cell bbox。

---

## 📊 Table A: Component-wise Results (TSR-only)

| Component | Metric | Old Pipeline (TT v1.1) | Docling (TableFormer) | Winner |
|-----------|--------|------------------------|----------------------|--------|
| **Detection** | P/R/F1 | N/A† | N/A† | - |
| **TSR-only** | TEDS_struct | **0.8336** ± 0.0877 | **0.8618** ± 0.0865 | Docling (+2.8%) |
| **OCR-only** | CER | N/A‡ | N/A‡ | - |
| | WER | N/A‡ | N/A‡ | - |

**Notes**:
- † FinTabNet provides pre-cropped table images, so table detection evaluation is not applicable.
- ‡ PaddleOCR import error on this system; OCR evaluation skipped.

---

## 📊 Table B: End-to-End Stage I Results

| Metric | Old Pipeline | Docling | Notes |
|--------|-------------|---------|-------|
| TEDS_full | N/A‡ | N/A‡ | OCR failed |
| Success Rate | 0% | 0% | OCR failed |

---

## 🔬 Key Findings

### TSR (Table Structure Recognition) Comparison

| Metric | Old Pipeline (TT v1.1) | Docling (TableFormer) |
|--------|------------------------|----------------------|
| **Mean TEDS** | 0.8336 | **0.8618** |
| **Std Dev** | 0.0877 | 0.0865 |
| **Median TEDS** | 0.8441 | **0.8745** |
| **Sample Count** | 100 | 100 |

**Analysis**:
1. **Docling TableFormer outperforms Table Transformer v1.1** on TSR-only evaluation
   - +2.82% improvement in mean TEDS (0.8618 vs 0.8336)
   - +3.04% improvement in median TEDS (0.8745 vs 0.8441)
2. Both methods show similar variance (std ~0.087)
3. Both methods operate in **image-only mode** without oracle GT cell bbox (fair comparison)

---

## 🆚 Comparison with Official Results

| Method | Evaluation Setting | TEDS_struct | Notes |
|--------|-------------------|-------------|-------|
| **Docling Official** | TSR + **Oracle GT cell bbox** | 0.90 | Uses GT cell bbox as input |
| **Our Docling Test** | TSR **image-only** | 0.8618 | Fair comparison, no GT bbox |
| **Our TT v1.1 Test** | TSR **image-only** | 0.8336 | Fair comparison |

**Key Insight**: 
- Official Docling TEDS=0.90 is inflated due to Oracle GT cell bbox usage
- When evaluated fairly (image-only), Docling achieves 0.86, Table Transformer achieves 0.83

---

## 📁 Generated Files

```
stage1_results/
├── samples_100.json                              # Reproducible sample IDs (seed=42)
├── stage1_results_20251227_001818.json           # Full results JSON
├── results_tsr_old_pipeline_20251227_001818.csv  # Per-sample TSR results (TT v1.1)
├── results_tsr_docling_20251227_001818.csv       # Per-sample TSR results (TableFormer)
├── results_e2e_old_pipeline_20251227_001818.csv  # Per-sample E2E results (TT v1.1)
└── results_e2e_docling_20251227_001818.csv       # Per-sample E2E results (TableFormer)
```

---

## 🔧 Configuration

```yaml
Dataset: D:/datasets/FinTabNet_OTSL/data
Split: val
Samples: 100
Random Seed: 42
Models:
  - Table Transformer: microsoft/table-transformer-structure-recognition-v1.1-all
  - TableFormer: docling-ibm-models (accurate mode)
```

---

## ⚠️ Known Issues

1. **PaddleOCR Import Error**:
   ```
   Error: Can not import paddle core while this file exists: 
   C:\Users\User\Documents\financial-document-ai\venv\Lib\site-packages\paddle\base\libpaddle.pyd
   ```
   - This prevented OCR and End-to-End evaluation
   - Solution: Reinstall PaddleOCR or use alternative OCR

2. **Detection N/A**:
   - FinTabNet provides pre-cropped table images
   - Full-page PDFs required for table detection evaluation

---

## 📈 Conclusion for FYP Report

Based on **fair, image-only evaluation** on FinTabNet (n=100):

| Aspect | Old Pipeline | Docling | Winner |
|--------|-------------|---------|--------|
| TSR Accuracy | 83.4% | **86.2%** | Docling |
| Variance | Similar | Similar | Tie |

**Recommendation**: 
- For TSR-only tasks, Docling TableFormer shows ~3% improvement over Table Transformer v1.1
- Official Docling TEDS=0.90 should be cited with caveat about Oracle GT cell bbox usage
- For end-to-end comparison, OCR integration needs to be fixed
