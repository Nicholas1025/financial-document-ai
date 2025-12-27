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
