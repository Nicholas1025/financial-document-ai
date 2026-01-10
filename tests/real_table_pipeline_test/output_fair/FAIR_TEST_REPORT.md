# Fair Pipeline Test Report
Generated: 2026-01-04 01:37:39

## Test Configuration
- **Image**: nanyang_sample1.png
- **Fair Test**: Yes (Step 6 uses OCR data only, not GT values)

## Pipeline Summary

| Step | Task | Time (ms) | Result |
|------|------|-----------|--------|
| 1 | Table Detection | 15382 | 1 tables, conf=99.99% |
| 2 | TSR | 1932 | 36 rows, 7 cols |
| 3 | OCR | 7219 | 107 cells, conf=0.88 |
| 4 | Numeric | 0 | 95% parse accuracy |
| 5 | Semantic | 1 | 107 cells classified |
| 6 | **QA** | 14 | **30.0%** accuracy (3/10) |

**Total Time**: 24549ms

## QA Results Detail

| # | Row Key | Col Key | Expected | Predicted | Status |
|---|---------|---------|----------|-----------|--------|
| 1 | Property, plant and ... | Group 2025... | 1,511,947 | 1,511,947 | ✅ |
| 2 | Total assets... | University Comp... | 8,516,127 | None | ❌ |
| 3 | Cash and cash equiva... | Group 2024... | 1,640,416 | 1,589,900 | ❌ |
| 4 | Other investments... | University Comp... | 5,467,851 | 11 | ❌ |
| 5 | Total current assets... | Group 2025... | 7,630,174 | None | ❌ |
| 6 | Loans and borrowings... | Group 2025... | 650,000 | 17 | ❌ |
| 7 | Intangible assets... | University Comp... | 1,411 | None | ❌ |
| 8 | Total non-current li... | Group 2024... | 1,895,006 | 533,963 | ❌ |
| 9 | - Sinking fund... | University Comp... | 302,307 | 302,307 | ✅ |
| 10 | Trade and other rece... | Group 2025... | 558,128 | 558,128 | ✅ |

## Fairness Statement

This test is **FAIR** because:
1. Step 1-5 extract data purely from the image
2. Step 6 QA only uses OCR-extracted grid data
3. Ground Truth is only used for **final answer verification**
4. No GT cell values are used during prediction

## Key Metrics

- **End-to-End QA Accuracy**: 30.0%
- **OCR Confidence**: 0.88
- **Total Processing Time**: 24549ms
