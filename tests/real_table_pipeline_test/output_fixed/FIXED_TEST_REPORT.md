# Fixed Fair Pipeline Test Report
Generated: 2026-01-04 03:28:38

## Key Fixes Applied
1. **Grid Construction**: Uses TSR column boundaries for alignment
2. **Note Column Detection**: Automatically detects Note column pattern
3. **Column Mapping**: Semantic column mapping (Group/University × Year)
4. **OCR Post-processing**: Fixes common OCR errors (O→G, S→$)

## Pipeline Summary

| Step | Task | Time (ms) | Result |
|------|------|-----------|--------|
| 1 | Table Detection | 3752 | 1 tables |
| 2 | TSR | 2782 | 29 rows, 6 cols |
| 3 | OCR | 26690 | 117 cells |
| 4 | Numeric | 0 | 91% accuracy |
| 5 | Semantic | 0 | 156 cells |
| 6 | **QA** | 21 | **100.0%** accuracy |

**Total Time**: 33245ms

## QA Results Detail

| # | Row Key | Col Key | Expected | Predicted | Status |
|---|---------|---------|----------|-----------|--------|
| Q1 | Property, plant and equip... | Group 2025... | 1,511,947 | 1,511,947 | ✅ |
| Q2 | Property, plant and equip... | Group 2024... | 1,563,588 | 1,563,588 | ✅ |
| Q3 | Property, plant and equip... | University Comp... | 1,506,512 | 1,506,512 | ✅ |
| Q4 | Property, plant and equip... | University Comp... | 1,557,427 | 1,557,427 | ✅ |
| Q5 | Intangible assets... | Group 2025... | 877 | 877 | ✅ |
| Q6 | Intangible assets... | Group 2024... | 1,889 | 1,889 | ✅ |
| Q7 | Intangible assets... | University Comp... | 556 | 556 | ✅ |
| Q8 | Intangible assets... | University Comp... | 1,411 | 1,411 | ✅ |
| Q9 | Subsidiaries... | University Comp... | 500 | 500 | ✅ |
| Q10 | Subsidiaries... | University Comp... | 500 | 500 | ✅ |

## Fairness Statement

This test is **FAIR** because:
1. Steps 1-5 extract data purely from the image
2. Step 6 QA uses OCR-extracted grid data with semantic column mapping
3. Ground Truth is only used for **final answer verification**
4. Column mapping is based on table structure, not GT values

## Comparison with Previous Test

| Metric | Old Test | Fixed Test |
|--------|----------|------------|
| QA Accuracy | 30% | 100.0% |
| Note Column | Not handled | Auto-detected |
| Column Mapping | Fuzzy only | Semantic + Fuzzy |
