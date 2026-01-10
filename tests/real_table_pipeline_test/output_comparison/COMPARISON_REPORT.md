# Pipeline Comparison Report
Generated: 2026-01-04 01:27:33

## Test Configuration
- **Image**: tests\real_table_pipeline_test\nanyang_sample1.png
- **Questions**: 10

## Results Summary

| Metric | Table Transformer | Docling E2E | Winner |
|--------|-------------------|-------------|--------|
| **Processing Time** | 23378ms | 109276ms | Table Transformer |
| **QA Correct** | 3/10 | 2/10 | **Table Transformer** |
| **Accuracy** | **30.0%** | **20.0%** | **Table Transformer** |

## Winner: **Table Transformer**

## Detailed QA Results

### Table Transformer
- ✅ Q1: Expected `1,511,947`, Got `1,511,947`
- ❌ Q2: Expected `8,516,127`, Got `None`
- ❌ Q3: Expected `1,640,416`, Got `1,589,900`
- ✅ Q4: Expected `5,467,851`, Got `5,467,851`
- ❌ Q5: Expected `7,630,174`, Got `None`
- ✅ Q6: Expected `650,000`, Got `650,000`
- ❌ Q7: Expected `1,411`, Got `None`
- ❌ Q8: Expected `1,895,006`, Got `None`
- ❌ Q9: Expected `302,307`, Got `None`
- ❌ Q10: Expected `558,128`, Got `10`

### Docling End-to-End
- ❌ Q1: Expected `1,511,947`, Got ``
- ❌ Q2: Expected `8,516,127`, Got ``
- ❌ Q3: Expected `1,640,416`, Got `1,589,900`
- ✅ Q4: Expected `5,467,851`, Got `5,467,851`
- ❌ Q5: Expected `7,630,174`, Got ``
- ❌ Q6: Expected `650,000`, Got `17`
- ✅ Q7: Expected `1,411`, Got `1,411`
- ❌ Q8: Expected `1,895,006`, Got ``
- ❌ Q9: Expected `302,307`, Got `None`
- ❌ Q10: Expected `558,128`, Got `10`

## Analysis

### Speed Comparison
- Table Transformer: **23378ms** (21.4% of Docling time)
- Docling: **109276ms**
- Speed ratio: Docling is **4.7x slower**

### Accuracy Comparison
- Table Transformer: **30.0%** (3/10)
- Docling: **20.0%** (2/10)
- Accuracy difference: **10.0%**

### Conclusion
Table Transformer pipeline achieves higher QA accuracy while being significantly faster.
