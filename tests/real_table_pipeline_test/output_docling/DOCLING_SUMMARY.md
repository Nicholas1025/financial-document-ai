
# Pipeline Test Results: Docling Detection
Generated: 2026-01-04 01:24:19

## Configuration
- **Step 1**: Docling Detection (instead of Table Transformer)
- **Step 2**: Table Transformer TSR
- **Step 3**: EasyOCR
- **Step 4-6**: Same as baseline

## Results Summary

| Step | Metric | Result | Status |
|------|--------|--------|--------|
| 1 | Detection | 1 tables | ✅ |
| 2 | Row Detection | 38/26 | 68.4% |
| 2 | Col Detection | 7/6 | 85.7% |
| 3 | OCR Match Rate | - | 93.2% |
| 4 | Numeric Parse | - | 93.6% |
| 5 | Semantic | - | 15.5% |
| 6 | QA Accuracy | 3/10 | 30.0% |

## Step 1: Docling Detection Details
- Method: Docling LayoutPredictor
- Tables detected: 1
- Inference time: 7827.2ms

## Step 6: QA Details
- ✅ Q1: Expected `1,511,947`, Got `1,511,947`
- ❌ Q2: Expected `8,516,127`, Got `None`
- ✅ Q3: Expected `1,640,416`, Got `1,640,416`
- ✅ Q4: Expected `5,467,851`, Got `5,467,851`
- ❌ Q5: Expected `7,630,174`, Got `None`
- ❌ Q6: Expected `650,000`, Got `and`
- ❌ Q7: Expected `1,411`, Got `None`
- ❌ Q8: Expected `1,895,006`, Got `7,011,294`
- ❌ Q9: Expected `302,307`, Got `451,845`
- ❌ Q10: Expected `558,128`, Got `Student loans`
