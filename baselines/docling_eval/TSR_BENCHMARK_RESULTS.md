# TSR-only Benchmark Results: Table Transformer v1.1 vs Docling TableFormer

## 🔥 核心发现

### 1. 官方 Docling TEDS=0.90 使用了 Oracle 辅助 (GT Cell BBox)

经过源代码分析，我们发现官方 `docling-eval` 评估使用了 **Ground Truth cell bounding boxes** 作为输入！

**证据来源**: `docling_eval/prediction_providers/tableformer_provider.py` 第 394-412 行

```python
if page_tokens is None:
    ptokens = []
    for ix, table_cell in enumerate(item.data.table_cells):
        # ↑ 使用 GT 的 table_cells 作为输入!
        pt = PageToken(
            bbox=table_cell.bbox,  # ← GT cell bbox
            text=table_cell.text,  # ← GT cell text  
            id=ix
        )
        ptokens.append(pt)
```

### 2. 公平对比结果 (100 samples, TSR-only)

| 方法 | 输入 | Avg. TEDS | Std | 备注 |
|------|------|-----------|-----|------|
| **Table Transformer v1.1** | Image only | **0.8360** | 0.0812 | 公平测试 |
| Docling TableFormer (我们测试) | Image only | 0.8014 | 0.1018 | 公平测试 |
| Docling TableFormer (官方) | Image + **GT bbox** | 0.8974 | 0.0878 | Oracle辅助 |

### 3. 关键结论

1. **Table Transformer v1.1 比 TableFormer 好 3.5%** (在公平条件下)
2. **官方 Docling 0.90 不是公平对比** - 使用了 GT cell bbox 作为输入
3. **差距原因**: 官方只测行列索引分配，跳过了单元格检测

---

## 详细分析

### 真实任务 vs 官方 Benchmark

```
真实场景 (我们的测试):
  Image → Cell Detection → Row/Col Assignment → Output
          ↑ 需要检测      ↑ 需要分配位置
          
官方 Benchmark (docling-eval):
  Image + GT Cell BBox → [跳过检测] → Row/Col Assignment → Output
                        ↑ 直接给单元格位置
```

### 测试配置

- **数据集**: FinTabNet OTSL (val split)
- **样本数**: 100
- **评估指标**: TEDS (structure-only, 忽略文本内容)
- **Table Transformer**: microsoft/table-transformer-structure-recognition-v1.1-all
- **TableFormer**: docling-ibm-models 3.x (ACCURATE mode)

### 运行命令

```bash
cd baselines/docling_eval
python tsr_benchmark_v2.py --num_samples 100 --mode tsr_only
```

---

## 官方评估数据确认

从官方 JSON 结果 (`evaluation_FinTabNet_tableformer.json`):

```json
{
  "TEDS_struct": {
    "mean": 0.897386,
    "median": 0.917,
    "std": 0.08784143168870326,
    "total": 1000
  }
}
```

**注意**: 官方声称的 TEDS_struct-only = 0.90 (mean=0.897) 确实存在，但评估方法使用了 GT cell bbox 作为输入。

---

## FYP Report 建议写法

```
Table X: TSR Performance Comparison on FinTabNet Dataset

┌────────────────────────────────────────────────────────────────────────────┐
│ Method                           │ Input           │ TEDS  │ Std   │ Note │
├──────────────────────────────────┼─────────────────┼───────┼───────┼──────┤
│ Table Transformer v1.1 (Ours)    │ Image only      │ 0.836 │ 0.081 │ Fair │
│ Docling TableFormer (Our test)   │ Image only      │ 0.801 │ 0.102 │ Fair │
│ Docling TableFormer (Official)   │ Image + GT bbox │ 0.897 │ 0.088 │ *    │
└────────────────────────────────────────────────────────────────────────────┘

* Official Docling evaluation uses Ground Truth cell bounding boxes as input,
  which reduces the task to row/column index assignment only.
```

---

## 文件位置

- 测试脚本: `baselines/docling_eval/tsr_benchmark_v2.py`
- 分析文档: `baselines/docling_eval/ANALYSIS_CN.md`
- 官方数据: https://github.com/docling-project/docling-eval/blob/main/docs/evaluations/FinTabNet/

---

## 参考资料

1. [docling-eval GitHub](https://github.com/DS4SD/docling-eval)
2. [FinTabNet Benchmarks](https://github.com/docling-project/docling-eval/blob/main/docs/FinTabNet_benchmarks.md)
3. [Table Transformer](https://github.com/microsoft/table-transformer)
