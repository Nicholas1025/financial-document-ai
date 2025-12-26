# Docling Baseline for Table Structure Recognition

This directory contains wrapper scripts and adapters for comparing our pipeline against docling's official FinTabNet benchmark.

## Overview

```
baselines/docling_eval/
├── __init__.py
├── requirements-baselines.txt      # Pinned dependencies
├── common_schema.py               # Common Table JSON intermediate format
├── adapter_docling.py             # Docling output → Common JSON
├── adapter_old_pipeline.py        # Old pipeline output → Common JSON
├── eval_metrics.py                # TEDS, Cell-F1, Structural Accuracy
├── run_docling_benchmark.py       # Docling FinTabNet benchmark wrapper
└── run_compare_structures.py      # Head-to-head comparison script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r baselines/docling_eval/requirements-baselines.txt
```

### 2. Run Docling FinTabNet Benchmark (Upstream Baseline)

```bash
# Full benchmark (1000 samples)
python baselines/docling_eval/run_docling_benchmark.py --benchmark FinTabNet --max-samples 1000

# Quick test (100 samples)
python baselines/docling_eval/run_docling_benchmark.py --benchmark FinTabNet --max-samples 100
```

### 3. Compare Structures (Docling vs Old Pipeline)

```bash
# Compare on local samples
python baselines/docling_eval/run_compare_structures.py --samples ./data/samples --max-samples 50

# Compare on FinTabNet samples
python baselines/docling_eval/run_compare_structures.py --use-fintabnet --max-samples 100
```

## Common Table JSON Schema

We define a system-agnostic intermediate format for comparing table structures:

```json
{
  "schema_version": "1.0.0",
  "source_system": "docling",
  "source_version": "docling=2.0.0",
  "table_id": "sample_001",
  "created_at": "2025-12-26T10:00:00",
  "structure": {
    "num_rows": 10,
    "num_cols": 5,
    "has_header": true,
    "num_header_rows": 1,
    "has_spanning_cells": false
  },
  "cells": [
    {
      "row": 0,
      "col": 0,
      "text": "Description",
      "is_header": true,
      "span": {"row_span": 1, "col_span": 1}
    }
  ],
  "html": "<table>...</table>",
  "grid": [["Description", "2024", "2023"], ...],
  "metadata": {}
}
```

## Adapters

### Docling → Common JSON

```python
from baselines.docling_eval.adapter_docling import DoclingAdapter

adapter = DoclingAdapter()

# From HTML (common output format)
common_table = adapter.from_html(html_string, table_id="sample_001")

# From OTSL (benchmark format)
common_table = adapter.from_otsl(otsl_tokens, cell_texts, table_id="sample_001")
```

### Old Pipeline → Common JSON

```python
from baselines.docling_eval.adapter_old_pipeline import OldPipelineAdapter

adapter = OldPipelineAdapter()

# From pipeline output dict
common_table = adapter.from_pipeline_output(pipeline_result, table_id="sample_001")

# From saved JSON
common_table = adapter.from_json_file("outputs/result.json")
```

## Evaluation Metrics

### TEDS (Tree-Edit-Distance-based Similarity)

```python
from baselines.docling_eval.eval_metrics import TEDSCalculator

calc = TEDSCalculator()
score = calc.calculate(pred_table, gt_table, structure_only=True)  # 0-1
```

### Cell F1

```python
from baselines.docling_eval.eval_metrics import calculate_cell_f1

metrics = calculate_cell_f1(pred_table, gt_table, match_text=True)
# {'precision': 0.95, 'recall': 0.92, 'f1': 0.93}
```

### Structural Accuracy

```python
from baselines.docling_eval.eval_metrics import calculate_structural_accuracy

metrics = calculate_structural_accuracy(pred_table, gt_table)
# {'row_count_match': True, 'col_count_match': True, ...}
```

## Output Example

```
======================================================================
STRUCTURE RECOGNITION COMPARISON REPORT
======================================================================
Timestamp: 2025-12-26T10:30:00
Samples: 100 / 100

----------------------------------------------------------------------
Metric                              Docling     OldPipeline
----------------------------------------------------------------------
TEDS (struct-only)                   0.9012          0.8756
TEDS (with text)                     0.8834          0.8521
Cell F1                              0.9123          0.8890
----------------------------------------------------------------------

Head-to-Head Results:
  Docling wins:         42 (42.0%)
  Old Pipeline wins:    35 (35.0%)
  Ties:                 23 (23.0%)

======================================================================
```

## Version Locking

All baseline dependencies are pinned in `requirements-baselines.txt`:

```
docling-eval>=0.3.0
# Commit: 629a451d7b75e274352a1f21710316e47fc7a80a
```

## Notes

- **Docling baseline is for reference only** - we don't modify its internal algorithms
- **Old pipeline remains unchanged** - baseline scripts only read its outputs
- **Common JSON format** enables fair comparison between any table extraction systems
