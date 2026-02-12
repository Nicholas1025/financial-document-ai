"""Debug script: run pipeline on CIMB sample and analyze validation."""
import json
from modules.pipeline import FinancialTablePipeline
from modules.validation import classify_all_rows, TableValidator

pipeline = FinancialTablePipeline(use_v1_1=True, ocr_backend='paddleocr')
result = pipeline.process_image('data/samples/CIMB_BANK-SAMPLE1.png')

grid = result.get('grid', [])
labels = result.get('labels', [])
norm_grid = result.get('normalized_grid', [])

print('=== GRID SHAPE ===')
print(f'Rows: {len(grid)}, Cols: {len(grid[0]) if grid else 0}')

print('\n=== HEADERS ===')
print(result.get('headers', []))

# Build value grid
value_grid = []
for row in norm_grid:
    vr = [cell.get('value') if isinstance(cell, dict) else None for cell in row]
    value_grid.append(vr)

# Classify rows
row_classes = classify_all_rows(labels, value_grid)

print('\n=== FULL GRID WITH CLASSIFICATION ===')
for i, (lbl, cls) in enumerate(zip(labels, row_classes)):
    vals = []
    for v in value_grid[i][1:]:
        if v is not None:
            vals.append(f'{v:>15,.0f}')
        else:
            vals.append(f'{"—":>15}')
    vals_str = ' | '.join(vals)
    print(f'Row {i:2d} [{cls:15s}] {lbl[:55]:55s} | {vals_str}')

print('\n=== CELL TYPES (first col) ===')
for i, row in enumerate(norm_grid):
    types = [cell.get('cell_type', '?') if isinstance(cell, dict) else '?' for cell in row]
    print(f'Row {i:2d}: {types}')

print('\n=== VALIDATION ===')
validator = TableValidator(tolerance=0.02)
results = validator.validate_column_sums_enhanced(value_grid, labels, row_classes, debug=True)
print(f'Total checks: {len(results)}')
for r in results:
    status = 'PASS' if r.get('passed') else 'FAIL'
    total_row = r.get('total_row', '')
    row_type = r.get('row_type', '')
    col = r.get('column', '')
    expected = r.get('expected', 0)
    actual = r.get('actual', 0)
    diff = r.get('diff', 0)
    print(f'  [{status}] {row_type}: Col {col} | Expected={expected:,.0f} Actual={actual:,.0f} Diff={diff:,.0f}')
    print(f'         Total row: {total_row}')
    if 'component_rows' in r:
        print(f'         Components ({r.get("component_count", 0)}): {r["component_rows"][:5]}...')

# Show equity checks from pipeline
print('\n=== EQUITY CHECKS (from pipeline) ===')
for ec in result.get('equity_checks', []):
    print(f'  Col {ec.get("column")}: total={ec.get("total_equity"):,.0f}, '
          f'attr={ec.get("attributable"):,.0f}, nci={ec.get("non_controlling"):,.0f}, '
          f'diff={ec.get("diff"):,.0f}, passed={ec.get("passed")}')

if not result.get('equity_checks'):
    print('  (none - no equity rows found)')
