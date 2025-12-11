"""
Test validation on CIMB BANK sample with enhanced features.

Tests:
1. Currency & unit propagation (RM'000)
2. Column type detection (skip years, notes, headers)
3. Validation engine output
4. Special row classification
5. OCR correction
"""
import json
import os
from datetime import datetime
from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator, classify_all_rows, RowClassification
from modules.semantic import correct_ocr_text, correct_financial_label

def main():
    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = FinancialTablePipeline(use_v1_1=True)
    
    # Process image
    image_path = 'data/samples/CIMB BANK-SAMPLE1.png'
    print(f"\nProcessing: {image_path}")
    result = pipeline.process_image(image_path)
    
    grid = result.get('grid', [])
    normalized = result.get('normalized_grid', [])
    col_metadata = result.get('column_metadata', [])
    
    # Show column metadata
    print("\n" + "="*60)
    print("COLUMN METADATA (Currency/Unit Propagation)")
    print("="*60)
    for i, meta in enumerate(col_metadata):
        print(f"  Column {i}: currency={meta.get('currency')}, scale={meta.get('scale')}, "
              f"type={meta.get('column_type')}, year={meta.get('year')}")
    
    # Show corrected labels
    print("\n" + "="*60)
    print("OCR CORRECTIONS")
    print("="*60)
    row_labels = [row[0] if row else '' for row in grid]
    for label in row_labels:
        if label:
            corrected = correct_financial_label(label)
            if corrected != label:
                print(f"  '{label}' -> '{corrected}'")
    
    # Show row classifications
    print("\n" + "="*60)
    print("ROW CLASSIFICATIONS")
    print("="*60)
    
    # Extract values for classification
    value_grid = []
    for row in normalized:
        value_row = []
        for cell in row:
            if isinstance(cell, dict):
                value_row.append(cell.get('value'))
            else:
                value_row.append(None)
        value_grid.append(value_row)
    
    row_classes = classify_all_rows(row_labels, value_grid)
    for i, (label, cls) in enumerate(zip(row_labels, row_classes)):
        if cls != RowClassification.DATA or not label:
            display_label = label[:50] if label else '(empty)'
            print(f"  Row {i:2d} [{cls:10s}]: {display_label}")
    
    # Show cell type detection
    print("\n" + "="*60)
    print("CELL TYPE DETECTION (Header Rows)")
    print("="*60)
    for row_idx in range(min(3, len(normalized))):
        row = normalized[row_idx]
        print(f"Row {row_idx}:")
        for col_idx, cell in enumerate(row):
            if isinstance(cell, dict):
                cell_type = cell.get('cell_type', 'unknown')
                raw = cell.get('raw', '')[:20]
                value = cell.get('value')
                print(f"    Col {col_idx}: type={cell_type:12s} raw='{raw}' value={value}")
    
    # Show normalized values (with correct scale)
    print("\n" + "="*60)
    print("NORMALIZED VALUES (with scale applied)")
    print("="*60)
    print("First 5 data rows:")
    for row_idx in range(3, min(8, len(normalized))):
        row = normalized[row_idx]
        label = row_labels[row_idx][:40] if row_idx < len(row_labels) else ''
        values_str = []
        for col_idx in range(2, len(row)):
            cell = row[col_idx]
            if isinstance(cell, dict) and cell.get('value') is not None:
                val = cell['value']
                currency = cell.get('currency', '')
                # Format large numbers
                if val >= 1_000_000_000:
                    values_str.append(f"{currency} {val/1e9:.2f}B")
                elif val >= 1_000_000:
                    values_str.append(f"{currency} {val/1e6:.2f}M")
                else:
                    values_str.append(f"{currency} {val:,.0f}")
        print(f"  {label:40s}: {', '.join(values_str)}")
    
    # Run validation
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    validator = TableValidator(tolerance=0.02)
    validations = validator.validate_grid(grid, row_labels, normalized)
    
    for v in validations:
        rule = v.get('rule', v.get('row_type', 'unknown'))
        passed = '✓ PASS' if v.get('passed') else '✗ FAIL'
        col = v.get('column', '?')
        total_row = v.get('total_row', 'N/A')
        expected = v.get('expected', 0)
        actual = v.get('actual', 0)
        diff = v.get('diff', 0)
        
        print(f"\n  [{rule}] Column {col} - {total_row}")
        print(f"    Expected: {expected:>20,.0f}")
        print(f"    Actual:   {actual:>20,.0f}")
        print(f"    Diff:     {diff:>20,.0f}  {passed}")
        
        if 'component_count' in v:
            print(f"    Components: {v['component_count']}")
        if 'subtotal_value' in v:
            print(f"    Subtotal: {v['subtotal_value']:,.0f}")
            print(f"    Additional items: {v.get('additional_items', 0)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Rows: {len(grid)}")
    print(f"Total Columns: {len(grid[0]) if grid else 0}")
    print(f"Validation Checks: {len(validations)}")
    
    passed = sum(1 for v in validations if v.get('passed'))
    failed = sum(1 for v in validations if not v.get('passed'))
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    # Save results to JSON
    output_dir = 'outputs/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/cimb_validation_{timestamp}.json"
    
    output_data = {
        'file': image_path,
        'timestamp': timestamp,
        'column_metadata': col_metadata,
        'row_count': len(grid),
        'column_count': len(grid[0]) if grid else 0,
        'row_classifications': [
            {'row': i, 'label': row_labels[i], 'class': row_classes[i]}
            for i in range(len(row_classes))
        ],
        'validations': validations,
        'summary': {
            'total_checks': len(validations),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(validations) if validations else 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
