"""
Quick test to verify the pipeline works with different bank samples.
"""
from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator

def test_sample(pipeline, image_path):
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)
    
    try:
        result = pipeline.process_image(image_path)
        
        # Column metadata
        col_meta = result.get('column_metadata', [])
        if len(col_meta) > 2:
            print(f"Currency: {col_meta[2].get('currency', 'N/A')}")
            print(f"Scale: {col_meta[2].get('scale', 1)}")
        
        # Grid info
        grid = result.get('grid', [])
        print(f"Rows: {len(grid)}, Columns: {len(grid[0]) if grid else 0}")
        
        # First few data labels
        print("Sample labels:")
        for i, row in enumerate(grid[2:7]):
            if row and row[0]:
                print(f"  - {row[0][:50]}")
        
        # Validation
        labels = [row[0] if row else '' for row in grid]
        validator = TableValidator(tolerance=0.02)
        validations = validator.validate_grid(grid, labels, result.get('normalized_grid'))
        
        passed = sum(1 for v in validations if v.get('passed'))
        failed = len(validations) - passed
        
        print(f"\nValidation Results: {len(validations)} checks")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        
        for v in validations[:3]:  # Show first 3
            status = '✓' if v.get('passed') else '✗'
            print(f"  {status} {v.get('total_row', 'N/A')[:40]}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Loading pipeline...")
    pipeline = FinancialTablePipeline(use_v1_1=True)
    
    samples = [
        'data/samples/CIMB BANK-SAMPLE1.png',
        'data/samples/ocbc_sample3.png',
        'data/samples/ocbc_127_1.png',
    ]
    
    results = {}
    for sample in samples:
        results[sample] = test_sample(pipeline, sample)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for sample, success in results.items():
        status = '✓' if success else '✗'
        print(f"  {status} {sample}")

if __name__ == "__main__":
    main()
