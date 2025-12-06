"""
Demo script for the FinancialTablePipeline module.
Runs the end-to-end pipeline on sample images.
"""
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pipeline import FinancialTablePipeline


def main():
    # Initialize pipeline
    print("Initializing Financial Table Pipeline...")
    pipeline = FinancialTablePipeline(use_v1_1=True)

    # Get sample images
    samples_dir = os.path.join('data', 'samples')
    if not os.path.exists(samples_dir):
        print(f"Samples directory not found: {samples_dir}")
        return

    image_files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
    image_files.sort()

    if not image_files:
        print("No PNG images found in samples directory.")
        return

    # Process images
    results = []
    for fname in image_files:
        path = os.path.join(samples_dir, fname)
        print(f"\nProcessing {fname}...")
        try:
            result = pipeline.process_image(path)
            results.append(result)
            
            # Print summary
            print(f"  Labels (first 5): {result['labels'][:5]}")
            print(f"  Headers: {result['headers']}")
            
            if result['equity_checks']:
                print(f"  Equity Checks ({len(result['equity_checks'])}):")
                for chk in result['equity_checks']:
                    status = "PASS" if chk['passed'] else "FAIL"
                    print(f"    Col {chk['column']}: {status} (Diff: {chk['diff']:.2f})")
            else:
                print("  Equity Checks: None triggered")
                
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join('outputs', 'results', f'pipeline_demo_{ts}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nPipeline processing complete. Results saved to {out_path}")


if __name__ == '__main__':
    main()
