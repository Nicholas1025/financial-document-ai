"""
Financial Document AI - Main Entry Point

End-to-end pipeline for financial table understanding.
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.utils import load_config, get_device


def process_single_image(image_path: str, config_path: str, output_dir: str, save: bool, 
                         verbose: bool = True, ocr_backend: str = "paddleocr"):
    """Process a single image through the pipeline."""
    from modules.pipeline import FinancialTablePipeline
    from modules.validation import TableValidator
    
    # Run pipeline
    pipeline = FinancialTablePipeline(config_path=config_path, ocr_backend=ocr_backend)
    result = pipeline.process_image(image_path)
    
    # Run validation
    validator = TableValidator(tolerance=0.02)
    validations = validator.validate_grid(
        result['grid'], result['labels'], result['normalized_grid']
    )
    result['full_validation'] = validations
    
    # Summary output
    grid = result['grid']
    passed = sum(1 for v in validations if v.get('passed'))
    total = len(validations)
    validation_summary = {
        'total': total,
        'passed': passed,
        'pass_rate': (passed / total) if total else 0.0,
        'all_passed': (passed == total) if total else None,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"File: {result['file']}")
        print(f"Grid: {len(grid)} rows × {len(grid[0]) if grid else 0} cols")
        print(f"Document Type: {result.get('document_type', 'N/A')}")
        print(f"\nHeaders: {result['headers']}")
        print(f"\nLabels (first 5): {result['labels'][:5]}")
        print(f"\nValidation: {passed}/{total} passed")
        for v in validations:
            status = "✓" if v.get('passed') else "✗"
            print(f"  {status} {v['rule']}: {v.get('description', '')}")
    
    # Save to file if requested
    output_path = None
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = Path(image_path).stem
        outdir = Path(output_dir) / "results"
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = outdir / f"{fname}_{ts}.json"
        
        # Prepare serializable output
        output = {
            'run_meta': result.get('run_meta', {}),
            'file': result['file'],
            'grid_shape': [len(grid), len(grid[0]) if grid else 0],
            'document_type': result.get('document_type'),
            'headers': result['headers'],
            'labels': result['labels'],
            'grid': result['grid'],
            'normalized_grid': result['normalized_grid'],
            'column_metadata': result['column_metadata'],
            'equity_checks': result['equity_checks'],
            # Human-friendly summary (what you called "verify")
            'verify': validation_summary,
            # Detailed checks
            'full_validation': validations,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            print(f"\nSaved to: {output_path}")
    
    return result, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Financial Document AI - Table Understanding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  C:/Users/User/Documents/financial-document-ai/venv/Scripts/python.exe main.py --input data/samples/OCBC_125.png --save
  
  
  # Process and save output to JSON
  python main.py --input data/samples/unliver_144.png --save
  
  # Process all images in a directory
  python main.py --input data/samples/ --save
  
  # Run official demo cases (CIMB/OCBC best samples)
  python main.py --mode demo
  
  # Run FinTabNet evaluation
  python main.py --mode eval --num_samples 100
  
  # Run baseline experiments (all datasets)
  python main.py --mode baseline --num_samples 100
  
  # Compare OCR backends
  python main.py --input data/samples/CIMB_BANK-SAMPLE1.png --ocr paddleocr --save
  python main.py --input data/samples/CIMB_BANK-SAMPLE1.png --ocr docling --save
        """
    )
    
    parser.add_argument("--mode", type=str, 
                       choices=["process", "demo", "eval", "baseline", "compare_ocr"],
                       default="process", help="Operation mode")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for eval/baseline")
    parser.add_argument("--input", type=str, 
                       help="Input image file or directory")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--save", action="store_true", 
                       help="Save output to JSON file")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--ocr", type=str, default="paddleocr",
                       choices=["paddleocr", "docling"],
                       help="OCR backend to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Financial Document AI")
    print("Table Understanding Pipeline")
    print("=" * 60)
    
    # Check device
    device = get_device()
    
    # ===== MODE: process =====
    if args.mode == "process":
        if not args.input:
            print("Error: --input required for process mode")
            print("Usage: python main.py --input <image.png or directory/>")
            sys.exit(1)
        
        input_path = Path(args.input)
        
        # Single file
        if input_path.is_file():
            print(f"\nProcessing: {args.input}")
            print(f"OCR Backend: {args.ocr}")
            process_single_image(args.input, args.config, args.output, args.save, 
                                ocr_backend=args.ocr)
        
        # Directory - batch processing
        elif input_path.is_dir():
            image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
            print(f"\nFound {len(image_files)} images in {args.input}")
            
            results = []
            for i, img_path in enumerate(image_files, 1):
                print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
                try:
                    result, out_path = process_single_image(
                        str(img_path), args.config, args.output, args.save, verbose=False,
                        ocr_backend=args.ocr
                    )
                    grid = result['grid']
                    passed = sum(1 for v in result['full_validation'] if v.get('passed'))
                    total = len(result['full_validation'])
                    status = "✓" if passed == total else f"{passed}/{total}"
                    print(f"  Grid: {len(grid)}×{len(grid[0]) if grid else 0}, Validation: {status}")
                    results.append({'file': img_path.name, 'passed': passed, 'total': total})
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    results.append({'file': img_path.name, 'error': str(e)})
            
            # Summary
            success = sum(1 for r in results if 'error' not in r)
            print(f"\n{'='*50}")
            print(f"Batch complete: {success}/{len(results)} processed successfully")
        
        else:
            print(f"Error: {args.input} not found")
            sys.exit(1)
    
    # ===== MODE: demo =====
    elif args.mode == "demo":
        print("\nRunning official demo cases...")
        from demo_official_cases import main as demo_main
        demo_main()
    
    # ===== MODE: eval =====
    elif args.mode == "eval":
        print(f"\nRunning FinTabNet evaluation ({args.num_samples} samples)...")
        from experiments.fintabnet_large_evaluation import FinTabNetEvaluator
        
        evaluator = FinTabNetEvaluator(
            config_path=args.config,
            num_samples=args.num_samples,
            seed=args.seed
        )
        results = evaluator.run()
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"FinTabNet Evaluation Results")
        print(f"Samples: {results['num_samples']}")
        print(f"Pass Rate: {results['pass_rate']:.1%}")
        print(f"Results saved to: {evaluator.output_path}")
    
    # ===== MODE: baseline =====
    elif args.mode == "baseline":
        print(f"\nRunning baseline experiments ({args.num_samples} samples)...")
        from experiments.run_all_baselines import run_all_baselines
        run_all_baselines(args.config, args.num_samples)
    
    # ===== MODE: compare_ocr =====
    elif args.mode == "compare_ocr":
        if not args.input:
            print("Error: --input required for compare_ocr mode")
            print("Usage: python main.py --mode compare_ocr --input <image.png>")
            sys.exit(1)
        
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"Error: {args.input} not found")
            sys.exit(1)
        
        print(f"\nComparing OCR backends on: {args.input}")
        print("=" * 60)
        
        from modules.pipeline import FinancialTablePipeline
        from modules.validation import TableValidator
        
        results_compare = {}
        backends = ["paddleocr", "docling"]
        
        for backend in backends:
            print(f"\n{'='*30}")
            print(f"Testing: {backend.upper()}")
            print("=" * 30)
            
            try:
                pipeline = FinancialTablePipeline(config_path=args.config, ocr_backend=backend)
                result = pipeline.process_image(str(input_path))
                
                # Run validation
                validator = TableValidator(tolerance=0.02)
                validations = validator.validate_grid(
                    result['grid'], result['labels'], result['normalized_grid']
                )
                
                passed = sum(1 for v in validations if v.get('passed'))
                total = len(validations)
                
                results_compare[backend] = {
                    'grid_shape': [len(result['grid']), len(result['grid'][0]) if result['grid'] else 0],
                    'num_cells': sum(len(row) for row in result['grid']),
                    'validation': {'passed': passed, 'total': total, 'rate': passed/total if total else 0},
                    'labels': result['labels'][:5],
                    'headers': result['headers']
                }
                
                print(f"Grid: {results_compare[backend]['grid_shape'][0]} × {results_compare[backend]['grid_shape'][1]}")
                print(f"Validation: {passed}/{total} ({results_compare[backend]['validation']['rate']:.1%})")
                
            except Exception as e:
                print(f"Error with {backend}: {e}")
                results_compare[backend] = {'error': str(e)}
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("OCR COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Backend':<15} {'Grid':<15} {'Validation':<15}")
        print("-" * 45)
        for backend, res in results_compare.items():
            if 'error' in res:
                print(f"{backend:<15} {'ERROR':<15} {res['error'][:20]}")
            else:
                grid_str = f"{res['grid_shape'][0]}×{res['grid_shape'][1]}"
                val_str = f"{res['validation']['passed']}/{res['validation']['total']}"
                print(f"{backend:<15} {grid_str:<15} {val_str:<15}")


if __name__ == "__main__":
    main()
