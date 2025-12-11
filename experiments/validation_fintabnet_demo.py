"""
Validation Demo using FinTabNet Dataset

Demonstrates the validation module capabilities on real financial tables
from the FinTabNet dataset (SEC filings).
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator, check_column_sum, validate_table
from modules.utils import ensure_dir


def find_fintabnet_samples(fintabnet_dir: str, num_samples: int = 10) -> list:
    """
    Find sample images from FinTabNet that are likely to have summable columns.
    
    Args:
        fintabnet_dir: Path to FinTabNet images directory
        num_samples: Number of samples to find
        
    Returns:
        List of image paths
    """
    images_dir = os.path.join(fintabnet_dir, 'images')
    if not os.path.exists(images_dir):
        # Try alternative paths
        if os.path.exists(fintabnet_dir):
            images_dir = fintabnet_dir
        else:
            print(f"FinTabNet images directory not found: {images_dir}")
            return []
    
    # Get all image files
    all_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_images.extend(Path(images_dir).glob(ext))
    
    if not all_images:
        print(f"No images found in {images_dir}")
        return []
    
    # Return a subset
    import random
    random.seed(42)  # For reproducibility
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    return [str(p) for p in samples]


def run_validation_demo(config_path: str = "configs/config.yaml",
                        num_samples: int = 10,
                        save_outputs: bool = True):
    """
    Run validation demo on FinTabNet samples.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to process
        save_outputs: Whether to save results
    """
    print("=" * 60)
    print("Financial Table Validation Demo")
    print("=" * 60)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = FinancialTablePipeline(config_path=config_path, use_v1_1=True)
    validator = TableValidator(tolerance=0.02)  # 2% tolerance
    
    # Try to find FinTabNet samples
    # Check common locations
    possible_paths = [
        'D:/datasets/fintabnet/images',
        'C:/datasets/fintabnet/images',
        '../datasets/fintabnet/images',
        'data/fintabnet/images',
    ]
    
    fintabnet_samples = []
    for path in possible_paths:
        if os.path.exists(path):
            fintabnet_samples = find_fintabnet_samples(path, num_samples)
            if fintabnet_samples:
                print(f"Found {len(fintabnet_samples)} samples in {path}")
                break
    
    # Fallback to local samples if FinTabNet not found
    if not fintabnet_samples:
        print("\nFinTabNet not found. Using local samples...")
        samples_dir = 'data/samples'
        if os.path.exists(samples_dir):
            fintabnet_samples = [
                os.path.join(samples_dir, f) 
                for f in os.listdir(samples_dir) 
                if f.endswith('.png')
            ]
    
    if not fintabnet_samples:
        print("No samples found. Please provide sample images.")
        return
    
    # Process each sample
    all_results = []
    validation_stats = {
        'total_tables': 0,
        'tables_with_validations': 0,
        'total_validations': 0,
        'passed_validations': 0,
        'failed_validations': 0
    }
    
    print(f"\nProcessing {len(fintabnet_samples)} samples...")
    
    for i, image_path in enumerate(fintabnet_samples):
        filename = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(fintabnet_samples)}] Processing: {filename}")
        
        try:
            # Run pipeline
            result = pipeline.process_image(image_path)
            
            # Run validation
            validations = validator.validate_grid(
                result['grid'],
                result['labels'],
                result.get('normalized_grid')
            )
            
            # Update stats
            validation_stats['total_tables'] += 1
            if validations:
                validation_stats['tables_with_validations'] += 1
                validation_stats['total_validations'] += len(validations)
                
                for v in validations:
                    if v.get('passed'):
                        validation_stats['passed_validations'] += 1
                    else:
                        validation_stats['failed_validations'] += 1
            
            # Store result
            sample_result = {
                'filename': filename,
                'num_rows': len(result['grid']),
                'num_cols': len(result['grid'][0]) if result['grid'] else 0,
                'labels': result['labels'],
                'validations': validations,
                'validation_summary': {
                    'total': len(validations),
                    'passed': sum(1 for v in validations if v.get('passed')),
                    'failed': sum(1 for v in validations if not v.get('passed'))
                }
            }
            all_results.append(sample_result)
            
            # Print validation results
            if validations:
                print(f"  Found {len(validations)} validation checks:")
                for v in validations:
                    status = "✓ PASS" if v.get('passed') else "✗ FAIL"
                    print(f"    {status}: {v.get('rule', 'unknown')} - {v.get('message', '')}")
            else:
                print("  No validation rules applicable (no total rows detected)")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tables processed: {validation_stats['total_tables']}")
    print(f"Tables with validations: {validation_stats['tables_with_validations']}")
    print(f"Total validation checks: {validation_stats['total_validations']}")
    print(f"  Passed: {validation_stats['passed_validations']}")
    print(f"  Failed: {validation_stats['failed_validations']}")
    
    if validation_stats['total_validations'] > 0:
        pass_rate = validation_stats['passed_validations'] / validation_stats['total_validations'] * 100
        print(f"  Pass Rate: {pass_rate:.1f}%")
    
    # Save results
    if save_outputs:
        output_dir = ensure_dir('outputs/results')
        figures_dir = ensure_dir('outputs/figures')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        results_file = os.path.join(output_dir, f'validation_demo_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': 'validation_demo',
                'timestamp': timestamp,
                'num_samples': len(all_results),
                'statistics': validation_stats,
                'detailed_results': all_results
            }, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate visualization
        if validation_stats['total_validations'] > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart of pass/fail
            passed = validation_stats['passed_validations']
            failed = validation_stats['failed_validations']
            
            if passed + failed > 0:
                axes[0].pie([passed, failed], 
                           labels=['Passed', 'Failed'],
                           colors=['#50E3C2', '#E74C3C'],
                           autopct='%1.1f%%',
                           startangle=90)
                axes[0].set_title('Validation Results Distribution')
            
            # Bar chart of validation types
            rule_counts = {}
            for result in all_results:
                for v in result.get('validations', []):
                    rule = v.get('rule', 'unknown')
                    if rule not in rule_counts:
                        rule_counts[rule] = {'passed': 0, 'failed': 0}
                    if v.get('passed'):
                        rule_counts[rule]['passed'] += 1
                    else:
                        rule_counts[rule]['failed'] += 1
            
            if rule_counts:
                rules = list(rule_counts.keys())
                passed_vals = [rule_counts[r]['passed'] for r in rules]
                failed_vals = [rule_counts[r]['failed'] for r in rules]
                
                x = np.arange(len(rules))
                width = 0.35
                
                axes[1].bar(x - width/2, passed_vals, width, label='Passed', color='#50E3C2')
                axes[1].bar(x + width/2, failed_vals, width, label='Failed', color='#E74C3C')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Validation Results by Rule Type')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(rules, rotation=45, ha='right')
                axes[1].legend()
            
            plt.tight_layout()
            fig_path = os.path.join(figures_dir, f'validation_demo_{timestamp}.png')
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            print(f"Figure saved to: {fig_path}")
    
    return validation_stats, all_results


def demo_manual_validation():
    """
    Demonstrate validation with manually constructed test cases.
    
    Uses the example from FinTabNet: Net Periodic Benefit Cost table.
    """
    print("\n" + "=" * 60)
    print("Manual Validation Demo - Net Periodic Benefit Cost")
    print("=" * 60)
    
    # Manually constructed test case based on the FinTabNet example
    # Components: Service cost, Interest cost, Expected return, Prior service cost, Unrecognized gain
    # Total: Net periodic benefit cost
    
    # 2002 column
    components_2002 = [77, 207, -9, -6, 0]  # Unrecognized gain is "-" which we treat as 0
    total_2002 = 269
    
    # 2001 column
    components_2001 = [66, 175, -9, -5, 0]
    total_2001 = 227
    
    # 2000 column
    components_2000 = [43, 108, -7, -5, -14]
    total_2000 = 125
    
    print("\nTest Case: Net Periodic Benefit Cost")
    print("-" * 40)
    
    # Validate each year
    for year, components, total in [
        (2002, components_2002, total_2002),
        (2001, components_2001, total_2001),
        (2000, components_2000, total_2000)
    ]:
        result = check_column_sum(components, total, tolerance=0.01)
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"\n{year}:")
        print(f"  Components: {components}")
        print(f"  Sum: {sum(components)}")
        print(f"  Expected Total: {total}")
        print(f"  Result: {status}")
        if not result['passed']:
            print(f"  Difference: {result['diff']}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation Demo")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to process")
    parser.add_argument("--manual", action="store_true",
                       help="Run manual validation demo only")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save outputs")
    
    args = parser.parse_args()
    
    # Always run manual demo first
    demo_manual_validation()
    
    if not args.manual:
        # Run full demo with actual images
        run_validation_demo(
            config_path=args.config,
            num_samples=args.num_samples,
            save_outputs=not args.no_save
        )
