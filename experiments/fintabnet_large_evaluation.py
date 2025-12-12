"""
Large-Scale FinTabNet Evaluation

Runs the Financial Document AI Pipeline on 5000 FinTabNet samples
to generate quantitative metrics for the thesis report.

Metrics:
1. Structure Recognition (rows/columns detection)
2. OCR Quality (text extraction)
3. Numeric Normalization (currency, scale detection)
4. Validation Pass Rate (column sum checks)
"""
import os
import sys
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set paths
FINTABNET_ROOT = r"D:\datasets\FinTabNet_c\FinTabNet.c-Structure\FinTabNet.c-Structure"
IMAGES_DIR = os.path.join(FINTABNET_ROOT, "images")
ANNOTATIONS_DIR = os.path.join(FINTABNET_ROOT, "train")  # Has JSON annotations

# Import pipeline modules
from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator, classify_all_rows, RowClassification
from modules.numeric import detect_cell_type


class FinTabNetEvaluator:
    """Evaluator for large-scale FinTabNet testing."""
    
    def __init__(self, sample_size: int = 5000):
        self.sample_size = sample_size
        self.pipeline = None
        self.validator = TableValidator(tolerance=0.02)
        
        # Metrics storage
        self.results = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            
            # Structure metrics
            'avg_rows': 0,
            'avg_cols': 0,
            'rows_detected': [],
            'cols_detected': [],
            
            # OCR metrics
            'total_cells': 0,
            'non_empty_cells': 0,
            'numeric_cells': 0,
            
            # Normalization metrics
            'currency_detected': 0,
            'scale_detected': 0,
            'currency_types': {},
            'scale_types': {},
            
            # Cell type metrics
            'cell_types': {},
            
            # Validation metrics
            'tables_with_validation': 0,
            'total_validations': 0,
            'validations_passed': 0,
            'validation_pass_rate': 0,
            
            # Row classification metrics
            'row_classifications': {},
            
            # Timing
            'total_time': 0,
            'avg_time_per_image': 0,
            
            # Per-sample details (sampled)
            'sample_details': [],
        }
    
    def load_pipeline(self):
        """Initialize the pipeline."""
        print("Loading pipeline...")
        self.pipeline = FinancialTablePipeline(use_v1_1=True)
        print("Pipeline loaded!")
    
    def get_sample_images(self) -> List[str]:
        """Get random sample of images."""
        print(f"\nScanning images directory: {IMAGES_DIR}")
        
        all_images = [f for f in os.listdir(IMAGES_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Total images found: {len(all_images)}")
        
        # Random sample
        if len(all_images) > self.sample_size:
            sample = random.sample(all_images, self.sample_size)
        else:
            sample = all_images
        
        print(f"Selected sample size: {len(sample)}")
        return sample
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and extract metrics."""
        result = {
            'file': os.path.basename(image_path),
            'success': False,
            'error': None,
            'metrics': {}
        }
        
        try:
            # Run pipeline
            pipeline_result = self.pipeline.process_image(image_path)
            
            grid = pipeline_result.get('grid', [])
            norm_grid = pipeline_result.get('normalized_grid', [])
            col_meta = pipeline_result.get('column_metadata', [])
            
            # Structure metrics
            num_rows = len(grid)
            num_cols = len(grid[0]) if grid else 0
            
            # Cell metrics
            total_cells = 0
            non_empty_cells = 0
            numeric_cells = 0
            cell_types = {}
            
            for row in norm_grid:
                for cell in row:
                    total_cells += 1
                    if isinstance(cell, dict):
                        raw = cell.get('raw', '')
                        if raw and raw.strip():
                            non_empty_cells += 1
                        if cell.get('value') is not None:
                            numeric_cells += 1
                        ct = cell.get('cell_type', 'unknown')
                        cell_types[ct] = cell_types.get(ct, 0) + 1
            
            # Currency/Scale detection
            currency_detected = any(m.get('currency') for m in col_meta)
            scale_detected = any(m.get('scale', 1) > 1 for m in col_meta)
            
            currencies = [m.get('currency') for m in col_meta if m.get('currency')]
            scales = [m.get('scale') for m in col_meta if m.get('scale', 1) > 1]
            
            # Row classification
            row_labels = [row[0] if row else '' for row in grid]
            value_grid = []
            for row in norm_grid:
                value_row = [c.get('value') if isinstance(c, dict) else None for c in row]
                value_grid.append(value_row)
            
            row_classes = classify_all_rows(row_labels, value_grid)
            row_class_counts = {}
            for c in row_classes:
                row_class_counts[c] = row_class_counts.get(c, 0) + 1
            
            # Validation
            validations = self.validator.validate_grid(grid, row_labels, norm_grid)
            validations_passed = sum(1 for v in validations if v.get('passed'))
            
            result['success'] = True
            result['metrics'] = {
                'rows': num_rows,
                'cols': num_cols,
                'total_cells': total_cells,
                'non_empty_cells': non_empty_cells,
                'numeric_cells': numeric_cells,
                'cell_types': cell_types,
                'currency_detected': currency_detected,
                'scale_detected': scale_detected,
                'currencies': currencies,
                'scales': scales,
                'row_classifications': row_class_counts,
                'num_validations': len(validations),
                'validations_passed': validations_passed,
            }
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def run_evaluation(self):
        """Run the full evaluation."""
        print("\n" + "="*70)
        print("FINTABNET LARGE-SCALE EVALUATION")
        print(f"Sample Size: {self.sample_size}")
        print("="*70)
        
        # Load pipeline
        self.load_pipeline()
        
        # Get samples
        image_files = self.get_sample_images()
        
        # Process with progress bar
        print(f"\nProcessing {len(image_files)} images...")
        start_time = time.time()
        
        for i, img_file in enumerate(tqdm(image_files, desc="Processing")):
            image_path = os.path.join(IMAGES_DIR, img_file)
            
            try:
                result = self.process_single_image(image_path)
                self.update_metrics(result)
                
                # Store sample details (every 100th)
                if i % 100 == 0:
                    self.results['sample_details'].append({
                        'index': i,
                        'file': img_file,
                        'success': result['success'],
                        'metrics': result.get('metrics', {})
                    })
                    
            except Exception as e:
                self.results['failed'] += 1
                self.results['errors'].append({
                    'file': img_file,
                    'error': str(e)
                })
            
            self.results['total_processed'] = i + 1
            
            # Progress update every 500
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(image_files) - i - 1) / rate
                print(f"\n  Progress: {i+1}/{len(image_files)} | "
                      f"Rate: {rate:.1f} img/s | "
                      f"ETA: {remaining/60:.1f} min")
        
        # Final timing
        total_time = time.time() - start_time
        self.results['total_time'] = total_time
        self.results['avg_time_per_image'] = total_time / len(image_files) if image_files else 0
        
        # Calculate final metrics
        self.finalize_metrics()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def update_metrics(self, result: Dict):
        """Update aggregate metrics from single result."""
        if result['success']:
            self.results['successful'] += 1
            m = result['metrics']
            
            # Structure
            self.results['rows_detected'].append(m['rows'])
            self.results['cols_detected'].append(m['cols'])
            
            # Cells
            self.results['total_cells'] += m['total_cells']
            self.results['non_empty_cells'] += m['non_empty_cells']
            self.results['numeric_cells'] += m['numeric_cells']
            
            # Cell types
            for ct, count in m['cell_types'].items():
                self.results['cell_types'][ct] = self.results['cell_types'].get(ct, 0) + count
            
            # Currency/Scale
            if m['currency_detected']:
                self.results['currency_detected'] += 1
            if m['scale_detected']:
                self.results['scale_detected'] += 1
            
            for curr in m.get('currencies', []):
                self.results['currency_types'][curr] = self.results['currency_types'].get(curr, 0) + 1
            for scale in m.get('scales', []):
                key = str(scale)
                self.results['scale_types'][key] = self.results['scale_types'].get(key, 0) + 1
            
            # Row classification
            for rc, count in m['row_classifications'].items():
                self.results['row_classifications'][rc] = \
                    self.results['row_classifications'].get(rc, 0) + count
            
            # Validation
            if m['num_validations'] > 0:
                self.results['tables_with_validation'] += 1
            self.results['total_validations'] += m['num_validations']
            self.results['validations_passed'] += m['validations_passed']
        else:
            self.results['failed'] += 1
            if result.get('error'):
                self.results['errors'].append({
                    'file': result['file'],
                    'error': result['error']
                })
    
    def finalize_metrics(self):
        """Calculate final aggregate metrics."""
        n = self.results['successful']
        if n > 0:
            self.results['avg_rows'] = sum(self.results['rows_detected']) / n
            self.results['avg_cols'] = sum(self.results['cols_detected']) / n
        
        if self.results['total_validations'] > 0:
            self.results['validation_pass_rate'] = \
                self.results['validations_passed'] / self.results['total_validations']
    
    def save_results(self):
        """Save results to JSON."""
        output_dir = 'outputs/results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/fintabnet_evaluation_{self.sample_size}_{timestamp}.json"
        
        # Clean up large lists for JSON
        output_data = {
            'timestamp': timestamp,
            'sample_size': self.sample_size,
            'summary': {
                'total_processed': self.results['total_processed'],
                'successful': self.results['successful'],
                'failed': self.results['failed'],
                'success_rate': self.results['successful'] / self.results['total_processed'] 
                               if self.results['total_processed'] > 0 else 0,
            },
            'structure_metrics': {
                'avg_rows': self.results['avg_rows'],
                'avg_cols': self.results['avg_cols'],
                'min_rows': min(self.results['rows_detected']) if self.results['rows_detected'] else 0,
                'max_rows': max(self.results['rows_detected']) if self.results['rows_detected'] else 0,
                'min_cols': min(self.results['cols_detected']) if self.results['cols_detected'] else 0,
                'max_cols': max(self.results['cols_detected']) if self.results['cols_detected'] else 0,
            },
            'cell_metrics': {
                'total_cells': self.results['total_cells'],
                'non_empty_cells': self.results['non_empty_cells'],
                'numeric_cells': self.results['numeric_cells'],
                'cell_fill_rate': self.results['non_empty_cells'] / self.results['total_cells']
                                 if self.results['total_cells'] > 0 else 0,
                'numeric_rate': self.results['numeric_cells'] / self.results['total_cells']
                               if self.results['total_cells'] > 0 else 0,
            },
            'cell_type_distribution': self.results['cell_types'],
            'normalization_metrics': {
                'currency_detection_rate': self.results['currency_detected'] / self.results['successful']
                                          if self.results['successful'] > 0 else 0,
                'scale_detection_rate': self.results['scale_detected'] / self.results['successful']
                                       if self.results['successful'] > 0 else 0,
                'currency_types': self.results['currency_types'],
                'scale_types': self.results['scale_types'],
            },
            'row_classification': self.results['row_classifications'],
            'validation_metrics': {
                'tables_with_validation': self.results['tables_with_validation'],
                'total_validations': self.results['total_validations'],
                'validations_passed': self.results['validations_passed'],
                'validation_pass_rate': self.results['validation_pass_rate'],
            },
            'timing': {
                'total_time_seconds': self.results['total_time'],
                'avg_time_per_image': self.results['avg_time_per_image'],
                'images_per_second': 1 / self.results['avg_time_per_image'] 
                                    if self.results['avg_time_per_image'] > 0 else 0,
            },
            'sample_details': self.results['sample_details'][:50],  # First 50 samples
            'errors': self.results['errors'][:20],  # First 20 errors
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        self.output_file = output_file
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        n = self.results['successful']
        total = self.results['total_processed']
        
        print(f"\n📊 PROCESSING")
        print(f"  Total Processed: {total}")
        print(f"  Successful: {n} ({n/total*100:.1f}%)")
        print(f"  Failed: {self.results['failed']}")
        
        print(f"\n📐 STRUCTURE RECOGNITION")
        print(f"  Average Rows: {self.results['avg_rows']:.1f}")
        print(f"  Average Columns: {self.results['avg_cols']:.1f}")
        if self.results['rows_detected']:
            print(f"  Row Range: {min(self.results['rows_detected'])} - {max(self.results['rows_detected'])}")
        
        print(f"\n📝 OCR & CELLS")
        print(f"  Total Cells: {self.results['total_cells']:,}")
        print(f"  Non-Empty: {self.results['non_empty_cells']:,} ({self.results['non_empty_cells']/self.results['total_cells']*100:.1f}%)")
        print(f"  Numeric: {self.results['numeric_cells']:,} ({self.results['numeric_cells']/self.results['total_cells']*100:.1f}%)")
        
        print(f"\n🔢 CELL TYPES")
        for ct, count in sorted(self.results['cell_types'].items(), key=lambda x: -x[1]):
            pct = count / self.results['total_cells'] * 100
            print(f"  {ct}: {count:,} ({pct:.1f}%)")
        
        print(f"\n💰 NORMALIZATION")
        print(f"  Currency Detected: {self.results['currency_detected']} ({self.results['currency_detected']/n*100:.1f}%)")
        print(f"  Scale Detected: {self.results['scale_detected']} ({self.results['scale_detected']/n*100:.1f}%)")
        print(f"  Currency Types: {dict(self.results['currency_types'])}")
        
        print(f"\n✓ VALIDATION")
        print(f"  Tables with Validations: {self.results['tables_with_validation']}")
        print(f"  Total Validation Checks: {self.results['total_validations']}")
        print(f"  Passed: {self.results['validations_passed']}")
        print(f"  Pass Rate: {self.results['validation_pass_rate']*100:.1f}%")
        
        print(f"\n⏱️ TIMING")
        print(f"  Total Time: {self.results['total_time']/60:.1f} minutes")
        print(f"  Avg per Image: {self.results['avg_time_per_image']:.2f} seconds")
        print(f"  Throughput: {1/self.results['avg_time_per_image']:.1f} images/second")
        
        print("\n" + "="*70)


def main():
    # Run evaluation with 500 samples (faster, ~30-50 min)
    evaluator = FinTabNetEvaluator(sample_size=500)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
