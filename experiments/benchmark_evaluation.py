"""
Benchmark Evaluation for Financial Document AI Pipeline

Evaluates OCR backends and pipeline components with metrics:
- Precision, Recall, F1 Score for OCR accuracy
- Grid extraction accuracy
- Validation pass rates

Usage:
  python experiments/benchmark_evaluation.py --dataset fintabnet --num_samples 50
  python experiments/benchmark_evaluation.py --dataset demo_cases --compare_ocr
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator
from modules.utils import load_config


def calculate_text_metrics(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate Precision, Recall, F1 for text comparison.
    
    Uses character-level comparison for simplicity.
    """
    pred_chars = set(enumerate(predicted))
    gt_chars = set(enumerate(ground_truth))
    
    # Word-level comparison (more meaningful)
    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())
    
    if not gt_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    true_positives = len(pred_words & gt_words)
    false_positives = len(pred_words - gt_words)
    false_negatives = len(gt_words - pred_words)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def calculate_grid_metrics(predicted_grid: List[List], ground_truth_grid: List[List]) -> Dict[str, float]:
    """
    Calculate metrics for grid extraction accuracy.
    
    Compares cell-by-cell for tables with ground truth.
    """
    if not ground_truth_grid or not predicted_grid:
        return {'cell_accuracy': 0.0, 'row_match': 0.0, 'col_match': 0.0}
    
    pred_rows, pred_cols = len(predicted_grid), len(predicted_grid[0]) if predicted_grid else 0
    gt_rows, gt_cols = len(ground_truth_grid), len(ground_truth_grid[0]) if ground_truth_grid else 0
    
    row_match = 1.0 if pred_rows == gt_rows else min(pred_rows, gt_rows) / max(pred_rows, gt_rows)
    col_match = 1.0 if pred_cols == gt_cols else min(pred_cols, gt_cols) / max(pred_cols, gt_cols)
    
    # Cell-level comparison
    correct_cells = 0
    total_cells = 0
    
    min_rows = min(pred_rows, gt_rows)
    min_cols = min(pred_cols, gt_cols)
    
    for i in range(min_rows):
        for j in range(min_cols):
            total_cells += 1
            pred_val = str(predicted_grid[i][j]).strip().lower()
            gt_val = str(ground_truth_grid[i][j]).strip().lower()
            
            # Fuzzy match - consider similar if normalized values match
            if pred_val == gt_val or _normalize_numeric(pred_val) == _normalize_numeric(gt_val):
                correct_cells += 1
    
    cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    
    return {
        'cell_accuracy': cell_accuracy,
        'row_match': row_match,
        'col_match': col_match,
        'predicted_shape': [pred_rows, pred_cols],
        'ground_truth_shape': [gt_rows, gt_cols],
        'correct_cells': correct_cells,
        'total_cells': total_cells
    }


def _normalize_numeric(val: str) -> str:
    """Normalize numeric strings for comparison."""
    # Remove common formatting
    val = val.replace(',', '').replace('$', '').replace('%', '')
    val = val.replace('(', '-').replace(')', '')
    try:
        num = float(val)
        return f"{num:.2f}"
    except ValueError:
        return val


class BenchmarkEvaluator:
    """Benchmark evaluator for pipeline components."""
    
    def __init__(self, config_path: str = "configs/config.yaml", 
                 output_dir: str = "outputs/benchmark"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'samples': [],
            'summary': {}
        }
    
    def evaluate_ocr_comparison(self, image_paths: List[str], 
                                backends: List[str] = ['paddleocr', 'docling']) -> Dict:
        """
        Compare OCR backends on the same images.
        
        Returns detailed metrics for each backend.
        """
        print(f"\n{'='*60}")
        print("OCR Backend Comparison Benchmark")
        print(f"Images: {len(image_paths)}")
        print(f"Backends: {backends}")
        print("=" * 60)
        
        comparison_results = {backend: {'samples': [], 'metrics': {}} for backend in backends}
        
        for img_path in image_paths:
            print(f"\nProcessing: {Path(img_path).name}")
            
            for backend in backends:
                try:
                    pipeline = FinancialTablePipeline(
                        config_path=self.config_path,
                        ocr_backend=backend
                    )
                    result = pipeline.process_image(img_path)
                    
                    # Run validation
                    validator = TableValidator(tolerance=0.02)
                    validations = validator.validate_grid(
                        result['grid'], result['labels'], result['normalized_grid']
                    )
                    
                    passed = sum(1 for v in validations if v.get('passed'))
                    total = len(validations)
                    
                    sample_result = {
                        'file': Path(img_path).name,
                        'grid_shape': [len(result['grid']), len(result['grid'][0]) if result['grid'] else 0],
                        'num_cells': sum(len(row) for row in result['grid']),
                        'validation_passed': passed,
                        'validation_total': total,
                        'validation_rate': passed / total if total > 0 else 0.0,
                        'labels_count': len(result['labels']),
                        'headers': result['headers']
                    }
                    
                    comparison_results[backend]['samples'].append(sample_result)
                    print(f"  {backend}: {sample_result['grid_shape'][0]}×{sample_result['grid_shape'][1]}, "
                          f"validation {passed}/{total}")
                    
                except Exception as e:
                    print(f"  {backend}: ERROR - {e}")
                    comparison_results[backend]['samples'].append({
                        'file': Path(img_path).name,
                        'error': str(e)
                    })
        
        # Calculate summary metrics for each backend
        for backend in backends:
            samples = [s for s in comparison_results[backend]['samples'] if 'error' not in s]
            if samples:
                comparison_results[backend]['metrics'] = {
                    'success_count': len(samples),
                    'error_count': len(comparison_results[backend]['samples']) - len(samples),
                    'avg_validation_rate': np.mean([s['validation_rate'] for s in samples]),
                    'avg_grid_rows': np.mean([s['grid_shape'][0] for s in samples]),
                    'avg_grid_cols': np.mean([s['grid_shape'][1] for s in samples]),
                    'total_cells': sum(s['num_cells'] for s in samples)
                }
        
        return comparison_results
    
    def evaluate_pipeline_accuracy(self, test_cases: List[Dict], 
                                   ocr_backend: str = 'paddleocr') -> Dict:
        """
        Evaluate pipeline accuracy on test cases with ground truth.
        
        Args:
            test_cases: List of {'image_path': str, 'ground_truth': {...}}
            ocr_backend: OCR backend to use
            
        Returns:
            Detailed accuracy metrics
        """
        print(f"\n{'='*60}")
        print(f"Pipeline Accuracy Evaluation ({ocr_backend})")
        print(f"Test cases: {len(test_cases)}")
        print("=" * 60)
        
        pipeline = FinancialTablePipeline(
            config_path=self.config_path,
            ocr_backend=ocr_backend
        )
        validator = TableValidator(tolerance=0.02)
        
        all_metrics = []
        
        for case in test_cases:
            img_path = case['image_path']
            gt = case.get('ground_truth', {})
            
            print(f"\nProcessing: {Path(img_path).name}")
            
            try:
                result = pipeline.process_image(img_path)
                
                # Validation metrics
                validations = validator.validate_grid(
                    result['grid'], result['labels'], result['normalized_grid']
                )
                passed = sum(1 for v in validations if v.get('passed'))
                total = len(validations)
                
                metrics = {
                    'file': Path(img_path).name,
                    'validation': {
                        'passed': passed,
                        'total': total,
                        'rate': passed / total if total > 0 else 0.0
                    },
                    'grid_shape': [len(result['grid']), len(result['grid'][0]) if result['grid'] else 0]
                }
                
                # Compare with ground truth if available
                if 'grid' in gt:
                    grid_metrics = calculate_grid_metrics(result['grid'], gt['grid'])
                    metrics['grid_metrics'] = grid_metrics
                    print(f"  Grid accuracy: {grid_metrics['cell_accuracy']:.1%}")
                
                if 'labels' in gt:
                    label_text = ' '.join(result['labels'])
                    gt_label_text = ' '.join(gt['labels'])
                    label_metrics = calculate_text_metrics(label_text, gt_label_text)
                    metrics['label_metrics'] = label_metrics
                    print(f"  Label F1: {label_metrics['f1']:.1%}")
                
                print(f"  Validation: {passed}/{total} ({metrics['validation']['rate']:.1%})")
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                all_metrics.append({
                    'file': Path(img_path).name,
                    'error': str(e)
                })
        
        # Calculate summary
        successful = [m for m in all_metrics if 'error' not in m]
        summary = {
            'total_samples': len(test_cases),
            'successful': len(successful),
            'failed': len(test_cases) - len(successful),
            'avg_validation_rate': np.mean([m['validation']['rate'] for m in successful]) if successful else 0.0,
        }
        
        if any('grid_metrics' in m for m in successful):
            summary['avg_cell_accuracy'] = np.mean([
                m['grid_metrics']['cell_accuracy'] 
                for m in successful if 'grid_metrics' in m
            ])
        
        if any('label_metrics' in m for m in successful):
            summary['avg_label_f1'] = np.mean([
                m['label_metrics']['f1'] 
                for m in successful if 'label_metrics' in m
            ])
            summary['avg_label_precision'] = np.mean([
                m['label_metrics']['precision'] 
                for m in successful if 'label_metrics' in m
            ])
            summary['avg_label_recall'] = np.mean([
                m['label_metrics']['recall'] 
                for m in successful if 'label_metrics' in m
            ])
        
        return {
            'ocr_backend': ocr_backend,
            'samples': all_metrics,
            'summary': summary
        }
    
    def run_demo_cases_benchmark(self, compare_ocr: bool = False) -> Dict:
        """
        Run benchmark on official demo cases.
        
        Uses CIMB, OCBC, Nanyang, Unilever samples.
        """
        demo_dir = Path("data/samples")
        if not demo_dir.exists():
            print(f"Error: {demo_dir} not found")
            return {}
        
        image_files = list(demo_dir.glob("*.png")) + list(demo_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} demo images")
        
        if compare_ocr:
            return self.evaluate_ocr_comparison([str(p) for p in image_files])
        else:
            return self.evaluate_pipeline_accuracy([
                {'image_path': str(p)} for p in image_files
            ])
    
    def run_fintabnet_benchmark(self, num_samples: int = 50, 
                                ocr_backend: str = 'paddleocr') -> Dict:
        """
        Run benchmark on FinTabNet dataset samples.
        """
        from modules.data_loaders import FinTabNetLoader
        from modules.utils import load_config
        
        try:
            config = load_config(self.config_path)
            loader = FinTabNetLoader(config)
            annotations = loader.load_annotations(split='val', num_samples=num_samples)
        except Exception as e:
            print(f"Error loading FinTabNet: {e}")
            return {'error': str(e)}
        
        test_cases = []
        for ann in annotations:
            image_path = loader.get_image_path(ann['filename'])
            if Path(image_path).exists():
                test_cases.append({
                    'image_path': image_path,
                    'ground_truth': {}  # FinTabNet XML doesn't have cell-level text ground truth
                })
        
        print(f"Loaded {len(test_cases)} FinTabNet samples")
        return self.evaluate_pipeline_accuracy(test_cases, ocr_backend)
    
    def save_results(self, results: Dict, name: str = "benchmark") -> Path:
        """Save benchmark results to JSON."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{name}_{ts}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_path}")
        return output_path


def print_comparison_summary(results: Dict):
    """Print formatted comparison summary."""
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    
    headers = ['Backend', 'Success', 'Errors', 'Avg Val Rate', 'Avg Grid', 'Total Cells']
    print(f"{headers[0]:<12} {headers[1]:<8} {headers[2]:<8} {headers[3]:<14} {headers[4]:<10} {headers[5]:<12}")
    print("-" * 70)
    
    for backend, data in results.items():
        metrics = data.get('metrics', {})
        if metrics:
            success = metrics.get('success_count', 0)
            errors = metrics.get('error_count', 0)
            val_rate = f"{metrics.get('avg_validation_rate', 0):.1%}"
            grid = f"{metrics.get('avg_grid_rows', 0):.0f}×{metrics.get('avg_grid_cols', 0):.0f}"
            cells = metrics.get('total_cells', 0)
            
            print(f"{backend:<12} {success:<8} {errors:<8} {val_rate:<14} {grid:<10} {cells:<12}")
        else:
            print(f"{backend:<12} {'N/A':<8} {'N/A':<8} {'N/A':<14} {'N/A':<10} {'N/A':<12}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation for Financial Document AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dataset", type=str, default="demo_cases",
                       choices=["demo_cases", "fintabnet"],
                       help="Dataset to evaluate")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples for dataset evaluation")
    parser.add_argument("--compare_ocr", action="store_true",
                       help="Compare PaddleOCR vs Docling")
    parser.add_argument("--ocr", type=str, default="paddleocr",
                       choices=["paddleocr", "docling"],
                       help="OCR backend for single-backend evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/benchmark",
                       help="Output directory for results")
    parser.add_argument("--save", action="store_true",
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Financial Document AI - Benchmark Evaluation")
    print("=" * 60)
    
    evaluator = BenchmarkEvaluator(
        config_path=args.config,
        output_dir=args.output
    )
    
    if args.dataset == "demo_cases":
        results = evaluator.run_demo_cases_benchmark(compare_ocr=args.compare_ocr)
    elif args.dataset == "fintabnet":
        if args.compare_ocr:
            # Run on both backends
            results = {}
            from modules.data_loaders import FinTabNetLoader
            try:
                loader = FinTabNetLoader()
                samples = loader.get_random_samples(args.num_samples)
                image_paths = [s['image_path'] for s in samples if 'image_path' in s]
                results = evaluator.evaluate_ocr_comparison(image_paths)
            except Exception as e:
                print(f"Error: {e}")
                return
        else:
            results = evaluator.run_fintabnet_benchmark(
                num_samples=args.num_samples,
                ocr_backend=args.ocr
            )
    
    # Print summary
    if args.compare_ocr:
        print_comparison_summary(results)
    elif 'summary' in results:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print("=" * 50)
        for key, val in results['summary'].items():
            if isinstance(val, float):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")
    
    # Save if requested
    if args.save:
        name = f"{args.dataset}_{'compare' if args.compare_ocr else args.ocr}"
        evaluator.save_results(results, name)


if __name__ == "__main__":
    main()
