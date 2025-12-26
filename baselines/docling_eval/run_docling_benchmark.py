"""
Docling FinTabNet Benchmark Wrapper

Wrapper script to run docling's official FinTabNet benchmark evaluation.
This preserves the original docling-eval API and simply wraps it for our comparison.

Usage:
    python run_docling_benchmark.py --benchmark FinTabNet --output-dir ./outputs/baseline
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Version Lock
# =============================================================================

BASELINE_VERSION = "0.1.0"
REQUIRED_DOCLING_EVAL_VERSION = ">=0.3.0"

VERSION_INFO = {
    "baseline_version": BASELINE_VERSION,
    "docling_eval_requirement": REQUIRED_DOCLING_EVAL_VERSION,
    "docling_eval_commit": "629a451d7b75e274352a1f21710316e47fc7a80a",
    "created_at": datetime.now().isoformat()
}


def check_docling_eval_version():
    """Check if docling-eval is installed with correct version."""
    try:
        import docling_eval
        version = getattr(docling_eval, '__version__', 'unknown')
        logger.info(f"docling-eval version: {version}")
        return version
    except ImportError:
        logger.error("docling-eval is not installed. Please install with:")
        logger.error("  pip install docling-eval>=0.3.0")
        sys.exit(1)


# =============================================================================
# Benchmark Runner
# =============================================================================

class DoclingBenchmarkRunner:
    """
    Wrapper for running docling-eval benchmarks.
    
    Preserves the original docling-eval API without modification.
    """
    
    SUPPORTED_BENCHMARKS = ['FinTabNet', 'PubTabNet', 'Pub1M', 'DPBench', 'OmniDocBench']
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.docling_eval_version = check_docling_eval_version()
        VERSION_INFO['docling_eval_version'] = self.docling_eval_version
    
    def create_ground_truth(
        self,
        benchmark: str,
        split: str = 'test',
        max_samples: Optional[int] = None
    ) -> Path:
        """
        Create ground truth dataset using docling-eval CLI.
        
        Args:
            benchmark: Benchmark name (e.g., 'FinTabNet')
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum samples (for quick testing)
            
        Returns:
            Path to ground truth directory
        """
        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {benchmark}. Choose from {self.SUPPORTED_BENCHMARKS}")
        
        gt_dir = self.output_dir / benchmark / 'ground_truth'
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            'docling-eval', 'create-gt',
            '--benchmark', benchmark,
            '--output-dir', str(gt_dir)
        ]
        
        if split:
            cmd.extend(['--split', split])
        
        if max_samples:
            cmd.extend(['--end-index', str(max_samples)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Ground truth creation failed: {result.stderr}")
            raise RuntimeError(f"docling-eval create-gt failed: {result.stderr}")
        
        logger.info(f"Ground truth created at: {gt_dir}")
        return gt_dir
    
    def create_predictions(
        self,
        benchmark: str,
        split: str = 'test',
        max_samples: Optional[int] = None,
        prediction_provider: str = 'TableFormer'
    ) -> Path:
        """
        Create predictions using docling-eval CLI.
        
        Args:
            benchmark: Benchmark name
            split: Dataset split
            max_samples: Maximum samples
            prediction_provider: Prediction method ('TableFormer', 'Docling')
            
        Returns:
            Path to predictions directory
        """
        pred_dir = self.output_dir / benchmark / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'docling-eval', 'create-eval',
            '--benchmark', benchmark,
            '--output-dir', str(pred_dir),
            '--prediction-provider', prediction_provider
        ]
        
        if split:
            cmd.extend(['--split', split])
        
        if max_samples:
            cmd.extend(['--end-index', str(max_samples)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Prediction creation failed: {result.stderr}")
            raise RuntimeError(f"docling-eval create-eval failed: {result.stderr}")
        
        logger.info(f"Predictions created at: {pred_dir}")
        return pred_dir
    
    def run_evaluation(
        self,
        benchmark: str,
        modality: str = 'table_structure'
    ) -> Dict[str, Any]:
        """
        Run evaluation using docling-eval CLI.
        
        Args:
            benchmark: Benchmark name
            modality: Evaluation modality ('table_structure', 'layout', etc.)
            
        Returns:
            Evaluation results dictionary
        """
        eval_dir = self.output_dir / benchmark
        
        cmd = [
            'docling-eval', 'evaluate',
            '--modality', modality,
            '--benchmark', benchmark,
            '--output-dir', str(eval_dir)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            raise RuntimeError(f"docling-eval evaluate failed: {result.stderr}")
        
        # Load evaluation results
        eval_json = eval_dir / f'evaluation_{benchmark}_tableformer.json'
        if eval_json.exists():
            with open(eval_json, 'r') as f:
                results = json.load(f)
        else:
            results = {'status': 'completed', 'output': result.stdout}
        
        logger.info(f"Evaluation completed for {benchmark}")
        return results
    
    def visualize(
        self,
        benchmark: str,
        modality: str = 'table_structure'
    ) -> List[Path]:
        """
        Generate visualization plots using docling-eval CLI.
        
        Args:
            benchmark: Benchmark name
            modality: Evaluation modality
            
        Returns:
            List of generated plot paths
        """
        vis_dir = self.output_dir / benchmark
        
        cmd = [
            'docling-eval', 'visualize',
            '--modality', modality,
            '--benchmark', benchmark,
            '--output-dir', str(vis_dir)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Visualization may have failed: {result.stderr}")
        
        # Find generated plots
        plots = list(vis_dir.glob('*.png'))
        logger.info(f"Generated {len(plots)} visualization plots")
        return plots
    
    def run_full_benchmark(
        self,
        benchmark: str = 'FinTabNet',
        split: str = 'test',
        max_samples: int = 1000,
        prediction_provider: str = 'TableFormer'
    ) -> Dict[str, Any]:
        """
        Run full benchmark pipeline: GT → Predictions → Evaluation → Visualization.
        
        Args:
            benchmark: Benchmark name
            split: Dataset split
            max_samples: Maximum samples to evaluate
            prediction_provider: Prediction method
            
        Returns:
            Complete results dictionary
        """
        logger.info(f"Running full {benchmark} benchmark (n={max_samples})")
        
        results = {
            'benchmark': benchmark,
            'split': split,
            'max_samples': max_samples,
            'prediction_provider': prediction_provider,
            'version_info': VERSION_INFO,
            'stages': {}
        }
        
        try:
            # Stage 1: Create ground truth
            logger.info("Stage 1/4: Creating ground truth...")
            gt_dir = self.create_ground_truth(benchmark, split, max_samples)
            results['stages']['ground_truth'] = {'status': 'success', 'path': str(gt_dir)}
        except Exception as e:
            results['stages']['ground_truth'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Ground truth creation failed: {e}")
            return results
        
        try:
            # Stage 2: Create predictions
            logger.info("Stage 2/4: Creating predictions...")
            pred_dir = self.create_predictions(benchmark, split, max_samples, prediction_provider)
            results['stages']['predictions'] = {'status': 'success', 'path': str(pred_dir)}
        except Exception as e:
            results['stages']['predictions'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Prediction creation failed: {e}")
            return results
        
        try:
            # Stage 3: Run evaluation
            logger.info("Stage 3/4: Running evaluation...")
            eval_results = self.run_evaluation(benchmark)
            results['stages']['evaluation'] = {'status': 'success', 'results': eval_results}
        except Exception as e:
            results['stages']['evaluation'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Evaluation failed: {e}")
        
        try:
            # Stage 4: Generate visualizations
            logger.info("Stage 4/4: Generating visualizations...")
            plots = self.visualize(benchmark)
            results['stages']['visualization'] = {'status': 'success', 'plots': [str(p) for p in plots]}
        except Exception as e:
            results['stages']['visualization'] = {'status': 'failed', 'error': str(e)}
            logger.warning(f"Visualization failed: {e}")
        
        # Save results
        results_path = self.output_dir / benchmark / f'benchmark_results_{benchmark}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark complete. Results saved to: {results_path}")
        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run docling-eval benchmark (wrapper)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FinTabNet benchmark (1000 samples)
  python run_docling_benchmark.py --benchmark FinTabNet --max-samples 1000
  
  # Run PubTabNet benchmark
  python run_docling_benchmark.py --benchmark PubTabNet --split val
  
  # Quick test (100 samples)
  python run_docling_benchmark.py --benchmark FinTabNet --max-samples 100
        """
    )
    
    parser.add_argument(
        '--benchmark',
        choices=['FinTabNet', 'PubTabNet', 'Pub1M', 'DPBench', 'OmniDocBench'],
        default='FinTabNet',
        help='Benchmark to run (default: FinTabNet)'
    )
    
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split (default: test)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples to evaluate (default: 1000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/baselines/docling',
        help='Output directory (default: ./outputs/baselines/docling)'
    )
    
    parser.add_argument(
        '--prediction-provider',
        choices=['TableFormer', 'Docling'],
        default='TableFormer',
        help='Prediction method (default: TableFormer)'
    )
    
    parser.add_argument(
        '--stage',
        choices=['all', 'gt', 'pred', 'eval', 'viz'],
        default='all',
        help='Which stage to run (default: all)'
    )
    
    args = parser.parse_args()
    
    runner = DoclingBenchmarkRunner(args.output_dir)
    
    if args.stage == 'all':
        results = runner.run_full_benchmark(
            benchmark=args.benchmark,
            split=args.split,
            max_samples=args.max_samples,
            prediction_provider=args.prediction_provider
        )
    elif args.stage == 'gt':
        runner.create_ground_truth(args.benchmark, args.split, args.max_samples)
    elif args.stage == 'pred':
        runner.create_predictions(args.benchmark, args.split, args.max_samples, args.prediction_provider)
    elif args.stage == 'eval':
        results = runner.run_evaluation(args.benchmark)
        print(json.dumps(results, indent=2))
    elif args.stage == 'viz':
        plots = runner.visualize(args.benchmark)
        print(f"Generated plots: {plots}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
