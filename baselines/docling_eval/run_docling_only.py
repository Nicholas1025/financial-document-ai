"""
Run Docling evaluation only (for Global Python environment).
Saves results to JSON for later comparison with Old Pipeline.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fintabnet_otsl(data_dir: str, num_samples: int):
    """Load samples from FinTabNet_OTSL parquet."""
    from baselines.docling_eval.fintabnet_otsl_loader import load_fintabnet_otsl as _load
    return _load(data_dir, 'test', num_samples)


def parse_ground_truth(ground_truth) -> Optional[List[List[str]]]:
    """Parse ground truth HTML to grid."""
    import numpy as np
    from bs4 import BeautifulSoup
    
    if ground_truth is None:
        return None
    
    # If it's a numpy array of tokens, join them
    if isinstance(ground_truth, np.ndarray):
        if len(ground_truth) > 0:
            first_elem = str(ground_truth[0])
            if first_elem.startswith('<') and len(first_elem) < 50:
                html_str = ''.join(str(x) for x in ground_truth)
            else:
                html_str = str(ground_truth[0])
        else:
            return None
    else:
        html_str = str(ground_truth)
    
    # Parse HTML
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table') or soup
        grid = []
        for tr in table.find_all('tr'):
            row = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
            if row:
                grid.append(row)
        return grid if grid else None
    except Exception:
        return None


class DoclingRunner:
    """Run Docling table extraction."""
    
    def __init__(self):
        self.version = "unknown"
        self._converter = None
    
    def _init_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            try:
                import docling
                self.version = getattr(docling, '__version__', 'unknown')
            except:
                pass
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return grid."""
        self._init_converter()
        
        start = datetime.now()
        try:
            result = self._converter.convert(image_path)
            tables = list(result.document.tables)
            
            if not tables:
                return {'error': 'No tables found', 'grid': []}
            
            # Get first table's grid
            table = tables[0]
            try:
                df = table.export_to_dataframe()
                grid = [df.columns.tolist()] + df.values.tolist()
                grid = [[str(c) if c is not None else '' for c in row] for row in grid]
            except:
                grid = []
            
            return {
                'grid': grid,
                'processing_time_ms': (datetime.now() - start).total_seconds() * 1000
            }
        except Exception as e:
            return {'error': str(e), 'grid': []}


def compute_teds_struct(pred_grid: List[List[str]], gt_grid: List[List[str]]) -> float:
    """Compute structure-only TEDS (based on grid dimensions match)."""
    if not pred_grid or not gt_grid:
        return 0.0
    
    pred_rows, pred_cols = len(pred_grid), max(len(r) for r in pred_grid) if pred_grid else 0
    gt_rows, gt_cols = len(gt_grid), max(len(r) for r in gt_grid) if gt_grid else 0
    
    # Simple structural similarity based on dimensions
    row_sim = 1.0 - abs(pred_rows - gt_rows) / max(pred_rows, gt_rows, 1)
    col_sim = 1.0 - abs(pred_cols - gt_cols) / max(pred_cols, gt_cols, 1)
    
    return (row_sim + col_sim) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='D:/datasets/FinTabNet_OTSL/data')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--output', default='outputs/docling_results.json')
    args = parser.parse_args()
    
    # Load samples
    logger.info(f"Loading {args.max_samples} samples from {args.data_dir}")
    samples = load_fintabnet_otsl(args.data_dir, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Initialize runner
    runner = DoclingRunner()
    
    results = []
    for i, (img_path, html_gt) in enumerate(samples):
        print(f"\rProgress: {i+1}/{len(samples)} ({100*(i+1)/len(samples):.1f}%)", end='')
        
        sample_id = Path(img_path).stem
        gt_grid = parse_ground_truth(html_gt)
        
        # Run Docling
        output = runner.process_image(img_path)
        pred_grid = output.get('grid', [])
        
        # Compute metrics
        teds = compute_teds_struct(pred_grid, gt_grid) if gt_grid else 0.0
        
        results.append({
            'sample_id': sample_id,
            'pred_grid': pred_grid,
            'gt_grid': gt_grid,
            'pred_rows': len(pred_grid) if pred_grid else 0,
            'pred_cols': max(len(r) for r in pred_grid) if pred_grid else 0,
            'gt_rows': len(gt_grid) if gt_grid else 0,
            'gt_cols': max(len(r) for r in gt_grid) if gt_grid else 0,
            'teds_struct': teds,
            'processing_time_ms': output.get('processing_time_ms', 0),
            'error': output.get('error')
        })
    
    print()  # newline after progress
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'system': 'docling',
        'version': runner.version,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(samples),
        'mean_teds_struct': sum(r['teds_struct'] for r in results) / len(results) if results else 0,
        'samples_with_results': sum(1 for r in results if r['pred_grid']),
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    print(f"\n=== Docling Results ===")
    print(f"Samples: {summary['samples_with_results']}/{summary['num_samples']}")
    print(f"Mean TEDS (struct): {summary['mean_teds_struct']:.4f}")


if __name__ == '__main__':
    main()
