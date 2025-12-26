"""
Run TableFormer evaluation directly (not through DocumentConverter).
This is how docling-eval actually evaluates FinTabNet - using TableFormer directly
on cropped table images, NOT full document conversion.

Key insight: 
- Official docling-eval uses TableFormerPredictionProvider which:
  1. Takes an image known to contain a table
  2. Directly applies TableFormer model to recognize structure
  3. Does NOT need layout detection (table location is already known)

- Our previous approach used DocumentConverter which:
  1. Tries to detect layout/tables in the image first
  2. Then applies TableFormer to detected tables
  3. Fails when layout detection doesn't find the table
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fintabnet_otsl(data_dir: str, num_samples: int):
    """Load samples from FinTabNet_OTSL parquet."""
    from baselines.docling_eval.fintabnet_otsl_loader import load_fintabnet_otsl as _load
    samples = _load(data_dir, 'test', num_samples)
    
    # Convert (image_path, html) tuples to list of dicts
    result = []
    for img_path, html_gt in samples:
        result.append({
            'image_path': img_path,
            'filename': Path(img_path).stem,
            'html': html_gt
        })
    return result


def parse_ground_truth(ground_truth) -> Optional[List[List[str]]]:
    """Parse ground truth HTML to grid."""
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


class TableFormerRunner:
    """Run TableFormer directly on table images."""
    
    def __init__(self):
        self.version = "unknown"
        self._model = None
        self._tf_predictor = None
    
    def _init_model(self):
        if self._model is None:
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.models.table_structure_model import TableStructureModel
            
            # Initialize TableFormer model directly
            table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
            accelerator_options = AcceleratorOptions(
                num_threads=16, 
                device=AcceleratorDevice.AUTO
            )
            
            self._model = TableStructureModel(
                enabled=True,
                artifacts_path=None,
                options=table_structure_options,
                accelerator_options=accelerator_options,
            )
            
            # Get the predictor
            self._tf_predictor = self._model.tf_predictor
            
            try:
                import docling
                self.version = getattr(docling, '__version__', 'unknown')
            except:
                pass
            
            logger.info(f"Initialized TableFormer model (Docling {self.version})")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image directly with TableFormer."""
        from PIL import Image
        
        self._init_model()
        
        start = datetime.now()
        try:
            # Load image
            pil_image = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_image)
            
            # Get image dimensions
            height, width = img_array.shape[:2]
            
            # Define the table bounding box as the entire image
            # (Since this IS a cropped table image)
            table_bbox = [0, 0, width, height]
            
            # Prepare input for TableFormer
            # Create empty page tokens (no OCR - structure only)
            ocr_page = {
                'tokens': [],
                'height': height,
                'width': width,
                'image': img_array,
                'table_bboxes': [table_bbox]
            }
            
            # Run TableFormer prediction
            tf_output = self._tf_predictor.multi_table_predict(
                ocr_page,
                table_bboxes=[table_bbox],
                do_matching=True,
                correct_overlapping_cells=False,
                sort_row_col_indexes=True,
            )
            
            # Extract results
            if tf_output and len(tf_output) > 0:
                table_out = tf_output[0]
                num_rows = table_out.get('predict_details', {}).get('num_rows', 0)
                num_cols = table_out.get('predict_details', {}).get('num_cols', 0)
                
                # Build grid from cells
                grid = []
                if num_rows > 0 and num_cols > 0:
                    # Initialize empty grid
                    grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
                    
                    # Fill in cells from tf_responses
                    for cell in table_out.get('tf_responses', []):
                        row_idx = cell.get('start_row_offset_idx', 0)
                        col_idx = cell.get('start_col_offset_idx', 0)
                        text = cell.get('text', '')
                        if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
                            grid[row_idx][col_idx] = text
            else:
                grid = []
                num_rows = 0
                num_cols = 0
            
            elapsed = (datetime.now() - start).total_seconds()
            
            return {
                'grid': grid,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'elapsed_time': elapsed,
                'error': None
            }
            
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'grid': [],
                'num_rows': 0,
                'num_cols': 0,
                'elapsed_time': elapsed,
                'error': str(e)
            }


def compute_teds_structure(pred_rows: int, pred_cols: int, gt_rows: int, gt_cols: int) -> float:
    """Simple TEDS approximation based on grid dimensions."""
    if gt_rows == 0 or gt_cols == 0:
        return 1.0 if (pred_rows == 0 and pred_cols == 0) else 0.0
    if pred_rows == 0 or pred_cols == 0:
        return 0.0
    
    row_diff = abs(pred_rows - gt_rows)
    col_diff = abs(pred_cols - gt_cols)
    max_rows = max(pred_rows, gt_rows)
    max_cols = max(pred_cols, gt_cols)
    
    row_sim = 1 - (row_diff / max_rows)
    col_sim = 1 - (col_diff / max_cols)
    
    return row_sim * col_sim


def main():
    parser = argparse.ArgumentParser(description='Run TableFormer evaluation on FinTabNet')
    parser.add_argument('--data-dir', type=str, default='D:/datasets/FinTabNet_OTSL/data',
                        help='Path to FinTabNet_OTSL data directory')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()
    
    # Generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'tableformer_results_{args.num_samples}samples_{timestamp}.json'
    
    logger.info(f"Loading {args.num_samples} samples from FinTabNet_OTSL...")
    samples = load_fintabnet_otsl(args.data_dir, args.num_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Initialize runner
    runner = TableFormerRunner()
    
    results = []
    total_teds = 0.0
    
    for i, sample in enumerate(samples):
        image_path = sample['image_path']
        filename = sample['filename']
        gt_html = sample.get('html')
        
        logger.info(f"[{i+1}/{len(samples)}] Processing: {filename}")
        
        # Get ground truth grid
        gt_grid = parse_ground_truth(gt_html)
        gt_rows = len(gt_grid) if gt_grid else 0
        gt_cols = len(gt_grid[0]) if gt_grid and gt_grid[0] else 0
        
        # Run TableFormer
        result = runner.process_image(image_path)
        
        pred_rows = result.get('num_rows', 0)
        pred_cols = result.get('num_cols', 0)
        
        # Compute TEDS
        teds = compute_teds_structure(pred_rows, pred_cols, gt_rows, gt_cols)
        total_teds += teds
        
        result_entry = {
            'filename': filename,
            'pred_rows': pred_rows,
            'pred_cols': pred_cols,
            'gt_rows': gt_rows,
            'gt_cols': gt_cols,
            'teds': teds,
            'elapsed_time': result.get('elapsed_time', 0),
            'error': result.get('error')
        }
        results.append(result_entry)
        
        status = 'OK' if pred_rows > 0 else 'FAIL'
        logger.info(f"  Pred: {pred_rows}x{pred_cols} | GT: {gt_rows}x{gt_cols} | TEDS: {teds:.3f} | {status}")
    
    # Summary
    mean_teds = total_teds / len(results) if results else 0
    successful = sum(1 for r in results if r['pred_rows'] > 0)
    
    output_data = {
        'config': {
            'method': 'TableFormer Direct',
            'description': 'Direct TableFormer prediction on table images (same as docling-eval)',
            'data_dir': args.data_dir,
            'num_samples': args.num_samples,
            'docling_version': runner.version,
        },
        'summary': {
            'mean_teds': mean_teds,
            'successful': successful,
            'total': len(results),
            'success_rate': successful / len(results) if results else 0
        },
        'results': results
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TableFormer Direct Results")
    print(f"{'='*60}")
    print(f"Samples: {len(results)}")
    print(f"Successful: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"Mean TEDS: {mean_teds:.4f}")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
