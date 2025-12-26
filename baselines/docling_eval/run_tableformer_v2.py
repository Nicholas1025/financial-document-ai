"""
Run TableFormer evaluation correctly - matching docling-eval's approach.

Key insight: TableFormer needs:
1. A DoclingDocument with table provenance (bounding box)
2. Page tokens (cell text + bbox from ground truth cells)
3. Page image

The official docling-eval:
1. Creates a DoclingDocument with ground truth table structure
2. Passes it to TableFormerPredictionProvider
3. TableFormer re-predicts the structure using the image and cell positions

For FinTabNet cropped table images:
- The whole image IS the table (bbox = 0, 0, width, height)
- We need to provide cell information from the 'cells' column
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import io

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fintabnet_samples(data_dir: str, num_samples: int) -> List[Dict]:
    """Load raw samples from FinTabNet_OTSL parquet with all columns."""
    import pandas as pd
    from PIL import Image
    import tempfile
    
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("test-*.parquet"))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {data_dir}")
        return []
    
    temp_dir = tempfile.mkdtemp(prefix="fintabnet_tf_")
    samples = []
    count = 0
    
    for parquet_file in parquet_files:
        if count >= num_samples:
            break
        
        logger.info(f"Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        
        for idx, row in df.iterrows():
            if count >= num_samples:
                break
            
            try:
                # Extract image
                image_data = row.get('image')
                if image_data is None:
                    continue
                
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    continue
                
                # Save image
                filename = row.get('filename', f'sample_{count}.png')
                image_path = Path(temp_dir) / f"{Path(filename).stem}.png"
                image.save(image_path)
                
                # Get all relevant columns
                sample = {
                    'image_path': str(image_path),
                    'filename': filename,
                    'image': image,
                    'html': row.get('html'),
                    'html_restored': row.get('html_restored'),
                    'cells': row.get('cells'),  # Important: cell bbox + text info
                    'rows': row.get('rows'),
                    'cols': row.get('cols'),
                }
                samples.append(sample)
                count += 1
                
            except Exception as e:
                logger.warning(f"Error loading row {idx}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def parse_html_to_grid(html) -> Tuple[int, int]:
    """Parse HTML to get grid dimensions."""
    from bs4 import BeautifulSoup
    
    if html is None:
        return 0, 0
    
    # Handle numpy array of tokens
    if isinstance(html, np.ndarray):
        if len(html) > 0:
            first = str(html[0])
            if first.startswith('<') and len(first) < 50:
                html_str = ''.join(str(x) for x in html)
            else:
                html_str = str(html[0])
        else:
            return 0, 0
    else:
        html_str = str(html)
    
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table') or soup
        rows = table.find_all('tr')
        if not rows:
            return 0, 0
        max_cols = 0
        for tr in rows:
            cols = len(tr.find_all(['td', 'th']))
            max_cols = max(max_cols, cols)
        return len(rows), max_cols
    except:
        return 0, 0


def create_page_tokens_from_cells(cells, height: float, width: float) -> Dict:
    """Create page tokens from cell data (matching docling-eval's approach)."""
    from docling_core.types.doc import BoundingBox, CoordOrigin
    
    tokens = []
    cnt = 0
    
    if cells is None:
        return {'tokens': [], 'height': height, 'width': width}
    
    # cells is a list of rows, each row is a list of cells
    # Each cell has 'tokens' (list of text) and 'bbox' (l, t, r, b)
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            
            cell_tokens = cell.get('tokens', [])
            cell_bbox = cell.get('bbox', [0, 0, 0, 0])
            
            text = ''.join(str(t) for t in cell_tokens) if cell_tokens else ''
            
            if len(cell_bbox) >= 4:
                token = {
                    'bbox': {
                        'l': float(cell_bbox[0]),
                        't': float(cell_bbox[1]),
                        'r': float(cell_bbox[2]),
                        'b': float(cell_bbox[3]),
                        'coord_origin': str(CoordOrigin.TOPLEFT.value),
                    },
                    'text': text,
                    'id': cnt,
                }
                tokens.append(token)
                cnt += 1
    
    return {'tokens': tokens, 'height': height, 'width': width}


class TableFormerEvaluator:
    """Evaluate using TableFormer with proper setup."""
    
    def __init__(self):
        self._model = None
        self.version = "unknown"
    
    def _init_model(self):
        if self._model is None:
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.models.table_structure_model import TableStructureModel
            
            table_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
            accel_options = AcceleratorOptions(num_threads=16, device=AcceleratorDevice.AUTO)
            
            self._model = TableStructureModel(
                enabled=True,
                artifacts_path=None,
                options=table_options,
                accelerator_options=accel_options,
            )
            
            try:
                import docling
                self.version = getattr(docling, '__version__', 'unknown')
            except:
                pass
            
            logger.info(f"Initialized TableFormer (Docling {self.version})")
    
    def predict(self, sample: Dict) -> Tuple[int, int, float]:
        """
        Predict table structure.
        Returns (predicted_rows, predicted_cols, elapsed_time)
        """
        from PIL import Image
        
        self._init_model()
        
        start = datetime.now()
        
        try:
            # Load image
            if 'image' in sample and sample['image'] is not None:
                pil_image = sample['image']
            else:
                pil_image = Image.open(sample['image_path']).convert('RGB')
            
            img_array = np.array(pil_image)
            height, width = img_array.shape[:2]
            
            # Create page tokens from cells
            page_tokens = create_page_tokens_from_cells(
                sample.get('cells'),
                height=float(height),
                width=float(width)
            )
            
            # Table bbox is the whole image (since these are cropped table images)
            table_bbox = [0, 0, width, height]
            
            # Prepare input
            ocr_page = {
                'tokens': page_tokens.get('tokens', []),
                'height': height,
                'width': width,
                'image': img_array,
                'table_bboxes': [table_bbox]
            }
            
            # Run TableFormer
            predictor = self._model.tf_predictor
            tf_output = predictor.multi_table_predict(
                ocr_page,
                table_bboxes=[table_bbox],
                do_matching=True,
                correct_overlapping_cells=False,
                sort_row_col_indexes=True,
            )
            
            elapsed = (datetime.now() - start).total_seconds()
            
            if tf_output and len(tf_output) > 0:
                table_out = tf_output[0]
                num_rows = table_out.get('predict_details', {}).get('num_rows', 0)
                num_cols = table_out.get('predict_details', {}).get('num_cols', 0)
                return num_rows, num_cols, elapsed
            else:
                return 0, 0, elapsed
                
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(f"Error: {e}")
            return 0, 0, elapsed


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
    parser.add_argument('--data-dir', type=str, default='D:/datasets/FinTabNet_OTSL/data')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'tableformer_v2_results_{args.num_samples}samples_{timestamp}.json'
    
    logger.info(f"Loading {args.num_samples} samples...")
    samples = load_fintabnet_samples(args.data_dir, args.num_samples)
    
    if not samples:
        logger.error("No samples loaded!")
        return
    
    # Check first sample's cells column
    first_sample = samples[0]
    cells = first_sample.get('cells')
    logger.info(f"First sample cells type: {type(cells)}")
    if cells is not None and len(cells) > 0:
        logger.info(f"First row type: {type(cells[0])}")
        if len(cells[0]) > 0:
            logger.info(f"First cell: {cells[0][0]}")
    
    evaluator = TableFormerEvaluator()
    results = []
    total_teds = 0.0
    
    for i, sample in enumerate(samples):
        filename = sample['filename']
        logger.info(f"[{i+1}/{len(samples)}] Processing: {filename}")
        
        # Get ground truth dimensions
        gt_rows, gt_cols = parse_html_to_grid(sample.get('html'))
        
        # Also check from columns
        if gt_rows == 0:
            gt_rows = sample.get('rows', 0)
            gt_cols = sample.get('cols', 0)
        
        # Run prediction
        pred_rows, pred_cols, elapsed = evaluator.predict(sample)
        
        # Compute TEDS
        teds = compute_teds_structure(pred_rows, pred_cols, gt_rows, gt_cols)
        total_teds += teds
        
        result = {
            'filename': filename,
            'pred_rows': pred_rows,
            'pred_cols': pred_cols,
            'gt_rows': gt_rows,
            'gt_cols': gt_cols,
            'teds': teds,
            'elapsed': elapsed,
        }
        results.append(result)
        
        status = 'OK' if pred_rows > 0 else 'FAIL'
        logger.info(f"  Pred: {pred_rows}x{pred_cols} | GT: {gt_rows}x{gt_cols} | TEDS: {teds:.3f} | {status}")
    
    # Summary
    mean_teds = total_teds / len(results) if results else 0
    successful = sum(1 for r in results if r['pred_rows'] > 0)
    
    output_data = {
        'config': {
            'method': 'TableFormer with PageTokens',
            'description': 'Using cell bbox info from FinTabNet for proper TableFormer input',
            'data_dir': args.data_dir,
            'num_samples': args.num_samples,
            'docling_version': evaluator.version,
        },
        'summary': {
            'mean_teds': mean_teds,
            'successful': successful,
            'total': len(results),
            'success_rate': successful / len(results) if results else 0
        },
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TableFormer v2 Results")
    print(f"{'='*60}")
    print(f"Samples: {len(results)}")
    print(f"Successful: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"Mean TEDS: {mean_teds:.4f}")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
