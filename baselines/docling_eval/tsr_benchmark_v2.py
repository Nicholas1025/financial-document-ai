"""
TSR Benchmark v2 - 直接使用底层 TableFormer 模型

这个版本直接调用 docling_ibm_models.tableformer 底层模型,
绕过 DocumentConverter 的 Layout Detection

实验设定:
- 输入: 裁剪好的表格图片
- 任务: Table Structure Recognition (TSR-only)
- 不使用 Layout Detection
- 假设整张图片就是一个表格
"""

import argparse
import json
import logging
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEDS Implementation
# ============================================================================

class TEDSCalculator:
    """TEDS (Tree Edit Distance Similarity) Calculator"""
    
    def __init__(self, structure_only: bool = True):
        self.structure_only = structure_only
        from apted import APTED, Config
        from apted.helpers import Tree
        self.APTED = APTED
        self.Tree = Tree
    
    def _html_to_tree(self, html: str):
        """Convert HTML table to tree structure."""
        
        class TableTree(self.Tree):
            def __init__(self, tag):
                super().__init__(tag)
        
        def _build_tree(element, structure_only):
            if element.name is None:
                return None
            
            tag = element.name
            colspan = int(element.get('colspan', 1))
            rowspan = int(element.get('rowspan', 1))
            
            if colspan > 1 or rowspan > 1:
                tag = f"{tag}[{colspan},{rowspan}]"
            
            if not structure_only and tag in ['td', 'th']:
                text = element.get_text(strip=True)
                if text:
                    tag = f"{tag}:{text}"
            
            node = TableTree(tag)
            
            for child in element.children:
                if hasattr(child, 'name') and child.name:
                    child_node = _build_tree(child, structure_only)
                    if child_node:
                        node.children.append(child_node)
            
            return node
        
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            soup = BeautifulSoup(f'<table>{html}</table>', 'html.parser')
            table = soup.find('table')
        
        if table is None:
            return TableTree('table')
        
        return _build_tree(table, self.structure_only)
    
    def _tree_size(self, tree) -> int:
        if tree is None:
            return 0
        count = 1
        for child in tree.children:
            count += self._tree_size(child)
        return count
    
    def compute(self, pred_html: str, gt_html: str) -> float:
        try:
            pred_tree = self._html_to_tree(pred_html)
            gt_tree = self._html_to_tree(gt_html)
            
            pred_size = self._tree_size(pred_tree)
            gt_size = self._tree_size(gt_tree)
            
            if pred_size == 0 and gt_size == 0:
                return 1.0
            if pred_size == 0 or gt_size == 0:
                return 0.0
            
            apted = self.APTED(pred_tree, gt_tree)
            ted = apted.compute_edit_distance()
            
            max_size = max(pred_size, gt_size)
            teds = 1.0 - ted / max_size
            
            return max(0.0, teds)
            
        except Exception as e:
            logger.warning(f"TEDS computation error: {e}")
            return 0.0


# ============================================================================
# Data Loading
# ============================================================================

def load_fintabnet_samples(data_dir: str, num_samples: int, split: str = 'test') -> List[Dict]:
    """Load FinTabNet samples."""
    import pandas as pd
    import io
    
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob(f"{split}-*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    temp_dir = tempfile.mkdtemp(prefix="tsr_benchmark_v2_")
    logger.info(f"Temp directory: {temp_dir}")
    
    samples = []
    count = 0
    
    for pf in parquet_files:
        if count >= num_samples:
            break
        
        logger.info(f"Loading {pf.name}...")
        df = pd.read_parquet(pf)
        
        for idx, row in df.iterrows():
            if count >= num_samples:
                break
            
            try:
                image_data = row.get('image')
                if image_data is None:
                    continue
                
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image = Image.open(io.BytesIO(image_data['bytes']))
                else:
                    continue
                
                filename = row.get('filename', f'sample_{count}.png')
                
                html_data = row.get('html')
                if html_data is None:
                    continue
                
                if isinstance(html_data, np.ndarray):
                    if len(html_data) > 0:
                        first = str(html_data[0])
                        if first.startswith('<') and len(first) < 50:
                            gt_html = ''.join(str(x) for x in html_data)
                        else:
                            gt_html = str(html_data[0])
                    else:
                        continue
                else:
                    gt_html = str(html_data)
                
                if not gt_html.strip().startswith('<table'):
                    gt_html = f'<table>{gt_html}</table>'
                
                image_path = Path(temp_dir) / f"{Path(filename).stem}.png"
                image = image.convert('RGB')
                image.save(image_path)
                
                samples.append({
                    'image_path': str(image_path),
                    'filename': filename,
                    'gt_html': gt_html,
                    'image_size': image.size,  # (width, height)
                })
                count += 1
                
            except Exception as e:
                logger.warning(f"Error loading row {idx}: {e}")
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


# ============================================================================
# TSR Methods
# ============================================================================

class TableTransformerTSR:
    """Table Transformer v1.1 for TSR-only"""
    
    def __init__(self):
        self._recognizer = None
        self._config = None
    
    def _init(self):
        if self._recognizer is not None:
            return
        
        import yaml
        from modules.structure import TableStructureRecognizer
        
        config_path = project_root / 'configs' / 'config.yaml'
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        
        self._recognizer = TableStructureRecognizer(
            config=self._config,
            use_v1_1=True
        )
        logger.info("Initialized Table Transformer v1.1")
    
    def predict(self, image_path: str) -> str:
        self._init()
        
        try:
            image = Image.open(image_path).convert('RGB')
            structure = self._recognizer.recognize(image)
            html = self._structure_to_html(structure)
            return html
        except Exception as e:
            logger.error(f"Table Transformer error: {e}")
            return "<table></table>"
    
    def _structure_to_html(self, structure: Dict) -> str:
        rows = structure.get('rows', [])
        cols = structure.get('columns', [])
        
        if not rows or not cols:
            return "<table></table>"
        
        num_rows = len(rows)
        num_cols = len(cols)
        
        html_parts = ['<table>']
        
        for row_idx in range(num_rows):
            if row_idx == 0:
                html_parts.append('<thead>')
            elif row_idx == 1:
                html_parts.append('<tbody>')
            
            html_parts.append('<tr>')
            for col_idx in range(num_cols):
                tag = 'th' if row_idx == 0 else 'td'
                html_parts.append(f'<{tag}></{tag}>')
            html_parts.append('</tr>')
            
            if row_idx == 0:
                html_parts.append('</thead>')
        
        if num_rows > 1:
            html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        return ''.join(html_parts)


class DoclingTableFormerDirect:
    """
    直接使用 Docling TableFormer 底层模型
    
    绕过 DocumentConverter，直接调用 TFPredictor
    假设整张图片就是一个表格
    """
    
    def __init__(self, mode: str = 'accurate'):
        self.mode = mode
        self._predictor = None
        self._config = None
    
    def _init(self):
        if self._predictor is not None:
            return
        
        from docling.models.table_structure_model import TableStructureModel
        
        # Download models if needed
        artifacts_path = TableStructureModel.download_models()
        model_path = artifacts_path / "model_artifacts" / "tableformer"
        
        if self.mode == 'accurate':
            model_path = model_path / "accurate"
        else:
            model_path = model_path / "fast"
        
        # Load config
        import docling_ibm_models.tableformer.common as c
        from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
        
        self._config = c.read_config(str(model_path / "tm_config.json"))
        self._config["model"]["save_dir"] = str(model_path)
        
        self._predictor = TFPredictor(self._config, device='cpu', num_threads=4)
        logger.info(f"Initialized TableFormer Direct ({self.mode})")
    
    def predict(self, image_path: str, image_size: Tuple[int, int] = None) -> str:
        """
        Predict table structure.
        
        由于我们假设整张图片就是一个表格，
        table_bbox 就是整张图片的边界。
        """
        self._init()
        
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # Scale factor (TableFormer uses 144 dpi, assume 72 dpi input)
            scale = 2.0
            
            # Prepare input (iocr_page format)
            page_input = {
                "width": width * scale,
                "height": height * scale,
                "image": np.asarray(image.resize((int(width*scale), int(height*scale)))),
                "tokens": []  # No OCR tokens - we only want structure
            }
            
            # Table bbox is the entire image
            table_bbox = [0, 0, width * scale, height * scale]
            
            # Run prediction
            tf_output = self._predictor.multi_table_predict(
                page_input,
                [table_bbox],
                do_matching=False  # No cell matching without tokens
            )
            
            if not tf_output:
                return "<table></table>"
            
            table_out = tf_output[0]
            
            # Extract structure from prediction
            predict_details = table_out.get("predict_details", {})
            num_rows = predict_details.get("num_rows", 0)
            num_cols = predict_details.get("num_cols", 0)
            
            # Build HTML from cell predictions
            html = self._build_html_from_cells(table_out, num_rows, num_cols)
            
            return html
            
        except Exception as e:
            logger.error(f"TableFormer Direct error: {e}")
            traceback.print_exc()
            return "<table></table>"
    
    def _build_html_from_cells(self, table_out: Dict, num_rows: int, num_cols: int) -> str:
        """Build HTML from TableFormer output."""
        
        if num_rows == 0 or num_cols == 0:
            return "<table></table>"
        
        # Get cell predictions
        cells = table_out.get("tf_responses", [])
        
        # Build grid
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        for cell in cells:
            row_start = cell.get("start_row_offset_idx", 0)
            col_start = cell.get("start_col_offset_idx", 0)
            row_end = cell.get("end_row_offset_idx", row_start + 1)
            col_end = cell.get("end_col_offset_idx", col_start + 1)
            
            if row_start < num_rows and col_start < num_cols:
                grid[row_start][col_start] = {
                    'rowspan': row_end - row_start,
                    'colspan': col_end - col_start,
                    'is_header': cell.get("column_header", False)
                }
        
        # Build HTML
        html_parts = ['<table>']
        
        # Track which cells are covered by spans
        covered = [[False for _ in range(num_cols)] for _ in range(num_rows)]
        
        for row_idx in range(num_rows):
            # Detect header row
            is_header_row = row_idx == 0 or any(
                grid[row_idx][c] and grid[row_idx][c].get('is_header', False)
                for c in range(num_cols) if grid[row_idx][c]
            )
            
            if row_idx == 0 and is_header_row:
                html_parts.append('<thead>')
            elif row_idx == 1 or (row_idx == 0 and not is_header_row):
                if row_idx == 1:
                    html_parts.append('</thead>')
                html_parts.append('<tbody>')
            
            html_parts.append('<tr>')
            
            for col_idx in range(num_cols):
                if covered[row_idx][col_idx]:
                    continue
                
                cell = grid[row_idx][col_idx]
                tag = 'th' if is_header_row else 'td'
                
                if cell:
                    rowspan = cell.get('rowspan', 1)
                    colspan = cell.get('colspan', 1)
                    
                    # Mark covered cells
                    for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                        for c in range(col_idx, min(col_idx + colspan, num_cols)):
                            if r != row_idx or c != col_idx:
                                covered[r][c] = True
                    
                    attrs = []
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    html_parts.append(f'<{tag}{attr_str}></{tag}>')
                else:
                    html_parts.append(f'<{tag}></{tag}>')
            
            html_parts.append('</tr>')
            
            if row_idx == 0 and is_header_row:
                pass  # Will close thead later
        
        if num_rows > 0:
            html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        return ''.join(html_parts)


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(samples: List[Dict], methods: Dict[str, Any], teds_calculator: TEDSCalculator) -> Dict[str, List[Dict]]:
    results = {name: [] for name in methods}
    
    for i, sample in enumerate(samples):
        filename = sample['filename']
        image_path = sample['image_path']
        gt_html = sample['gt_html']
        image_size = sample.get('image_size')
        
        logger.info(f"[{i+1}/{len(samples)}] Processing {filename}")
        
        for method_name, method in methods.items():
            try:
                if hasattr(method, 'predict') and 'image_size' in method.predict.__code__.co_varnames:
                    pred_html = method.predict(image_path, image_size)
                else:
                    pred_html = method.predict(image_path)
                
                teds_score = teds_calculator.compute(pred_html, gt_html)
                
                gt_soup = BeautifulSoup(gt_html, 'html.parser')
                gt_rows = len(gt_soup.find_all('tr'))
                
                pred_soup = BeautifulSoup(pred_html, 'html.parser')
                pred_rows = len(pred_soup.find_all('tr'))
                
                result = {
                    'filename': filename,
                    'teds': teds_score,
                    'gt_rows': gt_rows,
                    'pred_rows': pred_rows,
                    'success': pred_rows > 0,
                }
                
                results[method_name].append(result)
                
                status = 'OK' if pred_rows > 0 else 'FAIL'
                logger.info(f"  {method_name}: TEDS={teds_score:.4f} | rows: {pred_rows}/{gt_rows} | {status}")
                
            except Exception as e:
                logger.error(f"  {method_name}: Error - {e}")
                traceback.print_exc()
                results[method_name].append({
                    'filename': filename,
                    'teds': 0.0,
                    'success': False,
                    'error': str(e)
                })
    
    return results


def compute_statistics(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    summary = {}
    
    for method_name, result_list in results.items():
        if not result_list:
            continue
        
        teds_scores = [r['teds'] for r in result_list]
        success_count = sum(1 for r in result_list if r['success'])
        
        summary[method_name] = {
            'avg_teds': float(np.mean(teds_scores)),
            'std_teds': float(np.std(teds_scores)),
            'median_teds': float(np.median(teds_scores)),
            'min_teds': float(np.min(teds_scores)),
            'max_teds': float(np.max(teds_scores)),
            'num_samples': len(result_list),
            'num_success': success_count,
            'num_fail': len(result_list) - success_count,
            'success_rate': success_count / len(result_list),
        }
    
    return summary


def print_report(summary: Dict[str, Dict], config: Dict):
    print("\n" + "="*80)
    print("TSR BENCHMARK v2 - Direct TableFormer Comparison")
    print("="*80)
    print(f"Dataset: FinTabNet")
    print(f"Task: Table Structure Recognition (TSR-only)")
    print(f"Metric: TEDS (structure-only)")
    print(f"Samples: {config.get('num_samples', 'N/A')}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    print("\n### RESULTS TABLE (for FYP Report)")
    print("-"*80)
    print(f"| {'Method':<40} | {'Setting':<15} | {'Avg. TEDS':^10} | {'Std':^8} | {'#Samples':^10} | {'#Fail':^8} |")
    print(f"|{'-'*40}|{'-'*15}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*10}|")
    
    for method_name, stats in summary.items():
        print(f"| {method_name:<40} | {'TSR-only':<15} | {stats['avg_teds']:^10.4f} | "
              f"{stats['std_teds']:^8.4f} | {stats['num_samples']:^10} | {stats['num_fail']:^8} |")
    
    print(f"| {'Docling TableFormer (Official)':<40} | {'TSR-only':<15} | {'0.90':^10} | "
          f"{'0.09':^8} | {'1000':^10} | {'-':^8} |")
    
    print("-"*80)
    
    print("\n### DETAILED STATISTICS")
    for method_name, stats in summary.items():
        print(f"\n{method_name}:")
        print(f"  Average TEDS:  {stats['avg_teds']:.4f}")
        print(f"  Std Dev:       {stats['std_teds']:.4f}")
        print(f"  Median TEDS:   {stats['median_teds']:.4f}")
        print(f"  Success Rate:  {stats['success_rate']*100:.1f}% ({stats['num_success']}/{stats['num_samples']})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='TSR Benchmark v2')
    parser.add_argument('--data-dir', type=str, default='D:/datasets/FinTabNet_OTSL/data')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--methods', nargs='+', 
                        default=['table_transformer', 'tableformer_direct'],
                        choices=['table_transformer', 'tableformer_direct'])
    parser.add_argument('--tableformer-mode', type=str, default='accurate',
                        choices=['fast', 'accurate'])
    
    args = parser.parse_args()
    
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'tsr_benchmark_v2_{args.num_samples}samples_{timestamp}.json'
    
    logger.info(f"Loading {args.num_samples} samples from FinTabNet...")
    samples = load_fintabnet_samples(args.data_dir, args.num_samples)
    
    if not samples:
        logger.error("No samples loaded!")
        return
    
    teds_calculator = TEDSCalculator(structure_only=True)
    
    methods = {}
    if 'table_transformer' in args.methods:
        methods['Table Transformer v1.1'] = TableTransformerTSR()
    if 'tableformer_direct' in args.methods:
        methods[f'TableFormer Direct ({args.tableformer_mode})'] = DoclingTableFormerDirect(
            mode=args.tableformer_mode
        )
    
    logger.info(f"Running benchmark with methods: {list(methods.keys())}")
    
    results = run_benchmark(samples, methods, teds_calculator)
    summary = compute_statistics(results)
    
    config = {
        'data_dir': args.data_dir,
        'num_samples': len(samples),
        'methods': list(methods.keys()),
        'tableformer_mode': args.tableformer_mode,
    }
    print_report(summary, config)
    
    output_data = {
        'config': {
            **config,
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'TSR-only (structure-only TEDS)',
        },
        'summary': summary,
        'detailed_results': {name: results[name] for name in results}
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
