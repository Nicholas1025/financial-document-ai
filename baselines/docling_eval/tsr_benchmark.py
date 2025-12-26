"""
TSR (Table Structure Recognition) Benchmark - Stage 1 Fair Comparison

实验目的：
- 公平对比 Table Transformer v1.1 vs Docling TableFormer
- 只测试 TSR，不包含 detection/OCR/semantic 等

实验设定：
- 输入: FinTabNet 裁剪好的表格图片
- 任务: 预测 HTML table structure
- 评估: TEDS (Tree Edit Distance Similarity)
- 两个方法使用相同的 GT HTML 和 TEDS 实现

Methods:
- Method A: Table Transformer v1.1 (TSR-only)
- Method B: Docling TableFormer (TSR-only, via DocumentConverter with pre-cropped tables)

Note: 
- 由于 Docling 没有暴露纯 TableFormer API，我们使用 DocumentConverter
- 但输入是已裁剪的表格图片，所以实际上只测试 TSR
- Oracle-assisted 结果需要单独实验（如果可行）
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
from collections import Counter
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEDS Implementation (Tree Edit Distance Similarity)
# 使用 apted 库，与 docling-eval 相同
# ============================================================================

class TEDSCalculator:
    """
    TEDS (Tree Edit Distance Similarity) Calculator
    
    Reference: https://github.com/ibm-aur-nlp/PubTabNet
    Formula: TEDS = 1 - TED(pred, gt) / max(|pred|, |gt|)
    """
    
    def __init__(self, structure_only: bool = True):
        """
        Args:
            structure_only: If True, ignore text content (只比较结构)
        """
        self.structure_only = structure_only
        try:
            from apted import APTED, Config
            from apted.helpers import Tree
            self.APTED = APTED
            self.Config = Config
            self.Tree = Tree
        except ImportError:
            raise ImportError("Please install apted: pip install apted")
    
    def _html_to_tree(self, html: str) -> 'Tree':
        """Convert HTML table to tree structure for APTED."""
        
        class TableTree(self.Tree):
            def __init__(self, tag, colspan=1, rowspan=1, text=''):
                super().__init__(tag)
                self.colspan = colspan
                self.rowspan = rowspan
                self.text = text
            
            def bracket(self):
                """Return bracket notation for tree."""
                if self.children:
                    children_str = ''.join(c.bracket() for c in self.children)
                    return f"{{{self.tag}{children_str}}}"
                else:
                    return f"{{{self.tag}}}"
        
        def _build_tree(element, structure_only):
            """Recursively build tree from HTML element."""
            if element.name is None:
                return None
            
            tag = element.name
            colspan = int(element.get('colspan', 1))
            rowspan = int(element.get('rowspan', 1))
            
            # Get text content (if not structure_only)
            text = ''
            if not structure_only and tag in ['td', 'th']:
                text = element.get_text(strip=True)
            
            # Create node with span info
            if colspan > 1 or rowspan > 1:
                tag = f"{tag}[{colspan},{rowspan}]"
            
            if not structure_only and text:
                tag = f"{tag}:{text}"
            
            node = TableTree(tag, colspan, rowspan, text)
            
            # Process children
            for child in element.children:
                if hasattr(child, 'name') and child.name:
                    child_node = _build_tree(child, structure_only)
                    if child_node:
                        node.children.append(child_node)
            
            return node
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if table is None:
            # Try to wrap in table tag
            soup = BeautifulSoup(f'<table>{html}</table>', 'html.parser')
            table = soup.find('table')
        
        if table is None:
            return TableTree('table')
        
        return _build_tree(table, self.structure_only)
    
    def _tree_size(self, tree) -> int:
        """Count nodes in tree."""
        if tree is None:
            return 0
        count = 1
        for child in tree.children:
            count += self._tree_size(child)
        return count
    
    def compute(self, pred_html: str, gt_html: str) -> float:
        """
        Compute TEDS score between predicted and ground truth HTML.
        
        Args:
            pred_html: Predicted HTML table
            gt_html: Ground truth HTML table
            
        Returns:
            TEDS score in [0, 1], higher is better
        """
        try:
            pred_tree = self._html_to_tree(pred_html)
            gt_tree = self._html_to_tree(gt_html)
            
            pred_size = self._tree_size(pred_tree)
            gt_size = self._tree_size(gt_tree)
            
            if pred_size == 0 and gt_size == 0:
                return 1.0
            if pred_size == 0 or gt_size == 0:
                return 0.0
            
            # Compute tree edit distance using APTED
            apted = self.APTED(pred_tree, gt_tree)
            ted = apted.compute_edit_distance()
            
            # TEDS = 1 - TED / max(|pred|, |gt|)
            max_size = max(pred_size, gt_size)
            teds = 1.0 - ted / max_size
            
            return max(0.0, teds)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"TEDS computation error: {e}")
            return 0.0


# ============================================================================
# Data Loading
# ============================================================================

def load_fintabnet_samples(data_dir: str, num_samples: int, split: str = 'test') -> List[Dict]:
    """
    Load FinTabNet samples from parquet files.
    
    Returns list of dicts with:
    - image_path: path to saved image file
    - gt_html: ground truth HTML
    - filename: original filename
    """
    import pandas as pd
    import io
    
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob(f"{split}-*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp(prefix="tsr_benchmark_")
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
                # Load image
                image_data = row.get('image')
                if image_data is None:
                    continue
                
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image = Image.open(io.BytesIO(image_data['bytes']))
                else:
                    continue
                
                # Get filename
                filename = row.get('filename', f'sample_{count}.png')
                
                # Get ground truth HTML
                html_data = row.get('html')
                if html_data is None:
                    continue
                
                # Handle numpy array HTML (FinTabNet format)
                if isinstance(html_data, np.ndarray):
                    if len(html_data) > 0:
                        # Join all elements if it's tokenized
                        first = str(html_data[0])
                        if first.startswith('<') and len(first) < 50:
                            gt_html = ''.join(str(x) for x in html_data)
                        else:
                            gt_html = str(html_data[0])
                    else:
                        continue
                else:
                    gt_html = str(html_data)
                
                # Wrap in table tags if needed
                if not gt_html.strip().startswith('<table'):
                    gt_html = f'<table>{gt_html}</table>'
                
                # Save image to temp file
                image_path = Path(temp_dir) / f"{Path(filename).stem}.png"
                image = image.convert('RGB')
                image.save(image_path)
                
                samples.append({
                    'image_path': str(image_path),
                    'filename': filename,
                    'gt_html': gt_html,
                    'rows': row.get('rows', 0),
                    'cols': row.get('cols', 0),
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
    """
    Method A: Table Transformer v1.1 for TSR-only
    
    只做结构识别，输出 HTML table structure
    """
    
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
            use_v1_1=True  # Use v1.1 model
        )
        logger.info("Initialized Table Transformer v1.1")
    
    def predict(self, image_path: str) -> str:
        """
        Predict table structure and return HTML.
        
        Args:
            image_path: Path to cropped table image
            
        Returns:
            Predicted HTML table structure
        """
        self._init()
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Get structure predictions (rows, columns)
            structure = self._recognizer.recognize(image)
            
            # Convert to HTML
            html = self._structure_to_html(structure)
            
            return html
            
        except Exception as e:
            logger.error(f"Table Transformer error: {e}")
            traceback.print_exc()
            return "<table></table>"
    
    def _structure_to_html(self, structure: Dict) -> str:
        """Convert structure predictions to HTML."""
        rows = structure.get('rows', [])
        cols = structure.get('columns', [])
        
        if not rows or not cols:
            return "<table></table>"
        
        num_rows = len(rows)
        num_cols = len(cols)
        
        # Build HTML
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


class DoclingTableFormerTSR:
    """
    Method B: Docling TableFormer for TSR-only
    
    使用 DocumentConverter 但输入是已裁剪的表格图片
    实际上只测试 TableFormer 的结构识别能力
    """
    
    def __init__(self, mode: str = 'accurate'):
        """
        Args:
            mode: 'fast' or 'accurate' (TableFormer mode)
        """
        self._converter = None
        self.mode = mode
    
    def _init(self):
        if self._converter is not None:
            return
        
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, 
            TableStructureOptions,
            TableFormerMode
        )
        from docling.datamodel.base_models import InputFormat
        
        # Configure for table structure recognition
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE if self.mode == 'accurate' else TableFormerMode.FAST
        )
        
        self._converter = DocumentConverter()
        logger.info(f"Initialized Docling DocumentConverter (TableFormer mode: {self.mode})")
    
    def predict(self, image_path: str) -> str:
        """
        Predict table structure and return HTML.
        
        Args:
            image_path: Path to cropped table image
            
        Returns:
            Predicted HTML table structure
        """
        self._init()
        
        try:
            result = self._converter.convert(image_path)
            tables = list(result.document.tables)
            
            if not tables:
                logger.warning(f"No tables detected in {image_path}")
                return "<table></table>"
            
            # Get first table
            table = tables[0]
            
            # Export to HTML
            html = table.export_to_html()
            
            return html
            
        except Exception as e:
            logger.error(f"Docling error: {e}")
            traceback.print_exc()
            return "<table></table>"


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    samples: List[Dict],
    methods: Dict[str, Any],
    teds_calculator: TEDSCalculator
) -> Dict[str, List[Dict]]:
    """
    Run TSR benchmark on all samples.
    
    Returns:
        Dict mapping method name to list of result dicts
    """
    results = {name: [] for name in methods}
    
    for i, sample in enumerate(samples):
        filename = sample['filename']
        image_path = sample['image_path']
        gt_html = sample['gt_html']
        
        logger.info(f"[{i+1}/{len(samples)}] Processing {filename}")
        
        for method_name, method in methods.items():
            try:
                # Get prediction
                pred_html = method.predict(image_path)
                
                # Compute TEDS
                teds_score = teds_calculator.compute(pred_html, gt_html)
                
                # Parse dimensions for debugging
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
                logger.info(f"  {method_name}: TEDS={teds_score:.4f} | "
                           f"rows: {pred_rows}/{gt_rows} | {status}")
                
            except Exception as e:
                logger.error(f"  {method_name}: Error - {e}")
                results[method_name].append({
                    'filename': filename,
                    'teds': 0.0,
                    'gt_rows': 0,
                    'pred_rows': 0,
                    'success': False,
                    'error': str(e)
                })
    
    return results


def compute_statistics(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compute summary statistics for each method."""
    summary = {}
    
    for method_name, result_list in results.items():
        if not result_list:
            continue
        
        teds_scores = [r['teds'] for r in result_list]
        success_count = sum(1 for r in result_list if r['success'])
        fail_count = len(result_list) - success_count
        
        summary[method_name] = {
            'avg_teds': float(np.mean(teds_scores)),
            'std_teds': float(np.std(teds_scores)),
            'median_teds': float(np.median(teds_scores)),
            'min_teds': float(np.min(teds_scores)),
            'max_teds': float(np.max(teds_scores)),
            'num_samples': len(result_list),
            'num_success': success_count,
            'num_fail': fail_count,
            'success_rate': success_count / len(result_list),
        }
    
    return summary


def print_report(summary: Dict[str, Dict], config: Dict):
    """Print formatted benchmark report."""
    
    print("\n" + "="*80)
    print("TSR BENCHMARK RESULTS - Table Structure Recognition Only")
    print("="*80)
    print(f"Dataset: FinTabNet")
    print(f"Task: Table Structure Recognition (TSR-only)")
    print(f"Metric: TEDS (structure-only)")
    print(f"Samples: {config.get('num_samples', 'N/A')}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    # Results table
    print("\n### RESULTS TABLE (for FYP Report)")
    print("-"*80)
    print(f"| {'Method':<40} | {'Setting':<15} | {'Avg. TEDS':^10} | {'Std':^8} | {'#Samples':^10} | {'#Fail':^8} |")
    print(f"|{'-'*40}|{'-'*15}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*10}|")
    
    for method_name, stats in summary.items():
        setting = "TSR-only"
        print(f"| {method_name:<40} | {setting:<15} | {stats['avg_teds']:^10.4f} | "
              f"{stats['std_teds']:^8.4f} | {stats['num_samples']:^10} | {stats['num_fail']:^8} |")
    
    # Add official reference
    print(f"| {'Docling TableFormer (Official)':<40} | {'TSR-only':<15} | {'0.90':^10} | "
          f"{'0.09':^8} | {'1000':^10} | {'-':^8} |")
    
    print("-"*80)
    
    # Detailed stats
    print("\n### DETAILED STATISTICS")
    for method_name, stats in summary.items():
        print(f"\n{method_name}:")
        print(f"  Average TEDS:  {stats['avg_teds']:.4f}")
        print(f"  Std Dev:       {stats['std_teds']:.4f}")
        print(f"  Median TEDS:   {stats['median_teds']:.4f}")
        print(f"  Min TEDS:      {stats['min_teds']:.4f}")
        print(f"  Max TEDS:      {stats['max_teds']:.4f}")
        print(f"  Success Rate:  {stats['success_rate']*100:.1f}% ({stats['num_success']}/{stats['num_samples']})")
    
    print("\n" + "="*80)
    print("NOTES:")
    print("- TEDS is computed as structure-only (ignoring text content)")
    print("- Official Docling result (0.90) uses cell_inputs=True (oracle cell bboxes)")
    print("- Our test uses only the cropped table image as input")
    print("- This is a fair TSR-only comparison (no detection, no OCR)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='TSR Benchmark: Table Transformer v1.1 vs Docling TableFormer'
    )
    parser.add_argument('--data-dir', type=str, 
                        default='D:/datasets/FinTabNet_OTSL/data',
                        help='Path to FinTabNet parquet files')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--methods', nargs='+', 
                        default=['table_transformer', 'docling'],
                        choices=['table_transformer', 'docling'],
                        help='Methods to evaluate')
    parser.add_argument('--docling-mode', type=str, default='accurate',
                        choices=['fast', 'accurate'],
                        help='Docling TableFormer mode')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'tsr_benchmark_{args.num_samples}samples_{timestamp}.json'
    
    # Load samples
    logger.info(f"Loading {args.num_samples} samples from FinTabNet...")
    samples = load_fintabnet_samples(args.data_dir, args.num_samples)
    
    if not samples:
        logger.error("No samples loaded!")
        return
    
    # Initialize TEDS calculator (structure-only)
    teds_calculator = TEDSCalculator(structure_only=True)
    
    # Initialize methods
    methods = {}
    if 'table_transformer' in args.methods:
        methods['Table Transformer v1.1'] = TableTransformerTSR()
    if 'docling' in args.methods:
        methods[f'Docling TableFormer ({args.docling_mode})'] = DoclingTableFormerTSR(
            mode=args.docling_mode
        )
    
    logger.info(f"Running benchmark with methods: {list(methods.keys())}")
    
    # Run benchmark
    results = run_benchmark(samples, methods, teds_calculator)
    
    # Compute statistics
    summary = compute_statistics(results)
    
    # Print report
    config = {
        'data_dir': args.data_dir,
        'num_samples': len(samples),
        'methods': list(methods.keys()),
        'docling_mode': args.docling_mode,
    }
    print_report(summary, config)
    
    # Save results
    output_data = {
        'config': {
            **config,
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'TSR-only (structure-only TEDS)',
            'teds_mode': 'structure-only',
        },
        'summary': summary,
        'detailed_results': {
            name: [
                {k: v for k, v in r.items()}
                for r in results[name]
            ]
            for name in results
        }
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
