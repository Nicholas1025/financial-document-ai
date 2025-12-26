#!/usr/bin/env python3
"""
Stage I (Step 1-3) Evaluation Framework for FinTabNet
=====================================================

Strict, reproducible, ground-truth-based evaluation comparing:
- Old Pipeline (Table Transformer v1.1 + PaddleOCR)
- Docling (TableFormer + RapidOCR)

Experiments:
(A) Detection-only: P/R/F1@IoU0.5
(B) TSR-only: TEDS_struct (image-only, NO oracle GT cell bbox)
(C) OCR-only: CER/WER
(D) End-to-End: TEDS_full + success rate

Author: FYP Research
Date: 2025-12-26
"""

import os
import sys
import json
import argparse
import random
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import csv

import numpy as np
from PIL import Image
from tqdm import tqdm
from bs4 import BeautifulSoup

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    dataset_path: str = "D:/datasets/FinTabNet_OTSL/data"
    split: str = "val"
    num_samples: int = 100
    random_seed: int = 42
    output_dir: str = "./stage1_results"
    
    # Model configs
    table_transformer_model: str = "microsoft/table-transformer-structure-recognition-v1.1-all"
    tableformer_mode: str = "accurate"
    
    # Thresholds
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5


# ==============================================================================
# Data Classes for Results
# ==============================================================================

@dataclass
class DetectionResult:
    """Single sample detection result"""
    sample_id: str
    gt_bbox: List[float]  # [x1, y1, x2, y2]
    pred_bbox: Optional[List[float]]
    confidence: float
    iou: float
    is_tp: bool
    
@dataclass
class TSRResult:
    """Single sample TSR result"""
    sample_id: str
    teds_score: float
    pred_rows: int
    pred_cols: int
    gt_rows: int
    gt_cols: int
    success: bool
    error_msg: str = ""

@dataclass
class OCRResult:
    """Single sample OCR result"""
    sample_id: str
    cer: float
    wer: float
    gt_char_count: int
    pred_char_count: int
    gt_word_count: int
    pred_word_count: int
    coverage: float  # detected tokens / gt tokens
    success: bool
    error_msg: str = ""

@dataclass 
class EndToEndResult:
    """Single sample end-to-end result"""
    sample_id: str
    teds_full: float
    teds_struct: float
    success: bool
    error_msg: str = ""


# ==============================================================================
# Utility Functions
# ==============================================================================

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_cer(pred: str, gt: str) -> float:
    """Compute Character Error Rate using edit distance"""
    import editdistance
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)

def compute_wer(pred: str, gt: str) -> float:
    """Compute Word Error Rate"""
    import editdistance
    pred_words = pred.split()
    gt_words = gt.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, gt_words) / len(gt_words)


# ==============================================================================
# TEDS Calculator (Structure-only and Full)
# ==============================================================================

class TEDSCalculator:
    """TEDS (Tree Edit Distance-based Similarity) Calculator"""
    
    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only
        try:
            from apted import APTED, Config
            from apted.helpers import Tree
            self.APTED = APTED
            self.Config = Config
            self.Tree = Tree
        except ImportError:
            raise ImportError("Please install apted: pip install apted")
    
    def _html_to_tree(self, html: str):
        """Convert HTML table to tree structure."""
        from bs4 import BeautifulSoup
        
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
                    tag = f"{tag}:{text[:20]}"  # Truncate long text
            
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
        """Count nodes in tree"""
        if tree is None:
            return 0
        count = 1
        for child in tree.children:
            count += self._tree_size(child)
        return count
    
    def compute_html(self, pred_html: str, gt_html: str) -> float:
        """Compute TEDS score between two HTML tables"""
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
    
    def compute(self, pred_table: Dict, gt_table: Dict) -> float:
        """Compute TEDS score between predicted and ground truth tables"""
        # Convert to HTML and compute
        pred_html = self._table_to_html(pred_table)
        gt_html = self._table_to_html(gt_table)
        return self.compute_html(pred_html, gt_html)
    
    def _table_to_html(self, table_data: Dict) -> str:
        """Convert table data to HTML format"""
        # If it's already HTML, return it
        if isinstance(table_data, str):
            return table_data
        
        if 'html' in table_data and table_data['html']:
            return table_data['html']
        
        rows = table_data.get('num_rows', 0)
        cols = table_data.get('num_cols', 0)
        cells = table_data.get('cells', [])
        
        if rows == 0 or cols == 0:
            return "<table></table>"
        
        # Build grid
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        for cell in cells:
            row_start = cell.get('row_start', cell.get('start_row', 0))
            col_start = cell.get('col_start', cell.get('start_col', 0))
            if 0 <= row_start < rows and 0 <= col_start < cols:
                grid[row_start][col_start] = cell
        
        # Build HTML
        html_parts = ['<table>']
        covered = [[False for _ in range(cols)] for _ in range(rows)]
        
        for r in range(rows):
            html_parts.append('<tr>')
            for c in range(cols):
                if covered[r][c]:
                    continue
                
                cell = grid[r][c]
                tag = 'td'
                
                if cell:
                    rowspan = cell.get('row_span', cell.get('rowspan', 1))
                    colspan = cell.get('col_span', cell.get('colspan', 1))
                    
                    # Mark covered cells
                    for dr in range(rowspan):
                        for dc in range(colspan):
                            if r + dr < rows and c + dc < cols:
                                if dr > 0 or dc > 0:
                                    covered[r + dr][c + dc] = True
                    
                    attrs = []
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    
                    if self.structure_only:
                        html_parts.append(f'<{tag}{attr_str}></{tag}>')
                    else:
                        text = cell.get('text', '')
                        html_parts.append(f'<{tag}{attr_str}>{text}</{tag}>')
                else:
                    html_parts.append(f'<{tag}></{tag}>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        return ''.join(html_parts)


# ==============================================================================
# Dataset Loader
# ==============================================================================

class FinTabNetLoader:
    """Load FinTabNet dataset with reproducible sampling"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.split = config.split
        self.temp_img_dir = Path(config.output_dir) / "temp_images"
        self.temp_img_dir.mkdir(parents=True, exist_ok=True)
        
    def load_samples(self) -> List[Dict]:
        """Load and sample dataset from parquet files"""
        import pyarrow.parquet as pq
        import io
        
        samples = []
        
        # Find parquet files for the split
        parquet_pattern = f"{self.split}-*.parquet"
        parquet_files = sorted(self.dataset_path.glob(parquet_pattern))
        
        if not parquet_files:
            # Try alternative locations
            parquet_files = sorted(self.dataset_path.glob(f"*{self.split}*.parquet"))
        
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet files for {self.split} split")
            
            for pq_file in parquet_files:
                try:
                    table = pq.read_table(pq_file)
                    df = table.to_pandas()
                    
                    for idx, row in df.iterrows():
                        # Use filename or imgid as sample_id
                        sample_id = row.get('filename', row.get('imgid', f"{pq_file.stem}_{idx}"))
                        if hasattr(sample_id, '__iter__') and not isinstance(sample_id, str):
                            sample_id = str(sample_id)
                        
                        # Get image data
                        image_data = row.get('image', None)
                        image_path = None
                        
                        if image_data is not None:
                            # Save image to temp file
                            img_bytes = None
                            if isinstance(image_data, dict) and 'bytes' in image_data:
                                img_bytes = image_data['bytes']
                            elif hasattr(image_data, 'tobytes'):
                                img_bytes = image_data.tobytes()
                            elif isinstance(image_data, bytes):
                                img_bytes = image_data
                            
                            if img_bytes:
                                safe_id = str(sample_id).replace('/', '_').replace('\\', '_')
                                image_path = self.temp_img_dir / f"{safe_id}.png"
                                if not image_path.exists():
                                    try:
                                        img = Image.open(io.BytesIO(img_bytes))
                                        img.save(image_path)
                                    except Exception as e:
                                        logger.warning(f"Failed to save image {sample_id}: {e}")
                                        continue
                        
                        # Get HTML structure (ground truth)
                        html = row.get('html', '')
                        html_restored = row.get('html_restored', '')
                        
                        # Convert numpy arrays to strings if needed
                        if hasattr(html, 'item'):
                            html = html.item() if html.size == 1 else str(html)
                        if hasattr(html_restored, 'item'):
                            html_restored = html_restored.item() if html_restored.size == 1 else str(html_restored)
                        
                        # Get cells data
                        cells_data = row.get('cells', [])
                        if hasattr(cells_data, 'tolist'):
                            cells_data = cells_data.tolist()
                        
                        # Parse table structure
                        num_rows = row.get('rows', 0)
                        num_cols = row.get('cols', 0)
                        
                        # Handle numpy scalar values
                        if hasattr(num_rows, 'item'):
                            num_rows = num_rows.item()
                        if hasattr(num_cols, 'item'):
                            num_cols = num_cols.item()
                        
                        structure = {
                            'html': html_restored if html_restored else html,
                            'num_rows': int(num_rows) if num_rows else 0,
                            'num_cols': int(num_cols) if num_cols else 0,
                            'cells': cells_data if isinstance(cells_data, list) else []
                        }
                        
                        samples.append({
                            'id': str(sample_id),
                            'image_path': str(image_path) if image_path else None,
                            'structure': structure,
                            'html': html_restored if html_restored else html
                        })
                        
                except Exception as e:
                    logger.warning(f"Error reading {pq_file}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
        else:
            # Fallback: Try JSONL or image directory formats
            jsonl_file = self.dataset_path / f"FinTabNet_{self.split}.jsonl"
            if jsonl_file.exists():
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            else:
                # Scan for PNG files
                for img_path in sorted(self.dataset_path.rglob("*.png"))[:1000]:
                    samples.append({
                        'id': img_path.stem,
                        'image_path': str(img_path)
                    })
        
        logger.info(f"Found {len(samples)} total samples in {self.split}")
        
        # Reproducible sampling
        set_all_seeds(self.config.random_seed)
        if len(samples) > self.config.num_samples:
            samples = random.sample(samples, self.config.num_samples)
        
        logger.info(f"Selected {len(samples)} samples with seed={self.config.random_seed}")
        
        return samples
    
    def save_sample_ids(self, samples: List[Dict], output_path: Path):
        """Save sample IDs for reproducibility"""
        sample_ids = [s.get('id', s.get('filename', str(i))) for i, s in enumerate(samples)]
        with open(output_path, 'w') as f:
            json.dump({
                'seed': self.config.random_seed,
                'num_samples': len(samples),
                'split': self.split,
                'sample_ids': sample_ids
            }, f, indent=2)
        logger.info(f"Saved {len(sample_ids)} sample IDs to {output_path}")


# ==============================================================================
# Model Wrappers
# ==============================================================================

class TableTransformerWrapper:
    """Wrapper for Table Transformer v1.1 using project's existing recognizer"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name  # Not used, we use project's recognizer
        self._recognizer = None
        self._config = None
        
    def load(self):
        """Load model using project's TableStructureRecognizer"""
        import yaml
        from modules.structure import TableStructureRecognizer
        
        # Load project config
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        
        self._recognizer = TableStructureRecognizer(
            config=self._config,
            use_v1_1=True  # Use v1.1 model
        )
        logger.info("Table Transformer v1.1 loaded successfully")
    
    def predict_structure(self, image: Image.Image) -> Dict:
        """Predict table structure"""
        if self._recognizer is None:
            self.load()
        
        try:
            structure = self._recognizer.recognize(image)
            return self._convert_structure(structure)
        except Exception as e:
            logger.warning(f"Table Transformer error: {e}")
            return {'num_rows': 0, 'num_cols': 0, 'cells': []}
    
    def predict_structure_html(self, image: Image.Image) -> str:
        """Predict and return HTML directly"""
        if self._recognizer is None:
            self.load()
        
        try:
            structure = self._recognizer.recognize(image)
            return self._structure_to_html(structure)
        except Exception as e:
            logger.warning(f"Table Transformer error: {e}")
            return "<table></table>"
    
    def _convert_structure(self, structure: Dict) -> Dict:
        """Convert to standard format"""
        rows = structure.get('rows', [])
        cols = structure.get('columns', [])
        
        num_rows = len(rows)
        num_cols = len(cols)
        
        cells = []
        for r in range(num_rows):
            for c in range(num_cols):
                cells.append({
                    'row_start': r,
                    'col_start': c,
                    'row_span': 1,
                    'col_span': 1,
                    'text': ''
                })
        
        return {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'cells': cells
        }
    
    def _structure_to_html(self, structure: Dict) -> str:
        """Convert structure to HTML"""
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


class DoclingTableFormerWrapper:
    """Wrapper for Docling TableFormer (direct access, image-only mode)"""
    
    def __init__(self, mode: str = "accurate"):
        self.mode = mode
        self._predictor = None
        self._config = None
        
    def load(self):
        """Load TableFormer predictor (direct access)"""
        logger.info(f"Loading Docling TableFormer ({self.mode} mode)...")
        
        try:
            from docling.models.table_structure_model import TableStructureModel
            from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
            import docling_ibm_models.tableformer.common as c
            
            # Download models if needed
            artifacts_path = TableStructureModel.download_models()
            model_path = artifacts_path / "model_artifacts" / "tableformer"
            
            if self.mode == 'accurate':
                model_path = model_path / "accurate"
            else:
                model_path = model_path / "fast"
            
            # Load config
            self._config = c.read_config(str(model_path / "tm_config.json"))
            self._config["model"]["save_dir"] = str(model_path)
            
            # Initialize predictor (note: num_threads not num_workers)
            self._predictor = TFPredictor(self._config, device='cpu', num_threads=4)
            logger.info("Docling TableFormer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TableFormer: {e}")
            raise
    
    def predict_structure(self, image: Image.Image) -> Dict:
        """Predict table structure from image only (NO GT cell bbox)"""
        if self._predictor is None:
            self.load()
        
        try:
            width, height = image.size
            
            # Scale factor (TableFormer uses 144 dpi, assume 72 dpi input)
            scale = 2.0
            scaled_w = int(width * scale)
            scaled_h = int(height * scale)
            
            # Prepare input (iocr_page format)
            page_input = {
                "width": scaled_w,
                "height": scaled_h,
                "image": np.asarray(image.resize((scaled_w, scaled_h))),
                "tokens": []  # Empty - no GT cell bbox!
            }
            
            # Table bbox is the entire image
            table_bbox = [0, 0, scaled_w, scaled_h]
            
            # Run prediction
            tf_output = self._predictor.multi_table_predict(
                page_input,
                [table_bbox],
                do_matching=False  # No cell matching without tokens
            )
            
            if not tf_output:
                return {'num_rows': 0, 'num_cols': 0, 'cells': []}
            
            table_out = tf_output[0]
            
            # Extract structure
            predict_details = table_out.get("predict_details", {})
            num_rows = predict_details.get("num_rows", 0)
            num_cols = predict_details.get("num_cols", 0)
            
            cells = []
            for cell in table_out.get("tf_responses", []):
                cells.append({
                    'row_start': cell.get('start_row_offset_idx', 0),
                    'col_start': cell.get('start_col_offset_idx', 0),
                    'row_span': cell.get('end_row_offset_idx', 1) - cell.get('start_row_offset_idx', 0),
                    'col_span': cell.get('end_col_offset_idx', 1) - cell.get('start_col_offset_idx', 0),
                    'is_header': cell.get('column_header', False),
                    'bbox': cell.get('bbox', [0, 0, 0, 0]),
                    'text': ''
                })
            
            return {
                'num_rows': num_rows,
                'num_cols': num_cols,
                'cells': cells
            }
            
        except Exception as e:
            logger.warning(f"TableFormer prediction failed: {e}")
            return {'num_rows': 0, 'num_cols': 0, 'cells': []}
    
    def predict_structure_html(self, image: Image.Image) -> str:
        """Predict and return HTML directly"""
        struct = self.predict_structure(image)
        return self._build_html_from_structure(struct)
    
    def _build_html_from_structure(self, struct: Dict) -> str:
        """Build HTML from structure"""
        num_rows = struct.get('num_rows', 0)
        num_cols = struct.get('num_cols', 0)
        cells = struct.get('cells', [])
        
        if num_rows == 0 or num_cols == 0:
            return "<table></table>"
        
        # Build grid
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        for cell in cells:
            row_start = cell.get('row_start', 0)
            col_start = cell.get('col_start', 0)
            
            if row_start < num_rows and col_start < num_cols:
                grid[row_start][col_start] = cell
        
        # Build HTML
        html_parts = ['<table>']
        covered = [[False for _ in range(num_cols)] for _ in range(num_rows)]
        
        for row_idx in range(num_rows):
            is_header_row = row_idx == 0
            
            if row_idx == 0:
                html_parts.append('<thead>')
            elif row_idx == 1:
                html_parts.append('</thead><tbody>')
            
            html_parts.append('<tr>')
            
            for col_idx in range(num_cols):
                if covered[row_idx][col_idx]:
                    continue
                
                cell = grid[row_idx][col_idx]
                tag = 'th' if is_header_row else 'td'
                
                if cell:
                    rowspan = cell.get('row_span', 1)
                    colspan = cell.get('col_span', 1)
                    
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
        
        if num_rows > 0:
            html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        return ''.join(html_parts)


class PaddleOCRWrapper:
    """Wrapper for PaddleOCR"""
    
    def __init__(self):
        self.ocr = None
        
    def load(self):
        """Load PaddleOCR"""
        logger.info("Loading PaddleOCR...")
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        logger.info("PaddleOCR loaded successfully")
    
    def extract_text(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """Extract text from image"""
        if self.ocr is None:
            self.load()
        
        import numpy as np
        img_array = np.array(image)
        
        result = self.ocr.ocr(img_array, cls=True)
        
        text_parts = []
        tokens = []
        
        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                text_parts.append(text)
                tokens.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf
                })
        
        return ' '.join(text_parts), tokens


class RapidOCRWrapper:
    """Wrapper for RapidOCR (Docling's OCR backend)"""
    
    def __init__(self):
        self.ocr = None
        
    def load(self):
        """Load RapidOCR"""
        logger.info("Loading RapidOCR...")
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
            logger.info("RapidOCR loaded successfully")
        except ImportError:
            from rapidocr import RapidOCR
            self.ocr = RapidOCR()
            logger.info("RapidOCR (alternative) loaded successfully")
    
    def extract_text(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """Extract text from image"""
        if self.ocr is None:
            self.load()
        
        import numpy as np
        img_array = np.array(image)
        
        result, _ = self.ocr(img_array)
        
        text_parts = []
        tokens = []
        
        if result:
            for item in result:
                bbox, text, conf = item
                text_parts.append(text)
                tokens.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf
                })
        
        return ' '.join(text_parts), tokens


# ==============================================================================
# Evaluation Runners
# ==============================================================================

class Stage1Evaluator:
    """Main evaluator for Stage I experiments"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize calculators
        self.teds_struct = TEDSCalculator(structure_only=True)
        self.teds_full = TEDSCalculator(structure_only=False)
        
        # Model wrappers (lazy loading)
        self.tt_wrapper = None
        self.tf_wrapper = None
        self.paddle_ocr = None
        self.rapid_ocr = None
        
        # Results storage
        self.detection_results = {'old_pipeline': [], 'docling': []}
        self.tsr_results = {'old_pipeline': [], 'docling': []}
        self.ocr_results = {'old_pipeline': [], 'docling': []}
        self.e2e_results = {'old_pipeline': [], 'docling': []}
    
    def _get_tt_wrapper(self) -> TableTransformerWrapper:
        if self.tt_wrapper is None:
            self.tt_wrapper = TableTransformerWrapper(self.config.table_transformer_model)
        return self.tt_wrapper
    
    def _get_tf_wrapper(self) -> DoclingTableFormerWrapper:
        if self.tf_wrapper is None:
            self.tf_wrapper = DoclingTableFormerWrapper(self.config.tableformer_mode)
        return self.tf_wrapper
    
    def _get_paddle_ocr(self) -> PaddleOCRWrapper:
        if self.paddle_ocr is None:
            self.paddle_ocr = PaddleOCRWrapper()
        return self.paddle_ocr
    
    def _get_rapid_ocr(self) -> RapidOCRWrapper:
        if self.rapid_ocr is None:
            self.rapid_ocr = RapidOCRWrapper()
        return self.rapid_ocr
    
    # --------------------------------------------------------------------------
    # (A) Detection-only Evaluation
    # --------------------------------------------------------------------------
    
    def evaluate_detection(self, samples: List[Dict]) -> Dict:
        """
        Evaluate table detection (Step 1)
        Note: FinTabNet provides cropped table images, so detection is trivial.
        This evaluation is meaningful only for full-page images.
        """
        logger.info("=" * 60)
        logger.info("(A) Detection-only Evaluation")
        logger.info("=" * 60)
        logger.info("Note: FinTabNet provides cropped table images.")
        logger.info("Detection evaluation requires full-page images with GT bbox.")
        logger.info("Skipping detection evaluation for cropped images.")
        
        return {
            'old_pipeline': {'precision': 'N/A', 'recall': 'N/A', 'f1': 'N/A', 'note': 'Cropped images - detection N/A'},
            'docling': {'precision': 'N/A', 'recall': 'N/A', 'f1': 'N/A', 'note': 'Cropped images - detection N/A'}
        }
    
    # --------------------------------------------------------------------------
    # (B) TSR-only Evaluation
    # --------------------------------------------------------------------------
    
    def evaluate_tsr(self, samples: List[Dict]) -> Dict:
        """
        Evaluate Table Structure Recognition (Step 2)
        Image-only mode - NO GT cell bbox used!
        Uses HTML-based TEDS comparison for accuracy.
        """
        logger.info("=" * 60)
        logger.info("(B) TSR-only Evaluation (Image-only, NO oracle GT cell bbox)")
        logger.info("=" * 60)
        
        tt_wrapper = self._get_tt_wrapper()
        tf_wrapper = self._get_tf_wrapper()
        
        tt_scores = []
        tf_scores = []
        
        for sample in tqdm(samples, desc="TSR Evaluation"):
            sample_id = sample.get('id', sample.get('filename', 'unknown'))
            image_path = sample.get('image_path')
            
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image not found for sample {sample_id}")
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Get GT HTML
                gt_html = sample.get('html', '')
                if not gt_html:
                    gt_structure = sample.get('structure', {})
                    gt_html = gt_structure.get('html', '')
                
                if not gt_html:
                    logger.warning(f"No GT HTML for sample {sample_id}")
                    continue
                
                # Ensure proper HTML format
                if isinstance(gt_html, (list, np.ndarray)):
                    if len(gt_html) > 0:
                        first = str(gt_html[0])
                        if first.startswith('<') and len(first) < 50:
                            gt_html = ''.join(str(x) for x in gt_html)
                        else:
                            gt_html = str(gt_html[0])
                    else:
                        continue
                
                gt_html = str(gt_html)
                if not gt_html.strip().startswith('<table'):
                    gt_html = f'<table>{gt_html}</table>'
                
                # Count GT rows/cols
                gt_soup = BeautifulSoup(gt_html, 'html.parser')
                gt_rows = len(gt_soup.find_all('tr'))
                gt_cols = max((len(tr.find_all(['td', 'th'])) for tr in gt_soup.find_all('tr')), default=0)
                
                # Table Transformer prediction
                try:
                    tt_pred_html = tt_wrapper.predict_structure_html(image)
                    tt_score = self.teds_struct.compute_html(tt_pred_html, gt_html)
                    tt_scores.append(tt_score)
                    
                    # Count pred rows
                    pred_soup = BeautifulSoup(tt_pred_html, 'html.parser')
                    pred_rows = len(pred_soup.find_all('tr'))
                    pred_cols = max((len(tr.find_all(['td', 'th'])) for tr in pred_soup.find_all('tr')), default=0)
                    
                    self.tsr_results['old_pipeline'].append(TSRResult(
                        sample_id=sample_id,
                        teds_score=tt_score,
                        pred_rows=pred_rows,
                        pred_cols=pred_cols,
                        gt_rows=gt_rows,
                        gt_cols=gt_cols,
                        success=True
                    ))
                except Exception as e:
                    logger.warning(f"TT failed for {sample_id}: {e}")
                    self.tsr_results['old_pipeline'].append(TSRResult(
                        sample_id=sample_id, teds_score=0.0,
                        pred_rows=0, pred_cols=0,
                        gt_rows=gt_rows, gt_cols=gt_cols,
                        success=False, error_msg=str(e)
                    ))
                
                # TableFormer prediction (image-only!)
                try:
                    tf_pred_html = tf_wrapper.predict_structure_html(image)
                    tf_score = self.teds_struct.compute_html(tf_pred_html, gt_html)
                    tf_scores.append(tf_score)
                    
                    # Count pred rows
                    pred_soup = BeautifulSoup(tf_pred_html, 'html.parser')
                    pred_rows = len(pred_soup.find_all('tr'))
                    pred_cols = max((len(tr.find_all(['td', 'th'])) for tr in pred_soup.find_all('tr')), default=0)
                    
                    self.tsr_results['docling'].append(TSRResult(
                        sample_id=sample_id,
                        teds_score=tf_score,
                        pred_rows=pred_rows,
                        pred_cols=pred_cols,
                        gt_rows=gt_rows,
                        gt_cols=gt_cols,
                        success=True
                    ))
                except Exception as e:
                    logger.warning(f"TF failed for {sample_id}: {e}")
                    self.tsr_results['docling'].append(TSRResult(
                        sample_id=sample_id, teds_score=0.0,
                        pred_rows=0, pred_cols=0,
                        gt_rows=gt_rows, gt_cols=gt_cols,
                        success=False, error_msg=str(e)
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing {sample_id}: {e}")
        
        # Compute statistics
        tt_stats = self._compute_stats(tt_scores)
        tf_stats = self._compute_stats(tf_scores)
        
        return {
            'old_pipeline': tt_stats,
            'docling': tf_stats
        }
    
    def _parse_gt_structure(self, gt_structure: Dict) -> Dict:
        """Parse ground truth structure to standard format"""
        if not gt_structure:
            return {'num_rows': 0, 'num_cols': 0, 'cells': []}
        
        # Handle different GT formats
        if 'html' in gt_structure:
            return self._parse_html_structure(gt_structure['html'])
        elif 'cells' in gt_structure:
            return gt_structure
        elif 'num_rows' in gt_structure:
            return gt_structure
        else:
            # Try to extract from other formats
            return {'num_rows': 0, 'num_cols': 0, 'cells': []}
    
    def _parse_html_structure(self, html: str) -> Dict:
        """Parse HTML table structure"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return {'num_rows': 0, 'num_cols': 0, 'cells': []}
        
        rows = table.find_all('tr')
        cells = []
        num_rows = len(rows)
        num_cols = 0
        
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                text = cell.get_text(strip=True)
                
                cells.append({
                    'row_start': row_idx,
                    'col_start': col_idx,
                    'row_span': rowspan,
                    'col_span': colspan,
                    'text': text
                })
                
                col_idx += colspan
            
            num_cols = max(num_cols, col_idx)
        
        return {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'cells': cells
        }
    
    # --------------------------------------------------------------------------
    # (C) OCR-only Evaluation
    # --------------------------------------------------------------------------
    
    def evaluate_ocr(self, samples: List[Dict]) -> Dict:
        """Evaluate OCR (Step 3)"""
        logger.info("=" * 60)
        logger.info("(C) OCR-only Evaluation")
        logger.info("=" * 60)
        
        paddle_ocr = self._get_paddle_ocr()
        rapid_ocr = self._get_rapid_ocr()
        
        paddle_cer_list = []
        paddle_wer_list = []
        rapid_cer_list = []
        rapid_wer_list = []
        
        for sample in tqdm(samples, desc="OCR Evaluation"):
            sample_id = sample.get('id', sample.get('filename', 'unknown'))
            image_path = sample.get('image_path')
            
            if not image_path or not Path(image_path).exists():
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Get GT text
                gt_structure = sample.get('structure', {})
                gt_text = self._extract_gt_text(gt_structure)
                
                if not gt_text.strip():
                    continue
                
                # PaddleOCR
                try:
                    paddle_text, paddle_tokens = paddle_ocr.extract_text(image)
                    paddle_cer = compute_cer(paddle_text, gt_text)
                    paddle_wer = compute_wer(paddle_text, gt_text)
                    paddle_cer_list.append(paddle_cer)
                    paddle_wer_list.append(paddle_wer)
                    
                    self.ocr_results['old_pipeline'].append(OCRResult(
                        sample_id=sample_id,
                        cer=paddle_cer,
                        wer=paddle_wer,
                        gt_char_count=len(gt_text),
                        pred_char_count=len(paddle_text),
                        gt_word_count=len(gt_text.split()),
                        pred_word_count=len(paddle_text.split()),
                        coverage=len(paddle_tokens) / max(1, len(gt_text.split())),
                        success=True
                    ))
                except Exception as e:
                    logger.warning(f"PaddleOCR failed for {sample_id}: {e}")
                
                # RapidOCR
                try:
                    rapid_text, rapid_tokens = rapid_ocr.extract_text(image)
                    rapid_cer = compute_cer(rapid_text, gt_text)
                    rapid_wer = compute_wer(rapid_text, gt_text)
                    rapid_cer_list.append(rapid_cer)
                    rapid_wer_list.append(rapid_wer)
                    
                    self.ocr_results['docling'].append(OCRResult(
                        sample_id=sample_id,
                        cer=rapid_cer,
                        wer=rapid_wer,
                        gt_char_count=len(gt_text),
                        pred_char_count=len(rapid_text),
                        gt_word_count=len(gt_text.split()),
                        pred_word_count=len(rapid_text.split()),
                        coverage=len(rapid_tokens) / max(1, len(gt_text.split())),
                        success=True
                    ))
                except Exception as e:
                    logger.warning(f"RapidOCR failed for {sample_id}: {e}")
                    
            except Exception as e:
                logger.error(f"Error processing {sample_id}: {e}")
        
        return {
            'old_pipeline': {
                'cer': self._compute_stats(paddle_cer_list),
                'wer': self._compute_stats(paddle_wer_list)
            },
            'docling': {
                'cer': self._compute_stats(rapid_cer_list),
                'wer': self._compute_stats(rapid_wer_list)
            }
        }
    
    def _extract_gt_text(self, gt_structure: Dict) -> str:
        """Extract all text from GT structure"""
        texts = []
        
        if 'cells' in gt_structure:
            for cell in gt_structure['cells']:
                text = cell.get('text', '')
                if text:
                    texts.append(text)
        elif 'html' in gt_structure:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(gt_structure['html'], 'html.parser')
            texts.append(soup.get_text(separator=' ', strip=True))
        
        return ' '.join(texts)
    
    # --------------------------------------------------------------------------
    # (D) End-to-End Evaluation
    # --------------------------------------------------------------------------
    
    def evaluate_end_to_end(self, samples: List[Dict]) -> Dict:
        """Evaluate End-to-End Stage I (Step 1+2+3)"""
        logger.info("=" * 60)
        logger.info("(D) End-to-End Stage I Evaluation")
        logger.info("=" * 60)
        
        tt_wrapper = self._get_tt_wrapper()
        tf_wrapper = self._get_tf_wrapper()
        paddle_ocr = self._get_paddle_ocr()
        rapid_ocr = self._get_rapid_ocr()
        
        tt_teds_full = []
        tf_teds_full = []
        tt_success = 0
        tf_success = 0
        
        for sample in tqdm(samples, desc="End-to-End Evaluation"):
            sample_id = sample.get('id', sample.get('filename', 'unknown'))
            image_path = sample.get('image_path')
            
            if not image_path or not Path(image_path).exists():
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Get GT
                gt_structure = sample.get('structure', {})
                gt_table = self._parse_gt_structure(gt_structure)
                
                # Old Pipeline: TT + PaddleOCR
                try:
                    tt_struct = tt_wrapper.predict_structure(image)
                    paddle_text, paddle_tokens = paddle_ocr.extract_text(image)
                    
                    # Merge OCR text into structure
                    tt_with_text = self._merge_ocr_to_structure(tt_struct, paddle_tokens)
                    
                    tt_score = self.teds_full.compute(tt_with_text, gt_table)
                    tt_teds_full.append(tt_score)
                    tt_success += 1
                    
                    self.e2e_results['old_pipeline'].append(EndToEndResult(
                        sample_id=sample_id,
                        teds_full=tt_score,
                        teds_struct=self.teds_struct.compute(tt_struct, gt_table),
                        success=True
                    ))
                except Exception as e:
                    self.e2e_results['old_pipeline'].append(EndToEndResult(
                        sample_id=sample_id,
                        teds_full=0.0,
                        teds_struct=0.0,
                        success=False,
                        error_msg=str(e)
                    ))
                
                # Docling: TableFormer + RapidOCR
                try:
                    tf_struct = tf_wrapper.predict_structure(image)
                    rapid_text, rapid_tokens = rapid_ocr.extract_text(image)
                    
                    # Merge OCR text into structure
                    tf_with_text = self._merge_ocr_to_structure(tf_struct, rapid_tokens)
                    
                    tf_score = self.teds_full.compute(tf_with_text, gt_table)
                    tf_teds_full.append(tf_score)
                    tf_success += 1
                    
                    self.e2e_results['docling'].append(EndToEndResult(
                        sample_id=sample_id,
                        teds_full=tf_score,
                        teds_struct=self.teds_struct.compute(tf_struct, gt_table),
                        success=True
                    ))
                except Exception as e:
                    self.e2e_results['docling'].append(EndToEndResult(
                        sample_id=sample_id,
                        teds_full=0.0,
                        teds_struct=0.0,
                        success=False,
                        error_msg=str(e)
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing {sample_id}: {e}")
        
        total = len(samples)
        
        return {
            'old_pipeline': {
                'teds_full': self._compute_stats(tt_teds_full),
                'success_rate': tt_success / total if total > 0 else 0.0,
                'success_count': tt_success,
                'total': total
            },
            'docling': {
                'teds_full': self._compute_stats(tf_teds_full),
                'success_rate': tf_success / total if total > 0 else 0.0,
                'success_count': tf_success,
                'total': total
            }
        }
    
    def _merge_ocr_to_structure(self, structure: Dict, ocr_tokens: List[Dict]) -> Dict:
        """Merge OCR tokens into structure cells based on bbox overlap"""
        result = structure.copy()
        result['cells'] = []
        
        for cell in structure.get('cells', []):
            cell_copy = cell.copy()
            cell_bbox = cell.get('bbox', [0, 0, 0, 0])
            
            # Find overlapping OCR tokens
            cell_texts = []
            for token in ocr_tokens:
                token_bbox = token.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
                # Convert polygon to box if needed
                if isinstance(token_bbox[0], list):
                    x_coords = [p[0] for p in token_bbox]
                    y_coords = [p[1] for p in token_bbox]
                    token_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                else:
                    token_box = token_bbox
                
                # Check overlap
                if self._boxes_overlap(cell_bbox, token_box):
                    cell_texts.append(token.get('text', ''))
            
            cell_copy['text'] = ' '.join(cell_texts)
            result['cells'].append(cell_copy)
        
        return result
    
    def _boxes_overlap(self, box1: List[float], box2: List[float]) -> bool:
        """Check if two boxes overlap"""
        if len(box1) < 4 or len(box2) < 4:
            return False
        return not (box1[2] < box2[0] or box2[2] < box1[0] or
                   box1[3] < box2[1] or box2[3] < box1[1])
    
    # --------------------------------------------------------------------------
    # Statistics and Reporting
    # --------------------------------------------------------------------------
    
    def _compute_stats(self, scores: List[float]) -> Dict:
        """Compute statistics for a list of scores"""
        if not scores:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'count': 0}
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'count': len(scores)
        }
    
    def save_results(self, results: Dict):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results JSON
        results_file = self.output_dir / f"stage1_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_file}")
        
        # Save detailed CSVs
        self._save_csv(self.tsr_results, 'tsr', timestamp)
        self._save_csv(self.ocr_results, 'ocr', timestamp)
        self._save_csv(self.e2e_results, 'e2e', timestamp)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'config': asdict(self.config),
            'model_versions': {
                'table_transformer': self.config.table_transformer_model,
                'tableformer_mode': self.config.tableformer_mode
            }
        }
        metadata_file = self.output_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_csv(self, results_dict: Dict, prefix: str, timestamp: str):
        """Save results to CSV"""
        for method, results in results_dict.items():
            if not results:
                continue
            
            csv_file = self.output_dir / f"results_{prefix}_{method}_{timestamp}.csv"
            
            if results and hasattr(results[0], '__dict__'):
                fieldnames = list(asdict(results[0]).keys())
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in results:
                        writer.writerow(asdict(r))
                logger.info(f"Saved {prefix} results for {method} to {csv_file}")
    
    def print_summary_tables(self, results: Dict):
        """Print summary tables for report"""
        print("\n" + "=" * 80)
        print("STAGE I EVALUATION RESULTS - FINTABNET (n={})".format(self.config.num_samples))
        print("=" * 80)
        
        # Table A: Component-wise Results
        print("\n" + "-" * 80)
        print("Table A: Component-wise Results (FinTabNet, n={})".format(self.config.num_samples))
        print("-" * 80)
        print(f"{'Component':<20} {'Metric':<15} {'Old Pipeline':<20} {'Docling':<20}")
        print("-" * 80)
        
        # Detection
        det_old = results.get('detection', {}).get('old_pipeline', {})
        det_doc = results.get('detection', {}).get('docling', {})
        print(f"{'Detection':<20} {'P/R/F1':<15} {det_old.get('note', 'N/A'):<20} {det_doc.get('note', 'N/A'):<20}")
        
        # TSR
        tsr_old = results.get('tsr', {}).get('old_pipeline', {})
        tsr_doc = results.get('tsr', {}).get('docling', {})
        old_teds = f"{tsr_old.get('mean', 0):.4f}±{tsr_old.get('std', 0):.4f}"
        doc_teds = f"{tsr_doc.get('mean', 0):.4f}±{tsr_doc.get('std', 0):.4f}"
        print(f"{'TSR-only':<20} {'TEDS_struct':<15} {old_teds:<20} {doc_teds:<20}")
        
        # OCR
        ocr_old = results.get('ocr', {}).get('old_pipeline', {})
        ocr_doc = results.get('ocr', {}).get('docling', {})
        old_cer = f"{ocr_old.get('cer', {}).get('mean', 0):.4f}"
        doc_cer = f"{ocr_doc.get('cer', {}).get('mean', 0):.4f}"
        old_wer = f"{ocr_old.get('wer', {}).get('mean', 0):.4f}"
        doc_wer = f"{ocr_doc.get('wer', {}).get('mean', 0):.4f}"
        print(f"{'OCR-only':<20} {'CER':<15} {old_cer:<20} {doc_cer:<20}")
        print(f"{'':<20} {'WER':<15} {old_wer:<20} {doc_wer:<20}")
        
        print("-" * 80)
        
        # Table B: End-to-End Results
        print("\n" + "-" * 80)
        print("Table B: End-to-End Stage I Results (FinTabNet, n={})".format(self.config.num_samples))
        print("-" * 80)
        print(f"{'Metric':<25} {'Old Pipeline':<25} {'Docling':<25}")
        print("-" * 80)
        
        e2e_old = results.get('end_to_end', {}).get('old_pipeline', {})
        e2e_doc = results.get('end_to_end', {}).get('docling', {})
        
        old_teds_full = e2e_old.get('teds_full', {})
        doc_teds_full = e2e_doc.get('teds_full', {})
        
        old_full = f"{old_teds_full.get('mean', 0):.4f}±{old_teds_full.get('std', 0):.4f}"
        doc_full = f"{doc_teds_full.get('mean', 0):.4f}±{doc_teds_full.get('std', 0):.4f}"
        print(f"{'TEDS_full':<25} {old_full:<25} {doc_full:<25}")
        
        old_sr = f"{e2e_old.get('success_rate', 0)*100:.1f}% ({e2e_old.get('success_count', 0)}/{e2e_old.get('total', 0)})"
        doc_sr = f"{e2e_doc.get('success_rate', 0)*100:.1f}% ({e2e_doc.get('success_count', 0)}/{e2e_doc.get('total', 0)})"
        print(f"{'Success Rate':<25} {old_sr:<25} {doc_sr:<25}")
        
        print("-" * 80)
        
        # Fair comparison notice
        print("\n" + "=" * 80)
        print("公平条件：Docling 与 TT 均为 image-only；不使用 oracle GT cell bbox。")
        print("Fair Condition: Both Docling and TT use image-only mode; NO oracle GT cell bbox.")
        print("=" * 80)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage I Evaluation for FinTabNet")
    parser.add_argument("--dataset-path", type=str, default="D:/datasets/FinTabNet_OTSL/data",
                       help="Path to FinTabNet dataset")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./stage1_results",
                       help="Output directory")
    parser.add_argument("--skip-detection", action="store_true",
                       help="Skip detection evaluation")
    parser.add_argument("--skip-tsr", action="store_true",
                       help="Skip TSR evaluation")
    parser.add_argument("--skip-ocr", action="store_true",
                       help="Skip OCR evaluation")
    parser.add_argument("--skip-e2e", action="store_true",
                       help="Skip end-to-end evaluation")
    
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig(
        dataset_path=args.dataset_path,
        split=args.split,
        num_samples=args.num_samples,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Set seeds
    set_all_seeds(config.random_seed)
    
    # Load dataset
    logger.info("Loading FinTabNet dataset...")
    loader = FinTabNetLoader(config)
    samples = loader.load_samples()
    
    # Save sample IDs for reproducibility
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader.save_sample_ids(samples, output_dir / "samples_100.json")
    
    # Create evaluator
    evaluator = Stage1Evaluator(config)
    
    # Run evaluations
    results = {}
    
    if not args.skip_detection:
        results['detection'] = evaluator.evaluate_detection(samples)
    
    if not args.skip_tsr:
        results['tsr'] = evaluator.evaluate_tsr(samples)
    
    if not args.skip_ocr:
        results['ocr'] = evaluator.evaluate_ocr(samples)
    
    if not args.skip_e2e:
        results['end_to_end'] = evaluator.evaluate_end_to_end(samples)
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    evaluator.print_summary_tables(results)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
