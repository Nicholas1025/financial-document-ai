"""
Compare Structure Recognition: Docling Baseline vs Old Pipeline

This script runs the same set of samples through both:
1. Docling's TableFormer (via docling-eval)
2. Our old pipeline (Table Transformer + PaddleOCR)

Then computes comparative metrics: TEDS, Cell-F1, Structural Accuracy.

Usage:
    python run_compare_structures.py --samples ./data/samples --output-dir ./outputs/comparison
    python run_compare_structures.py --use-fintabnet --max-samples 100
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Comparison Results
# =============================================================================

@dataclass
class SystemResult:
    """Results from a single system."""
    system_name: str
    version: str
    teds_struct: float
    teds_text: float
    cell_f1_struct: float
    cell_f1_text: float
    row_accuracy: float
    col_accuracy: float
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class SampleComparison:
    """Comparison results for a single sample."""
    sample_id: str
    ground_truth_source: str
    docling_result: Optional[SystemResult] = None
    old_pipeline_result: Optional[SystemResult] = None
    
    def docling_wins_teds_struct(self) -> Optional[bool]:
        if self.docling_result and self.old_pipeline_result:
            return self.docling_result.teds_struct > self.old_pipeline_result.teds_struct
        return None
    
    def teds_struct_diff(self) -> Optional[float]:
        if self.docling_result and self.old_pipeline_result:
            return self.docling_result.teds_struct - self.old_pipeline_result.teds_struct
        return None


@dataclass
class ComparisonReport:
    """Full comparison report."""
    timestamp: str
    num_samples: int
    samples_compared: int
    
    # Aggregated metrics for docling
    docling_mean_teds_struct: float = 0.0
    docling_mean_teds_text: float = 0.0
    docling_mean_cell_f1: float = 0.0
    
    # Aggregated metrics for old pipeline
    pipeline_mean_teds_struct: float = 0.0
    pipeline_mean_teds_text: float = 0.0
    pipeline_mean_cell_f1: float = 0.0
    
    # Comparison stats
    docling_wins_count: int = 0
    pipeline_wins_count: int = 0
    tie_count: int = 0
    
    # Version info
    docling_version: str = ""
    pipeline_version: str = ""
    
    # Individual comparisons
    comparisons: List[SampleComparison] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['comparisons'] = [asdict(c) for c in self.comparisons]
        return d
    
    def summary_table(self) -> str:
        """Generate comparison summary table."""
        lines = [
            "=" * 70,
            "STRUCTURE RECOGNITION COMPARISON REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Samples: {self.samples_compared} / {self.num_samples}",
            "",
            "-" * 70,
            f"{'Metric':<35} {'Docling':>15} {'OldPipeline':>15}",
            "-" * 70,
            f"{'TEDS (struct-only)':<35} {self.docling_mean_teds_struct:>15.4f} {self.pipeline_mean_teds_struct:>15.4f}",
            f"{'TEDS (with text)':<35} {self.docling_mean_teds_text:>15.4f} {self.pipeline_mean_teds_text:>15.4f}",
            f"{'Cell F1':<35} {self.docling_mean_cell_f1:>15.4f} {self.pipeline_mean_cell_f1:>15.4f}",
            "-" * 70,
            "",
            "Head-to-Head Results:",
            f"  Docling wins:      {self.docling_wins_count:>5} ({self.docling_wins_count/max(1,self.samples_compared)*100:.1f}%)",
            f"  Old Pipeline wins: {self.pipeline_wins_count:>5} ({self.pipeline_wins_count/max(1,self.samples_compared)*100:.1f}%)",
            f"  Ties:              {self.tie_count:>5} ({self.tie_count/max(1,self.samples_compared)*100:.1f}%)",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


# =============================================================================
# Pipeline Runners
# =============================================================================

class DoclingRunner:
    """Run table extraction using Docling."""
    
    def __init__(self):
        self.version = self._get_version()
        self._check_availability()
    
    def _get_version(self) -> str:
        try:
            import docling
            return getattr(docling, '__version__', 'unknown')
        except ImportError:
            return 'not_installed'
    
    def _check_availability(self):
        if self.version == 'not_installed':
            logger.warning("Docling is not installed. Docling results will be skipped.")
            self.available = False
        else:
            self.available = True
            logger.info(f"Docling version: {self.version}")
    
    def process_image(self, image_path: str) -> Optional[Dict]:
        """Process image with Docling and return table grid."""
        if not self.available:
            return None
        
        try:
            from docling.document_converter import DocumentConverter
            
            start_time = datetime.now()
            
            converter = DocumentConverter()
            result = converter.convert(image_path)
            
            # Extract tables - using new Docling API
            tables = []
            html_tables = []
            
            for table in result.document.tables:
                # Try multiple ways to extract table data
                grid = []
                
                # Method 1: Export to HTML and parse
                try:
                    html = table.export_to_html()
                    html_tables.append(html)
                    
                    # Parse HTML to grid
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    for tr in soup.find_all('tr'):
                        row_data = []
                        for cell in tr.find_all(['td', 'th']):
                            row_data.append(cell.get_text(strip=True))
                        if row_data:
                            grid.append(row_data)
                except Exception as e:
                    logger.debug(f"HTML export failed: {e}")
                
                # Method 2: Try export_to_dataframe
                if not grid:
                    try:
                        df = table.export_to_dataframe()
                        grid = [df.columns.tolist()] + df.values.tolist()
                    except Exception as e:
                        logger.debug(f"DataFrame export failed: {e}")
                
                if grid:
                    tables.append(grid)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'grid': tables[0] if tables else [],
                'html': html_tables[0] if html_tables else None,
                'num_tables': len(tables),
                'processing_time_ms': processing_time
            }
        except Exception as e:
            logger.error(f"Docling processing failed: {e}")
            return {'error': str(e)}


class OldPipelineRunner:
    """Run table extraction using our old pipeline."""
    
    def __init__(self, config_path: str = None):
        # Use absolute path to config
        if config_path is None:
            config_path = str(project_root / 'configs' / 'config.yaml')
        self.config_path = config_path
        self.version = self._get_version()
        self._pipeline = None
    
    def _get_version(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5,
                cwd=str(project_root)
            )
            if result.returncode == 0:
                return f"git:{result.stdout.strip()}"
        except Exception:
            pass
        return "1.0.0"
    
    def _init_pipeline(self):
        """Lazy initialization of pipeline components."""
        if self._pipeline is None:
            try:
                # Initialize structure recognizer directly
                from modules.structure import TableStructureRecognizer
                from modules.utils import load_config
                
                config_path = str(project_root / 'configs' / 'config.yaml')
                config = load_config(config_path)
                
                self._structure = TableStructureRecognizer(config, use_v1_1=True)
                
                # Initialize RapidOCR for text extraction
                from rapidocr import RapidOCR
                # Use torch engine (already installed)
                self._ocr = RapidOCR()
                
                self._pipeline = True  # Mark as initialized
                logger.info(f"Old pipeline initialized: Table Transformer v1.1 + RapidOCR")
            except Exception as e:
                logger.error(f"Failed to initialize old pipeline: {e}")
                import traceback
                traceback.print_exc()
                self._pipeline = None
    
    def process_image(self, image_path: str) -> Optional[Dict]:
        """Process image with Table Transformer + RapidOCR."""
        self._init_pipeline()
        
        if self._pipeline is None:
            return {'error': 'Pipeline not initialized'}
        
        try:
            from PIL import Image
            from datetime import datetime
            
            start_time = datetime.now()
            
            # 1. Structure Recognition (Table Transformer v1.1)
            image = Image.open(image_path).convert('RGB')
            structure = self._structure.recognize(image)
            
            rows = structure.get('rows', [])
            columns = structure.get('columns', [])
            
            if not rows or not columns:
                return {'error': 'No table structure detected', 'grid': []}
            
            # 2. OCR (RapidOCR)
            ocr_results, _ = self._ocr(image_path)
            
            # Convert RapidOCR results to our format
            ocr_items = []
            if ocr_results:
                for item in ocr_results:
                    box, text, conf = item
                    # RapidOCR box format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    ocr_items.append({'text': text, 'bbox': bbox, 'confidence': conf})
            
            # 3. Grid Alignment
            grid = self._align_to_grid(ocr_items, rows, columns)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'grid': grid,
                'structure': structure,
                'processing_time_ms': processing_time
            }
        except Exception as e:
            logger.error(f"Old pipeline processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _align_to_grid(self, ocr_items, rows, columns):
        """Align OCR results to grid using center point matching."""
        num_rows = len(rows)
        num_cols = len(columns)
        
        if num_rows == 0 or num_cols == 0:
            return []
        
        grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Sort by y then x
        sorted_ocr = sorted(ocr_items, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        for item in sorted_ocr:
            text = item['text']
            bbox = item['bbox']
            
            # Find best matching cell using center point
            text_center_y = (bbox[1] + bbox[3]) / 2
            text_center_x = (bbox[0] + bbox[2]) / 2
            
            row_idx = -1
            col_idx = -1
            
            # Find row
            for i, row in enumerate(rows):
                row_bbox = row.get('bbox', row) if isinstance(row, dict) else row
                if row_bbox[1] <= text_center_y <= row_bbox[3]:
                    row_idx = i
                    break
            
            # Find column
            for j, col in enumerate(columns):
                col_bbox = col.get('bbox', col) if isinstance(col, dict) else col
                if col_bbox[0] <= text_center_x <= col_bbox[2]:
                    col_idx = j
                    break
            
            if row_idx >= 0 and col_idx >= 0:
                if grid[row_idx][col_idx]:
                    grid[row_idx][col_idx] += ' ' + text
                else:
                    grid[row_idx][col_idx] = text
        
        return grid


# =============================================================================
# Comparison Engine
# =============================================================================

class StructureComparisonEngine:
    """Engine for comparing structure recognition systems."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.docling_runner = DoclingRunner()
        self.pipeline_runner = OldPipelineRunner()
        
        # Import adapters and metrics
        from baselines.docling_eval.adapter_docling import DoclingAdapter
        from baselines.docling_eval.adapter_old_pipeline import OldPipelineAdapter
        from baselines.docling_eval.eval_metrics import evaluate_table_pair
        
        self.docling_adapter = DoclingAdapter()
        self.pipeline_adapter = OldPipelineAdapter()
        self.evaluate_pair = evaluate_table_pair
    
    def _parse_ground_truth(self, ground_truth: Any) -> Optional[List[List[str]]]:
        """Parse ground truth from various formats (HTML, grid, etc.)."""
        if ground_truth is None:
            return None
        
        # If it's already a grid (list of lists)
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            if isinstance(ground_truth[0], list):
                return ground_truth
        
        # If it's HTML string
        if isinstance(ground_truth, str) and ('<table' in ground_truth.lower() or '<tr' in ground_truth.lower()):
            return self._html_to_grid(ground_truth)
        
        # If it's a numpy array or similar (from parquet) - could be tokenized HTML
        try:
            import numpy as np
            if isinstance(ground_truth, np.ndarray):
                if len(ground_truth) > 0:
                    # Check if it's tokenized HTML (array of tags like '<tr>', '<td>', etc.)
                    first_elem = str(ground_truth[0])
                    if first_elem.startswith('<') and len(first_elem) < 50:
                        # Tokenized HTML - join all tokens
                        html_str = ''.join(str(x) for x in ground_truth)
                        return self._html_to_grid(html_str)
                    elif isinstance(ground_truth[0], str):
                        # Single HTML string in array
                        return self._html_to_grid(ground_truth[0])
        except ImportError:
            pass
        
        # Try converting to string and parsing as HTML
        try:
            html_str = str(ground_truth)
            if '<' in html_str:
                return self._html_to_grid(html_str)
        except Exception:
            pass
        
        return None
    
    def _html_to_grid(self, html: str) -> Optional[List[List[str]]]:
        """Convert HTML table to grid format."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')
            if not table:
                table = soup
            
            grid = []
            for tr in table.find_all('tr'):
                row = []
                for cell in tr.find_all(['td', 'th']):
                    text = cell.get_text(strip=True)
                    row.append(text)
                if row:
                    grid.append(row)
            
            return grid if grid else None
            
        except Exception as e:
            logger.debug(f"Failed to parse HTML: {e}")
            return None
    
    def compare_on_image(
        self,
        image_path: str,
        ground_truth: Optional[Any] = None
    ) -> SampleComparison:
        """
        Compare both systems on a single image.
        
        Args:
            image_path: Path to image file
            ground_truth: Optional ground truth (HTML string, grid list, or None)
            
        Returns:
            SampleComparison result
        """
        sample_id = Path(image_path).stem
        
        # Parse ground truth - could be HTML string or grid
        ground_truth_grid = None
        if ground_truth is not None:
            ground_truth_grid = self._parse_ground_truth(ground_truth)
        
        comparison = SampleComparison(
            sample_id=sample_id,
            ground_truth_source='provided' if ground_truth_grid else 'cross_compare'
        )
        
        docling_grid = None
        pipeline_grid = None
        docling_time = 0
        pipeline_time = 0
        
        # Run Docling
        if self.docling_runner.available:
            docling_output = self.docling_runner.process_image(image_path)
            if docling_output and 'error' not in docling_output:
                docling_grid = docling_output.get('grid', [])
                docling_time = docling_output.get('processing_time_ms', 0)
        
        # Run Old Pipeline
        pipeline_output = self.pipeline_runner.process_image(image_path)
        if pipeline_output and 'error' not in pipeline_output:
            pipeline_grid = pipeline_output.get('grid', [])
            pipeline_time = pipeline_output.get('processing_time_ms', 0)
        
        # Determine reference for evaluation
        if ground_truth_grid:
            # Use provided ground truth
            reference_grid = ground_truth_grid
            reference_source = 'ground_truth'
        elif docling_grid and pipeline_grid:
            # Cross-compare: use each other as reference
            # This gives us a measure of agreement between systems
            reference_grid = None  # Will compare both ways
            reference_source = 'cross_compare'
        else:
            return comparison  # Can't compare
        
        from baselines.docling_eval.common_schema import create_common_table
        
        if reference_source == 'ground_truth':
            gt_table = create_common_table(
                table_id=sample_id,
                source_system='ground_truth',
                grid=reference_grid
            )
            
            # Evaluate Docling against GT
            if docling_grid:
                pred_table = create_common_table(
                    table_id=sample_id,
                    source_system='docling',
                    grid=docling_grid
                )
                eval_result = self.evaluate_pair(pred_table, gt_table)
                comparison.docling_result = SystemResult(
                    system_name='docling',
                    version=self.docling_runner.version,
                    teds_struct=eval_result.teds_struct,
                    teds_text=eval_result.teds_text,
                    cell_f1_struct=eval_result.cell_f1_struct.get('f1', 0),
                    cell_f1_text=eval_result.cell_f1_text.get('f1', 0),
                    row_accuracy=1.0 if eval_result.structural_acc.get('row_count_match') else 0.0,
                    col_accuracy=1.0 if eval_result.structural_acc.get('col_count_match') else 0.0,
                    processing_time_ms=docling_time
                )
            
            # Evaluate Old Pipeline against GT
            if pipeline_grid:
                pred_table = create_common_table(
                    table_id=sample_id,
                    source_system='old_pipeline',
                    grid=pipeline_grid
                )
                eval_result = self.evaluate_pair(pred_table, gt_table)
                comparison.old_pipeline_result = SystemResult(
                    system_name='old_pipeline',
                    version=self.pipeline_runner.version,
                    teds_struct=eval_result.teds_struct,
                    teds_text=eval_result.teds_text,
                    cell_f1_struct=eval_result.cell_f1_struct.get('f1', 0),
                    cell_f1_text=eval_result.cell_f1_text.get('f1', 0),
                    row_accuracy=1.0 if eval_result.structural_acc.get('row_count_match') else 0.0,
                    col_accuracy=1.0 if eval_result.structural_acc.get('col_count_match') else 0.0,
                    processing_time_ms=pipeline_time
                )
        
        elif reference_source == 'cross_compare' and docling_grid and pipeline_grid:
            # Cross-compare: measure agreement between systems
            docling_table = create_common_table(
                table_id=sample_id,
                source_system='docling',
                grid=docling_grid
            )
            pipeline_table = create_common_table(
                table_id=sample_id,
                source_system='old_pipeline',
                grid=pipeline_grid
            )
            
            # Compare Docling vs Pipeline (using pipeline as reference)
            eval_vs_pipeline = self.evaluate_pair(docling_table, pipeline_table)
            # Compare Pipeline vs Docling (using docling as reference)
            eval_vs_docling = self.evaluate_pair(pipeline_table, docling_table)
            
            # For cross-compare, we report agreement scores
            comparison.docling_result = SystemResult(
                system_name='docling',
                version=self.docling_runner.version,
                teds_struct=eval_vs_pipeline.teds_struct,
                teds_text=eval_vs_pipeline.teds_text,
                cell_f1_struct=eval_vs_pipeline.cell_f1_struct.get('f1', 0),
                cell_f1_text=eval_vs_pipeline.cell_f1_text.get('f1', 0),
                row_accuracy=1.0 if eval_vs_pipeline.structural_acc.get('row_count_match') else 0.0,
                col_accuracy=1.0 if eval_vs_pipeline.structural_acc.get('col_count_match') else 0.0,
                processing_time_ms=docling_time
            )
            comparison.old_pipeline_result = SystemResult(
                system_name='old_pipeline',
                version=self.pipeline_runner.version,
                teds_struct=eval_vs_docling.teds_struct,
                teds_text=eval_vs_docling.teds_text,
                cell_f1_struct=eval_vs_docling.cell_f1_struct.get('f1', 0),
                cell_f1_text=eval_vs_docling.cell_f1_text.get('f1', 0),
                row_accuracy=1.0 if eval_vs_docling.structural_acc.get('row_count_match') else 0.0,
                col_accuracy=1.0 if eval_vs_docling.structural_acc.get('col_count_match') else 0.0,
                processing_time_ms=pipeline_time
            )
        
        return comparison
    
    def compare_batch(
        self,
        samples: List[Tuple[str, Optional[Any]]],
        progress_callback=None
    ) -> ComparisonReport:
        """
        Compare both systems on a batch of samples.
        
        Args:
            samples: List of (image_path, ground_truth) tuples
                     ground_truth can be HTML string, grid list, or None
            progress_callback: Optional callback for progress updates
            
        Returns:
            ComparisonReport with aggregated results
        """
        comparisons = []
        
        for i, (image_path, gt) in enumerate(samples):
            if progress_callback:
                progress_callback(i + 1, len(samples))
            
            logger.info(f"Processing sample {i+1}/{len(samples)}: {Path(image_path).name}")
            
            try:
                comparison = self.compare_on_image(image_path, gt)
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                comparisons.append(SampleComparison(
                    sample_id=Path(image_path).stem,
                    ground_truth_source='error'
                ))
        
        # Aggregate results
        report = self._aggregate_results(comparisons)
        
        # Save report
        report_path = self.output_dir / f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save summary
        summary_path = self.output_dir / f'comparison_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(summary_path, 'w') as f:
            f.write(report.summary_table())
        
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return report
    
    def _aggregate_results(self, comparisons: List[SampleComparison]) -> ComparisonReport:
        """Aggregate individual comparisons into a report."""
        valid_comparisons = [
            c for c in comparisons
            if c.docling_result or c.old_pipeline_result
        ]
        
        # Docling stats
        docling_teds_struct = []
        docling_teds_text = []
        docling_cell_f1 = []
        
        # Pipeline stats
        pipeline_teds_struct = []
        pipeline_teds_text = []
        pipeline_cell_f1 = []
        
        # Win counts
        docling_wins = 0
        pipeline_wins = 0
        ties = 0
        
        for c in valid_comparisons:
            if c.docling_result:
                docling_teds_struct.append(c.docling_result.teds_struct)
                docling_teds_text.append(c.docling_result.teds_text)
                docling_cell_f1.append(c.docling_result.cell_f1_struct)
            
            if c.old_pipeline_result:
                pipeline_teds_struct.append(c.old_pipeline_result.teds_struct)
                pipeline_teds_text.append(c.old_pipeline_result.teds_text)
                pipeline_cell_f1.append(c.old_pipeline_result.cell_f1_struct)
            
            # Determine winner
            diff = c.teds_struct_diff()
            if diff is not None:
                if diff > 0.01:
                    docling_wins += 1
                elif diff < -0.01:
                    pipeline_wins += 1
                else:
                    ties += 1
        
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        return ComparisonReport(
            timestamp=datetime.now().isoformat(),
            num_samples=len(comparisons),
            samples_compared=len(valid_comparisons),
            docling_mean_teds_struct=round(safe_mean(docling_teds_struct), 4),
            docling_mean_teds_text=round(safe_mean(docling_teds_text), 4),
            docling_mean_cell_f1=round(safe_mean(docling_cell_f1), 4),
            pipeline_mean_teds_struct=round(safe_mean(pipeline_teds_struct), 4),
            pipeline_mean_teds_text=round(safe_mean(pipeline_teds_text), 4),
            pipeline_mean_cell_f1=round(safe_mean(pipeline_cell_f1), 4),
            docling_wins_count=docling_wins,
            pipeline_wins_count=pipeline_wins,
            tie_count=ties,
            docling_version=self.docling_runner.version,
            pipeline_version=self.pipeline_runner.version,
            comparisons=comparisons
        )


# =============================================================================
# Sample Loaders
# =============================================================================

def load_local_samples(samples_dir: str, max_samples: int = None) -> List[Tuple[str, Optional[List]]]:
    """Load samples from local directory."""
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        raise ValueError(f"Samples directory not found: {samples_dir}")
    
    samples = []
    for img_path in samples_path.glob('*.png'):
        samples.append((str(img_path), None))  # No GT for local samples
    
    for img_path in samples_path.glob('*.jpg'):
        samples.append((str(img_path), None))
    
    if max_samples:
        samples = samples[:max_samples]
    
    logger.info(f"Loaded {len(samples)} samples from {samples_dir}")
    return samples


def load_fintabnet_local(max_samples: int = 100) -> List[Tuple[str, Optional[List]]]:
    """Load samples from locally downloaded FinTabNet dataset.
    
    Uses the existing data_loaders.py infrastructure and config.yaml paths.
    """
    try:
        # Import local dataset loader
        from local_dataset_loader import load_fintabnet_local as _load_fintabnet
        
        logger.info("Loading FinTabNet samples from local dataset...")
        
        # Call with correct parameters
        samples = _load_fintabnet(split='test', num_samples=max_samples)
        
        logger.info(f"Loaded {len(samples)} FinTabNet samples from local dataset")
        return samples
    
    except ImportError as e:
        logger.warning(f"local_dataset_loader not available: {e}")
        logger.info("Falling back to HuggingFace...")
        return load_fintabnet_huggingface(max_samples)
    except Exception as e:
        logger.error(f"Failed to load local FinTabNet: {e}")
        logger.info("Falling back to HuggingFace...")
        return load_fintabnet_huggingface(max_samples)


def load_pubtabnet_local(max_samples: int = 100) -> List[Tuple[str, Optional[List]]]:
    """Load samples from locally downloaded PubTabNet dataset.
    
    Uses the existing data_loaders.py infrastructure and config.yaml paths.
    """
    try:
        # Import local dataset loader
        from local_dataset_loader import load_pubtabnet_local as _load_pubtabnet
        
        logger.info("Loading PubTabNet samples from local dataset...")
        
        # Call with correct parameters
        samples = _load_pubtabnet(split='val', num_samples=max_samples)
        
        logger.info(f"Loaded {len(samples)} PubTabNet samples from local dataset")
        return samples
    
    except ImportError as e:
        logger.warning(f"local_dataset_loader not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load local PubTabNet: {e}")
        return []


def load_fintabnet_huggingface(max_samples: int = 100) -> List[Tuple[str, Optional[str]]]:
    """Load samples from FinTabNet dataset (via HuggingFace).
    
    Fallback when local dataset is not available.
    """
    try:
        from datasets import load_dataset
        
        logger.info("Loading FinTabNet samples from HuggingFace...")
        dataset = load_dataset('ds4sd/FinTabNet_OTSL', split='test')
        
        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            # FinTabNet provides image and HTML ground truth
            image = item.get('image')
            html_gt = item.get('html')
            
            # Save image temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                samples.append((f.name, html_gt))
        
        logger.info(f"Loaded {len(samples)} FinTabNet samples from HuggingFace")
        return samples
    
    except Exception as e:
        logger.error(f"Failed to load FinTabNet from HuggingFace: {e}")
        return []


def load_fintabnet_otsl_local(
    max_samples: int = 100,
    data_dir: str = "D:/datasets/FinTabNet_OTSL/data"
) -> List[Tuple[str, Optional[str]]]:
    """Load FinTabNet_OTSL samples from local parquet files.
    
    This version has HTML ground truth for proper TEDS evaluation.
    """
    try:
        from fintabnet_otsl_loader import load_fintabnet_otsl
        
        logger.info(f"Loading FinTabNet_OTSL from local parquet: {data_dir}")
        samples = load_fintabnet_otsl(data_dir, split='test', num_samples=max_samples)
        
        logger.info(f"Loaded {len(samples)} FinTabNet_OTSL samples with HTML ground truth")
        return samples
    
    except ImportError as e:
        logger.warning(f"fintabnet_otsl_loader not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load FinTabNet_OTSL: {e}")
        return []


def load_fintabnet_samples(max_samples: int = 100, prefer_local: bool = True) -> List[Tuple[str, Optional[str]]]:
    """Load FinTabNet samples, preferring local OTSL dataset.
    
    Args:
        max_samples: Maximum number of samples to load
        prefer_local: If True, try local OTSL dataset first
    
    Returns:
        List of (image_path, html_ground_truth) tuples
    """
    if prefer_local:
        # Try local OTSL parquet first (has HTML GT)
        samples = load_fintabnet_otsl_local(max_samples)
        if samples:
            return samples
        
        # Fall back to old local dataset (no HTML GT)
        samples = load_fintabnet_local(max_samples)
        if samples:
            return samples
        
        logger.info("Local FinTabNet not available, falling back to HuggingFace")
    
    return load_fintabnet_huggingface(max_samples)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare structure recognition: Docling vs Old Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare on local samples directory
  python run_compare_structures.py --samples ./data/samples --output-dir ./outputs/comparison
  
  # Compare on local FinTabNet dataset (recommended - uses config.yaml paths)
  python run_compare_structures.py --use-fintabnet --max-samples 100
  
  # Compare on local PubTabNet dataset
  python run_compare_structures.py --use-pubtabnet --max-samples 100
  
  # Force HuggingFace download instead of local dataset
  python run_compare_structures.py --use-fintabnet --no-local --max-samples 50
  
  # Quick test
  python run_compare_structures.py --samples ./data/samples --max-samples 10
        """
    )
    
    parser.add_argument(
        '--samples',
        type=str,
        help='Path to directory containing sample images'
    )
    
    parser.add_argument(
        '--use-fintabnet',
        action='store_true',
        help='Use FinTabNet dataset samples (prefers local, falls back to HuggingFace)'
    )
    
    parser.add_argument(
        '--use-pubtabnet',
        action='store_true',
        help='Use PubTabNet dataset samples (local only)'
    )
    
    parser.add_argument(
        '--no-local',
        action='store_true',
        help='Skip local dataset, force HuggingFace download'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum samples to compare (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/baselines/comparison',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Load samples
    if args.use_fintabnet:
        prefer_local = not args.no_local
        samples = load_fintabnet_samples(args.max_samples, prefer_local=prefer_local)
    elif args.use_pubtabnet:
        samples = load_pubtabnet_local(args.max_samples)
    elif args.samples:
        samples = load_local_samples(args.samples, args.max_samples)
    else:
        # Default: look for samples in data/samples
        default_samples = project_root / 'data' / 'samples'
        if default_samples.exists():
            samples = load_local_samples(str(default_samples), args.max_samples)
        else:
            logger.error("No samples specified. Use --samples, --use-fintabnet, or --use-pubtabnet")
            sys.exit(1)
    
    if not samples:
        logger.error("No samples found")
        sys.exit(1)
    
    # Run comparison
    engine = StructureComparisonEngine(args.output_dir)
    
    def progress(current, total):
        print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%)", end='', flush=True)
    
    report = engine.compare_batch(samples, progress_callback=progress)
    print()  # New line after progress
    
    # Print summary
    print(report.summary_table())


if __name__ == '__main__':
    main()
