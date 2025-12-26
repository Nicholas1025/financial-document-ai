"""
Structure Evaluation Metrics

Implements standard table structure evaluation metrics:
- TEDS (Tree-Edit-Distance-based Similarity)
- Cell F1 Score
- Structural Accuracy (row/col count)

Uses docling-eval's TEDS implementation when available,
falls back to simplified implementation otherwise.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json

from .common_schema import CommonTable

logger = logging.getLogger(__name__)


# =============================================================================
# TEDS Implementation
# =============================================================================

def _try_import_docling_teds():
    """Try to import docling-eval's TEDS implementation."""
    try:
        from docling_eval.evaluators.table.teds import TEDScorer
        return TEDScorer
    except ImportError:
        return None


class TEDSCalculator:
    """
    TEDS (Tree-Edit-Distance-based Similarity) Calculator.
    
    Uses docling-eval's implementation when available,
    otherwise falls back to a simplified version.
    """
    
    def __init__(self):
        self._docling_teds = _try_import_docling_teds()
        if self._docling_teds:
            self._scorer = self._docling_teds()
            logger.info("Using docling-eval TEDS implementation")
        else:
            self._scorer = None
            logger.warning("docling-eval not available, using simplified TEDS")
    
    def calculate(
        self,
        pred_table: CommonTable,
        gt_table: CommonTable,
        structure_only: bool = False
    ) -> float:
        """
        Calculate TEDS score between predicted and ground truth tables.
        
        Args:
            pred_table: Predicted table (CommonTable)
            gt_table: Ground truth table (CommonTable)
            structure_only: If True, ignore cell text content
            
        Returns:
            TEDS score (0-1, higher is better)
        """
        pred_html = pred_table.to_html()
        gt_html = gt_table.to_html()
        
        if self._scorer:
            # Use docling-eval implementation
            try:
                from lxml import html
                pred_html_obj = html.fromstring(pred_html)
                gt_html_obj = html.fromstring(gt_html)
                score = self._scorer(
                    gt_table=gt_html_obj,
                    pred_table=pred_html_obj,
                    structure_only=structure_only
                )
                return round(score, 4)
            except Exception as e:
                logger.warning(f"docling TEDS failed, using fallback: {e}")
        
        # Fallback: simplified TEDS
        return self._simplified_teds(pred_html, gt_html, structure_only)
    
    def _simplified_teds(
        self,
        pred_html: str,
        gt_html: str,
        structure_only: bool = False
    ) -> float:
        """Simplified TEDS implementation."""
        from html.parser import HTMLParser
        
        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.nodes = []
                self.current_text = ""
                
            def handle_starttag(self, tag, attrs):
                if tag in ['table', 'tr', 'td', 'th']:
                    self.nodes.append(('start', tag, dict(attrs)))
                    
            def handle_endtag(self, tag):
                if tag in ['table', 'tr', 'td', 'th']:
                    self.nodes.append(('end', tag, self.current_text.strip()))
                    self.current_text = ""
                    
            def handle_data(self, data):
                self.current_text += data
        
        pred_parser = TableParser()
        gt_parser = TableParser()
        pred_parser.feed(pred_html)
        gt_parser.feed(gt_html)
        
        pred_nodes = pred_parser.nodes
        gt_nodes = gt_parser.nodes
        
        if structure_only:
            # Only compare structure (tags), not text
            pred_struct = [(n[0], n[1]) for n in pred_nodes]
            gt_struct = [(n[0], n[1]) for n in gt_nodes]
            pred_nodes = pred_struct
            gt_nodes = gt_struct
        
        # Calculate Levenshtein distance (simplified tree edit)
        m, n = len(pred_nodes), len(gt_nodes)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_nodes[i-1] == gt_nodes[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_dist = dp[m][n]
        max_size = max(m, n)
        
        if max_size == 0:
            return 1.0
        
        teds = 1 - (edit_dist / max_size)
        return round(max(0.0, teds), 4)


# =============================================================================
# Cell F1 Score
# =============================================================================

def calculate_cell_f1(
    pred_table: CommonTable,
    gt_table: CommonTable,
    match_text: bool = True
) -> Dict[str, float]:
    """
    Calculate cell-level F1 score.
    
    A cell is considered a match if:
    - Position (row, col) matches
    - If match_text=True, text content must also match
    
    Args:
        pred_table: Predicted table
        gt_table: Ground truth table
        match_text: Whether to also match cell text
        
    Returns:
        Dict with precision, recall, f1 scores
    """
    # Build cell sets
    def cell_key(cell, include_text: bool):
        if include_text:
            return (cell.row, cell.col, cell.text.strip().lower())
        return (cell.row, cell.col)
    
    pred_cells = {cell_key(c, match_text) for c in pred_table.cells}
    gt_cells = {cell_key(c, match_text) for c in gt_table.cells}
    
    # Calculate metrics
    true_positives = len(pred_cells & gt_cells)
    false_positives = len(pred_cells - gt_cells)
    false_negatives = len(gt_cells - pred_cells)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


# =============================================================================
# Structural Accuracy
# =============================================================================

def calculate_structural_accuracy(
    pred_table: CommonTable,
    gt_table: CommonTable
) -> Dict[str, Any]:
    """
    Calculate structural accuracy metrics.
    
    Compares:
    - Row count
    - Column count
    - Header row count
    - Whether spans match
    
    Args:
        pred_table: Predicted table
        gt_table: Ground truth table
        
    Returns:
        Dict with structural metrics
    """
    pred_struct = pred_table.structure
    gt_struct = gt_table.structure
    
    row_diff = abs(pred_struct.num_rows - gt_struct.num_rows)
    col_diff = abs(pred_struct.num_cols - gt_struct.num_cols)
    
    row_match = pred_struct.num_rows == gt_struct.num_rows
    col_match = pred_struct.num_cols == gt_struct.num_cols
    structure_match = row_match and col_match
    
    return {
        'row_count_match': row_match,
        'col_count_match': col_match,
        'structure_match': structure_match,
        'row_diff': row_diff,
        'col_diff': col_diff,
        'pred_rows': pred_struct.num_rows,
        'pred_cols': pred_struct.num_cols,
        'gt_rows': gt_struct.num_rows,
        'gt_cols': gt_struct.num_cols,
        'header_row_match': pred_struct.num_header_rows == gt_struct.num_header_rows
    }


# =============================================================================
# Combined Evaluation
# =============================================================================

@dataclass
class TableEvaluationResult:
    """Results from table structure evaluation."""
    table_id: str
    pred_source: str
    gt_source: str
    
    # TEDS scores
    teds_struct: float = 0.0
    teds_text: float = 0.0
    
    # Cell F1 scores
    cell_f1_struct: Dict[str, float] = field(default_factory=dict)
    cell_f1_text: Dict[str, float] = field(default_factory=dict)
    
    # Structural accuracy
    structural_acc: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def evaluate_table_pair(
    pred_table: CommonTable,
    gt_table: CommonTable
) -> TableEvaluationResult:
    """
    Perform full evaluation of a predicted table against ground truth.
    
    Args:
        pred_table: Predicted table (CommonTable)
        gt_table: Ground truth table (CommonTable)
        
    Returns:
        TableEvaluationResult with all metrics
    """
    teds_calc = TEDSCalculator()
    
    # TEDS scores
    teds_struct = teds_calc.calculate(pred_table, gt_table, structure_only=True)
    teds_text = teds_calc.calculate(pred_table, gt_table, structure_only=False)
    
    # Cell F1 scores
    cell_f1_struct = calculate_cell_f1(pred_table, gt_table, match_text=False)
    cell_f1_text = calculate_cell_f1(pred_table, gt_table, match_text=True)
    
    # Structural accuracy
    structural_acc = calculate_structural_accuracy(pred_table, gt_table)
    
    return TableEvaluationResult(
        table_id=pred_table.table_id,
        pred_source=pred_table.source_system,
        gt_source=gt_table.source_system,
        teds_struct=teds_struct,
        teds_text=teds_text,
        cell_f1_struct=cell_f1_struct,
        cell_f1_text=cell_f1_text,
        structural_acc=structural_acc
    )


# =============================================================================
# Batch Evaluation
# =============================================================================

@dataclass
class BatchEvaluationResult:
    """Results from batch table evaluation."""
    num_samples: int
    
    # Aggregated TEDS
    mean_teds_struct: float = 0.0
    mean_teds_text: float = 0.0
    
    # Aggregated Cell F1
    mean_cell_f1_struct: float = 0.0
    mean_cell_f1_text: float = 0.0
    
    # Structural accuracy rates
    row_match_rate: float = 0.0
    col_match_rate: float = 0.0
    structure_match_rate: float = 0.0
    
    # Individual results
    results: List[TableEvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['results'] = [r.to_dict() for r in self.results]
        return d
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary_table(self) -> str:
        """Generate summary table string."""
        lines = [
            "=" * 60,
            f"Batch Evaluation Summary (n={self.num_samples})",
            "=" * 60,
            f"{'Metric':<30} {'Value':>15}",
            "-" * 60,
            f"{'TEDS (struct-only)':<30} {self.mean_teds_struct:>15.4f}",
            f"{'TEDS (with text)':<30} {self.mean_teds_text:>15.4f}",
            f"{'Cell F1 (struct)':<30} {self.mean_cell_f1_struct:>15.4f}",
            f"{'Cell F1 (with text)':<30} {self.mean_cell_f1_text:>15.4f}",
            f"{'Row Match Rate':<30} {self.row_match_rate:>15.2%}",
            f"{'Col Match Rate':<30} {self.col_match_rate:>15.2%}",
            f"{'Structure Match Rate':<30} {self.structure_match_rate:>15.2%}",
            "=" * 60,
        ]
        return "\n".join(lines)


def evaluate_batch(
    pred_tables: List[CommonTable],
    gt_tables: List[CommonTable]
) -> BatchEvaluationResult:
    """
    Evaluate a batch of table pairs.
    
    Args:
        pred_tables: List of predicted tables
        gt_tables: List of ground truth tables (matched by index)
        
    Returns:
        BatchEvaluationResult with aggregated metrics
    """
    if len(pred_tables) != len(gt_tables):
        raise ValueError(f"Mismatched lengths: {len(pred_tables)} vs {len(gt_tables)}")
    
    results = []
    for pred, gt in zip(pred_tables, gt_tables):
        result = evaluate_table_pair(pred, gt)
        results.append(result)
    
    n = len(results)
    if n == 0:
        return BatchEvaluationResult(num_samples=0)
    
    # Aggregate metrics
    mean_teds_struct = sum(r.teds_struct for r in results) / n
    mean_teds_text = sum(r.teds_text for r in results) / n
    mean_cell_f1_struct = sum(r.cell_f1_struct.get('f1', 0) for r in results) / n
    mean_cell_f1_text = sum(r.cell_f1_text.get('f1', 0) for r in results) / n
    
    row_matches = sum(1 for r in results if r.structural_acc.get('row_count_match', False))
    col_matches = sum(1 for r in results if r.structural_acc.get('col_count_match', False))
    struct_matches = sum(1 for r in results if r.structural_acc.get('structure_match', False))
    
    return BatchEvaluationResult(
        num_samples=n,
        mean_teds_struct=round(mean_teds_struct, 4),
        mean_teds_text=round(mean_teds_text, 4),
        mean_cell_f1_struct=round(mean_cell_f1_struct, 4),
        mean_cell_f1_text=round(mean_cell_f1_text, 4),
        row_match_rate=round(row_matches / n, 4),
        col_match_rate=round(col_matches / n, 4),
        structure_match_rate=round(struct_matches / n, 4),
        results=results
    )
