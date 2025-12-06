"""
Financial Document Processing Pipeline

Encapsulates the end-to-end logic for processing financial table images:
1. Structure Recognition
2. OCR
3. Grid Alignment (with fallback)
4. Numeric Normalization
5. Semantic Mapping
6. Validation Rules
"""
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from .structure import TableStructureRecognizer
from .ocr import TableOCR
from .numeric import normalize_numeric
from .semantic import map_alias, correct_ocr_text
from .utils import load_config


class FinancialTablePipeline:
    """
    End-to-end pipeline for financial table processing.
    """

    def __init__(self, config_path: str = 'configs/config.yaml', use_v1_1: bool = True):
        """
        Initialize the pipeline with models.
        
        Args:
            config_path: Path to configuration file
            use_v1_1: Whether to use v1.1 structure model (better for complex tables)
        """
        self.config = load_config(config_path)
        self.structure_recognizer = TableStructureRecognizer(self.config, use_v1_1=use_v1_1)
        self.ocr = TableOCR(lang='en')

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing processed results (grid, headers, validation, etc.)
        """
        # Ensure consistent 3-channel input for structure model
        image = Image.open(image_path).convert('RGB')
        
        # 1. Structure Recognition
        structure = self.structure_recognizer.recognize(image)
        
        # 2. OCR
        ocr_results = self.ocr.extract_text(image)
        
        # 3. Grid Alignment
        grid = self.ocr.align_text_to_grid(
            ocr_results,
            structure.get('rows', []),
            structure.get('columns', []),
            method='hybrid'
        )

        # Fallback: if structure failed to produce a sensible grid, build rows/cols from OCR only
        headers = []
        if not grid or len(grid) < 2 or (grid and len(grid[0]) < 2):
            grid, headers = self._build_grid_from_ocr(ocr_results)
        else:
            headers = self._collect_headers(grid)
            
        # Apply OCR corrections to labels (first column)
        if grid:
            for row in grid:
                if row and row[0]:
                    row[0] = correct_ocr_text(row[0])

        labels = self._collect_row_labels(grid)
        
        # 4. Numeric Normalization
        norm_grid = self._normalize_grid(grid)
        
        # 6. Validation Rules (Equity Check)
        equity_checks = self._equity_rule_check(labels, norm_grid)

        # Second-chance fallback if rule checks are empty (structure might have been bad)
        if not equity_checks:
            fb_grid, fb_headers = self._build_grid_from_ocr(ocr_results)
            if fb_grid:
                # Check if fallback grid actually yields results
                fb_labels = self._collect_row_labels(fb_grid)
                fb_norm_grid = self._normalize_grid(fb_grid)
                fb_checks = self._equity_rule_check(fb_labels, fb_norm_grid)
                
                # If fallback yields checks or better structure, use it
                if fb_checks or (len(fb_grid) > len(grid)):
                    grid = fb_grid
                    headers = fb_headers
                    labels = fb_labels
                    norm_grid = fb_norm_grid
                    equity_checks = fb_checks

        # 5. Semantic Mapping (Examples)
        mapping_examples = []
        for lbl in labels[:10]:
            if lbl:
                mapping_examples.append({'raw': lbl, 'mapped': map_alias(lbl)})

        return {
            'file': os.path.basename(image_path),
            'headers': headers,
            'labels': labels,
            'grid': grid,
            'normalized_grid': norm_grid,
            'equity_checks': equity_checks,
            'semantic_mapping': mapping_examples
        }

    def _collect_headers(self, grid: List[List[str]]) -> List[str]:
        if not grid:
            return []
        return [cell for cell in grid[0]]

    def _collect_row_labels(self, grid: List[List[str]]) -> List[str]:
        labels = []
        for r in grid:
            labels.append(r[0] if r else '')
        return labels

    def _normalize_grid(self, grid: List[List[str]]) -> List[List[Dict[str, Any]]]:
        norm_grid = []
        for row in grid:
            norm_row = []
            for cell in row:
                if cell and any(ch.isdigit() for ch in cell):
                    norm_row.append(normalize_numeric(cell, default_currency='USD'))
                else:
                    norm_row.append({'raw': cell, 'value': None})
            norm_grid.append(norm_row)
        return norm_grid

    def _cluster_centers(self, values: List[float], threshold: float) -> List[float]:
        centers = []
        for v in sorted(values):
            if not centers or abs(v - centers[-1]) > threshold:
                centers.append(v)
            else:
                centers[-1] = (centers[-1] + v) / 2
        return centers

    def _build_grid_from_ocr(self, ocr_results: List[Dict]) -> Tuple[List[List[str]], List[str]]:
        """Construct a coarse row/column grid directly from OCR boxes when structure fails."""
        if not ocr_results:
            return [], []

        # Prepare geometry stats
        heights = [box['bbox'][3] - box['bbox'][1] for box in ocr_results]
        median_h = np.median(heights) if heights else 12
        row_thresh = max(10.0, 0.6 * median_h)

        # Cluster by y-center into rows
        ocr_sorted = sorted(ocr_results, key=lambda r: ( (r['bbox'][1] + r['bbox'][3]) / 2, r['bbox'][0]))
        rows = []
        current = []
        last_y = None
        for item in ocr_sorted:
            y_center = (item['bbox'][1] + item['bbox'][3]) / 2
            if last_y is None or abs(y_center - last_y) <= row_thresh:
                current.append(item)
            else:
                rows.append(current)
                current = [item]
            last_y = y_center
        if current:
            rows.append(current)

        # Determine column centers from numeric boxes
        numeric_centers = []
        for r in rows:
            for item in r:
                if any(ch.isdigit() for ch in item['text']):
                    x_center = (item['bbox'][0] + item['bbox'][2]) / 2
                    numeric_centers.append(x_center)
        if not numeric_centers:
            return [], []

        col_thresh = 40.0
        col_centers = self._cluster_centers(numeric_centers, threshold=col_thresh)

        # Build grid: first column labels, subsequent columns per detected numeric center
        grid = []
        headers = ['label'] + [f'col_{i+1}' for i in range(len(col_centers))]

        for r in rows:
            # sort row items left-to-right
            r_sorted = sorted(r, key=lambda x: x['bbox'][0])
            label_parts = []
            values = [''] * len(col_centers)
            for item in r_sorted:
                text = item['text']
                x_center = (item['bbox'][0] + item['bbox'][2]) / 2
                if any(ch.isdigit() for ch in text):
                    # assign to nearest column center
                    nearest = min(range(len(col_centers)), key=lambda idx: abs(x_center - col_centers[idx]))
                    if values[nearest]:
                        values[nearest] += ' ' + text
                    else:
                        values[nearest] = text
                else:
                    label_parts.append(text)
            label = ' '.join(label_parts).strip()
            grid.append([label] + values)

        return grid, headers

    def _find_row_indices(self, labels: List[str], key: str) -> List[int]:
        idxs = []
        k = key.lower()
        for i, lbl in enumerate(labels):
            if lbl and k in lbl.lower():
                idxs.append(i)
        return idxs

    def _equity_rule_check(self, labels: List[str], norm_grid: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Check total equity ≈ attributable + non-controlling, per column."""
        results = []
        if not norm_grid:
            return results
        num_cols = len(norm_grid[0]) if norm_grid else 0
        
        # find rows with flexible heuristics
        r_total = self._find_row_indices(labels, 'total equity')
        r_attr = self._find_row_indices(labels, 'equity holders') or self._find_row_indices(labels, 'attributable')
        r_nci = self._find_row_indices(labels, 'non-controlling')

        # fallback heuristics if missing
        if not r_attr:
            r_attr = [i for i, lbl in enumerate(labels) if 'equity' in (lbl or '').lower()]
        if not r_nci:
            r_nci = [i for i, lbl in enumerate(labels) if 'interest' in (lbl or '').lower()]
        if not r_total:
            r_total = [i for i, lbl in enumerate(labels) if 'total' in (lbl or '').lower() and 'equity' in (lbl or '').lower()]
            if not r_total:
                # choose row with largest numeric magnitude as a last resort
                best_idx = -1
                best_val = -1
                for i, row in enumerate(norm_grid):
                    values = [cell.get('value') for cell in row[1:] if cell and cell.get('value') is not None]
                    if not values:
                        continue
                    max_val = max(abs(v) for v in values)
                    if max_val > best_val:
                        best_val = max_val
                        best_idx = i
                if best_idx >= 0:
                    r_total = [best_idx]

        if not r_total or not r_attr or not r_nci:
            return results

        r_total = r_total[0]
        r_attr = r_attr[0]
        r_nci = r_nci[0]
        
        for c in range(1, num_cols):  # skip first column (labels)
            v_total = norm_grid[r_total][c]['value'] if c < len(norm_grid[r_total]) else None
            v_attr = norm_grid[r_attr][c]['value'] if c < len(norm_grid[r_attr]) else None
            v_nci = norm_grid[r_nci][c]['value'] if c < len(norm_grid[r_nci]) else None
            
            if v_total is None or v_attr is None or v_nci is None:
                continue
                
            diff = v_total - (v_attr + v_nci)
            denom = max(abs(v_total), 1.0)
            passed = abs(diff) <= 0.01 * denom
            
            results.append({
                'column': c,
                'total_equity': v_total,
                'attributable': v_attr,
                'non_controlling': v_nci,
                'diff': diff,
                'passed': passed
            })
        return results
