"""
Financial Document Processing Pipeline

Encapsulates the end-to-end logic for processing financial table images:
1. Table Detection (DETR)
2. Structure Recognition (Table Transformer)
3. OCR (PaddleOCR / Docling)
4. Grid Alignment (with fallback)
5. Numeric Normalization
6. Semantic Mapping
7. Rule-based Validation
8. LLM Validation (optional, using Gemini)
"""
import os
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from .ocr import TableOCR
from .numeric import normalize_numeric, normalize_grid_with_metadata, extract_column_metadata
from .semantic import map_alias, correct_ocr_text
from .utils import load_config


def _get_git_commit() -> Optional[str]:
    """Try to get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_device_info() -> Dict[str, Any]:
    """Gather device information for run metadata."""
    info = {'torch_available': False}
    try:
        import torch
        info['torch_available'] = True
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_count'] = torch.cuda.device_count()
    except Exception:
        pass
    return info


class FinancialTablePipeline:
    """
    End-to-end pipeline for financial table processing.
    
    Pipeline Steps:
        1. Table Detection - Detect and crop table region from full page
        2. Structure Recognition - Detect table structure (rows, columns, cells)
        3. OCR - Extract text from cropped table image
        4. Grid Alignment - Map OCR results to table structure
        5. Numeric Normalization - Parse and normalize numbers
        6. Semantic Mapping - Map labels to standard terms
        7. Rule-based Validation - Check accounting rules (e.g., equity balance)
        8. LLM Validation (optional) - Cross-validate using Gemini
    """

    def __init__(self, config_path: str = 'configs/config.yaml', use_v1_1: bool = True,
                 ocr_backend: str = 'paddleocr', enable_llm_validation: bool = False):
        """
        Initialize the pipeline with models.
        
        Args:
            config_path: Path to configuration file
            use_v1_1: Whether to use v1.1 structure model (better for complex tables)
            ocr_backend: OCR backend to use ('paddleocr' or 'docling')
            enable_llm_validation: Whether to enable LLM-based validation (Step 7)
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        self.use_v1_1 = use_v1_1
        self.ocr_backend = ocr_backend.lower()
        self.enable_llm_validation = enable_llm_validation
        
        # Keep structure recognizer in the main process (PyTorch).
        from .structure import TableStructureRecognizer
        from .detection import TableDetector
        self.detector = TableDetector(self.config)
        self.structure_recognizer = TableStructureRecognizer(self.config, use_v1_1=use_v1_1)
        
        # Initialize OCR backend
        from .ocr import get_ocr_backend, TableOCR
        if self.ocr_backend in ('paddleocr', 'paddle'):
            # Stable default: keep PyTorch on GPU, run PaddleOCR on CPU (in a separate process).
            self.ocr = TableOCR(lang='en', use_gpu=False)
        else:
            self.ocr = get_ocr_backend(ocr_backend)
        
        # Initialize LLM validator if enabled
        self.llm_validator = None
        if enable_llm_validation:
            self._init_llm_validator()
        
        # Capture run metadata once at init
        self._run_meta = self._build_run_meta()
    
    def _init_llm_validator(self):
        """Initialize LLM validator (Gemini) for Step 7."""
        try:
            from .gemini_validation import GeminiTableValidator
            self.llm_validator = GeminiTableValidator()
            print("LLM Validation enabled (Gemini 2.5 Flash)")
        except Exception as e:
            print(f"Warning: Could not initialize LLM validator: {e}")
            self.llm_validator = None

    def _build_run_meta(self) -> Dict[str, Any]:
        """Build metadata about this pipeline run for reproducibility."""
        return {
            'timestamp': datetime.now().isoformat(),
            'git_commit': _get_git_commit(),
            'config_path': self.config_path,
            'model_version': 'v1.1' if self.use_v1_1 else 'v1.0',
            'ocr_backend': self.ocr_backend,
            'ocr_mode': 'isolated_cpu' if self.ocr_backend in ('paddleocr', 'paddle') else 'direct',
            'llm_validation': self.enable_llm_validation,
            'device_info': _get_device_info(),
        }

    def get_run_meta(self) -> Dict[str, Any]:
        """Return a copy of run metadata."""
        return dict(self._run_meta)

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
        
        # 0. Table Detection - crop the best table region
        det_results = self.detector.detect(image)
        if len(det_results['boxes']) > 0:
            best_idx = int(np.argmax(det_results['scores']))
            box = det_results['boxes'][best_idx]
            w, h = image.size
            padding = 10
            x1 = max(0, int(box[0]) - padding)
            y1 = max(0, int(box[1]) - padding)
            x2 = min(w, int(box[2]) + padding)
            y2 = min(h, int(box[3]) + padding)
            table_image = image.crop((x1, y1, x2, y2))
        else:
            # Fallback: treat entire image as table
            table_image = image
        
        # Check if using Docling - it provides direct table extraction
        use_docling_tables = (self.ocr_backend == 'docling' and 
                             hasattr(self.ocr, 'extract_table_grid'))
        
        if use_docling_tables:
            # Docling path: Use Docling's built-in table structure recognition
            # This bypasses our Table Transformer since Docling has its own TableFormer
            grid = self.ocr.extract_table_grid(image_path)
            ocr_results = []  # Not needed for Docling path
            
            # Still run structure for metadata (optional)
            structure = self.structure_recognizer.recognize(table_image)
        else:
            # PaddleOCR path: Use Table Transformer + OCR + Grid Alignment
            # 1. Structure Recognition (on CROPPED table)
            structure = self.structure_recognizer.recognize(table_image)
            
            # 2. OCR (on CROPPED table)
            # Save cropped table for OCR (needs file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                table_image.save(tmp.name)
                crop_path = tmp.name
            try:
                ocr_results = self.ocr.extract_text(crop_path)
            finally:
                try:
                    os.remove(crop_path)
                except:
                    pass
            
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
            if not use_docling_tables:
                grid, headers = self._build_grid_from_ocr(ocr_results)
        else:
            headers = self._collect_headers(grid)
            
        # Apply OCR corrections to labels (first column)
        if grid:
            for row in grid:
                if row and row[0]:
                    row[0] = correct_ocr_text(row[0])

        labels = self._collect_row_labels(grid)
        
        # 4. Numeric Normalization (with column metadata propagation)
        norm_grid, col_metadata = self._normalize_grid(grid)
        
        # 6. Validation Rules (Equity Check)
        equity_checks = self._equity_rule_check(labels, norm_grid)

        # Second-chance fallback if rule checks are empty (structure might have been bad)
        if not equity_checks:
            fb_grid, fb_headers = self._build_grid_from_ocr(ocr_results)
            if fb_grid:
                # Check if fallback grid actually yields results
                fb_labels = self._collect_row_labels(fb_grid)
                fb_norm_grid, fb_col_meta = self._normalize_grid(fb_grid)
                fb_checks = self._equity_rule_check(fb_labels, fb_norm_grid)
                
                # If fallback yields checks or better structure, use it
                if fb_checks or (len(fb_grid) > len(grid)):
                    grid = fb_grid
                    headers = fb_headers
                    labels = fb_labels
                    norm_grid = fb_norm_grid
                    col_metadata = fb_col_meta
                    equity_checks = fb_checks

        # 5. Semantic Mapping (Examples)
        mapping_examples = []
        for lbl in labels[:10]:
            if lbl:
                mapping_examples.append({'raw': lbl, 'mapped': map_alias(lbl)})

        result = {
            'run_meta': self._run_meta,
            'file': os.path.basename(image_path),
            'headers': headers,
            'labels': labels,
            'grid': grid,
            'normalized_grid': norm_grid,
            'column_metadata': [{'currency': m.currency, 'scale': m.scale, 'unit': m.unit, 
                                'column_type': m.column_type, 'year': m.year} 
                               for m in col_metadata],
            'equity_checks': equity_checks,
            'semantic_mapping': mapping_examples
        }
        
        # 7. LLM Validation (optional)
        if self.llm_validator and self.enable_llm_validation:
            llm_result = self._run_llm_validation(image_path, result)
            result['llm_validation'] = llm_result
        
        return result
    
    def _run_llm_validation(self, image_path: str, pipeline_result: Dict) -> Dict[str, Any]:
        """
        Run LLM-based validation (Step 7) to cross-check pipeline extraction.
        
        Strategy:
        - Sample a few cells from the extracted grid
        - Ask LLM to verify those values directly from the image
        - Compare LLM answers with pipeline extraction
        - Report discrepancies (potential OCR/alignment errors)
        
        Args:
            image_path: Path to the original image
            pipeline_result: Results from Steps 1-6
            
        Returns:
            LLM validation results with accuracy and discrepancies
        """
        if not self.llm_validator:
            return {'enabled': False, 'error': 'LLM validator not initialized'}
        
        grid = pipeline_result.get('grid', [])
        headers = pipeline_result.get('headers', [])
        labels = pipeline_result.get('labels', [])
        
        if not grid or len(grid) < 2:
            return {'enabled': True, 'skipped': True, 'reason': 'Grid too small'}
        
        # Find actual data columns (columns with year labels like "2024", "2023")
        year_columns = []
        for col_idx, header in enumerate(headers):
            if header and any(str(year) in str(header) for year in range(2015, 2030)):
                year_columns.append((col_idx, header))
        
        # If no year columns found, use columns 2 and 3 (typical for financial tables)
        if not year_columns and len(headers) >= 3:
            year_columns = [(2, headers[2] if len(headers) > 2 else 'Column 2'),
                           (3, headers[3] if len(headers) > 3 else 'Column 3')]
        
        # Sample cells for verification (up to 5 cells)
        verification_results = []
        cells_to_check = []
        
        # Select cells with numeric values for verification
        # Skip first few rows which might be headers
        data_start_row = 1
        for row_idx in range(len(grid)):
            row_label = labels[row_idx] if row_idx < len(labels) else ''
            # Skip header-like rows
            if row_label and any(c.isdigit() for c in str(grid[row_idx][1] if len(grid[row_idx]) > 1 else '')):
                data_start_row = row_idx
                break
        
        for row_idx in range(data_start_row, min(len(grid), data_start_row + 6)):
            row_label = labels[row_idx] if row_idx < len(labels) else ''
            # Skip empty or section header rows
            if not row_label or row_label.upper() in ['ASSETS', 'LIABILITIES', 'EQUITY', '']:
                continue
                
            for col_idx, col_label in year_columns[:2]:  # Check first 2 year columns
                if col_idx >= len(grid[row_idx]):
                    continue
                cell_value = grid[row_idx][col_idx]
                if cell_value and any(c.isdigit() for c in str(cell_value)):
                    cells_to_check.append({
                        'row_idx': row_idx,
                        'col_idx': col_idx,
                        'row_label': row_label,
                        'col_label': col_label,
                        'pipeline_value': cell_value
                    })
        
        # Limit to 5 verifications to save API calls
        cells_to_check = cells_to_check[:5]
        
        correct = 0
        for cell in cells_to_check:
            try:
                llm_answer = self.llm_validator.ask_cell_value(
                    image_path,
                    cell['row_label'],
                    cell['col_label']
                )
                
                # Compare values (normalize for comparison)
                pipeline_val = str(cell['pipeline_value']).replace(',', '').replace(' ', '')
                llm_val = str(llm_answer).replace(',', '').replace(' ', '')
                
                match = (pipeline_val == llm_val)
                if match:
                    correct += 1
                
                verification_results.append({
                    'row': cell['row_label'],
                    'column': cell['col_label'],
                    'pipeline_value': cell['pipeline_value'],
                    'llm_value': llm_answer,
                    'match': match
                })
            except Exception as e:
                verification_results.append({
                    'row': cell['row_label'],
                    'column': cell['col_label'],
                    'pipeline_value': cell['pipeline_value'],
                    'llm_value': None,
                    'error': str(e)
                })
        
        total = len(verification_results)
        accuracy = correct / total if total > 0 else 0.0
        
        # Get token usage
        token_usage = self.llm_validator.get_token_usage()
        
        return {
            'enabled': True,
            'model': self.llm_validator.model_name,
            'cells_verified': total,
            'cells_matched': correct,
            'accuracy': accuracy,
            'token_usage': token_usage,
            'details': verification_results,
            'discrepancies': [r for r in verification_results if not r.get('match', False)]
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

    def _normalize_grid(self, grid: List[List[str]]) -> Tuple[List[List[Dict[str, Any]]], List]:
        """
        Normalize grid values using column metadata propagation.
        
        Returns:
            Tuple of (normalized_grid, column_metadata)
        """
        return normalize_grid_with_metadata(grid, header_rows=3)

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
