"""
Process sample financial table images (in data/samples/*.png):
- run structure recognition + OCR + hybrid alignment
- normalize numeric cells
- attempt simple equity rule check per column: total equity ≈ attributable + non-controlling interests
- print summary and save JSON
"""
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.structure import TableStructureRecognizer
from modules.ocr import TableOCR
from modules.numeric import normalize_numeric
from modules.semantic import map_alias
from modules.utils import load_config


def collect_headers(grid: List[List[str]]) -> List[str]:
    if not grid:
        return []
    return [cell for cell in grid[0]]


def collect_row_labels(grid: List[List[str]]) -> List[str]:
    labels = []
    for r in grid:
        labels.append(r[0] if r else '')
    return labels


def normalize_grid(grid: List[List[str]]) -> List[List[Dict[str, Any]]]:
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


def _cluster_centers(values: List[float], threshold: float) -> List[float]:
    centers = []
    for v in sorted(values):
        if not centers or abs(v - centers[-1]) > threshold:
            centers.append(v)
        else:
            centers[-1] = (centers[-1] + v) / 2
    return centers


def build_grid_from_ocr(ocr_results: List[Dict]) -> Tuple[List[List[str]], List[str]]:
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
    col_centers = _cluster_centers(numeric_centers, threshold=col_thresh)

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


def find_row_indices(labels: List[str], key: str) -> List[int]:
    idxs = []
    k = key.lower()
    for i, lbl in enumerate(labels):
        if lbl and k in lbl.lower():
            idxs.append(i)
    return idxs


def equity_rule_check(labels: List[str], norm_grid: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Check total equity ≈ attributable + non-controlling, per column."""
    results = []
    if not norm_grid:
        return results
    num_cols = len(norm_grid[0]) if norm_grid else 0
    # find rows with flexible heuristics
    r_total = find_row_indices(labels, 'total equity')
    r_attr = find_row_indices(labels, 'equity holders') or find_row_indices(labels, 'attributable')
    r_nci = find_row_indices(labels, 'non-controlling')

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


def process_image(path: str, recognizer: TableStructureRecognizer, ocr: TableOCR) -> Dict[str, Any]:
    # ensure consistent 3-channel input for structure model
    image = Image.open(path).convert('RGB')
    structure = recognizer.recognize(image)
    ocr_results = ocr.extract_text(image)
    grid = ocr.align_text_to_grid(
        ocr_results,
        structure.get('rows', []),
        structure.get('columns', []),
        method='hybrid'
    )

    # Fallback: if structure failed to produce a sensible grid, build rows/cols from OCR only
    if not grid or len(grid) < 2 or (grid and len(grid[0]) < 2):
        grid, headers = build_grid_from_ocr(ocr_results)
    else:
        headers = collect_headers(grid)
    labels = collect_row_labels(grid)
    norm_grid = normalize_grid(grid)
    equity_checks = equity_rule_check(labels, norm_grid)

    # Second-chance fallback if rule checks are empty
    if not equity_checks:
        fb_grid, fb_headers = build_grid_from_ocr(ocr_results)
        if fb_grid:
            grid = fb_grid
            headers = fb_headers
            labels = collect_row_labels(grid)
            norm_grid = normalize_grid(grid)
            equity_checks = equity_rule_check(labels, norm_grid)

    # gather examples
    numeric_examples = []
    for row in norm_grid:
        for cell in row:
            if cell.get('value') is not None and len(numeric_examples) < 6:
                numeric_examples.append(cell)
    mapping_examples = []
    for lbl in labels[:8]:
        if lbl:
            mapping_examples.append({'raw': lbl, 'mapped': map_alias(lbl)})
            if len(mapping_examples) >= 6:
                break

    return {
        'file': os.path.basename(path),
        'headers': headers,
        'labels': labels,
        'numeric_examples': numeric_examples,
        'equity_checks': equity_checks,
        'grid': grid,
        'fallback_headers': headers,
    }


def main():
    config = load_config('configs/config.yaml')
    recognizer = TableStructureRecognizer(config, use_v1_1=True)
    ocr = TableOCR(lang='en')

    samples_dir = os.path.join('data', 'samples')
    image_files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
    image_files.sort()

    summaries = []
    for fname in image_files:
        path = os.path.join(samples_dir, fname)
        print(f"\nProcessing {fname}...")
        try:
            result = process_image(path, recognizer, ocr)
            summaries.append(result)
            print(f"  Labels (first 8): {result['labels'][:8]}")
            print(f"  Headers: {result['headers']}")
            if result['equity_checks']:
                for chk in result['equity_checks']:
                    print(f"  Equity check col {chk['column']}: total={chk['total_equity']}, attrib={chk['attributable']}, nci={chk['non_controlling']}, diff={chk['diff']:.2f}, passed={chk['passed']}")
            else:
                print("  Equity check: not enough labels detected")
        except Exception as e:
            print(f"  Error processing {fname}: {e}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join('outputs', 'results', f'financial_samples_demo_{ts}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == '__main__':
    main()
