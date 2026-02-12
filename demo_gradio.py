"""
Financial Document AI - Interactive Visual Demo (Gradio)

ITEX 2026 Demo: Visual pipeline walkthrough for financial table understanding.

Usage:
    python demo_gradio.py
    python demo_gradio.py --share        # Public link for demo
    python demo_gradio.py --port 7861    # Custom port
"""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO

import gradio as gr
import yaml

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Globals (lazy-loaded)
# ─────────────────────────────────────────────────────────────
_pipeline = None
_detector = None
_structure = None
_ocr = None
_config = None

STAGE_COLORS = {
    'detection': '#FF6B6B',
    'structure_row': '#4ECDC4',
    'structure_col': '#45B7D1',
    'structure_header': '#96CEB4',
    'ocr': '#FFEAA7',
    'numeric': '#DDA0DD',
    'semantic': '#98D8C8',
    'validation_pass': '#2ECC71',
    'validation_fail': '#E74C3C',
}

CSS = """
.stage-title { 
    font-size: 1.3em; font-weight: bold; 
    color: #2c3e50; margin-bottom: 8px; 
}
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 15px; border-radius: 10px;
    text-align: center; margin: 5px;
}
.metric-value { font-size: 2em; font-weight: bold; }
.metric-label { font-size: 0.85em; opacity: 0.9; }
.pass-badge { 
    background: #2ECC71; color: white; padding: 3px 10px; 
    border-radius: 12px; font-weight: bold; 
}
.fail-badge { 
    background: #E74C3C; color: white; padding: 3px 10px; 
    border-radius: 12px; font-weight: bold; 
}
footer { display: none !important; }
"""


def load_config():
    global _config
    if _config is None:
        with open('configs/config.yaml', 'r') as f:
            _config = yaml.safe_load(f)
    return _config


def get_pipeline():
    """Lazy-load pipeline (heavy models)."""
    global _pipeline
    if _pipeline is None:
        from modules.pipeline import FinancialTablePipeline
        _pipeline = FinancialTablePipeline(use_v1_1=True, ocr_backend='paddleocr')
    return _pipeline


def get_detector():
    global _detector
    if _detector is None:
        from modules.detection import TableDetector
        _detector = TableDetector(load_config())
    return _detector


def get_structure():
    global _structure
    if _structure is None:
        from modules.structure import TableStructureRecognizer
        _structure = TableStructureRecognizer(load_config(), use_v1_1=True)
    return _structure


def get_ocr():
    global _ocr
    if _ocr is None:
        from modules.ocr import TableOCR
        _ocr = TableOCR(lang='en', use_gpu=False)
    return _ocr


# ─────────────────────────────────────────────────────────────
# Drawing utilities
# ─────────────────────────────────────────────────────────────

def draw_detection_boxes(image: Image.Image, detections: Dict) -> Image.Image:
    """Draw table detection bounding boxes on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    for i, (box, score) in enumerate(zip(detections['boxes'], detections['scores'])):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = '#FF6B6B'
        
        # Draw box with thick border
        for offset in range(3):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)
        
        # Label background
        label = f"Table {i+1}  {score:.1%}"
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2], fill=color)
        draw.text((x1, y1 - 25), label, fill='white', font=font)
    
    return img


def draw_structure_overlay(image: Image.Image, structure: Dict) -> Image.Image:
    """Draw rows, columns, and headers on table image."""
    img = image.copy()
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw rows (horizontal bands)
    for i, row in enumerate(structure.get('rows', [])):
        bbox = row['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (78, 205, 196, 40)  # Teal, semi-transparent
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(78, 205, 196, 180), width=2)
    
    # Draw columns (vertical bands)
    for i, col in enumerate(structure.get('columns', [])):
        bbox = col['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (69, 183, 209, 35)  # Blue, semi-transparent
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(69, 183, 209, 180), width=2)
    
    # Draw column headers
    for header in structure.get('column_headers', []):
        bbox = header['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], fill=(150, 206, 180, 60), outline=(150, 206, 180, 200), width=2)
    
    # Composite
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


def draw_ocr_boxes(image: Image.Image, ocr_results: List[Dict]) -> Image.Image:
    """Draw OCR text detections on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        font = ImageFont.load_default()
    
    for item in ocr_results:
        bbox = item.get('bbox', item.get('box', []))
        text = item.get('text', '')
        conf = item.get('confidence', item.get('score', 0))
        
        if not bbox or len(bbox) < 4:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        
        # Color based on confidence
        if conf > 0.9:
            color = '#2ECC71'  # Green
        elif conf > 0.7:
            color = '#F39C12'  # Orange
        else:
            color = '#E74C3C'  # Red
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Show text above box (truncate if too long)
        display_text = text[:20] + '...' if len(text) > 20 else text
        draw.text((x1, max(0, y1-12)), display_text, fill=color, font=font)
    
    return img


# ─────────────────────────────────────────────────────────────
# Pipeline stage runners
# ─────────────────────────────────────────────────────────────

def run_stage1_detection(image: Image.Image) -> Tuple[Image.Image, str, Dict, Image.Image]:
    """Stage 1: Table Detection → Crop the best table for downstream stages."""
    detector = get_detector()
    t0 = time.time()
    results = detector.detect(image)
    elapsed = time.time() - t0
    
    n_tables = len(results['boxes'])
    scores = results['scores']
    
    vis_image = draw_detection_boxes(image, results)
    
    # Crop the highest-confidence table for Stage 2+
    cropped_table = None
    crop_box = None
    if n_tables > 0:
        best_idx = int(np.argmax(scores))
        box = results['boxes'][best_idx]
        w, h = image.size
        padding = 10
        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(w, int(box[2]) + padding)
        y2 = min(h, int(box[3]) + padding)
        cropped_table = image.crop((x1, y1, x2, y2))
        crop_box = [x1, y1, x2, y2]
    
    info_lines = [
        f"### Stage 1: Table Detection",
        f"**Model:** Microsoft Table Transformer (DETR)",
        f"**Tables detected:** {n_tables}",
        f"**Inference time:** {elapsed:.2f}s",
        ""
    ]
    for i, (box, score) in enumerate(zip(results['boxes'], scores)):
        x1, y1, x2, y2 = [int(v) for v in box]
        best_marker = " ⭐ (selected)" if n_tables > 1 and i == int(np.argmax(scores)) else ""
        info_lines.append(f"| Table {i+1} | Confidence: **{score:.1%}** | Box: ({x1}, {y1}) → ({x2}, {y2}) |{best_marker}")
    
    if cropped_table is not None:
        cw, ch = cropped_table.size
        info_lines.append(f"\n**Cropped table region:** {cw} × {ch} px → passed to Stage 2 (TSR)")
    
    return vis_image, "\n".join(info_lines), results, cropped_table


def run_stage2_tsr(table_image: Image.Image) -> Tuple[Image.Image, str, Dict]:
    """Stage 2: Table Structure Recognition (runs on CROPPED table)."""
    recognizer = get_structure()
    t0 = time.time()
    structure = recognizer.recognize(table_image)
    elapsed = time.time() - t0
    
    n_rows = len(structure.get('rows', []))
    n_cols = len(structure.get('columns', []))
    n_headers = len(structure.get('column_headers', []))
    
    vis_image = draw_structure_overlay(table_image, structure)
    
    tw, th = table_image.size
    info_lines = [
        f"### Stage 2: Table Structure Recognition (TSR)",
        f"**Model:** Table Transformer v1.1-all",
        f"**Input:** Cropped table region ({tw} × {th} px)",
        f"**Grid detected:** {n_rows} rows × {n_cols} columns",
        f"**Column headers:** {n_headers}",
        f"**Inference time:** {elapsed:.2f}s",
        "",
        "| Element | Count | Avg Confidence |",
        "|---------|-------|----------------|",
    ]
    for key in ['rows', 'columns', 'column_headers']:
        items = structure.get(key, [])
        if items:
            avg_conf = np.mean([it['score'] for it in items])
            info_lines.append(f"| {key.replace('_', ' ').title()} | {len(items)} | {avg_conf:.1%} |")
    
    return vis_image, "\n".join(info_lines), structure


def run_stage3_ocr(table_image: Image.Image, structure: Dict) -> Tuple[Image.Image, str, List, List]:
    """Stage 3: OCR Extraction + Grid Alignment (runs on CROPPED table)."""
    ocr = get_ocr()
    
    # Save cropped table to temp for OCR (PaddleOCR needs file path)
    tmp_crop_path = os.path.join('outputs', '_demo_crop_temp.png')
    os.makedirs('outputs', exist_ok=True)
    table_image.save(tmp_crop_path)
    
    t0 = time.time()
    ocr_results = ocr.extract_text(tmp_crop_path)
    elapsed_ocr = time.time() - t0
    
    t1 = time.time()
    grid = ocr.align_text_to_grid(
        ocr_results,
        structure.get('rows', []),
        structure.get('columns', []),
        method='hybrid'
    )
    elapsed_align = time.time() - t1
    
    # Apply OCR corrections
    from modules.semantic import correct_ocr_text
    if grid:
        for row in grid:
            if row and row[0]:
                row[0] = correct_ocr_text(row[0])
    
    vis_image = draw_ocr_boxes(table_image, ocr_results)
    
    n_words = len(ocr_results)
    n_rows = len(grid) if grid else 0
    n_cols = len(grid[0]) if grid and grid[0] else 0
    tw, th = table_image.size
    
    info_lines = [
        f"### Stage 3: OCR Text Extraction",
        f"**Engine:** PaddleOCR v2.7",
        f"**Input:** Cropped table region ({tw} × {th} px)",
        f"**Words detected:** {n_words}",
        f"**Grid aligned:** {n_rows} × {n_cols}",
        f"**OCR time:** {elapsed_ocr:.2f}s | **Alignment time:** {elapsed_align:.2f}s",
    ]
    
    # Clean up
    try:
        os.remove(tmp_crop_path)
    except:
        pass
    
    return vis_image, "\n".join(info_lines), ocr_results, grid


def run_stage4_numeric(grid: List[List[str]]) -> Tuple[str, List, List]:
    """Stage 4: Numeric Normalisation."""
    from modules.numeric import normalize_grid_with_metadata
    
    t0 = time.time()
    norm_grid, col_metadata = normalize_grid_with_metadata(grid, header_rows=3)
    elapsed = time.time() - t0
    
    # Build display table
    rows_data = []
    for r_idx, (raw_row, norm_row) in enumerate(zip(grid, norm_grid)):
        for c_idx, (raw_cell, norm_cell) in enumerate(zip(raw_row, norm_row)):
            if norm_cell.get('cell_type') == 'numeric' and norm_cell.get('value') is not None:
                rows_data.append({
                    'Row': r_idx,
                    'Col': c_idx,
                    'Raw Text': raw_cell,
                    'Parsed Value': f"{norm_cell['value']:,.0f}",
                    'Currency': norm_cell.get('currency', ''),
                    'Scale': f"×{norm_cell.get('scale', 1):,}",
                    'Type': norm_cell.get('cell_type', ''),
                })
    
    # Column metadata summary
    meta_lines = [
        f"### Stage 4: Numeric Normalisation",
        f"**Method:** Rule-based parser",
        f"**Processing time:** {elapsed*1000:.1f}ms",
        f"**Numeric cells found:** {len(rows_data)}",
        "",
        "#### Column Metadata (auto-detected)",
        "| Column | Type | Currency | Scale | Year |",
        "|--------|------|----------|-------|------|",
    ]
    for i, meta in enumerate(col_metadata):
        meta_lines.append(
            f"| {i} | {meta.column_type} | {meta.currency or '-'} | ×{meta.scale:,} | {meta.year or '-'} |"
        )
    
    return "\n".join(meta_lines), norm_grid, col_metadata


def run_stage5_semantic(grid: List[List[str]], norm_grid: List) -> str:
    """Stage 5: Semantic Classification."""
    from modules.semantic import map_alias, correct_financial_label
    from modules.validation import classify_all_rows
    
    t0 = time.time()
    
    # Extract labels
    labels = [row[0] if row else '' for row in grid]
    
    # Get value grid for row classification
    value_grid = []
    for row in norm_grid:
        value_row = []
        for cell in row:
            value_row.append(cell.get('value') if isinstance(cell, dict) else None)
        value_grid.append(value_row)
    
    row_classes = classify_all_rows(labels, value_grid)
    
    # Semantic mapping
    mappings = []
    for lbl in labels:
        if lbl:
            mapped = map_alias(lbl)
            corrected = correct_financial_label(lbl)
            mappings.append({
                'label': lbl,
                'corrected': corrected if corrected != lbl else '-',
                'semantic_key': mapped,
            })
    
    elapsed = time.time() - t0
    
    lines = [
        f"### Stage 5: Semantic Classification",
        f"**Method:** Heuristic classifier + alias dictionary",
        f"**Processing time:** {elapsed*1000:.1f}ms",
        "",
        "#### Row Classifications",
        "| Row | Label | Class | Semantic Key |",
        "|-----|-------|-------|-------------|",
    ]
    
    for i, (lbl, cls) in enumerate(zip(labels, row_classes)):
        mapped = map_alias(lbl) if lbl else '-'
        cls_badge = cls
        if cls in ('total', 'subtotal'):
            cls_badge = f"**{cls}**"
        elif cls == 'header':
            cls_badge = f"*{cls}*"
        lines.append(f"| {i} | {lbl[:40] if lbl else '(empty)'} | {cls_badge} | {mapped} |")
    
    return "\n".join(lines), row_classes


def run_stage6_validation(grid: List, norm_grid: List, labels: List[str], 
                          row_classes: List[str]) -> str:
    """Stage 6: Rule-based Validation."""
    from modules.validation import TableValidator
    
    t0 = time.time()
    validator = TableValidator(tolerance=0.02)
    
    # Extract value grid
    value_grid = []
    for row in norm_grid:
        value_row = []
        for cell in row:
            value_row.append(cell.get('value') if isinstance(cell, dict) else None)
        value_grid.append(value_row)
    
    results = validator.validate_column_sums_enhanced(value_grid, labels, row_classes)
    elapsed = time.time() - t0
    
    passed = sum(1 for r in results if r.get('passed'))
    failed = sum(1 for r in results if not r.get('passed'))
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    lines = [
        f"### Stage 6: Financial Validation",
        f"**Method:** Column-sum consistency checks",
        f"**Processing time:** {elapsed*1000:.1f}ms",
        "",
    ]
    
    # Summary metrics
    if pass_rate == 1.0:
        lines.append(f"### ✅ All {total} validation rules PASSED ({pass_rate:.0%})")
    elif pass_rate > 0.5:
        lines.append(f"### ⚠️ {passed}/{total} rules passed ({pass_rate:.0%})")
    else:
        lines.append(f"### ❌ {passed}/{total} rules passed ({pass_rate:.0%})")
    
    lines.extend(["", "#### Detailed Results",
                   "| # | Rule | Column | Expected | Actual | Diff | Status |",
                   "|---|------|--------|----------|--------|------|--------|"])
    
    for i, r in enumerate(results):
        status = "✅ PASS" if r.get('passed') else "❌ FAIL"
        expected = f"{r.get('expected', 0):,.0f}"
        actual = f"{r.get('actual', 0):,.0f}"
        diff = f"{r.get('diff', 0):,.0f}"
        rule_type = r.get('row_type', 'sum')
        total_row = r.get('total_row', '')[:30]
        col = r.get('column', '')
        lines.append(f"| {i+1} | {rule_type} | Col {col} | {expected} | {actual} | {diff} | {status} |")
        
        # Show component rows for first few
        if i < 3 and 'component_rows' in r:
            comp_count = r.get('component_count', len(r['component_rows']))
            lines.append(f"|   | ↳ Components: {comp_count} rows → **{total_row}** | | | | | |")
    
    return "\n".join(lines), results


# ─────────────────────────────────────────────────────────────
# Grid display helpers
# ─────────────────────────────────────────────────────────────

def grid_to_html_table(grid: List[List[str]], norm_grid: List = None, 
                       row_classes: List[str] = None) -> str:
    """Convert grid to styled HTML table."""
    if not grid:
        return "<p>No grid data</p>"
    
    # Cell type → color mapping
    type_colors = {
        'numeric': '#E8F5E9',
        'header': '#E3F2FD',
        'year': '#FFF3E0',
        'unit_header': '#FCE4EC',
        'note': '#F3E5F5',
        'text': '#FAFAFA',
        'empty': '#F5F5F5',
    }
    
    row_class_colors = {
        'header': '#BBDEFB',
        'total': '#C8E6C9',
        'subtotal': '#DCEDC8',
        'section_header': '#FFE0B2',
        'data': 'transparent',
        'empty': '#F5F5F5',
    }
    
    html = ['<div style="overflow-x:auto; max-height:500px; overflow-y:auto; background:#fff; border-radius:8px; padding:4px;">']
    html.append('<table style="border-collapse:collapse; width:100%; font-size:13px; font-family:Consolas,monospace; color:#222;">')
    
    for r_idx, row in enumerate(grid):
        # Row background based on classification
        row_bg = 'transparent'
        if row_classes and r_idx < len(row_classes):
            row_bg = row_class_colors.get(row_classes[r_idx], 'transparent')
        
        html.append(f'<tr style="background:{row_bg};">')
        
        for c_idx, cell in enumerate(row):
            # Cell styling
            cell_bg = 'transparent'
            font_weight = 'normal'
            text_align = 'left'
            
            if norm_grid and r_idx < len(norm_grid) and c_idx < len(norm_grid[r_idx]):
                nc = norm_grid[r_idx][c_idx]
                if isinstance(nc, dict):
                    ct = nc.get('cell_type', '')
                    cell_bg = type_colors.get(ct, 'transparent')
                    if ct == 'numeric':
                        text_align = 'right'
                    if nc.get('value') is not None and nc.get('scale', 1) > 1:
                        font_weight = 'bold'
            
            # Total/subtotal rows get bold
            if row_classes and r_idx < len(row_classes):
                if row_classes[r_idx] in ('total', 'subtotal'):
                    font_weight = 'bold'
            
            display = cell if cell else ''
            html.append(
                f'<td style="border:1px solid #ccc; padding:4px 8px; '
                f'background:{cell_bg}; color:#222; font-weight:{font_weight}; '
                f'text-align:{text_align}; white-space:nowrap;">{display}</td>'
            )
        
        html.append('</tr>')
    
    html.append('</table></div>')
    return '\n'.join(html)


def format_metrics_html(metrics: Dict[str, Any]) -> str:
    """Create metric cards HTML."""
    cards = []
    for label, value in metrics.items():
        if isinstance(value, float):
            display = f"{value:.1%}" if value <= 1 else f"{value:,.0f}"
        else:
            display = str(value)
        cards.append(
            f'<div style="display:inline-block; background:linear-gradient(135deg,#667eea,#764ba2); '
            f'color:white; padding:12px 20px; border-radius:10px; margin:4px; text-align:center; min-width:100px;">'
            f'<div style="font-size:1.8em; font-weight:bold;">{display}</div>'
            f'<div style="font-size:0.8em; opacity:0.9;">{label}</div></div>'
        )
    return '<div style="display:flex; flex-wrap:wrap; gap:5px;">' + ''.join(cards) + '</div>'


# ─────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────

def process_full_pipeline(image_input, progress=gr.Progress()):
    """Run the full 6-stage pipeline with visual output for each stage."""
    
    if image_input is None:
        return [None]*4 + [""]*7 + [""]
    
    # Save uploaded image to temp path
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert('RGB')
    elif isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')
    
    total_start = time.time()
    
    # ── Stage 1: Detection ──
    progress(0.05, desc="Stage 1: Detecting tables...")
    det_image, det_info, det_results, cropped_table = run_stage1_detection(image)
    
    # Use cropped table for subsequent stages; fall back to full image if detection fails
    if cropped_table is None:
        cropped_table = image  # fallback: treat entire image as table
    
    # ── Stage 2: TSR (on CROPPED table) ──
    progress(0.20, desc="Stage 2: Recognizing structure...")
    tsr_image, tsr_info, structure = run_stage2_tsr(cropped_table)
    
    # ── Stage 3: OCR (on CROPPED table) ──
    progress(0.40, desc="Stage 3: Extracting text (OCR)...")
    ocr_image, ocr_info, ocr_results, grid = run_stage3_ocr(cropped_table, structure)
    
    if not grid or len(grid) < 2:
        return (
            det_image, tsr_image, ocr_image, None,
            det_info, tsr_info, ocr_info,
            "⚠️ Grid extraction failed - insufficient structure detected",
            "", "", "",
            format_metrics_html({'Tables': 0, 'Status': 'Failed'})
        )
    
    # ── Stage 4: Numeric ──
    progress(0.60, desc="Stage 4: Normalising numbers...")
    num_info, norm_grid, col_metadata = run_stage4_numeric(grid)
    
    # ── Stage 5: Semantic ──
    progress(0.75, desc="Stage 5: Classifying cells...")
    labels = [row[0] if row else '' for row in grid]
    sem_info, row_classes = run_stage5_semantic(grid, norm_grid)
    
    # ── Stage 6: Validation ──
    progress(0.90, desc="Stage 6: Validating results...")
    val_info, val_results = run_stage6_validation(grid, norm_grid, labels, row_classes)
    
    total_elapsed = time.time() - total_start
    
    # Build grid HTML
    grid_html = grid_to_html_table(grid, norm_grid, row_classes)
    
    # Summary metrics
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    n_numeric = sum(1 for row in norm_grid for cell in row 
                    if isinstance(cell, dict) and cell.get('cell_type') == 'numeric')
    val_passed = sum(1 for r in val_results if r.get('passed'))
    val_total = len(val_results)
    
    # Currency info
    currencies = set()
    for meta in col_metadata:
        if meta.currency:
            currencies.add(meta.currency)
    currency_str = ', '.join(currencies) if currencies else 'N/A'
    
    metrics = {
        'Grid': f"{n_rows}×{n_cols}",
        'Numeric Cells': n_numeric,
        'Currency': currency_str,
        'Validation': f"{val_passed}/{val_total}",
        'Time': f"{total_elapsed:.1f}s",
    }
    metrics_html = format_metrics_html(metrics)
    
    progress(1.0, desc="Complete!")
    
    return (
        det_image,      # Stage 1 image
        tsr_image,      # Stage 2 image
        ocr_image,      # Stage 3 image
        grid_html,      # Extracted table
        det_info,       # Stage 1 info
        tsr_info,       # Stage 2 info
        ocr_info,       # Stage 3 info
        num_info,       # Stage 4 info
        sem_info,       # Stage 5 info
        val_info,       # Stage 6 info
        metrics_html,   # Summary
    )


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(
        title="Financial Document AI",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        js="() => { document.querySelector('body').classList.remove('dark'); }",
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 📊 Financial Document AI
        ### Automated Financial Table Understanding Pipeline
        
        Upload a financial document image (annual report, balance sheet, income statement) 
        to see the **6-stage extraction pipeline** in action.
        
        **Pipeline:** Table Detection → Structure Recognition → OCR → Numeric Normalisation → Semantic Classification → Validation
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📄 Upload Financial Document Image",
                    type="numpy",
                    height=350,
                )
                
                # Sample images
                sample_dir = Path("data/samples")
                samples = []
                if sample_dir.exists():
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        samples.extend(sorted(sample_dir.glob(ext)))
                
                if samples:
                    gr.Examples(
                        examples=[[str(s)] for s in samples[:6]],
                        inputs=image_input,
                        label="📁 Sample Documents",
                    )
                
                run_btn = gr.Button(
                    "🚀 Run Full Pipeline", 
                    variant="primary", 
                    size="lg",
                )
            
            with gr.Column(scale=1):
                summary_html = gr.HTML(
                    label="Pipeline Summary",
                    value='<div style="text-align:center; color:#999; padding:40px;">Upload an image and click "Run Full Pipeline" to begin</div>'
                )
        
        gr.Markdown("---")
        
        # Stage outputs in tabs
        with gr.Tabs() as tabs:
            
            # Stage 1
            with gr.Tab("1️⃣ Detection", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        det_output = gr.Image(label="Detection Result", height=400)
                    with gr.Column(scale=1):
                        det_info = gr.Markdown("*Waiting for input...*")
            
            # Stage 2
            with gr.Tab("2️⃣ Structure (TSR)", id=1):
                with gr.Row():
                    with gr.Column(scale=2):
                        tsr_output = gr.Image(label="Structure Recognition", height=400)
                    with gr.Column(scale=1):
                        tsr_info = gr.Markdown("*Waiting for input...*")
            
            # Stage 3
            with gr.Tab("3️⃣ OCR", id=2):
                with gr.Row():
                    with gr.Column(scale=2):
                        ocr_output = gr.Image(label="OCR Text Detection", height=400)
                    with gr.Column(scale=1):
                        ocr_info = gr.Markdown("*Waiting for input...*")
            
            # Stage 4
            with gr.Tab("4️⃣ Numeric", id=3):
                num_info = gr.Markdown("*Waiting for input...*")
            
            # Stage 5
            with gr.Tab("5️⃣ Semantic", id=4):
                sem_info = gr.Markdown("*Waiting for input...*")
            
            # Stage 6
            with gr.Tab("6️⃣ Validation", id=5):
                val_info = gr.Markdown("*Waiting for input...*")
            
            # Extracted Table
            with gr.Tab("📋 Extracted Table", id=6):
                grid_html = gr.HTML(
                    value='<div style="text-align:center; color:#999; padding:40px;">No table extracted yet</div>'
                )
        
        # Wire up
        run_btn.click(
            fn=process_full_pipeline,
            inputs=[image_input],
            outputs=[
                det_output, tsr_output, ocr_output,  # Images
                grid_html,                             # Table HTML
                det_info, tsr_info, ocr_info,          # Stage 1-3 info
                num_info, sem_info, val_info,           # Stage 4-6 info
                summary_html,                          # Summary
            ],
            show_progress="full",
        )
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        
        **Financial Document AI** | FYP Project | Powered by Table Transformer, PaddleOCR, and Rule-based NLP
        
        *Detection: Table Transformer (DETR) • TSR: TT v1.1 • OCR: PaddleOCR v2.7 • Validation: Column-Sum Rules*
        
        </center>
        """)
    
    return demo


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Financial Document AI - Gradio Demo')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--share', action='store_true', help='Create public share link')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Financial Document AI - Interactive Demo")
    print("  ITEX 2026 Demonstration")
    print("=" * 60)
    
    demo = build_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
