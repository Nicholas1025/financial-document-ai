"""
End-to-End Pipeline Demo: Step 1-5
==================================

This script runs the complete pipeline on a single image:
- Step 1: Table Detection (Docling)
- Step 2: Table Structure Recognition (Table Transformer v1.1)
- Step 3: OCR (PaddleOCR via subprocess to avoid CUDA conflicts)
- Step 4: Numeric Normalization
- Step 5: Semantic Classification

Usage:
    python end_to_end_demo.py --image path/to/image.png [--gt path/to/gt.json]
"""

import os
import sys
import json
import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Step 1: Table Detection
# ============================================================================

def run_step1_detection(image_path: str) -> List[Dict]:
    """Detect tables in the image using Docling LayoutPredictor."""
    print("\n" + "="*60)
    print("STEP 1: TABLE DETECTION")
    print("="*60)
    
    from PIL import Image
    from docling.models.layout_model import LayoutModel
    from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
    
    # Load model - use official download method
    print("Loading Docling LayoutPredictor...")
    artifact_path = LayoutModel.download_models()
    predictor = LayoutPredictor(
        artifact_path=str(artifact_path),
        device="cpu",
        num_threads=4
    )
    print("Docling LayoutPredictor loaded successfully")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Image size: {width} x {height}")
    
    # Predict - convert image to numpy for predictor
    import numpy as np
    image_np = np.array(image)
    predictions = list(predictor.predict(image_np))
    
    # Filter for tables (prediction is a dict with 'label', 'confidence', 'l', 't', 'r', 'b')
    tables = []
    for pred in predictions:
        label = pred['label'].lower() if isinstance(pred, dict) else pred.label.lower()
        confidence = pred['confidence'] if isinstance(pred, dict) else pred.confidence
        
        if label == "table":
            if isinstance(pred, dict):
                bbox = [pred['l'], pred['t'], pred['r'], pred['b']]
            else:
                bbox = [pred.l, pred.t, pred.r, pred.b]
            tables.append({
                "bbox": bbox,
                "confidence": confidence,
                "label": label
            })
    
    print(f"Detected {len(tables)} table(s)")
    for i, t in enumerate(tables):
        print(f"  Table {i+1}: bbox={t['bbox']}, conf={t['confidence']:.3f}")
    
    return tables, image


# ============================================================================
# Step 2: Table Structure Recognition
# ============================================================================

def run_step2_tsr(image: Image.Image, table_bbox: List[float]) -> Dict:
    """Recognize table structure using Table Transformer v1.1."""
    print("\n" + "="*60)
    print("STEP 2: TABLE STRUCTURE RECOGNITION")
    print("="*60)
    
    import torch
    import torchvision.transforms as T
    from transformers import TableTransformerForObjectDetection
    
    # Crop table region
    x1, y1, x2, y2 = [int(c) for c in table_bbox]
    table_crop = image.crop((x1, y1, x2, y2))
    crop_width, crop_height = table_crop.size
    print(f"Table crop size: {crop_width} x {crop_height}")
    
    # Load model (without processor - v1.1 has issues with AutoImageProcessor)
    print("Loading Table Transformer v1.1...")
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition-v1.1-all"
    )
    model.eval()
    
    # Manual preprocessing for v1.1 model
    max_size = 800
    w, h = table_crop.size
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = table_crop.resize((new_w, new_h), Image.BILINEAR)
    
    # Normalize with DETR defaults
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_tensor = normalize(T.ToTensor()(resized)).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
    
    # Manual post-processing
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes
    
    # Get predictions above threshold
    probs = logits.softmax(-1)[0, :, :-1]  # Remove no-object class
    scores, labels = probs.max(-1)
    keep = scores > 0.5
    
    scores = scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[0][keep]
    
    # Convert boxes from cxcywh to xyxy
    def box_cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    boxes = box_cxcywh_to_xyxy(boxes)
    
    # Scale to original crop size
    scale_factor = torch.tensor([crop_width, crop_height, crop_width, crop_height])
    boxes = boxes * scale_factor
    
    # Extract cells
    cells = []
    rows = []
    cols = []
    
    for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        label_name = model.config.id2label[label]
        
        if label_name == "table cell":
            cells.append({
                "bbox": box,
                "confidence": score
            })
        elif label_name == "table row":
            rows.append({"bbox": box, "confidence": score})
        elif label_name == "table column":
            cols.append({"bbox": box, "confidence": score})
    
    # Sort rows by y-coordinate
    rows = sorted(rows, key=lambda r: r["bbox"][1])
    cols = sorted(cols, key=lambda c: c["bbox"][0])
    
    print(f"Detected: {len(rows)} rows, {len(cols)} columns")
    
    # Generate cells from row/column intersections (v1.1 doesn't output cells directly)
    if len(cells) == 0 and len(rows) > 0 and len(cols) > 0:
        print("Generating cells from row/column intersections...")
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                # Cell bbox is intersection of row and column
                cell_bbox = [
                    max(row["bbox"][0], col["bbox"][0]),  # x1
                    row["bbox"][1],                        # y1 (from row)
                    min(row["bbox"][2], col["bbox"][2]),  # x2
                    row["bbox"][3]                         # y2 (from row)
                ]
                cells.append({
                    "bbox": cell_bbox,
                    "row": row_idx,
                    "col": col_idx,
                    "confidence": (row["confidence"] + col["confidence"]) / 2
                })
        print(f"Generated {len(cells)} cells from {len(rows)} rows x {len(cols)} cols")
    else:
        # Assign cells to rows/cols if cells were detected directly
        for cell in cells:
            cx = (cell["bbox"][0] + cell["bbox"][2]) / 2
            cy = (cell["bbox"][1] + cell["bbox"][3]) / 2
            
            # Find row
            cell["row"] = 0
            for i, row in enumerate(rows):
                if row["bbox"][1] <= cy <= row["bbox"][3]:
                    cell["row"] = i
                    break
            
            # Find col
            cell["col"] = 0
            for i, col in enumerate(cols):
                if col["bbox"][0] <= cx <= col["bbox"][2]:
                    cell["col"] = i
                    break
    
    return {
        "num_rows": len(rows),
        "num_cols": len(cols),
        "cells": cells,
        "rows": rows,
        "cols": cols,
        "table_crop": table_crop,
        "offset": (x1, y1)  # Offset for converting back to original coords
    }


# ============================================================================
# Step 3: OCR
# ============================================================================

def run_step3_ocr(table_crop: Image.Image, cells: List[Dict]) -> List[Dict]:
    """Extract text from cells using PaddleOCR via subprocess (avoids CUDA conflicts)."""
    print("\n" + "="*60)
    print("STEP 3: OCR")
    print("="*60)
    
    import tempfile
    import subprocess
    import json
    import numpy as np
    
    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_image_path = f.name
        table_crop.save(temp_image_path)
    
    # Create OCR worker script
    ocr_script = '''
import sys
import json
import os
# Force CPU mode for PaddlePaddle
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
result = ocr.ocr(image_np, cls=True)

ocr_boxes = []
if result and result[0]:
    for line in result[0]:
        bbox_points = line[0]
        text = line[1][0]
        conf = line[1][1]
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        ocr_boxes.append({"bbox": bbox, "text": text, "confidence": conf})

print(json.dumps(ocr_boxes))
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_script_path = f.name
        f.write(ocr_script)
    
    try:
        # Run OCR in subprocess
        print("Running PaddleOCR in subprocess...")
        python_path = sys.executable
        result = subprocess.run(
            [python_path, temp_script_path, temp_image_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"OCR subprocess error: {result.stderr}")
            # Try to extract any JSON from stdout anyway
            ocr_boxes = []
        else:
            # Parse JSON output - find the last complete JSON array (OCR results)
            output = result.stdout.strip()
            # Get just the last line which should be our JSON output
            lines = output.strip().split('\n')
            json_line = lines[-1] if lines else ''
            
            try:
                ocr_boxes = json.loads(json_line)
            except json.JSONDecodeError:
                # Try to find any JSON array in output
                import re
                arrays = re.findall(r'\[.*?\]', output, re.DOTALL)
                if arrays:
                    # Get the longest array (likely the OCR results)
                    longest = max(arrays, key=len)
                    try:
                        ocr_boxes = json.loads(longest)
                    except:
                        ocr_boxes = []
                else:
                    ocr_boxes = []
        
        print(f"OCR detected {len(ocr_boxes)} text regions")
        
    except subprocess.TimeoutExpired:
        print("OCR timeout - using empty results")
        ocr_boxes = []
    finally:
        # Cleanup temp files
        import os
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    
    # Match OCR boxes to cells
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    # For each cell, find overlapping OCR boxes and merge text
    for cell in cells:
        cell_bbox = cell["bbox"]
        matched_texts = []
        
        for ocr_box in ocr_boxes:
            # Check if OCR box overlaps with cell
            overlap = iou(cell_bbox, ocr_box["bbox"])
            if overlap > 0.1:  # 10% overlap threshold
                matched_texts.append((ocr_box["bbox"][0], ocr_box["text"]))
        
        # Sort by x-coordinate and join
        matched_texts.sort(key=lambda x: x[0])
        cell["text"] = " ".join([t[1] for t in matched_texts])
    
    # Print sample
    print("\nSample cell texts:")
    for cell in cells[:5]:
        print(f"  Row {cell.get('row', '?')}, Col {cell.get('col', '?')}: '{cell.get('text', '')}'")
    
    return cells


# ============================================================================
# Step 4: Numeric Normalization
# ============================================================================

def run_step4_numeric(cells: List[Dict]) -> List[Dict]:
    """Parse and normalize numeric values."""
    print("\n" + "="*60)
    print("STEP 4: NUMERIC NORMALIZATION")
    print("="*60)
    
    def parse_numeric(text: str) -> Optional[float]:
        """Parse a numeric string to float."""
        if not text or text.strip() in ['', '-', '—', '–']:
            return None
        
        text = text.strip()
        
        # Check for percentage
        is_percentage = '%' in text
        
        # Check for negative (parentheses)
        is_negative = text.startswith('(') and text.endswith(')')
        if is_negative:
            text = text[1:-1]
        
        # Remove currency symbols and whitespace
        text = re.sub(r'[RM$£€¥,\s]', '', text)
        text = text.replace("'", "")
        
        # Try to parse
        try:
            value = float(text)
            if is_negative:
                value = -value
            if is_percentage:
                value = value / 100
            return value
        except ValueError:
            return None
    
    numeric_count = 0
    for cell in cells:
        text = cell.get("text", "")
        value = parse_numeric(text)
        cell["numeric_value"] = value
        cell["is_numeric"] = value is not None
        if value is not None:
            numeric_count += 1
    
    print(f"Parsed {numeric_count} numeric values out of {len(cells)} cells")
    
    # Print sample
    print("\nSample numeric values:")
    for cell in cells:
        if cell.get("is_numeric"):
            print(f"  '{cell['text']}' → {cell['numeric_value']}")
            if sum(1 for c in cells if c.get('is_numeric')) > 5:
                break
    
    return cells


# ============================================================================
# Step 5: Semantic Classification
# ============================================================================

def run_step5_semantic(cells: List[Dict], num_rows: int, num_cols: int) -> List[Dict]:
    """Classify cells into semantic types."""
    print("\n" + "="*60)
    print("STEP 5: SEMANTIC CLASSIFICATION")
    print("="*60)
    
    for cell in cells:
        row = cell.get("row", 0)
        col = cell.get("col", 0)
        text = cell.get("text", "").lower()
        is_numeric = cell.get("is_numeric", False)
        
        # Classification rules
        if row == 0:
            cell["semantic_type"] = "column_header"
        elif col == 0:
            # Check for section titles or totals
            if any(kw in text for kw in ['total', 'assets', 'liabilities', 'equity']):
                if 'total' in text:
                    cell["semantic_type"] = "total_row"
                else:
                    cell["semantic_type"] = "section_title"
            else:
                cell["semantic_type"] = "row_header"
        elif "rm'000" in text or "rm" in text and "000" in text:
            cell["semantic_type"] = "currency_unit"
        elif is_numeric:
            cell["semantic_type"] = "data"
        else:
            cell["semantic_type"] = "data"
    
    # Count types
    type_counts = {}
    for cell in cells:
        t = cell.get("semantic_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("Semantic type distribution:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    
    return cells


# ============================================================================
# Evaluation against Ground Truth
# ============================================================================

def evaluate_against_gt(cells: List[Dict], gt: Dict) -> Dict:
    """Compare pipeline output against ground truth."""
    print("\n" + "="*60)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("="*60)
    
    results = {
        "ocr": {"correct": 0, "total": 0, "cer": []},
        "numeric": {"correct": 0, "total": 0},
        "semantic": {"correct": 0, "total": 0}
    }
    
    gt_cells = gt.get("cells", [])
    
    # Build a lookup by (row, col)
    pred_lookup = {}
    for cell in cells:
        key = (cell.get("row", -1), cell.get("col", -1))
        pred_lookup[key] = cell
    
    for gt_cell in gt_cells:
        row = gt_cell.get("row")
        col = gt_cell.get("col")
        key = (row, col)
        
        if key not in pred_lookup:
            continue
        
        pred_cell = pred_lookup[key]
        
        # OCR evaluation
        gt_text = gt_cell.get("text", "").strip()
        pred_text = pred_cell.get("text", "").strip()
        if gt_text:
            results["ocr"]["total"] += 1
            if gt_text == pred_text:
                results["ocr"]["correct"] += 1
            # Calculate CER
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, gt_text, pred_text).ratio()
            results["ocr"]["cer"].append(1 - ratio)
        
        # Numeric evaluation
        gt_value = gt_cell.get("numeric_value")
        pred_value = pred_cell.get("numeric_value")
        if gt_value is not None:
            results["numeric"]["total"] += 1
            if pred_value is not None and abs(gt_value - pred_value) < 0.01:
                results["numeric"]["correct"] += 1
        
        # Semantic evaluation
        gt_type = gt_cell.get("semantic_type")
        pred_type = pred_cell.get("semantic_type")
        if gt_type:
            results["semantic"]["total"] += 1
            if gt_type == pred_type:
                results["semantic"]["correct"] += 1
    
    # Calculate metrics
    print("\nResults:")
    
    if results["ocr"]["total"] > 0:
        ocr_acc = results["ocr"]["correct"] / results["ocr"]["total"] * 100
        avg_cer = np.mean(results["ocr"]["cer"]) * 100 if results["ocr"]["cer"] else 0
        print(f"  OCR Exact Match: {results['ocr']['correct']}/{results['ocr']['total']} ({ocr_acc:.1f}%)")
        print(f"  OCR Avg CER: {avg_cer:.1f}%")
    
    if results["numeric"]["total"] > 0:
        num_acc = results["numeric"]["correct"] / results["numeric"]["total"] * 100
        print(f"  Numeric Match: {results['numeric']['correct']}/{results['numeric']['total']} ({num_acc:.1f}%)")
    
    if results["semantic"]["total"] > 0:
        sem_acc = results["semantic"]["correct"] / results["semantic"]["total"] * 100
        print(f"  Semantic Match: {results['semantic']['correct']}/{results['semantic']['total']} ({sem_acc:.1f}%)")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--gt", type=str, help="Path to ground truth JSON (optional)")
    parser.add_argument("--output", type=str, help="Path to save output JSON")
    args = parser.parse_args()
    
    print("="*60)
    print("END-TO-END PIPELINE DEMO")
    print("="*60)
    print(f"Image: {args.image}")
    
    # Step 1: Detection
    tables, image = run_step1_detection(args.image)
    
    if not tables:
        print("\nNo tables detected! Exiting.")
        return
    
    # Use the first (or largest) table
    table = tables[0]
    
    # Step 2: TSR
    structure = run_step2_tsr(image, table["bbox"])
    
    # Step 3: OCR
    cells = run_step3_ocr(structure["table_crop"], structure["cells"])
    
    # Step 4: Numeric
    cells = run_step4_numeric(cells)
    
    # Step 5: Semantic
    cells = run_step5_semantic(cells, structure["num_rows"], structure["num_cols"])
    
    # Build output
    output = {
        "image": args.image,
        "table": {
            "bbox": table["bbox"],
            "num_rows": structure["num_rows"],
            "num_cols": structure["num_cols"],
            "cells": cells
        }
    }
    
    # Evaluate against GT if provided
    if args.gt:
        with open(args.gt, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        evaluate_against_gt(cells, gt)
    
    # Save output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            # Remove non-serializable items
            output_clean = json.loads(json.dumps(output, default=str))
            json.dump(output_clean, f, indent=2, ensure_ascii=False)
        print(f"\nOutput saved to: {args.output}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL OUTPUT SUMMARY")
    print("="*60)
    print(f"Table bbox: {table['bbox']}")
    print(f"Structure: {structure['num_rows']} rows × {structure['num_cols']} cols")
    print(f"Total cells: {len(cells)}")
    
    # Print table preview
    print("\nTable Preview (first 10 rows):")
    print("-" * 80)
    
    # Group by row
    rows_data = {}
    for cell in cells:
        r = cell.get("row", 0)
        if r not in rows_data:
            rows_data[r] = {}
        rows_data[r][cell.get("col", 0)] = cell
    
    for r in sorted(rows_data.keys())[:10]:
        row_texts = []
        for c in sorted(rows_data[r].keys()):
            cell = rows_data[r][c]
            text = cell.get("text", "")[:20]
            row_texts.append(text)
        print(f"Row {r}: {' | '.join(row_texts)}")


if __name__ == "__main__":
    main()
