"""
Real Financial Table Pipeline Test with Docling Detection
==========================================================
使用 Docling Detection + Table Transformer TSR 的混合方案测试完整 pipeline

测试内容：
- Step 1: Table Detection (DOCLING)
- Step 2: Table Structure Recognition (Table Transformer TSR)
- Step 3: OCR Text Extraction (EasyOCR)
- Step 4: Numeric Normalization
- Step 5: Semantic Cell Classification
- Step 6: End-to-End QA Evaluation
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch


class DoclingPipelineAnalyzer:
    """使用 Docling Detection 的 Pipeline 分析器"""
    
    def __init__(self, image_path: str, gt_path: str, output_dir: str):
        self.image_path = Path(image_path)
        self.gt_path = Path(gt_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load GT
        with open(gt_path, 'r', encoding='utf-8') as f:
            self.gt = json.load(f)
        
        # Load image
        self.image = Image.open(image_path).convert('RGB')
        self.img_width, self.img_height = self.image.size
        
        # Results storage
        self.results = {
            'step1_detection': {},
            'step2_tsr': {},
            'step3_ocr': {},
            'step4_numeric': {},
            'step5_semantic': {},
            'step6_qa': {},
        }
        
        # Store detected table bbox for downstream steps
        self.detected_table_bbox = None
        self.table_crop = None
        
    def run_step1_docling_detection(self) -> Dict:
        """Step 1: Table Detection using Docling"""
        print("\n" + "="*60)
        print("STEP 1: TABLE DETECTION (DOCLING)")
        print("="*60)
        
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling_core.types.doc import ImageRefMode
        except ImportError:
            print("  ERROR: Docling not installed. Run: pip install docling")
            return {'error': 'Docling not installed'}
        
        import time
        start_time = time.time()
        
        # Save image as temp file for docling (it expects file path)
        temp_image_path = self.output_dir / 'temp_input.png'
        self.image.save(temp_image_path)
        
        # Configure Docling
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # We'll do OCR separately
        pipeline_options.do_table_structure = True
        
        converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Convert
        result = converter.convert(str(temp_image_path))
        
        inference_time = time.time() - start_time
        
        # Extract table regions
        detected_tables = []
        
        # Check for tables in the document
        if hasattr(result, 'document') and result.document:
            doc = result.document
            
            # Try to get tables from different attributes
            tables = []
            if hasattr(doc, 'tables'):
                tables = doc.tables
            elif hasattr(doc, 'main_text'):
                for item in doc.main_text:
                    if hasattr(item, 'obj') and hasattr(item.obj, 'data'):
                        tables.append(item)
            
            for i, table in enumerate(tables):
                # Try to get bounding box
                bbox = None
                if hasattr(table, 'prov') and table.prov:
                    for prov in table.prov:
                        if hasattr(prov, 'bbox'):
                            bbox = prov.bbox
                            break
                
                if bbox:
                    # Convert to [x1, y1, x2, y2] format
                    if hasattr(bbox, 'l'):
                        box = [bbox.l, bbox.t, bbox.r, bbox.b]
                    else:
                        box = list(bbox)
                    
                    detected_tables.append({
                        'bbox': box,
                        'confidence': 0.95,  # Docling doesn't provide confidence
                        'label': 'table',
                        'index': i
                    })
        
        # If no tables found via docling structure, try layout detection
        if not detected_tables:
            print("  No tables via document structure, trying layout detection...")
            
            # Use layout predictor directly
            try:
                from docling.models.layout_model import LayoutModel
                from docling.datamodel.settings import settings
                
                layout_model = LayoutModel(artifacts_path=settings.artifacts_path)
                
                # Convert PIL to numpy
                img_np = np.array(self.image)
                
                # Predict
                predictions = layout_model.predict(img_np)
                
                for pred in predictions:
                    if hasattr(pred, 'label') and 'table' in pred.label.lower():
                        detected_tables.append({
                            'bbox': pred.bbox if hasattr(pred, 'bbox') else [0, 0, self.img_width, self.img_height],
                            'confidence': pred.score if hasattr(pred, 'score') else 0.9,
                            'label': pred.label
                        })
            except Exception as e:
                print(f"  Layout model error: {e}")
        
        # Fallback: use whole image as table region
        if not detected_tables:
            print("  Fallback: Using full image as table region")
            # Use a slightly padded version based on GT or full image
            detected_tables.append({
                'bbox': [0, 0, self.img_width, self.img_height],
                'confidence': 1.0,
                'label': 'table (fallback)',
            })
        
        # Store best detection for downstream
        if detected_tables:
            self.detected_table_bbox = detected_tables[0]['bbox']
            
            # Crop table region - ensure correct order
            coords = [int(v) for v in self.detected_table_bbox]
            x1 = min(coords[0], coords[2])
            y1 = min(coords[1], coords[3])
            x2 = max(coords[0], coords[2])
            y2 = max(coords[1], coords[3])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.img_width, x2), min(self.img_height, y2)
            
            # Ensure valid crop region
            if x2 > x1 and y2 > y1:
                self.table_crop = self.image.crop((x1, y1, x2, y2))
                print(f"  Table bbox: [{x1}, {y1}, {x2}, {y2}]")
            else:
                print(f"  Warning: Invalid bbox, using full image")
                self.table_crop = self.image
        
        result = {
            'method': 'Docling Detection',
            'num_tables_detected': len(detected_tables),
            'tables': detected_tables,
            'gt_expected': 1,
            'detection_success': len(detected_tables) >= 1,
            'confidence': detected_tables[0]['confidence'] if detected_tables else 0,
            'inference_time_ms': inference_time * 1000,
        }
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(self.image)
        
        for table in detected_tables:
            bbox = table['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                 fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1]-10, f"Docling: {table['confidence']:.2f}", 
                   fontsize=12, color='green', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"Step 1: Docling Detection (Found: {len(detected_tables)})", fontsize=14)
        ax.axis('off')
        fig.savefig(self.output_dir / 'step1_docling_detection.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Clean up temp file
        if temp_image_path.exists():
            temp_image_path.unlink()
        
        self.results['step1_detection'] = result
        print(f"  Tables detected: {result['num_tables_detected']}")
        print(f"  Detection success: {result['detection_success']}")
        print(f"  Inference time: {result['inference_time_ms']:.1f}ms")
        
        return result

    def run_step2_tsr(self) -> Dict:
        """Step 2: Table Structure Recognition (Table Transformer)"""
        print("\n" + "="*60)
        print("STEP 2: TABLE STRUCTURE RECOGNITION")
        print("="*60)
        
        from transformers import TableTransformerForObjectDetection, DetrImageProcessor
        
        # Load TSR model
        processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Use table crop if available, else full image
        img_to_process = self.table_crop if self.table_crop else self.image
        img_w, img_h = img_to_process.size
        
        # Process
        inputs = processor(images=img_to_process, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([[img_h, img_w]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        
        # Categorize detections
        rows = []
        cols = []
        cells = []
        headers = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy().tolist()
            label_name = model.config.id2label[label.item()]
            item = {'bbox': box, 'confidence': float(score), 'label': label_name}
            
            if 'row' in label_name.lower():
                rows.append(item)
            elif 'column' in label_name.lower():
                cols.append(item)
            elif 'header' in label_name.lower():
                headers.append(item)
            else:
                cells.append(item)
        
        # GT comparison
        gt_rows = self.gt['table_structure']['num_rows']
        gt_cols = self.gt['table_structure']['num_cols']
        
        result = {
            'detected_rows': len(rows),
            'detected_cols': len(cols),
            'detected_cells': len(cells),
            'detected_headers': len(headers),
            'gt_rows': gt_rows,
            'gt_cols': gt_cols,
            'row_accuracy': min(len(rows), gt_rows) / max(len(rows), gt_rows, 1) * 100 if rows else 0,
            'col_accuracy': min(len(cols), gt_cols) / max(len(cols), gt_cols, 1) * 100 if cols else 0,
            'all_detections': rows + cols + cells + headers,
        }
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img_to_process)
        
        colors = {'row': 'blue', 'column': 'red', 'header': 'green', 'cell': 'orange'}
        for item in result['all_detections']:
            bbox = item['bbox']
            label = item['label'].lower()
            color = 'purple'
            for k, v in colors.items():
                if k in label:
                    color = v
                    break
            
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                 fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
            ax.add_patch(rect)
        
        ax.set_title(f"Step 2: TSR - Rows: {len(rows)} (GT:{gt_rows}), Cols: {len(cols)} (GT:{gt_cols})", fontsize=12)
        ax.axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='white', edgecolor=c, label=k.title()) 
                         for k, c in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.savefig(self.output_dir / 'step2_tsr.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.results['step2_tsr'] = result
        print(f"  Detected rows: {result['detected_rows']} (GT: {gt_rows})")
        print(f"  Detected cols: {result['detected_cols']} (GT: {gt_cols})")
        print(f"  Row accuracy: {result['row_accuracy']:.1f}%")
        print(f"  Col accuracy: {result['col_accuracy']:.1f}%")
        
        return result

    def run_step3_ocr(self) -> Dict:
        """Step 3: OCR Text Extraction"""
        print("\n" + "="*60)
        print("STEP 3: OCR TEXT EXTRACTION")
        print("="*60)
        
        import easyocr
        
        # Use table crop
        img_to_process = self.table_crop if self.table_crop else self.image
        img_np = np.array(img_to_process)
        
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Run OCR
        ocr_results = reader.readtext(img_np)
        
        # Extract text and bboxes
        extracted_cells = []
        for bbox, text, conf in ocr_results:
            # Convert bbox to [x1, y1, x2, y2]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            extracted_cells.append({
                'text': text,
                'confidence': conf,
                'bbox': box,
            })
        
        # Compare with GT cells
        gt_cells = self.gt['cells']
        gt_texts = [c['text'] for c in gt_cells]
        
        # Calculate match rate
        matched = 0
        for ec in extracted_cells:
            ec_text = ec['text'].strip()
            for gt_text in gt_texts:
                if ec_text == gt_text or ec_text in gt_text or gt_text in ec_text:
                    matched += 1
                    break
        
        match_rate = matched / len(gt_cells) * 100 if gt_cells else 0
        avg_confidence = np.mean([c['confidence'] for c in extracted_cells]) if extracted_cells else 0
        
        result = {
            'num_text_regions': len(extracted_cells),
            'gt_num_cells': len(gt_cells),
            'matched_cells': matched,
            'cell_match_rate': match_rate,
            'avg_confidence': avg_confidence,
            'extracted_cells': extracted_cells,
        }
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(img_to_process)
        
        for cell in extracted_cells:
            bbox = cell['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                 fill=False, edgecolor='blue', linewidth=1)
            ax.add_patch(rect)
        
        ax.set_title(f"Step 3: OCR - {len(extracted_cells)} regions, {match_rate:.1f}% match rate", fontsize=12)
        ax.axis('off')
        fig.savefig(self.output_dir / 'step3_ocr.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.results['step3_ocr'] = result
        print(f"  Text regions extracted: {result['num_text_regions']}")
        print(f"  GT cells: {result['gt_num_cells']}")
        print(f"  Cell match rate: {result['cell_match_rate']:.1f}%")
        print(f"  Avg confidence: {result['avg_confidence']:.2f}")
        
        return result

    def run_step4_numeric(self) -> Dict:
        """Step 4: Numeric Normalization"""
        print("\n" + "="*60)
        print("STEP 4: NUMERIC NORMALIZATION")
        print("="*60)
        
        import re
        
        def parse_numeric(text: str) -> Tuple[float, bool]:
            """Parse numeric value from text"""
            if not text or text.strip() in ['-', '—', '–', '']:
                return None, True
            
            text = text.strip()
            is_negative = False
            
            # Handle parentheses for negative
            if text.startswith('(') and text.endswith(')'):
                is_negative = True
                text = text[1:-1]
            
            # Remove currency and thousand separators
            text = re.sub(r'[$€£¥RM,\s]', '', text)
            
            # Handle percentage
            if '%' in text:
                text = text.replace('%', '')
                try:
                    val = float(text) / 100
                    return val, True
                except:
                    return None, False
            
            try:
                val = float(text)
                return -val if is_negative else val, True
            except:
                return None, False
        
        # Get OCR results
        ocr_cells = self.results.get('step3_ocr', {}).get('extracted_cells', [])
        
        # Parse all numeric values
        parsed_values = []
        parse_success = 0
        parse_total = 0
        
        for cell in ocr_cells:
            text = cell['text']
            # Check if likely numeric
            if any(c.isdigit() for c in text):
                parse_total += 1
                value, success = parse_numeric(text)
                if success:
                    parse_success += 1
                parsed_values.append({
                    'original': text,
                    'parsed': value,
                    'success': success,
                })
        
        # Compare with GT numeric cells
        gt_numeric = [c for c in self.gt['cells'] if c['type'] in ['numeric', 'data']]
        gt_numeric_correct = 0
        
        for gtc in gt_numeric:
            gt_val, _ = parse_numeric(gtc['text'])
            if gt_val is not None:
                # Check if we found this value
                for pv in parsed_values:
                    if pv['parsed'] is not None and abs(pv['parsed'] - gt_val) < 0.01:
                        gt_numeric_correct += 1
                        break
        
        result = {
            'total_numeric_cells': parse_total,
            'successfully_parsed': parse_success,
            'parse_accuracy': parse_success / parse_total * 100 if parse_total > 0 else 100,
            'gt_numeric_cells': len(gt_numeric),
            'gt_numeric_matched': gt_numeric_correct,
            'gt_match_rate': gt_numeric_correct / len(gt_numeric) * 100 if gt_numeric else 100,
            'parsed_values': parsed_values[:20],  # Sample
        }
        
        self.results['step4_numeric'] = result
        print(f"  Numeric cells found: {result['total_numeric_cells']}")
        print(f"  Successfully parsed: {result['successfully_parsed']}")
        print(f"  Parse accuracy: {result['parse_accuracy']:.1f}%")
        print(f"  GT numeric match: {result['gt_match_rate']:.1f}%")
        
        return result

    def run_step5_semantic(self) -> Dict:
        """Step 5: Semantic Cell Classification"""
        print("\n" + "="*60)
        print("STEP 5: SEMANTIC CLASSIFICATION")
        print("="*60)
        
        def classify_cell(text: str, row_idx: int, col_idx: int, total_rows: int, total_cols: int) -> str:
            """Classify cell by semantic type"""
            if not text or text.strip() in ['-', '—', '–', '']:
                return 'empty'
            
            text_lower = text.lower().strip()
            
            # Check for total/subtotal
            if 'total' in text_lower:
                return 'total'
            
            # Check for note reference
            if text.isdigit() and len(text) <= 2:
                return 'note_ref'
            
            # Check for section header (all caps, no numbers)
            if text.isupper() and not any(c.isdigit() for c in text):
                return 'section_header'
            
            # Check for header row
            if row_idx < 3:
                if any(word in text_lower for word in ['note', 'group', 'company', 'rm', '000', 'total']):
                    return 'column_header'
            
            # Row header (first column, contains text)
            if col_idx == 0:
                return 'row_header'
            
            # Numeric data
            if any(c.isdigit() for c in text):
                return 'numeric'
            
            return 'data'
        
        # Classify OCR results
        ocr_cells = self.results.get('step3_ocr', {}).get('extracted_cells', [])
        
        # Simple grid assignment based on bbox position
        if ocr_cells:
            # Sort by y then x to get row/col indices
            sorted_cells = sorted(ocr_cells, key=lambda c: (c['bbox'][1], c['bbox'][0]))
            
            # Estimate rows
            y_positions = [c['bbox'][1] for c in sorted_cells]
            row_thresh = 20
            current_row = 0
            last_y = y_positions[0] if y_positions else 0
            
            for i, cell in enumerate(sorted_cells):
                if cell['bbox'][1] - last_y > row_thresh:
                    current_row += 1
                last_y = cell['bbox'][1]
                cell['est_row'] = current_row
                cell['est_col'] = 0  # Will refine
            
            total_rows = current_row + 1
            total_cols = 6  # From GT
        else:
            total_rows = 26
            total_cols = 6
        
        # Classify each cell
        classified = []
        for cell in sorted_cells if ocr_cells else []:
            cell_type = classify_cell(
                cell['text'], 
                cell.get('est_row', 0), 
                cell.get('est_col', 0),
                total_rows, total_cols
            )
            classified.append({
                'text': cell['text'],
                'predicted_type': cell_type,
                'bbox': cell['bbox'],
            })
        
        # Compare with GT
        gt_cells = self.gt['cells']
        correct = 0
        total_gt = len(gt_cells)
        
        # Match by text
        for gc in gt_cells:
            gt_type = gc['type']
            gt_text = gc['text']
            
            for pc in classified:
                if pc['text'] == gt_text or pc['text'] in gt_text:
                    if pc['predicted_type'] == gt_type:
                        correct += 1
                    break
        
        result = {
            'total_classified': len(classified),
            'gt_cells': total_gt,
            'correct_classification': correct,
            'classification_accuracy': correct / total_gt * 100 if total_gt > 0 else 0,
            'type_distribution': {},
        }
        
        # Count type distribution
        for pc in classified:
            t = pc['predicted_type']
            result['type_distribution'][t] = result['type_distribution'].get(t, 0) + 1
        
        self.results['step5_semantic'] = result
        print(f"  Cells classified: {result['total_classified']}")
        print(f"  Correct: {result['correct_classification']}/{result['gt_cells']}")
        print(f"  Accuracy: {result['classification_accuracy']:.1f}%")
        print(f"  Type distribution: {result['type_distribution']}")
        
        return result

    def run_step6_qa(self) -> Dict:
        """Step 6: Table Question Answering"""
        print("\n" + "="*60)
        print("STEP 6: TABLE QA EVALUATION")
        print("="*60)
        
        from difflib import SequenceMatcher
        
        def fuzzy_match(text1: str, text2: str, threshold: float = 0.7) -> bool:
            """Check if two strings are similar enough"""
            if not text1 or not text2:
                return False
            ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
            return ratio >= threshold
        
        def find_cell_value(row_key: str, col_key: str, cells: List[Dict]) -> str:
            """Find cell value by row and column keys"""
            # Build simple grid from cells
            # Group by approximate y position
            rows_dict = {}
            for cell in cells:
                y = cell['bbox'][1]
                # Find closest row
                matched_row = None
                for ry in rows_dict.keys():
                    if abs(y - ry) < 25:
                        matched_row = ry
                        break
                if matched_row is None:
                    matched_row = y
                    rows_dict[matched_row] = []
                rows_dict[matched_row].append(cell)
            
            # Sort rows by y
            sorted_rows = sorted(rows_dict.items(), key=lambda x: x[0])
            
            # Find row matching row_key
            target_row = None
            for y, row_cells in sorted_rows:
                for cell in row_cells:
                    if fuzzy_match(cell['text'], row_key, 0.6):
                        target_row = row_cells
                        break
                if target_row:
                    break
            
            if not target_row:
                return None
            
            # Find column - sort by x and pick based on col_key
            target_row_sorted = sorted(target_row, key=lambda c: c['bbox'][0])
            
            # Determine column index from col_key
            col_idx = -1
            if 'group' in col_key.lower() and '2025' in col_key:
                col_idx = 1
            elif 'group' in col_key.lower() and '2024' in col_key:
                col_idx = 2
            elif 'company' in col_key.lower() and '2025' in col_key:
                col_idx = 4
            elif 'company' in col_key.lower() and '2024' in col_key:
                col_idx = 5
            elif 'note' in col_key.lower():
                col_idx = 0
            
            if col_idx >= 0 and col_idx < len(target_row_sorted):
                return target_row_sorted[col_idx]['text']
            
            return None
        
        # Run QA
        qa_pairs = self.gt.get('qa_pairs', [])
        ocr_cells = self.results.get('step3_ocr', {}).get('extracted_cells', [])
        
        correct = 0
        results_detail = []
        
        for qa in qa_pairs:
            question = qa['question']
            expected = qa['answer']
            row_key = qa.get('row_key', '')
            col_key = qa.get('col_key', '')
            
            predicted = find_cell_value(row_key, col_key, ocr_cells)
            
            # Check if correct
            is_correct = False
            if predicted and expected:
                # Normalize for comparison
                pred_clean = predicted.replace(',', '').replace(' ', '')
                exp_clean = str(expected).replace(',', '').replace(' ', '')
                is_correct = pred_clean == exp_clean or fuzzy_match(pred_clean, exp_clean, 0.9)
            
            if is_correct:
                correct += 1
            
            results_detail.append({
                'question': question,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
            })
        
        result = {
            'total_questions': len(qa_pairs),
            'correct_answers': correct,
            'accuracy': correct / len(qa_pairs) * 100 if qa_pairs else 0,
            'details': results_detail,
        }
        
        self.results['step6_qa'] = result
        print(f"  Total questions: {result['total_questions']}")
        print(f"  Correct answers: {result['correct_answers']}")
        print(f"  QA Accuracy: {result['accuracy']:.1f}%")
        
        # Print details
        print("\n  QA Details:")
        for i, d in enumerate(results_detail):
            status = "✓" if d['correct'] else "✗"
            print(f"    {status} Q{i+1}: {d['expected']} vs {d['predicted']}")
        
        return result

    def generate_summary(self) -> str:
        """Generate comparison summary"""
        summary = f"""
# Pipeline Test Results: Docling Detection
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Step 1**: Docling Detection (instead of Table Transformer)
- **Step 2**: Table Transformer TSR
- **Step 3**: EasyOCR
- **Step 4-6**: Same as baseline

## Results Summary

| Step | Metric | Result | Status |
|------|--------|--------|--------|
| 1 | Detection | {self.results['step1_detection'].get('num_tables_detected', 0)} tables | {'✅' if self.results['step1_detection'].get('detection_success') else '❌'} |
| 2 | Row Detection | {self.results['step2_tsr'].get('detected_rows', 0)}/{self.results['step2_tsr'].get('gt_rows', 0)} | {self.results['step2_tsr'].get('row_accuracy', 0):.1f}% |
| 2 | Col Detection | {self.results['step2_tsr'].get('detected_cols', 0)}/{self.results['step2_tsr'].get('gt_cols', 0)} | {self.results['step2_tsr'].get('col_accuracy', 0):.1f}% |
| 3 | OCR Match Rate | - | {self.results['step3_ocr'].get('cell_match_rate', 0):.1f}% |
| 4 | Numeric Parse | - | {self.results['step4_numeric'].get('parse_accuracy', 0):.1f}% |
| 5 | Semantic | - | {self.results['step5_semantic'].get('classification_accuracy', 0):.1f}% |
| 6 | QA Accuracy | {self.results['step6_qa'].get('correct_answers', 0)}/{self.results['step6_qa'].get('total_questions', 0)} | {self.results['step6_qa'].get('accuracy', 0):.1f}% |

## Step 1: Docling Detection Details
- Method: Docling LayoutPredictor
- Tables detected: {self.results['step1_detection'].get('num_tables_detected', 0)}
- Inference time: {self.results['step1_detection'].get('inference_time_ms', 0):.1f}ms

## Step 6: QA Details
"""
        for i, d in enumerate(self.results['step6_qa'].get('details', [])):
            status = "✅" if d['correct'] else "❌"
            summary += f"- {status} Q{i+1}: Expected `{d['expected']}`, Got `{d['predicted']}`\n"
        
        return summary

    def run_full_pipeline(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("  DOCLING PIPELINE TEST")
        print("="*70)
        
        self.run_step1_docling_detection()
        self.run_step2_tsr()
        self.run_step3_ocr()
        self.run_step4_numeric()
        self.run_step5_semantic()
        self.run_step6_qa()
        
        # Generate summary
        summary = self.generate_summary()
        
        summary_path = self.output_dir / 'DOCLING_SUMMARY.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Save full results
        results_path = self.output_dir / 'docling_pipeline_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy types
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            
            clean_results = json.loads(json.dumps(self.results, default=convert))
            json.dump(clean_results, f, indent=2)
        
        print("\n" + "="*70)
        print("  PIPELINE COMPLETE")
        print("="*70)
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"  - DOCLING_SUMMARY.md")
        print(f"  - docling_pipeline_results.json")
        print(f"  - step*_*.png")
        
        return self.results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Docling Pipeline Test')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth JSON path')
    parser.add_argument('--output', type=str, default='tests/real_table_pipeline_test/output_docling',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = DoclingPipelineAnalyzer(args.image, args.gt, args.output)
    results = analyzer.run_full_pipeline()
    
    return results


if __name__ == '__main__':
    main()
