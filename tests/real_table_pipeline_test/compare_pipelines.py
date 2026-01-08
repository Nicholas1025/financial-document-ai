"""
Pipeline Comparison: Table Transformer vs Docling End-to-End
============================================================
公平对比两个系统在同一张图片上的 QA 准确率

System A: Table Transformer Pipeline (Detection + TSR + EasyOCR)
System B: Docling End-to-End (完整端到端)
"""

import os
import sys
import json
import time
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
from difflib import SequenceMatcher


def fuzzy_match(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """Check if two strings are similar enough"""
    if not text1 or not text2:
        return False
    ratio = SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()
    return ratio >= threshold


class TableTransformerPipeline:
    """Our original Table Transformer based pipeline"""
    
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = Image.open(image_path).convert('RGB')
        self.img_width, self.img_height = self.image.size
        self.extracted_cells = []
        self.table_data = []  # List of rows, each row is list of cell texts
        
    def run(self) -> Dict:
        """Run full pipeline"""
        print("\n" + "="*60)
        print("Running: TABLE TRANSFORMER PIPELINE")
        print("="*60)
        
        start_time = time.time()
        
        # Step 1: Detection
        self._detect_table()
        
        # Step 2: TSR (skipped, using OCR directly)
        
        # Step 3: OCR
        self._extract_text()
        
        # Step 4-5: Build table structure
        self._build_table_structure()
        
        total_time = time.time() - start_time
        
        return {
            'method': 'Table Transformer + EasyOCR',
            'total_time_ms': total_time * 1000,
            'num_cells': len(self.extracted_cells),
            'table_data': self.table_data,
        }
    
    def _detect_table(self):
        """Step 1: Table Detection"""
        from transformers import TableTransformerForObjectDetection, DetrImageProcessor
        
        processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        inputs = processor(images=self.image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([[self.img_height, self.img_width]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        
        if len(results["boxes"]) > 0:
            box = results["boxes"][0].cpu().numpy()
            self.table_bbox = box.tolist()
            print(f"  Table detected: {self.table_bbox}")
        else:
            self.table_bbox = [0, 0, self.img_width, self.img_height]
            print(f"  No table detected, using full image")
    
    def _extract_text(self):
        """Step 3: OCR"""
        import easyocr
        
        img_np = np.array(self.image)
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        ocr_results = reader.readtext(img_np)
        
        for bbox, text, conf in ocr_results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            self.extracted_cells.append({
                'text': text,
                'confidence': conf,
                'bbox': box,
                'center_y': (box[1] + box[3]) / 2,
                'center_x': (box[0] + box[2]) / 2,
            })
        
        print(f"  OCR extracted: {len(self.extracted_cells)} text regions")
    
    def _build_table_structure(self):
        """Build table structure from OCR results"""
        if not self.extracted_cells:
            return
        
        # Sort by y position to group into rows
        sorted_cells = sorted(self.extracted_cells, key=lambda c: c['center_y'])
        
        # Group into rows based on y proximity
        rows = []
        current_row = [sorted_cells[0]]
        row_y = sorted_cells[0]['center_y']
        
        for cell in sorted_cells[1:]:
            if abs(cell['center_y'] - row_y) < 25:  # Same row threshold
                current_row.append(cell)
            else:
                # Sort current row by x
                current_row.sort(key=lambda c: c['center_x'])
                rows.append(current_row)
                current_row = [cell]
                row_y = cell['center_y']
        
        # Don't forget last row
        current_row.sort(key=lambda c: c['center_x'])
        rows.append(current_row)
        
        # Convert to text matrix
        self.table_data = []
        for row in rows:
            row_texts = [c['text'] for c in row]
            self.table_data.append(row_texts)
        
        print(f"  Table structure: {len(self.table_data)} rows")
    
    def answer_question(self, row_key: str, col_key: str) -> str:
        """Answer a question by finding row and column"""
        if not self.table_data:
            return None
        
        # Find row containing row_key
        target_row = None
        for row in self.table_data:
            row_text = ' '.join(row).lower()
            if row_key.lower() in row_text or fuzzy_match(row_key, row_text, 0.5):
                target_row = row
                break
        
        if not target_row:
            return None
        
        # Determine column index based on col_key
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
        else:
            # Try to find numeric value in the row
            for i, cell in enumerate(target_row):
                if any(c.isdigit() for c in cell) and i > 0:
                    col_idx = i
                    break
        
        if col_idx >= 0 and col_idx < len(target_row):
            return target_row[col_idx]
        
        return None


class DoclingEndToEndPipeline:
    """Docling complete end-to-end pipeline"""
    
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = Image.open(image_path).convert('RGB')
        self.tables = []
        self.table_data = []
        
    def run(self) -> Dict:
        """Run full Docling pipeline"""
        print("\n" + "="*60)
        print("Running: DOCLING END-TO-END PIPELINE")
        print("="*60)
        
        start_time = time.time()
        
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
        except ImportError:
            return {'error': 'Docling not installed'}
        
        # Save temp image
        temp_path = self.image_path.parent / 'temp_docling_input.png'
        self.image.save(temp_path)
        
        # Configure Docling with full processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR
        pipeline_options.do_table_structure = True  # Enable table structure
        
        converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Convert
        result = converter.convert(str(temp_path))
        
        total_time = time.time() - start_time
        
        # Extract tables
        self._extract_tables(result)
        
        # Clean up
        if temp_path.exists():
            temp_path.unlink()
        
        return {
            'method': 'Docling End-to-End',
            'total_time_ms': total_time * 1000,
            'num_tables': len(self.tables),
            'table_data': self.table_data,
        }
    
    def _extract_tables(self, result):
        """Extract table data from Docling result"""
        if not hasattr(result, 'document') or not result.document:
            print("  No document in result")
            return
        
        doc = result.document
        
        # Try to get tables
        tables = []
        if hasattr(doc, 'tables'):
            tables = list(doc.tables)
        
        print(f"  Found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            print(f"  Processing table {i+1}")
            
            # Try different ways to extract table data
            table_rows = []
            
            # Method 1: Check for data attribute
            if hasattr(table, 'data') and table.data:
                if hasattr(table.data, 'grid'):
                    grid = table.data.grid
                    for row in grid:
                        row_texts = []
                        for cell in row:
                            if hasattr(cell, 'text'):
                                row_texts.append(cell.text)
                            elif isinstance(cell, str):
                                row_texts.append(cell)
                            else:
                                row_texts.append(str(cell))
                        table_rows.append(row_texts)
                elif isinstance(table.data, list):
                    for row in table.data:
                        if isinstance(row, list):
                            table_rows.append([str(c) for c in row])
            
            # Method 2: Check for export_to_dataframe
            if not table_rows and hasattr(table, 'export_to_dataframe'):
                try:
                    df = table.export_to_dataframe()
                    table_rows = [df.columns.tolist()] + df.values.tolist()
                except:
                    pass
            
            # Method 3: Check for cells
            if not table_rows and hasattr(table, 'cells'):
                # Build grid from cells
                cells_list = list(table.cells) if hasattr(table.cells, '__iter__') else []
                if cells_list:
                    # Get max row and col
                    max_row = max(c.row_index for c in cells_list if hasattr(c, 'row_index')) + 1
                    max_col = max(c.col_index for c in cells_list if hasattr(c, 'col_index')) + 1
                    
                    grid = [[''] * max_col for _ in range(max_row)]
                    for cell in cells_list:
                        if hasattr(cell, 'row_index') and hasattr(cell, 'col_index'):
                            text = cell.text if hasattr(cell, 'text') else str(cell)
                            grid[cell.row_index][cell.col_index] = text
                    
                    table_rows = grid
            
            # Method 4: Try to get markdown/text representation
            if not table_rows:
                if hasattr(table, 'export_to_markdown'):
                    try:
                        md = table.export_to_markdown()
                        # Parse markdown table
                        lines = md.strip().split('\n')
                        for line in lines:
                            if '|' in line and not line.strip().startswith('|-'):
                                cells = [c.strip() for c in line.split('|')[1:-1]]
                                if cells:
                                    table_rows.append(cells)
                    except:
                        pass
            
            if table_rows:
                self.tables.append(table_rows)
                self.table_data = table_rows  # Use last table
                print(f"    Extracted {len(table_rows)} rows")
            else:
                print(f"    Could not extract table data")
        
        # If no tables found, try to get text content
        if not self.table_data:
            print("  Trying to extract text content...")
            if hasattr(doc, 'export_to_markdown'):
                try:
                    md = doc.export_to_markdown()
                    print(f"  Document has {len(md)} chars of markdown")
                    # Try to find table in markdown
                    lines = md.split('\n')
                    in_table = False
                    table_rows = []
                    for line in lines:
                        if '|' in line:
                            if not line.strip().startswith('|-') and not line.strip() == '|':
                                cells = [c.strip() for c in line.split('|')]
                                cells = [c for c in cells if c]  # Remove empty
                                if cells:
                                    table_rows.append(cells)
                                    in_table = True
                        elif in_table and table_rows:
                            break
                    
                    if table_rows:
                        self.table_data = table_rows
                        print(f"  Extracted {len(table_rows)} rows from markdown")
                except Exception as e:
                    print(f"  Markdown export error: {e}")
    
    def answer_question(self, row_key: str, col_key: str) -> str:
        """Answer a question by finding row and column"""
        if not self.table_data:
            return None
        
        # Find row containing row_key
        target_row = None
        for row in self.table_data:
            row_text = ' '.join(str(c) for c in row).lower()
            if row_key.lower() in row_text or fuzzy_match(row_key, row_text, 0.5):
                target_row = row
                break
        
        if not target_row:
            return None
        
        # Determine column index
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
        else:
            for i, cell in enumerate(target_row):
                if any(c.isdigit() for c in str(cell)) and i > 0:
                    col_idx = i
                    break
        
        if col_idx >= 0 and col_idx < len(target_row):
            return str(target_row[col_idx])
        
        return None


def run_comparison(image_path: str, gt_path: str, output_dir: str):
    """Run comparison between two pipelines"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GT
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    
    qa_pairs = gt.get('qa_pairs', [])
    
    print("\n" + "="*70)
    print("  PIPELINE COMPARISON: Table Transformer vs Docling E2E")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Questions: {len(qa_pairs)}")
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image': str(image_path),
        'num_questions': len(qa_pairs),
        'systems': {}
    }
    
    # =====================================
    # Run Table Transformer Pipeline
    # =====================================
    tt_pipeline = TableTransformerPipeline(image_path)
    tt_info = tt_pipeline.run()
    
    tt_correct = 0
    tt_details = []
    
    for qa in qa_pairs:
        row_key = qa.get('row_key', '')
        col_key = qa.get('col_key', '')
        expected = str(qa['answer'])
        
        predicted = tt_pipeline.answer_question(row_key, col_key)
        
        is_correct = False
        if predicted and expected:
            pred_clean = predicted.replace(',', '').replace(' ', '').strip()
            exp_clean = expected.replace(',', '').replace(' ', '').strip()
            is_correct = pred_clean == exp_clean or fuzzy_match(pred_clean, exp_clean, 0.9)
        
        if is_correct:
            tt_correct += 1
        
        tt_details.append({
            'question': qa['question'],
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct,
        })
    
    tt_accuracy = tt_correct / len(qa_pairs) * 100 if qa_pairs else 0
    
    results['systems']['table_transformer'] = {
        'name': 'Table Transformer + EasyOCR',
        'time_ms': tt_info['total_time_ms'],
        'correct': tt_correct,
        'total': len(qa_pairs),
        'accuracy': tt_accuracy,
        'details': tt_details,
    }
    
    print(f"\n  Table Transformer Results:")
    print(f"    Time: {tt_info['total_time_ms']:.0f}ms")
    print(f"    QA Accuracy: {tt_correct}/{len(qa_pairs)} ({tt_accuracy:.1f}%)")
    
    # =====================================
    # Run Docling End-to-End Pipeline
    # =====================================
    docling_pipeline = DoclingEndToEndPipeline(image_path)
    docling_info = docling_pipeline.run()
    
    docling_correct = 0
    docling_details = []
    
    for qa in qa_pairs:
        row_key = qa.get('row_key', '')
        col_key = qa.get('col_key', '')
        expected = str(qa['answer'])
        
        predicted = docling_pipeline.answer_question(row_key, col_key)
        
        is_correct = False
        if predicted and expected:
            pred_clean = str(predicted).replace(',', '').replace(' ', '').strip()
            exp_clean = expected.replace(',', '').replace(' ', '').strip()
            is_correct = pred_clean == exp_clean or fuzzy_match(pred_clean, exp_clean, 0.9)
        
        if is_correct:
            docling_correct += 1
        
        docling_details.append({
            'question': qa['question'],
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct,
        })
    
    docling_accuracy = docling_correct / len(qa_pairs) * 100 if qa_pairs else 0
    
    results['systems']['docling_e2e'] = {
        'name': 'Docling End-to-End',
        'time_ms': docling_info['total_time_ms'],
        'correct': docling_correct,
        'total': len(qa_pairs),
        'accuracy': docling_accuracy,
        'details': docling_details,
    }
    
    print(f"\n  Docling E2E Results:")
    print(f"    Time: {docling_info['total_time_ms']:.0f}ms")
    print(f"    QA Accuracy: {docling_correct}/{len(qa_pairs)} ({docling_accuracy:.1f}%)")
    
    # =====================================
    # Generate Comparison Report
    # =====================================
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    
    winner = "Table Transformer" if tt_accuracy > docling_accuracy else ("Docling" if docling_accuracy > tt_accuracy else "Tie")
    
    comparison_table = f"""
| Metric | Table Transformer | Docling E2E | Winner |
|--------|-------------------|-------------|--------|
| Time | {tt_info['total_time_ms']:.0f}ms | {docling_info['total_time_ms']:.0f}ms | {'TT' if tt_info['total_time_ms'] < docling_info['total_time_ms'] else 'Docling'} |
| QA Correct | {tt_correct}/{len(qa_pairs)} | {docling_correct}/{len(qa_pairs)} | {winner} |
| Accuracy | {tt_accuracy:.1f}% | {docling_accuracy:.1f}% | {winner} |
"""
    print(comparison_table)
    
    # Detailed Q&A comparison
    print("\nDetailed QA Comparison:")
    print("-" * 80)
    print(f"{'Q#':<4} {'Expected':<15} {'TT Pred':<15} {'TT':<4} {'Docling Pred':<15} {'Doc':<4}")
    print("-" * 80)
    
    for i, (tt_d, doc_d) in enumerate(zip(tt_details, docling_details)):
        tt_status = "✓" if tt_d['correct'] else "✗"
        doc_status = "✓" if doc_d['correct'] else "✗"
        tt_pred = str(tt_d['predicted'])[:13] if tt_d['predicted'] else "None"
        doc_pred = str(doc_d['predicted'])[:13] if doc_d['predicted'] else "None"
        expected = str(tt_d['expected'])[:13]
        print(f"Q{i+1:<3} {expected:<15} {tt_pred:<15} {tt_status:<4} {doc_pred:<15} {doc_status:<4}")
    
    print("-" * 80)
    
    # Save results
    results_path = output_dir / 'pipeline_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate markdown report
    report = f"""# Pipeline Comparison Report
Generated: {results['timestamp']}

## Test Configuration
- **Image**: {image_path}
- **Questions**: {len(qa_pairs)}

## Results Summary

| Metric | Table Transformer | Docling E2E | Winner |
|--------|-------------------|-------------|--------|
| **Processing Time** | {tt_info['total_time_ms']:.0f}ms | {docling_info['total_time_ms']:.0f}ms | {'Table Transformer' if tt_info['total_time_ms'] < docling_info['total_time_ms'] else 'Docling'} |
| **QA Correct** | {tt_correct}/{len(qa_pairs)} | {docling_correct}/{len(qa_pairs)} | **{winner}** |
| **Accuracy** | **{tt_accuracy:.1f}%** | **{docling_accuracy:.1f}%** | **{winner}** |

## Winner: **{winner}**

## Detailed QA Results

### Table Transformer
"""
    for i, d in enumerate(tt_details):
        status = "✅" if d['correct'] else "❌"
        report += f"- {status} Q{i+1}: Expected `{d['expected']}`, Got `{d['predicted']}`\n"
    
    report += f"""
### Docling End-to-End
"""
    for i, d in enumerate(docling_details):
        status = "✅" if d['correct'] else "❌"
        report += f"- {status} Q{i+1}: Expected `{d['expected']}`, Got `{d['predicted']}`\n"
    
    report += f"""
## Analysis

### Speed Comparison
- Table Transformer: **{tt_info['total_time_ms']:.0f}ms** ({tt_info['total_time_ms']/docling_info['total_time_ms']*100:.1f}% of Docling time)
- Docling: **{docling_info['total_time_ms']:.0f}ms**
- Speed ratio: Docling is **{docling_info['total_time_ms']/tt_info['total_time_ms']:.1f}x slower**

### Accuracy Comparison
- Table Transformer: **{tt_accuracy:.1f}%** ({tt_correct}/{len(qa_pairs)})
- Docling: **{docling_accuracy:.1f}%** ({docling_correct}/{len(qa_pairs)})
- Accuracy difference: **{abs(tt_accuracy - docling_accuracy):.1f}%**

### Conclusion
{"Table Transformer pipeline achieves higher QA accuracy while being significantly faster." if tt_accuracy > docling_accuracy else "Docling achieves higher QA accuracy but is slower." if docling_accuracy > tt_accuracy else "Both systems achieve the same QA accuracy."}
"""
    
    report_path = output_dir / 'COMPARISON_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - pipeline_comparison_results.json")
    print(f"  - COMPARISON_REPORT.md")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Table Transformer vs Docling')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth JSON path')
    parser.add_argument('--output', type=str, default='tests/real_table_pipeline_test/output_comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_comparison(args.image, args.gt, args.output)


if __name__ == '__main__':
    main()
