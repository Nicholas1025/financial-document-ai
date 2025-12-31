"""
Visualize Step Results with Image Comparison
=============================================
为每个 step 生成可视化对比图，展示：
- 原始图片
- 处理结果（检测框、OCR文字、cell分类等）
- 输出的 JSON 结构

输出到各 step 文件夹：
- comparison_*.png - 对比图
- output_sample_*.json - 详细输出结构
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
import random

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Dataset paths
DATASETS = {
    'doclaynet': 'D:/datasets/DocLayNet',
    'synfintabs': 'D:/datasets/synfintabs/data',
    'fintabnet': 'D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure',
    'pubtabnet': 'D:/datasets/PubTabNet/pubtabnet/pubtabnet',
}

# Colors for visualization
COLORS = {
    'detection': '#00FF00',  # Green for detected boxes
    'gt': '#FF0000',  # Red for ground truth
    'row_header': '#FFD700',  # Gold
    'column_header': '#00BFFF',  # Deep Sky Blue
    'data': '#90EE90',  # Light Green
    'section_title': '#FF69B4',  # Hot Pink
    'currency_unit': '#FFA500',  # Orange
}


def visualize_step1_detection(output_dir: Path, num_samples: int = 1):
    """
    Step 1: Table Detection 可视化
    显示：原图 + 检测框 + GT框
    """
    print("\n" + "="*60)
    print("STEP 1: TABLE DETECTION VISUALIZATION")
    print("="*60)
    
    from baselines.docling_eval.stage2_detection import (
        load_doclaynet_samples,
        TableTransformerDetector,
    )
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = load_doclaynet_samples(DATASETS['doclaynet'], 'val', num_samples)
    detector = TableTransformerDetector(device, threshold=0.5)
    
    for i, sample in enumerate(samples):
        img_path = sample['image_path']
        gt_boxes = sample.get('gt_boxes', sample.get('boxes', []))  # Handle both field names
        
        # Load image first
        img = Image.open(img_path).convert('RGB')
        
        # Run detection - pass PIL image not path
        pred_boxes, scores = detector.detect(img)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original image with GT boxes
        axes[0].imshow(img)
        axes[0].set_title('Ground Truth (Red)', fontsize=14)
        for box in gt_boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=3, edgecolor='red', facecolor='none'
            )
            axes[0].add_patch(rect)
        axes[0].axis('off')
        
        # Right: Image with predicted boxes
        axes[1].imshow(img)
        axes[1].set_title(f'Predictions (Green) - {len(pred_boxes)} detected', fontsize=14)
        for j, box in enumerate(pred_boxes):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=3, edgecolor='green', facecolor='none'
            )
            axes[1].add_patch(rect)
            # Add confidence score
            axes[1].text(box[0], box[1]-5, f'{scores[j]:.2f}', 
                        color='green', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison image
        comp_path = output_dir / f'comparison_detection_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 1,
            'task': 'Table Detection',
            'input': {
                'image_path': str(img_path),
                'image_size': list(img.size),
            },
            'ground_truth': {
                'num_tables': len(gt_boxes),
                'boxes': [[float(x) for x in box] for box in gt_boxes],
            },
            'predictions': {
                'num_detected': len(pred_boxes),
                'boxes': [[float(x) for x in box] for box in pred_boxes],
                'confidence_scores': [float(s) for s in scores],
            },
            'evaluation': {
                'description': 'Bounding boxes detected by Table Transformer',
                'format': '[x1, y1, x2, y2] in pixels',
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        
        print(f"  Sample {i+1}: GT={len(gt_boxes)} tables, Pred={len(pred_boxes)} tables")
        print(f"    Saved: {comp_path}")
        print(f"    Saved: {json_path}")


def visualize_step2_tsr(output_dir: Path, num_samples: int = 1):
    """
    Step 2: Table Structure Recognition 可视化 (使用 FinTabNet_c)
    显示：原图 + Cell 边界框 (GT)
    """
    print("\n" + "="*60)
    print("STEP 2: TSR (Table Structure Recognition) VISUALIZATION")
    print("="*60)
    
    import xml.etree.ElementTree as ET
    from pathlib import Path
    
    fintabnet_path = Path(DATASETS['fintabnet'])
    images_dir = fintabnet_path / 'images'
    test_dir = fintabnet_path / 'test'
    
    # Get test XML files
    xml_files = list(test_dir.glob('*.xml'))[:num_samples]
    
    if not xml_files:
        print("  No XML files found in FinTabNet test directory")
        return
    
    for i, xml_file in enumerate(xml_files):
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image filename
        filename = root.find('filename').text
        img_path = images_dir / filename
        
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Parse objects (cells)
        cells = []
        table_box = None
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            box = [
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text),
            ]
            
            if name == 'table':
                table_box = box
            else:
                cells.append({
                    'type': name,
                    'box': box,
                })
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Table Image\n({filename})', fontsize=12)
        axes[0].axis('off')
        
        # Right: Image with cell boxes
        axes[1].imshow(img)
        axes[1].set_title(f'Ground Truth Structure\n({len(cells)} cells)', fontsize=12)
        
        # Color map for cell types
        cell_colors = {
            'table column': '#FF6B6B',
            'table row': '#4ECDC4',
            'table spanning cell': '#FFE66D',
            'table column header': '#95E1D3',
            'table projected row header': '#F38181',
        }
        
        # Draw table box
        if table_box:
            rect = patches.Rectangle(
                (table_box[0], table_box[1]), 
                table_box[2]-table_box[0], table_box[3]-table_box[1],
                linewidth=3, edgecolor='blue', facecolor='none', linestyle='--'
            )
            axes[1].add_patch(rect)
        
        # Draw cell boxes
        for cell in cells:
            box = cell['box']
            color = cell_colors.get(cell['type'], '#AAAAAA')
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
            )
            axes[1].add_patch(rect)
        
        axes[1].axis('off')
        
        # Add legend
        legend_text = "Cell Types:\n"
        for ctype, color in cell_colors.items():
            legend_text += f"  {color}: {ctype}\n"
        axes[1].text(1.02, 0.98, legend_text, transform=axes[1].transAxes, fontsize=9,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = output_dir / f'comparison_tsr_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 2,
            'task': 'Table Structure Recognition (TSR)',
            'dataset': 'FinTabNet_c',
            'input': {
                'image_file': filename,
                'image_size': [img_width, img_height],
            },
            'ground_truth': {
                'table_box': table_box,
                'num_cells': len(cells),
                'cells': cells[:20],  # First 20
                'cell_types': list(set(c['type'] for c in cells)),
            },
            'evaluation': {
                'description': 'TSR identifies table structure: rows, columns, spanning cells',
                'metrics': ['TEDS (Tree Edit Distance Score)', 'Cell Detection F1'],
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        
        print(f"  Sample {i+1}: {filename}")
        print(f"    Table: {table_box is not None}, Cells: {len(cells)}")
        print(f"    Saved: {comp_path}")


def visualize_step3_ocr_pubtabnet(output_dir: Path, num_samples: int = 1):
    """
    Step 3: OCR 可视化 (使用 PubTabNet)
    显示：原图 + OCR 结果对比
    """
    print("\n" + "="*60)
    print("STEP 3: OCR VISUALIZATION (PubTabNet)")
    print("="*60)
    
    import jsonlines
    from pathlib import Path
    
    pubtabnet_path = Path(DATASETS['pubtabnet'])
    jsonl_path = pubtabnet_path / 'PubTabNet_2.0.0.jsonl'
    val_dir = pubtabnet_path / 'val'
    
    # Read samples from jsonl
    samples = []
    with jsonlines.open(jsonl_path) as reader:
        for item in reader:
            if item.get('split') == 'val':
                samples.append(item)
                if len(samples) >= num_samples:
                    break
    
    if not samples:
        print("  No val samples found in PubTabNet")
        return
    
    for i, sample in enumerate(samples):
        filename = sample['filename']
        img_path = val_dir / filename
        
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get HTML structure (contains text)
        html = sample.get('html', {})
        html_cells = html.get('cells', [])
        structure = html.get('structure', {})
        
        # Extract text from cells
        gt_texts = []
        for cell in html_cells:
            tokens = cell.get('tokens', [])
            text = ''.join(tokens).strip()
            if text and text not in ['<b>', '</b>', '<i>', '</i>']:
                # Clean HTML tags
                import re
                text = re.sub(r'<[^>]+>', '', text).strip()
                if text:
                    gt_texts.append(text)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # Left: Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Table Image\n({filename})', fontsize=12)
        axes[0].axis('off')
        
        # Right: Cell text content
        axes[1].axis('off')
        axes[1].set_title('Ground Truth Cell Texts', fontsize=12)
        
        result_text = f"PubTabNet Sample: {filename}\n"
        result_text += f"Total Cells: {len(html_cells)}\n"
        result_text += f"Cells with Text: {len(gt_texts)}\n\n"
        result_text += "Cell Texts (first 25):\n"
        result_text += "-" * 40 + "\n"
        
        for j, text in enumerate(gt_texts[:25]):
            result_text += f"{j+1}. \"{text[:40]}\"\n"
        
        if len(gt_texts) > 25:
            result_text += f"... (+{len(gt_texts)-25} more)\n"
        
        axes[1].text(0.05, 0.95, result_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = output_dir / f'comparison_ocr_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 3,
            'task': 'OCR (Optical Character Recognition)',
            'dataset': 'PubTabNet',
            'input': {
                'image_file': filename,
                'image_size': list(img.size),
            },
            'ground_truth': {
                'num_cells': len(html_cells),
                'num_text_cells': len(gt_texts),
                'cell_texts': gt_texts[:30],
                'html_structure': structure.get('tokens', [])[:20] if structure else [],
            },
            'evaluation': {
                'description': 'OCR extracts text from table cells',
                'metrics': ['CER (Character Error Rate)', 'Word Error Rate'],
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        print(f"  Sample {i+1}: {filename}")
        print(f"    Cells: {len(html_cells)}, Text cells: {len(gt_texts)}")
        print(f"    Saved: {comp_path}")


def visualize_step3_ocr_fintabnet(output_dir: Path, num_samples: int = 1):
    """
    Step 3: OCR 可视化 (使用 FinTabNet_c - 有 word-level bbox)
    显示：原图 + 每个word的边界框和文字
    """
    print("\n" + "="*60)
    print("STEP 3: OCR VISUALIZATION (FinTabNet_c)")
    print("="*60)
    
    from pathlib import Path
    
    fintabnet_path = Path(DATASETS['fintabnet'])
    images_dir = fintabnet_path / 'images'
    words_dir = fintabnet_path / 'words'
    
    # Get word JSON files
    word_files = list(words_dir.glob('*.json'))[:num_samples]
    
    if not word_files:
        print("  No word JSON files found in FinTabNet")
        return
    
    for i, word_file in enumerate(word_files):
        # Get corresponding image
        # word file: AAL_2003_page_25_table_0_words.json
        # image file: AAL_2003_page_25_table_0.jpg
        base_name = word_file.stem.replace('_words', '')
        img_path = images_dir / f'{base_name}.jpg'
        
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Load words
        with open(word_file, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        
        # Extract words and boxes
        words = []
        for w in words_data:
            words.append({
                'text': w['text'],
                'bbox': w['bbox'],  # [x1, y1, x2, y2]
            })
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # Left: Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Table Image\n({base_name}.jpg)', fontsize=12)
        axes[0].axis('off')
        
        # Right: Image with word bounding boxes
        axes[1].imshow(img)
        axes[1].set_title(f'OCR Word Annotations\n({len(words)} words)', fontsize=12)
        
        # Draw word boxes (sample some to avoid clutter)
        colors = plt.cm.tab20.colors
        for j, word in enumerate(words[:50]):  # Draw first 50 words
            bbox = word['bbox']
            color = colors[j % len(colors)]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.3
            )
            axes[1].add_patch(rect)
        
        axes[1].axis('off')
        
        # Add word list on the side
        word_list = '\n'.join([f"{j+1}. \"{w['text']}\"" for j, w in enumerate(words[:25])])
        if len(words) > 25:
            word_list += f'\n... (+{len(words)-25} more)'
        
        axes[1].text(1.02, 0.98, f'Words:\n{word_list}', transform=axes[1].transAxes, 
                    fontsize=8, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = output_dir / f'comparison_ocr_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 3,
            'task': 'OCR (Optical Character Recognition)',
            'dataset': 'FinTabNet_c',
            'input': {
                'image_file': f'{base_name}.jpg',
                'image_size': [img_width, img_height],
            },
            'ground_truth': {
                'num_words': len(words),
                'words': [w['text'] for w in words[:50]],
                'word_boxes': [w['bbox'] for w in words[:20]],
                'format': 'Each word has text and bbox [x1, y1, x2, y2]',
            },
            'evaluation': {
                'description': 'FinTabNet provides word-level bounding boxes for OCR evaluation',
                'metrics': ['CER (Character Error Rate)', 'Word Detection F1'],
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        print(f"  Sample {i+1}: {base_name}")
        print(f"    Words: {len(words)}")
        print(f"    Saved: {comp_path}")


def visualize_step3_ocr(output_dir: Path, num_samples: int = 1):
    """
    Step 3: OCR 可视化
    显示：原图 + OCR识别的文字标注
    """
    print("\n" + "="*60)
    print("STEP 3: OCR VISUALIZATION")
    print("="*60)
    
    from datasets import load_dataset
    
    np.random.seed(42)
    ds = load_dataset('parquet', data_dir=DATASETS['synfintabs'], split='test')
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        sample_id = item['id']
        
        # Get image
        img_data = item.get('image')
        if img_data is None:
            continue
        
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
        elif hasattr(img_data, 'convert'):
            img = img_data.convert('RGB')
        else:
            img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # Get OCR results
        ocr_results = item.get('ocr_results', {})
        ocr_words = ocr_results.get('words', []) if isinstance(ocr_results, dict) else []
        
        # Get GT cell texts
        rows_data = item['rows']
        gt_texts = []
        for row in rows_data:
            cells = row.get('cells', [])
            for cell in cells:
                if isinstance(cell, dict):
                    text = cell.get('text', '').strip()
                    if text:
                        gt_texts.append(text)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # Left: Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Image\n(Sample: {sample_id[:15]}...)', fontsize=12)
        axes[0].axis('off')
        
        # Right: Image with OCR words overlay
        axes[1].imshow(img)
        axes[1].set_title(f'OCR Results\n({len(ocr_words)} words detected)', fontsize=12)
        
        # Show OCR words as text list on the side
        ocr_text_display = '\n'.join(ocr_words[:30])  # First 30 words
        if len(ocr_words) > 30:
            ocr_text_display += f'\n... (+{len(ocr_words)-30} more)'
        
        axes[1].text(1.02, 0.98, f'OCR Words:\n{ocr_text_display}', 
                    transform=axes[1].transAxes, fontsize=8,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = output_dir / f'comparison_ocr_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 3,
            'task': 'OCR (Optical Character Recognition)',
            'input': {
                'sample_id': sample_id,
                'image_size': list(img.size),
            },
            'ground_truth': {
                'num_cells': len(gt_texts),
                'cell_texts': gt_texts[:20],  # First 20
            },
            'ocr_output': {
                'num_words': len(ocr_words),
                'words': ocr_words[:50],  # First 50
                'format': 'List of recognized text strings',
            },
            'evaluation': {
                'description': 'Compare OCR words with GT cell texts',
                'metrics': ['CER (Character Error Rate)', 'Exact Match Rate'],
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        print(f"  Sample {i+1}: {sample_id}")
        print(f"    GT cells: {len(gt_texts)}, OCR words: {len(ocr_words)}")
        print(f"    Saved: {comp_path}")


def visualize_step5_semantic(output_dir: Path, num_samples: int = 1):
    """
    Step 5: Semantic Cell Classification 可视化
    显示：原图 + 每个 cell 的分类结果（用颜色标注）
    """
    print("\n" + "="*60)
    print("STEP 5: SEMANTIC CLASSIFICATION VISUALIZATION")
    print("="*60)
    
    from datasets import load_dataset
    from baselines.docling_eval.stage2_step5_semantic import HeuristicCellClassifier
    
    np.random.seed(42)
    ds = load_dataset('parquet', data_dir=DATASETS['synfintabs'], split='test')
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    classifier = HeuristicCellClassifier()
    
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        sample_id = item['id']
        
        # Get image
        img_data = item.get('image')
        if img_data is None:
            continue
        
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
        elif hasattr(img_data, 'convert'):
            img = img_data.convert('RGB')
        else:
            img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # Process cells
        rows_data = item['rows']
        num_rows = len(rows_data)
        num_cols = max(len(row.get('cells', [])) for row in rows_data) if rows_data else 0
        
        cell_results = []
        for row_idx, row_dict in enumerate(rows_data):
            cells = row_dict.get('cells', [])
            for col_idx, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    continue
                
                text = cell.get('text', '')
                gt_label = cell.get('label', 'data')
                pred_label = classifier.classify(text, row_idx, col_idx, num_rows, num_cols, cells)
                
                cell_results.append({
                    'row': row_idx,
                    'col': col_idx,
                    'text': text[:30],
                    'gt_label': gt_label,
                    'pred_label': pred_label,
                    'correct': gt_label == pred_label,
                })
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        
        # Left: Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Image\n({sample_id[:15]}...)', fontsize=12)
        axes[0].axis('off')
        
        # Right: Classification results as table
        axes[1].axis('off')
        axes[1].set_title('Cell Classification Results', fontsize=12)
        
        # Create a text representation of the table with colors
        result_text = "Cell Classification Output:\n\n"
        result_text += f"Grid Size: {num_rows} rows x {num_cols} cols\n"
        result_text += f"Total Cells: {len(cell_results)}\n"
        result_text += f"Correct: {sum(1 for c in cell_results if c['correct'])}/{len(cell_results)}\n\n"
        result_text += "Legend:\n"
        result_text += "  [RH]=row_header  [CH]=column_header\n"
        result_text += "  [D]=data         [ST]=section_title\n"
        result_text += "  [CU]=currency_unit\n\n"
        result_text += "Sample Results (first 15 cells):\n"
        result_text += "-" * 50 + "\n"
        
        label_emoji = {
            'row_header': '[RH]',
            'column_header': '[CH]',
            'data': '[D]',
            'section_title': '[ST]',
            'currency_unit': '[CU]',
        }
        
        for c in cell_results[:15]:
            emoji = label_emoji.get(c['pred_label'], '⚪')
            status = '✓' if c['correct'] else '✗'
            result_text += f"{status} [{c['row']},{c['col']}] {emoji} {c['pred_label']:<15} \"{c['text']}\"\n"
        
        if len(cell_results) > 15:
            result_text += f"... (+{len(cell_results)-15} more cells)\n"
        
        axes[1].text(0.05, 0.95, result_text, transform=axes[1].transAxes, 
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = output_dir / f'comparison_semantic_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 5,
            'task': 'Semantic Cell Classification',
            'input': {
                'sample_id': sample_id,
                'grid_size': f'{num_rows}x{num_cols}',
                'total_cells': len(cell_results),
            },
            'classification_output': {
                'cell_types': ['section_title', 'column_header', 'row_header', 'data', 'currency_unit'],
                'results': cell_results[:30],  # First 30
                'format': 'For each cell: position, text, GT label, predicted label',
            },
            'evaluation': {
                'accuracy': sum(1 for c in cell_results if c['correct']) / len(cell_results) if cell_results else 0,
                'correct': sum(1 for c in cell_results if c['correct']),
                'total': len(cell_results),
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        print(f"  Sample {i+1}: {sample_id}")
        print(f"    Accuracy: {output_json['evaluation']['accuracy']:.1%}")
        print(f"    Saved: {comp_path}")


def visualize_step6_qa(output_dir: Path, num_samples: int = 1):
    """
    Step 6: QA Validation 可视化
    显示：原图 + QA问答过程 + 答案对比
    """
    print("\n" + "="*60)
    print("STEP 6: QA VALIDATION VISUALIZATION")
    print("="*60)
    
    from datasets import load_dataset
    
    np.random.seed(42)
    ds = load_dataset('parquet', data_dir=DATASETS['synfintabs'], split='test')
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    def normalize_answer(answer: str) -> str:
        if not answer:
            return ""
        text = str(answer).strip().replace(" ", "").replace(",", "").replace("'", "")
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]
        return text.lower()
    
    def fuzzy_match(text1: str, text2: str) -> bool:
        if not text1 or not text2:
            return False
        t1, t2 = text1.lower().strip(), text2.lower().strip()
        return t1 == t2 or t1 in t2 or t2 in t1
    
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        sample_id = item['id']
        
        # Get image
        img_data = item.get('image')
        if img_data is None:
            continue
        
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
        elif hasattr(img_data, 'convert'):
            img = img_data.convert('RGB')
        else:
            img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # Build grid
        rows_data = item['rows']
        questions = item['questions'][:5]
        
        grid = []
        row_labels = []
        col_headers = []
        
        if rows_data and len(rows_data) > 0:
            first_row = rows_data[0]
            first_cells = first_row.get('cells', [])
            col_headers = [cell.get('text', '') if isinstance(cell, dict) else '' for cell in first_cells]
            
            for row in rows_data:
                cells = row.get('cells', [])
                row_data = [cell.get('text', '') if isinstance(cell, dict) else '' for cell in cells]
                if row_data:
                    grid.append(row_data)
                    row_labels.append(row_data[0] if row_data else '')
        
        # Process QA
        qa_results = []
        for q in questions:
            gt_answer = q['answer']
            answer_keys = q.get('answer_keys', {})
            row_key = answer_keys.get('row', '')
            col_key = answer_keys.get('col', '')
            
            # Find row
            row_idx = None
            for j, label in enumerate(row_labels):
                if fuzzy_match(label, row_key):
                    row_idx = j
                    break
            
            # Find col - need exact match or close match
            col_idx = None
            for j, header in enumerate(col_headers):
                if header and col_key:  # Skip empty headers
                    col_key_norm = col_key.lower().strip()
                    header_norm = header.lower().strip()
                    if col_key_norm == header_norm or col_key_norm in header_norm or header_norm in col_key_norm:
                        col_idx = j
                        break
            
            # Get cell value
            pred_answer = ""
            if row_idx is not None and col_idx is not None:
                if row_idx < len(grid) and col_idx < len(grid[row_idx]):
                    pred_answer = grid[row_idx][col_idx]
            
            is_correct = normalize_answer(pred_answer) == normalize_answer(gt_answer)
            
            qa_results.append({
                'question': q.get('question', ''),  # 完整问题文本
                'row_key': row_key,
                'col_key': col_key,
                'row_found': row_idx,
                'col_found': col_idx,
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'correct': is_correct,
            })
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        
        # Left: Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Table Image\n({sample_id[:15]}...)', fontsize=12)
        axes[0].axis('off')
        
        # Right: QA results
        axes[1].axis('off')
        axes[1].set_title('QA Validation Process', fontsize=12)
        
        result_text = "QA Validation Output:\n\n"
        result_text += f"Table Grid: {len(grid)} rows x {len(col_headers)} cols\n"
        result_text += f"Questions: {len(qa_results)}\n"
        result_text += f"Correct: {sum(1 for q in qa_results if q['correct'])}/{len(qa_results)}\n\n"
        result_text += "Process:\n"
        result_text += "1. Parse row_key and col_key from question\n"
        result_text += "2. Find matching row label (fuzzy match)\n"
        result_text += "3. Find matching column header\n"
        result_text += "4. Extract cell value at [row, col]\n"
        result_text += "5. Compare with GT answer\n\n"
        result_text += "-" * 55 + "\n"
        result_text += "QA Results:\n"
        
        for j, qa in enumerate(qa_results):
            status = '✓' if qa['correct'] else '✗'
            result_text += f"\nQ{j+1}: {status}\n"
            result_text += f"  \"{qa['question'][:50]}...\"\n"
            result_text += f"  Row Key: \"{qa['row_key'][:20]}\" → idx={qa['row_found']}\n"
            result_text += f"  Col Key: \"{qa['col_key']}\" → idx={qa['col_found']}\n"
            result_text += f"  GT:   \"{qa['gt_answer']}\"  |  Pred: \"{qa['pred_answer']}\"\n"
        
        axes[1].text(0.05, 0.95, result_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = output_dir / f'comparison_qa_{i+1}.png'
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save output JSON
        output_json = {
            'step': 6,
            'task': 'End-to-End QA Validation',
            'input': {
                'sample_id': sample_id,
                'grid_size': f'{len(grid)}x{len(col_headers)}',
                'num_questions': len(qa_results),
            },
            'table_structure': {
                'column_headers': col_headers[:10],
                'row_labels': row_labels[:10],
                'sample_grid': [row[:5] for row in grid[:5]],  # First 5x5
            },
            'qa_output': {
                'results': qa_results,
                'process': [
                    '1. Extract row_key and col_key from question',
                    '2. Find row index by fuzzy matching row_key with row labels',
                    '3. Find col index by matching col_key with column headers',
                    '4. Retrieve cell value at grid[row_idx][col_idx]',
                    '5. Normalize and compare with ground truth answer',
                ],
            },
            'evaluation': {
                'accuracy': sum(1 for q in qa_results if q['correct']) / len(qa_results) if qa_results else 0,
                'correct': sum(1 for q in qa_results if q['correct']),
                'total': len(qa_results),
            }
        }
        
        json_path = output_dir / f'output_sample_{i+1}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        print(f"  Sample {i+1}: {sample_id}")
        print(f"    Accuracy: {output_json['evaluation']['accuracy']:.0%}")
        print(f"    Saved: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Step Results')
    parser.add_argument('--num-samples', type=int, default=1, help='Samples per step')
    parser.add_argument('--output-dir', type=str, default='./outputs/thesis_figures', help='Base output directory')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3, 5, 6], help='Steps to visualize')
    parser.add_argument('--ocr-dataset', type=str, default='synfintabs', 
                       choices=['synfintabs', 'pubtabnet', 'fintabnet'],
                       help='Dataset for Step 3 OCR: synfintabs, pubtabnet, or fintabnet')
    
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    
    # Choose OCR function based on dataset
    ocr_functions = {
        'synfintabs': visualize_step3_ocr,
        'pubtabnet': visualize_step3_ocr_pubtabnet,
        'fintabnet': visualize_step3_ocr_fintabnet,
    }
    ocr_func = ocr_functions[args.ocr_dataset]
    
    step_functions = {
        1: ('step1_detection', visualize_step1_detection),
        2: ('step2_tsr', visualize_step2_tsr),
        3: ('step3_ocr', ocr_func),
        5: ('step5_semantic', visualize_step5_semantic),
        6: ('step6_qa_validation', visualize_step6_qa),
    }
    
    print(f"\n{'='*60}")
    print("VISUALIZING STEP RESULTS")
    print(f"{'='*60}")
    print(f"Samples per step: {args.num_samples}")
    print(f"Steps to visualize: {args.steps}")
    if 3 in args.steps:
        print(f"Step 3 OCR dataset: {args.ocr_dataset}")
    
    for step in args.steps:
        if step not in step_functions:
            print(f"  Step {step}: Not implemented for visualization")
            continue
        
        folder_name, func = step_functions[step]
        step_dir = base_dir / folder_name
        step_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            func(step_dir, args.num_samples)
        except Exception as e:
            print(f"  ERROR in Step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETED")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
