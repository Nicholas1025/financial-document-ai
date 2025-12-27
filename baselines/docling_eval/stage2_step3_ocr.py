"""
Stage 2 Step 3: OCR Evaluation using PubTabNet

PubTabNet provides character-level ground truth for table cells.
We evaluate OCR accuracy at the cell level by comparing:
- Ground truth text (from JSONL cells)
- OCR predicted text (from PaddleOCR / Docling)

Metrics:
- Character Error Rate (CER): Edit distance / GT length
- Word Error Rate (WER): Word-level edit distance
- Cell-level Exact Match: Percentage of cells with perfect match
- Normalized Edit Distance (NED): 1 - (edit_dist / max(len_gt, len_pred))

Dataset: PubTabNet (val split, ~9115 images)
Path: D:/datasets/PubTabNet/pubtabnet/pubtabnet/
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random
import csv

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class OCRMetrics:
    """Metrics for OCR evaluation."""
    total_cells: int = 0
    matched_cells: int = 0  # Cells where OCR found corresponding text
    exact_match: int = 0
    
    # Error rates (lower is better)
    total_cer: float = 0.0
    total_wer: float = 0.0
    total_ned: float = 0.0  # Normalized Edit Distance
    
    # Aggregated
    avg_cer: float = 0.0
    avg_wer: float = 0.0
    avg_ned: float = 0.0
    exact_match_rate: float = 0.0
    cell_detection_rate: float = 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def word_error_rate(gt_words: List[str], pred_words: List[str]) -> float:
    """Calculate Word Error Rate."""
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    
    distance = levenshtein_distance(gt_words, pred_words)
    return distance / len(gt_words)


def character_error_rate(gt: str, pred: str) -> float:
    """Calculate Character Error Rate."""
    if not gt:
        return 0.0 if not pred else 1.0
    
    distance = levenshtein_distance(gt, pred)
    return distance / len(gt)


def normalized_edit_distance(gt: str, pred: str) -> float:
    """Calculate Normalized Edit Distance (1 - NED = similarity)."""
    if not gt and not pred:
        return 0.0
    
    distance = levenshtein_distance(gt, pred)
    max_len = max(len(gt), len(pred))
    return distance / max_len if max_len > 0 else 0.0


class PubTabNetLoader:
    """Load PubTabNet dataset for OCR evaluation."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.jsonl_path = self.data_dir / "PubTabNet_2.0.0.jsonl"
        self.annotations = {}
        
    def load_annotations(self, split: str = 'val', max_samples: Optional[int] = None):
        """Load annotations for a specific split."""
        print(f"Loading PubTabNet annotations for split: {split}")
        
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                data = json.loads(line)
                if data.get('split') == split:
                    self.annotations[data['filename']] = data
                    count += 1
                    if max_samples and count >= max_samples:
                        break
        
        print(f"Loaded {len(self.annotations)} annotations")
        return self.annotations
    
    def get_image_path(self, filename: str, split: str = 'val') -> Path:
        """Get full path to image file."""
        return self.data_dir / split / filename
    
    def extract_cell_texts(self, annotation: Dict) -> List[Dict[str, Any]]:
        """Extract cell texts and bboxes from annotation."""
        cells = []
        html_data = annotation.get('html', {})
        
        for cell in html_data.get('cells', []):
            tokens = cell.get('tokens', [])
            bbox = cell.get('bbox')
            
            # Skip empty cells
            if not tokens or not bbox:
                continue
            
            # Reconstruct text from tokens, filtering out HTML tags
            text_parts = []
            for token in tokens:
                if not token.startswith('<') and not token.startswith('</'):
                    text_parts.append(token)
            
            text = ''.join(text_parts).strip()
            
            if text:  # Only include non-empty cells
                cells.append({
                    'text': text,
                    'bbox': bbox,  # [x1, y1, x2, y2]
                })
        
        return cells


class OCREvaluator:
    """Evaluate OCR performance against PubTabNet ground truth."""
    
    def __init__(self, ocr_backend: str = 'paddleocr'):
        self.ocr_backend = ocr_backend
        self.ocr = None
        
    def init_ocr(self):
        """Initialize OCR backend."""
        if self.ocr_backend == 'paddleocr':
            from modules.ocr import TableOCR
            self.ocr = TableOCR(lang='en', use_gpu=False)
        elif self.ocr_backend == 'docling':
            from modules.ocr import get_ocr_backend
            self.ocr = get_ocr_backend('docling')
        else:
            raise ValueError(f"Unknown OCR backend: {self.ocr_backend}")
    
    def run_ocr(self, image_path: str) -> List[Dict[str, Any]]:
        """Run OCR on an image and return results."""
        if self.ocr is None:
            self.init_ocr()
        
        results = self.ocr.extract_text(str(image_path))
        return results
    
    def match_ocr_to_gt(self, gt_cells: List[Dict], ocr_results: List[Dict],
                        iou_threshold: float = 0.3) -> List[Tuple[Dict, Optional[Dict]]]:
        """Match OCR results to ground truth cells based on bbox IoU."""
        matches = []
        
        for gt_cell in gt_cells:
            gt_bbox = gt_cell['bbox']  # [x1, y1, x2, y2]
            best_match = None
            best_iou = 0.0
            
            for ocr_result in ocr_results:
                ocr_bbox = ocr_result['bbox']  # [x1, y1, x2, y2]
                iou = self._calculate_iou(gt_bbox, ocr_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = ocr_result
            
            if best_iou >= iou_threshold:
                matches.append((gt_cell, best_match))
            else:
                matches.append((gt_cell, None))
        
        return matches
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_single(self, gt_cells: List[Dict], ocr_results: List[Dict]) -> Dict[str, Any]:
        """Evaluate OCR on a single image."""
        matches = self.match_ocr_to_gt(gt_cells, ocr_results)
        
        metrics = {
            'total_gt_cells': len(gt_cells),
            'matched_cells': 0,
            'exact_matches': 0,
            'cer_sum': 0.0,
            'wer_sum': 0.0,
            'ned_sum': 0.0,
            'cell_results': []
        }
        
        for gt_cell, ocr_match in matches:
            gt_text = gt_cell['text']
            
            if ocr_match is None:
                # No OCR match found
                pred_text = ""
                cer = 1.0
                wer = 1.0
                ned = 1.0
            else:
                pred_text = ocr_match.get('text', '')
                
                # Normalize texts for comparison
                gt_norm = self._normalize_text(gt_text)
                pred_norm = self._normalize_text(pred_text)
                
                cer = character_error_rate(gt_norm, pred_norm)
                
                gt_words = gt_norm.split()
                pred_words = pred_norm.split()
                wer = word_error_rate(gt_words, pred_words)
                
                ned = normalized_edit_distance(gt_norm, pred_norm)
                
                metrics['matched_cells'] += 1
                if gt_norm == pred_norm:
                    metrics['exact_matches'] += 1
            
            metrics['cer_sum'] += cer
            metrics['wer_sum'] += wer
            metrics['ned_sum'] += ned
            
            metrics['cell_results'].append({
                'gt_text': gt_text,
                'pred_text': pred_text,
                'cer': cer,
                'wer': wer,
                'ned': ned,
                'matched': ocr_match is not None
            })
        
        return metrics
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Lowercase for case-insensitive comparison
        text = text.lower()
        return text


def run_ocr_evaluation(
    data_dir: str = "D:/datasets/PubTabNet/pubtabnet/pubtabnet",
    split: str = 'val',
    ocr_backend: str = 'paddleocr',
    num_samples: int = 100,
    output_dir: str = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run OCR evaluation on PubTabNet.
    
    Args:
        data_dir: Path to PubTabNet dataset
        split: Dataset split to use ('val' or 'test')
        ocr_backend: OCR backend ('paddleocr' or 'docling')
        num_samples: Number of samples to evaluate
        output_dir: Output directory for results
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation results
    """
    random.seed(seed)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "stage2_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    loader = PubTabNetLoader(data_dir)
    
    # Load more annotations than needed, then sample
    loader.load_annotations(split=split, max_samples=num_samples * 2)
    
    # Sample annotations
    all_filenames = list(loader.annotations.keys())
    if len(all_filenames) > num_samples:
        sampled_filenames = random.sample(all_filenames, num_samples)
    else:
        sampled_filenames = all_filenames
    
    print(f"\nEvaluating {len(sampled_filenames)} samples with {ocr_backend}")
    
    # Initialize evaluator
    evaluator = OCREvaluator(ocr_backend=ocr_backend)
    evaluator.init_ocr()
    
    # Aggregate metrics
    total_metrics = OCRMetrics()
    all_results = []
    
    for filename in tqdm(sampled_filenames, desc=f"OCR Evaluation ({ocr_backend})"):
        annotation = loader.annotations[filename]
        image_path = loader.get_image_path(filename, split)
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Extract ground truth cells
        gt_cells = loader.extract_cell_texts(annotation)
        
        if not gt_cells:
            continue
        
        # Run OCR
        try:
            ocr_results = evaluator.run_ocr(str(image_path))
        except Exception as e:
            print(f"OCR error on {filename}: {e}")
            continue
        
        # Evaluate
        metrics = evaluator.evaluate_single(gt_cells, ocr_results)
        
        # Aggregate
        total_metrics.total_cells += metrics['total_gt_cells']
        total_metrics.matched_cells += metrics['matched_cells']
        total_metrics.exact_match += metrics['exact_matches']
        total_metrics.total_cer += metrics['cer_sum']
        total_metrics.total_wer += metrics['wer_sum']
        total_metrics.total_ned += metrics['ned_sum']
        
        all_results.append({
            'filename': filename,
            'gt_cells': metrics['total_gt_cells'],
            'matched_cells': metrics['matched_cells'],
            'exact_matches': metrics['exact_matches'],
            'avg_cer': metrics['cer_sum'] / metrics['total_gt_cells'] if metrics['total_gt_cells'] > 0 else 0,
            'avg_wer': metrics['wer_sum'] / metrics['total_gt_cells'] if metrics['total_gt_cells'] > 0 else 0,
            'avg_ned': metrics['ned_sum'] / metrics['total_gt_cells'] if metrics['total_gt_cells'] > 0 else 0,
        })
    
    # Calculate final averages
    if total_metrics.total_cells > 0:
        total_metrics.avg_cer = total_metrics.total_cer / total_metrics.total_cells
        total_metrics.avg_wer = total_metrics.total_wer / total_metrics.total_cells
        total_metrics.avg_ned = total_metrics.total_ned / total_metrics.total_cells
        total_metrics.exact_match_rate = total_metrics.exact_match / total_metrics.total_cells
        total_metrics.cell_detection_rate = total_metrics.matched_cells / total_metrics.total_cells
    
    # Prepare results
    results = {
        'config': {
            'dataset': 'PubTabNet',
            'split': split,
            'ocr_backend': ocr_backend,
            'num_samples': len(sampled_filenames),
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        },
        'metrics': {
            'total_cells': total_metrics.total_cells,
            'matched_cells': total_metrics.matched_cells,
            'exact_match_count': total_metrics.exact_match,
            'cell_detection_rate': round(total_metrics.cell_detection_rate * 100, 2),
            'exact_match_rate': round(total_metrics.exact_match_rate * 100, 2),
            'avg_cer': round(total_metrics.avg_cer * 100, 2),  # As percentage
            'avg_wer': round(total_metrics.avg_wer * 100, 2),
            'avg_ned': round(total_metrics.avg_ned * 100, 2),
            'accuracy': round((1 - total_metrics.avg_cer) * 100, 2),  # 1 - CER
        },
        'per_image_results': all_results
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ocr_eval_{ocr_backend}_{num_samples}samples_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")
    
    # Save CSV summary
    csv_path = output_dir / f"ocr_eval_{ocr_backend}_{num_samples}samples_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'gt_cells', 'matched_cells', 'exact_matches', 'avg_cer', 'avg_wer', 'avg_ned'])
        for r in all_results:
            writer.writerow([
                r['filename'], r['gt_cells'], r['matched_cells'], r['exact_matches'],
                f"{r['avg_cer']:.4f}", f"{r['avg_wer']:.4f}", f"{r['avg_ned']:.4f}"
            ])
    print(f"CSV saved to: {csv_path}")
    
    return results


def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    metrics = results['metrics']
    config = results['config']
    
    print("\n" + "=" * 60)
    print(f"OCR Evaluation Results - {config['ocr_backend'].upper()}")
    print("=" * 60)
    print(f"Dataset: {config['dataset']} ({config['split']} split)")
    print(f"Samples: {config['num_samples']}")
    print(f"Total Cells Evaluated: {metrics['total_cells']}")
    print("-" * 60)
    print(f"Cell Detection Rate:   {metrics['cell_detection_rate']:.2f}%")
    print(f"Exact Match Rate:      {metrics['exact_match_rate']:.2f}%")
    print(f"Character Error Rate:  {metrics['avg_cer']:.2f}%")
    print(f"Word Error Rate:       {metrics['avg_wer']:.2f}%")
    print(f"Normalized Edit Dist:  {metrics['avg_ned']:.2f}%")
    print(f"Character Accuracy:    {metrics['accuracy']:.2f}%")
    print("=" * 60)


def compare_ocr_backends(
    data_dir: str = "D:/datasets/PubTabNet/pubtabnet/pubtabnet",
    split: str = 'val',
    num_samples: int = 100,
    output_dir: str = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Compare PaddleOCR vs Docling OCR on the same samples."""
    
    random.seed(seed)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "stage2_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run both evaluations with same seed
    results = {}
    
    print("\n" + "=" * 60)
    print("Running OCR Comparison: PaddleOCR vs Docling")
    print("=" * 60)
    
    for backend in ['paddleocr', 'docling']:
        print(f"\n>>> Evaluating {backend}...")
        results[backend] = run_ocr_evaluation(
            data_dir=data_dir,
            split=split,
            ocr_backend=backend,
            num_samples=num_samples,
            output_dir=output_dir,
            seed=seed  # Same seed ensures same samples
        )
        print_results(results[backend])
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'PaddleOCR':>15} {'Docling':>15} {'Difference':>15}")
    print("-" * 70)
    
    paddle = results['paddleocr']['metrics']
    docling = results['docling']['metrics']
    
    for metric in ['cell_detection_rate', 'exact_match_rate', 'avg_cer', 'avg_wer', 'accuracy']:
        p_val = paddle[metric]
        d_val = docling[metric]
        diff = d_val - p_val
        sign = '+' if diff > 0 else ''
        print(f"{metric:<25} {p_val:>14.2f}% {d_val:>14.2f}% {sign}{diff:>13.2f}%")
    
    print("=" * 60)
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison = {
        'config': {
            'dataset': 'PubTabNet',
            'split': split,
            'num_samples': num_samples,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        },
        'paddleocr': paddle,
        'docling': docling,
    }
    
    json_path = output_dir / f"ocr_comparison_{num_samples}samples_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to: {json_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR Evaluation on PubTabNet")
    parser.add_argument("--data-dir", type=str, 
                        default="D:/datasets/PubTabNet/pubtabnet/pubtabnet",
                        help="Path to PubTabNet dataset")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--backend", type=str, default="paddleocr",
                        choices=["paddleocr", "docling", "compare"],
                        help="OCR backend or 'compare' for both")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.backend == "compare":
        compare_ocr_backends(
            data_dir=args.data_dir,
            split=args.split,
            num_samples=args.num_samples,
            seed=args.seed
        )
    else:
        results = run_ocr_evaluation(
            data_dir=args.data_dir,
            split=args.split,
            ocr_backend=args.backend,
            num_samples=args.num_samples,
            seed=args.seed
        )
        print_results(results)
