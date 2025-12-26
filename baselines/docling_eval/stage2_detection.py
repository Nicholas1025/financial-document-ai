#!/usr/bin/env python3
"""
Stage II: Table Detection Evaluation on DocLayNet

Compares:
1. Table Transformer (microsoft/table-transformer-detection)
2. Docling LayoutPredictor

Metrics:
- mAP@0.5 (COCO-style)
- Precision/Recall/F1@0.5
- Average IoU

Reference: Docling official DocLayNet results
- Table Detection mAP[0.5:0.95]: 0.8726
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Our modules
from modules.utils import load_config, get_device


@dataclass
class DetectionResult:
    """Detection evaluation result for one sample."""
    filename: str
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    avg_iou: float
    inference_time: float


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_detections(
    pred_boxes: List[List[float]], 
    gt_boxes: List[List[float]], 
    iou_threshold: float = 0.5
) -> Tuple[int, int, int, float]:
    """
    Match predictions to ground truth using Hungarian-style greedy matching.
    
    Returns:
        tp, fp, fn, avg_iou
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, 1.0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0.0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    
    while True:
        # Find max IoU
        max_iou = 0
        max_idx = (-1, -1)
        for i in range(len(pred_boxes)):
            if i in matched_pred:
                continue
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_idx = (i, j)
        
        if max_iou < iou_threshold:
            break
            
        matched_pred.add(max_idx[0])
        matched_gt.add(max_idx[1])
        matched_ious.append(max_iou)
    
    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    avg_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    return tp, fp, fn, avg_iou


class TableTransformerDetector:
    """Table Transformer Detection model."""
    
    def __init__(self, device: torch.device, threshold: float = 0.5):
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        
        model_name = "microsoft/table-transformer-detection"
        print(f"Loading Table Transformer Detection: {model_name}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        
    @torch.no_grad()
    def detect(self, image: Image.Image) -> Tuple[List[List[float]], List[float]]:
        """
        Detect tables in image.
        
        Returns:
            boxes: List of [x1, y1, x2, y2]
            scores: List of confidence scores
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.threshold
        )[0]
        
        boxes = results['boxes'].cpu().numpy().tolist()
        scores = results['scores'].cpu().numpy().tolist()
        
        return boxes, scores


class DoclingLayoutDetector:
    """Docling LayoutPredictor for table detection."""
    
    def __init__(self, device: torch.device, threshold: float = 0.5):
        try:
            from docling.models.layout_model import LayoutModel
            from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
            
            print("Loading Docling LayoutPredictor...")
            
            # Get artifact path
            artifact_path = LayoutModel.download_models()
            
            # Initialize predictor - note: parameter is artifact_path (singular)
            device_str = "cuda" if torch.cuda.is_available() and 'cuda' in str(device) else "cpu"
            self.predictor = LayoutPredictor(
                artifact_path=str(artifact_path),
                device=device_str,
                num_threads=4
            )
            self.threshold = threshold
            self.available = True
            print("Docling LayoutPredictor loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load Docling LayoutPredictor: {e}")
            print("Docling detection will be skipped")
            self.available = False
    
    def detect(self, image: Image.Image) -> Tuple[List[List[float]], List[float]]:
        """
        Detect tables using Docling LayoutPredictor.
        
        Returns:
            boxes: List of [x1, y1, x2, y2]
            scores: List of confidence scores
        """
        if not self.available:
            return [], []
        
        import numpy as np
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Run prediction - returns a generator of dicts
        predictions = list(self.predictor.predict(image_np))
        
        boxes = []
        scores = []
        
        for pred in predictions:
            # pred is a dict: {'l', 't', 'r', 'b', 'label', 'confidence'}
            label = pred['label'].lower() if isinstance(pred, dict) else pred.label.lower()
            confidence = pred['confidence'] if isinstance(pred, dict) else pred.confidence
            
            # Filter for table class
            if label == 'table' and confidence >= self.threshold:
                # Get bbox coordinates
                if isinstance(pred, dict):
                    bbox = [pred['l'], pred['t'], pred['r'], pred['b']]
                else:
                    bbox = [pred.bbox.l, pred.bbox.t, pred.bbox.r, pred.bbox.b]
                boxes.append(bbox)
                scores.append(confidence)
        
        return boxes, scores


def load_doclaynet_samples(
    data_dir: str,
    split: str = 'val',
    num_samples: int = 100
) -> List[Dict]:
    """Load DocLayNet samples with table annotations."""
    
    coco_path = os.path.join(data_dir, 'COCO', f'{split}.json')
    images_dir = os.path.join(data_dir, 'PNG')
    
    print(f"Loading DocLayNet from: {coco_path}")
    
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build mappings
    id_to_image = {img['id']: img for img in coco_data['images']}
    
    # Find category ID for 'table'
    table_category_id = None
    for cat in coco_data['categories']:
        if cat['name'].lower() == 'table':
            table_category_id = cat['id']
            break
    
    if table_category_id is None:
        raise ValueError("Could not find 'table' category in DocLayNet")
    
    print(f"Table category ID: {table_category_id}")
    
    # Get table annotations grouped by image
    image_tables = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['category_id'] == table_category_id:
            img_id = ann['image_id']
            # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
            bbox = ann['bbox']
            image_tables[img_id].append({
                'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                'area': ann['area']
            })
    
    # Build samples (only images with tables)
    samples = []
    for img_id, tables in image_tables.items():
        img_info = id_to_image[img_id]
        samples.append({
            'image_id': img_id,
            'filename': img_info['file_name'],
            'image_path': os.path.join(images_dir, img_info['file_name']),
            'width': img_info['width'],
            'height': img_info['height'],
            'tables': tables
        })
    
    # Random sample
    if num_samples and len(samples) > num_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, num_samples)
    
    print(f"Loaded {len(samples)} images with tables")
    return samples


def evaluate_detector(
    detector,
    detector_name: str,
    samples: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """Evaluate a detector on samples."""
    
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    total_time = 0
    
    for sample in tqdm(samples, desc=f"Evaluating {detector_name}"):
        try:
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Get GT boxes
            gt_boxes = [t['bbox'] for t in sample['tables']]
            
            # Run detection
            start_time = time.time()
            pred_boxes, scores = detector.detect(image)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Match predictions to GT
            tp, fp, fn, avg_iou = match_detections(pred_boxes, gt_boxes, iou_threshold)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if avg_iou > 0:
                all_ious.append(avg_iou)
            
            # Compute per-sample metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            result = DetectionResult(
                filename=sample['filename'],
                gt_count=len(gt_boxes),
                pred_count=len(pred_boxes),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                avg_iou=avg_iou,
                inference_time=inference_time
            )
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing {sample['filename']}: {e}")
            continue
    
    # Compute overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0
    
    return {
        'detector': detector_name,
        'iou_threshold': iou_threshold,
        'num_samples': len(results),
        'metrics': {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1),
            'avg_iou': float(avg_iou),
            'total_tp': int(total_tp),
            'total_fp': int(total_fp),
            'total_fn': int(total_fn)
        },
        'timing': {
            'total_time': float(total_time),
            'avg_time_per_image': float(total_time / len(results)) if results else 0
        },
        'per_sample_results': [asdict(r) for r in results]
    }


def main():
    parser = argparse.ArgumentParser(description='Stage II: Detection Evaluation on DocLayNet')
    parser.add_argument('--data-dir', type=str, default='D:/datasets/DocLayNet',
                        help='Path to DocLayNet dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split (train/val/test)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--output-dir', type=str, default='./stage2_results',
                        help='Output directory')
    parser.add_argument('--skip-docling', action='store_true',
                        help='Skip Docling detector')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load samples
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    samples = load_doclaynet_samples(
        args.data_dir,
        split=args.split,
        num_samples=args.num_samples
    )
    
    # Initialize detectors
    print("\n" + "="*60)
    print("INITIALIZING DETECTORS")
    print("="*60)
    
    detectors = {}
    
    # Table Transformer
    print("\n[1] Table Transformer Detection")
    detectors['table_transformer'] = TableTransformerDetector(
        device=device,
        threshold=args.conf_threshold
    )
    
    # Docling
    if not args.skip_docling:
        print("\n[2] Docling LayoutPredictor")
        docling_detector = DoclingLayoutDetector(
            device=device,
            threshold=args.conf_threshold
        )
        if docling_detector.available:
            detectors['docling'] = docling_detector
    
    # Evaluate
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    all_results = {
        'experiment': 'stage2_detection',
        'dataset': 'DocLayNet',
        'split': args.split,
        'num_samples': args.num_samples,
        'iou_threshold': args.iou_threshold,
        'conf_threshold': args.conf_threshold,
        'timestamp': timestamp,
        'results': {}
    }
    
    for name, detector in detectors.items():
        print(f"\n--- Evaluating {name} ---")
        results = evaluate_detector(
            detector,
            name,
            samples,
            iou_threshold=args.iou_threshold
        )
        all_results['results'][name] = results
        
        # Print summary
        m = results['metrics']
        print(f"\n{name} Results @ IoU={args.iou_threshold}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  Avg IoU:   {m['avg_iou']:.4f}")
        print(f"  TP: {m['total_tp']}, FP: {m['total_fp']}, FN: {m['total_fn']}")
        print(f"  Avg time:  {results['timing']['avg_time_per_image']*1000:.1f}ms")
    
    # Save results
    output_file = os.path.join(args.output_dir, f'detection_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg IoU':<12}")
    print("-" * 73)
    
    for name, results in all_results['results'].items():
        m = results['metrics']
        print(f"{name:<25} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['avg_iou']:.4f}")
    
    # Reference
    print("\n" + "-"*73)
    print("Reference: Docling Official DocLayNet Results")
    print(f"{'Table mAP[0.5:0.95]':<25} {'0.8726':>12}")
    print("-" * 73)
    
    return all_results


if __name__ == '__main__':
    main()
