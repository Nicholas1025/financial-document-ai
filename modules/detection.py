"""
Table Detection Module using Microsoft Table Transformer

Uses pretrained DETR-based model for detecting tables in document images.
"""
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from .utils import get_device, rescale_bboxes, box_cxcywh_to_xyxy


class TableDetector:
    """
    Table Detection using Microsoft Table Transformer.
    
    Model: microsoft/table-transformer-detection
    """
    
    def __init__(self, config: Dict, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or get_device()
        self.threshold = config['models']['table_detection']['threshold']
        
        # Load model and processor
        model_name = config['models']['table_detection']['name']
        print(f"Loading Table Detection model: {model_name}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    @torch.no_grad()
    def detect(self, image: Image.Image) -> Dict:
        """
        Detect tables in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dict with 'boxes', 'scores', 'labels'
        """
        # Prepare input
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.threshold
        )[0]
        
        return {
            'boxes': results['boxes'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy(),
            'labels': results['labels'].cpu().numpy()
        }
    
    def detect_and_crop(self, image: Image.Image, padding: int = 10) -> List[Tuple[Image.Image, Dict]]:
        """
        Detect tables and crop them from the image.
        
        Args:
            image: PIL Image
            padding: Pixels to add around detected table
            
        Returns:
            List of (cropped_image, detection_info) tuples
        """
        results = self.detect(image)
        cropped_tables = []
        
        w, h = image.size
        
        for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
            # Add padding and clip to image bounds
            x1 = max(0, int(box[0]) - padding)
            y1 = max(0, int(box[1]) - padding)
            x2 = min(w, int(box[2]) + padding)
            y2 = min(h, int(box[3]) + padding)
            
            # Crop table
            cropped = image.crop((x1, y1, x2, y2))
            
            info = {
                'box': [x1, y1, x2, y2],
                'original_box': box.tolist(),
                'score': float(score),
                'table_id': i
            }
            
            cropped_tables.append((cropped, info))
        
        return cropped_tables


class TableDetectionEvaluator:
    """
    Evaluator for table detection performance.
    
    Computes Precision, Recall, F1 at various IoU thresholds.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.total_iou = 0
        self.num_matches = 0
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, pred_boxes: List[List[float]], gt_boxes: List[List[float]]):
        """
        Update metrics with predictions and ground truth for one image.
        
        Args:
            pred_boxes: List of predicted boxes [x1, y1, x2, y2]
            gt_boxes: List of ground truth boxes [x1, y1, x2, y2]
        """
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold:
                self.tp += 1
                self.total_iou += best_iou
                self.num_matches += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1
        
        # Count unmatched ground truths as false negatives
        self.fn += len(gt_boxes) - len(matched_gt)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dict with precision, recall, f1, avg_iou
        """
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = self.total_iou / self.num_matches if self.num_matches > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn
        }
