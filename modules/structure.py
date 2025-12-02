"""
Table Structure Recognition Module

Recognizes table structure (rows, columns, cells) from table images.
Uses Microsoft Table Transformer for structure recognition.
"""
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Dict, List, Tuple, Optional
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from .utils import get_device


class TableStructureRecognizer:
    """
    Table Structure Recognition using Microsoft Table Transformer.
    
    Models:
    - microsoft/table-transformer-structure-recognition
    - microsoft/table-transformer-structure-recognition-v1.1-all (better for complex tables)
    """
    
    # Structure element labels
    STRUCTURE_LABELS = {
        0: 'table',
        1: 'table column',
        2: 'table row',
        3: 'table column header',
        4: 'table projected row header',
        5: 'table spanning cell'
    }
    
    def __init__(self, config: Dict, device: Optional[torch.device] = None, use_v1_1: bool = False):
        self.config = config
        self.device = device or get_device()
        
        # Select model version
        if use_v1_1:
            model_config = config['models']['table_structure_v1_1']
        else:
            model_config = config['models']['table_structure']
        
        self.threshold = model_config['threshold']
        model_name = model_config['name']
        self.use_v1_1 = use_v1_1
        
        print(f"Loading Table Structure model: {model_name}")
        
        # For v1.1 model, use manual preprocessing to avoid size dict issues
        if use_v1_1:
            self.processor = None
            # Standard DETR normalization
            self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.max_size = 800
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manual preprocessing for v1.1 model."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio
        width, height = image.size
        scale = self.max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Convert to tensor and normalize
        img_tensor = T.ToTensor()(image)
        img_tensor = self.normalize(img_tensor)
        
        return img_tensor.unsqueeze(0), torch.tensor([[new_height, new_width]])
    
    def _box_cxcywh_to_xyxy(self, boxes):
        """Convert boxes from center format to corner format."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    @torch.no_grad()
    def recognize(self, image: Image.Image) -> Dict:
        """
        Recognize table structure from a table image.
        
        Args:
            image: PIL Image of a cropped table
            
        Returns:
            Dict with structure elements (rows, columns, cells, headers)
        """
        original_size = image.size  # (width, height)
        
        if self.use_v1_1:
            # Manual preprocessing for v1.1 model
            pixel_values, target_sizes = self._preprocess_image(image)
            pixel_values = pixel_values.to(self.device)
            target_sizes = target_sizes.to(self.device)
            
            # Run inference
            outputs = self.model(pixel_values=pixel_values)
            
            # Manual post-processing
            logits = outputs.logits
            pred_boxes = outputs.pred_boxes
            
            # Get predictions above threshold
            probs = logits.softmax(-1)[0, :, :-1]  # Remove no-object class
            scores, labels = probs.max(-1)
            keep = scores > self.threshold
            
            scores = scores[keep]
            labels = labels[keep]
            boxes = pred_boxes[0][keep]
            
            # Convert boxes from normalized cxcywh to xyxy
            boxes = self._box_cxcywh_to_xyxy(boxes)
            
            # Scale to original image size
            img_h, img_w = original_size[1], original_size[0]
            scale = torch.tensor([img_w, img_h, img_w, img_h], device=self.device)
            boxes = boxes * scale
            
            results = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }
        else:
            # Use processor for standard model
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            outputs = self.model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.threshold
            )[0]
        
        # Organize by element type
        structure = {
            'table': [],
            'columns': [],
            'rows': [],
            'column_headers': [],
            'projected_row_headers': [],
            'spanning_cells': []
        }
        
        for box, score, label in zip(
            results['boxes'].cpu().numpy(),
            results['scores'].cpu().numpy(),
            results['labels'].cpu().numpy()
        ):
            element = {
                'bbox': box.tolist(),
                'score': float(score)
            }
            
            label_name = self.STRUCTURE_LABELS.get(int(label), 'unknown')
            
            if label_name == 'table':
                structure['table'].append(element)
            elif label_name == 'table column':
                structure['columns'].append(element)
            elif label_name == 'table row':
                structure['rows'].append(element)
            elif label_name == 'table column header':
                structure['column_headers'].append(element)
            elif label_name == 'table projected row header':
                structure['projected_row_headers'].append(element)
            elif label_name == 'table spanning cell':
                structure['spanning_cells'].append(element)
        
        # Sort rows and columns by position
        structure['rows'].sort(key=lambda x: x['bbox'][1])  # Sort by y1
        structure['columns'].sort(key=lambda x: x['bbox'][0])  # Sort by x1
        
        return structure
    
    def structure_to_grid(self, structure: Dict) -> List[List[Dict]]:
        """
        Convert structure elements to a 2D grid of cells.
        
        Args:
            structure: Output from recognize()
            
        Returns:
            2D list representing table grid
        """
        rows = structure['rows']
        cols = structure['columns']
        
        if not rows or not cols:
            return []
        
        # Create grid
        grid = [[None for _ in range(len(cols))] for _ in range(len(rows))]
        
        # Assign cells to grid positions
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                # Find intersection
                cell_bbox = [
                    max(row['bbox'][0], col['bbox'][0]),
                    row['bbox'][1],
                    min(row['bbox'][2], col['bbox'][2]),
                    row['bbox'][3]
                ]
                
                grid[row_idx][col_idx] = {
                    'bbox': cell_bbox,
                    'row': row_idx,
                    'col': col_idx,
                    'is_header': row_idx == 0 or col_idx == 0  # Simple heuristic
                }
        
        return grid
    
    def structure_to_html(self, structure: Dict, cell_texts: Optional[List[List[str]]] = None) -> str:
        """
        Convert structure to HTML table.
        
        Args:
            structure: Output from recognize()
            cell_texts: Optional 2D list of cell text content
            
        Returns:
            HTML string
        """
        grid = self.structure_to_grid(structure)
        
        if not grid:
            return "<table></table>"
        
        html = "<table>\n"
        
        for row_idx, row in enumerate(grid):
            if row_idx == 0:
                html += "  <thead>\n"
            elif row_idx == 1:
                html += "  <tbody>\n"
            
            html += "    <tr>\n"
            
            for col_idx, cell in enumerate(row):
                tag = "th" if row_idx == 0 else "td"
                text = ""
                
                if cell_texts and row_idx < len(cell_texts) and col_idx < len(cell_texts[row_idx]):
                    text = cell_texts[row_idx][col_idx] or ""
                
                html += f"      <{tag}>{text}</{tag}>\n"
            
            html += "    </tr>\n"
            
            if row_idx == 0:
                html += "  </thead>\n"
        
        if len(grid) > 1:
            html += "  </tbody>\n"
        
        html += "</table>"
        
        return html


class StructureEvaluator:
    """
    Evaluator for table structure recognition.
    
    Computes metrics for row/column detection accuracy.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        self.row_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        self.col_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        self.cell_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _match_boxes(self, pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> Tuple[int, int, int]:
        """Match predicted and ground truth boxes."""
        matched_gt = set()
        tp = 0
        
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
                tp += 1
                matched_gt.add(best_gt_idx)
        
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)
        
        return tp, fp, fn
    
    def update(self, pred_structure: Dict, gt_structure: Dict):
        """Update metrics with predictions and ground truth."""
        # Evaluate rows
        pred_rows = [r['bbox'] for r in pred_structure.get('rows', [])]
        gt_rows = [r['bbox'] for r in gt_structure.get('rows', [])]
        tp, fp, fn = self._match_boxes(pred_rows, gt_rows)
        self.row_metrics['tp'] += tp
        self.row_metrics['fp'] += fp
        self.row_metrics['fn'] += fn
        
        # Evaluate columns
        pred_cols = [c['bbox'] for c in pred_structure.get('columns', [])]
        gt_cols = [c['bbox'] for c in gt_structure.get('columns', [])]
        tp, fp, fn = self._match_boxes(pred_cols, gt_cols)
        self.col_metrics['tp'] += tp
        self.col_metrics['fp'] += fp
        self.col_metrics['fn'] += fn
    
    def _compute_prf(self, metrics: Dict) -> Dict[str, float]:
        """Compute precision, recall, F1."""
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def compute(self) -> Dict:
        """Compute final metrics."""
        return {
            'rows': self._compute_prf(self.row_metrics),
            'columns': self._compute_prf(self.col_metrics),
            'row_counts': self.row_metrics,
            'column_counts': self.col_metrics
        }
