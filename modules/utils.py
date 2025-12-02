"""
Utility functions for Financial Document AI
"""
import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(image_path: str, max_size: int = 1000) -> Image.Image:
    """
    Load and optionally resize an image.
    
    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height)
        
    Returns:
        PIL Image in RGB format
    """
    image = Image.open(image_path).convert("RGB")
    
    # Resize if too large (for GPU memory)
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center format to corner format.
    
    Args:
        boxes: Tensor of shape (N, 4) in format [cx, cy, w, h]
        
    Returns:
        Tensor of shape (N, 4) in format [x1, y1, x2, y2]
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def rescale_bboxes(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Rescale bounding boxes to image size.
    
    Args:
        boxes: Normalized boxes (0-1)
        size: Image size (width, height)
        
    Returns:
        Rescaled boxes in pixel coordinates
    """
    w, h = size
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_results(results: Dict, output_path: str, format: str = "json"):
    """
    Save results to file.
    
    Args:
        results: Dictionary of results
        output_path: Path to save file
        format: Output format ("json" or "yaml")
    """
    ensure_dir(os.path.dirname(output_path))
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif format == "yaml":
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False)
    
    print(f"Results saved to: {output_path}")


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
