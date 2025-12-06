"""
OCR Module for Table Text Extraction

Uses PaddleOCR for extracting text from table images.
Aligns OCR results to detected cell regions using IoU-based matching.

v2.0 Changes:
- Improved alignment using IoU instead of center-point only
- Added multiple alignment strategies
- Better handling of text spanning multiple cells
"""
import numpy as np
from PIL import Image, ImageOps
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Suppress PaddleOCR verbose logging
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Lazy load PaddleOCR to avoid import delays
_ocr_instance = None


def get_ocr_instance(lang: str = 'en'):
    """Get or create PaddleOCR instance (singleton pattern)."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR with English model
        # use_angle_cls=True for rotated text detection
        # use_gpu=True if available
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            use_gpu=True
        )
    return _ocr_instance


class TableOCR:
    """
    OCR module for extracting text from table images.
    
    Uses PaddleOCR for text detection and recognition.
    """
    
    def __init__(self, lang: str = 'en', use_gpu: bool = True):
        """
        Initialize TableOCR.
        
        Args:
            lang: Language for OCR ('en' for English)
            use_gpu: Whether to use GPU acceleration
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None
    
    @property
    def ocr(self):
        """Lazy load OCR model."""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            import warnings
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            # PaddleOCR v3 API - simplified parameters
            self._ocr = PaddleOCR(lang=self.lang)
        return self._ocr
    
    def extract_text(
        self,
        image: Union[Image.Image, np.ndarray, str],
        padding: int = 32
    ) -> List[Dict]:
        """
        Extract all text from an image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            
        Returns:
            List of dicts with 'text', 'bbox', 'confidence'
            bbox format: [x1, y1, x2, y2]
        """
        pad = padding
        pad_tuple = (pad, pad, pad, pad)
        undo_padding = False
        if isinstance(image, Image.Image):
            image = ImageOps.expand(image, border=pad, fill='white')
            undo_padding = True
            np_image = np.array(image)
        elif isinstance(image, np.ndarray):
            image = ImageOps.expand(Image.fromarray(image), border=pad, fill='white')
            undo_padding = True
            np_image = np.array(image)
        else:
            np_image = image
        
        # Run OCR - PaddleOCR v3 uses predict() method
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        results = self.ocr.predict(np_image)
        
        if not results:
            return []
        
        extracted = []
        
        # PaddleOCR v3 output format
        for result in results:
            if 'rec_texts' in result and 'rec_polys' in result and 'rec_scores' in result:
                texts = result['rec_texts']
                polys = result['rec_polys']
                scores = result['rec_scores']
                
                for i, (text, poly, score) in enumerate(zip(texts, polys, scores)):
                    # Convert polygon to bounding box [x1, y1, x2, y2]
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    if undo_padding:
                        bbox = [
                            bbox[0] - pad,
                            bbox[1] - pad,
                            bbox[2] - pad,
                            bbox[3] - pad,
                        ]
                    
                    extracted.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': float(score)
                    })
        
        return extracted
    
    def align_text_to_cells(
        self,
        ocr_results: List[Dict],
        cell_bboxes: List[List[float]],
        iou_threshold: float = 0.1
    ) -> List[str]:
        """
        Align OCR text to cell bounding boxes using IoU matching.
        
        Args:
            ocr_results: List of OCR results from extract_text()
            cell_bboxes: List of cell bounding boxes [x1, y1, x2, y2]
            iou_threshold: Minimum IoU to assign text to cell
            
        Returns:
            List of text strings, one per cell
        """
        cell_texts = [''] * len(cell_bboxes)
        
        for ocr_item in ocr_results:
            text = ocr_item['text']
            text_bbox = ocr_item['bbox']
            
            # Find best matching cell using IoU
            best_cell_idx = -1
            best_iou = 0
            
            for cell_idx, cell_bbox in enumerate(cell_bboxes):
                iou = self._calculate_iou(text_bbox, cell_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_cell_idx = cell_idx
            
            # If no good IoU match, try containment check
            if best_cell_idx < 0:
                text_center = [
                    (text_bbox[0] + text_bbox[2]) / 2,
                    (text_bbox[1] + text_bbox[3]) / 2
                ]
                for cell_idx, cell_bbox in enumerate(cell_bboxes):
                    if (cell_bbox[0] <= text_center[0] <= cell_bbox[2] and
                        cell_bbox[1] <= text_center[1] <= cell_bbox[3]):
                        best_cell_idx = cell_idx
                        break
            
            # Assign text to cell
            if best_cell_idx >= 0:
                if cell_texts[best_cell_idx]:
                    cell_texts[best_cell_idx] += ' ' + text
                else:
                    cell_texts[best_cell_idx] = text
        
        return cell_texts
    
    def align_text_to_grid(
        self,
        ocr_results: List[Dict],
        rows: List[Dict],
        columns: List[Dict],
        method: str = 'iou'
    ) -> List[List[str]]:
        """
        Align OCR text to a row-column grid structure using improved matching.
        
        Args:
            ocr_results: List of OCR results from extract_text()
            rows: List of row dicts with 'bbox' key
            columns: List of column dicts with 'bbox' key
            method: Alignment method - 'iou', 'center', or 'hybrid'
            
        Returns:
            2D list of text strings [row][col]
        """
        num_rows = len(rows)
        num_cols = len(columns)
        
        if num_rows == 0 or num_cols == 0:
            return []
        
        # Initialize grid
        grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Sort OCR results by y-coordinate (top to bottom), then x (left to right)
        sorted_ocr = sorted(ocr_results, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        for ocr_item in sorted_ocr:
            text = ocr_item['text']
            text_bbox = ocr_item['bbox']
            
            if method == 'iou':
                row_idx, col_idx = self._find_cell_by_iou(text_bbox, rows, columns)
            elif method == 'center':
                row_idx, col_idx = self._find_cell_by_center(text_bbox, rows, columns)
            else:  # hybrid
                row_idx, col_idx = self._find_cell_hybrid(text_bbox, rows, columns)
            
            # Assign to grid
            if row_idx >= 0 and col_idx >= 0:
                if grid[row_idx][col_idx]:
                    grid[row_idx][col_idx] += ' ' + text
                else:
                    grid[row_idx][col_idx] = text
        
        return grid
    
    def _find_cell_by_center(
        self,
        text_bbox: List[float],
        rows: List[Dict],
        columns: List[Dict]
    ) -> Tuple[int, int]:
        """Find cell using center point matching (original method)."""
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
        # Find matching row
        row_idx = -1
        for i, row in enumerate(rows):
            row_bbox = row['bbox']
            if row_bbox[1] <= text_center_y <= row_bbox[3]:
                row_idx = i
                break
        
        # Find matching column
        col_idx = -1
        for j, col in enumerate(columns):
            col_bbox = col['bbox']
            if col_bbox[0] <= text_center_x <= col_bbox[2]:
                col_idx = j
                break
        
        return row_idx, col_idx
    
    def _find_cell_by_iou(
        self,
        text_bbox: List[float],
        rows: List[Dict],
        columns: List[Dict]
    ) -> Tuple[int, int]:
        """Find cell using IoU matching."""
        best_row_idx = -1
        best_row_iou = 0
        
        # Find best matching row by IoU
        for i, row in enumerate(rows):
            row_bbox = row['bbox']
            # Create a horizontal slice for the row
            row_slice = [text_bbox[0], row_bbox[1], text_bbox[2], row_bbox[3]]
            iou = self._calculate_iou(text_bbox, row_slice)
            # Also consider overlap ratio (how much of text is in this row)
            overlap_ratio = self._calculate_overlap_ratio(text_bbox, row_bbox, axis='y')
            combined_score = iou * 0.3 + overlap_ratio * 0.7
            
            if combined_score > best_row_iou:
                best_row_iou = combined_score
                best_row_idx = i
        
        best_col_idx = -1
        best_col_iou = 0
        
        # Find best matching column by IoU
        for j, col in enumerate(columns):
            col_bbox = col['bbox']
            # Create a vertical slice for the column
            col_slice = [col_bbox[0], text_bbox[1], col_bbox[2], text_bbox[3]]
            iou = self._calculate_iou(text_bbox, col_slice)
            # Also consider overlap ratio
            overlap_ratio = self._calculate_overlap_ratio(text_bbox, col_bbox, axis='x')
            combined_score = iou * 0.3 + overlap_ratio * 0.7
            
            if combined_score > best_col_iou:
                best_col_iou = combined_score
                best_col_idx = j
        
        return best_row_idx, best_col_idx
    
    def _find_cell_hybrid(
        self,
        text_bbox: List[float],
        rows: List[Dict],
        columns: List[Dict]
    ) -> Tuple[int, int]:
        """
        Hybrid method: try IoU first, fall back to center if no good match.
        """
        # Try IoU first
        row_idx, col_idx = self._find_cell_by_iou(text_bbox, rows, columns)
        
        # If either failed, try center method
        if row_idx < 0 or col_idx < 0:
            center_row, center_col = self._find_cell_by_center(text_bbox, rows, columns)
            if row_idx < 0:
                row_idx = center_row
            if col_idx < 0:
                col_idx = center_col
        
        # Last resort: find closest row/column
        if row_idx < 0:
            row_idx = self._find_closest_row(text_bbox, rows)
        if col_idx < 0:
            col_idx = self._find_closest_column(text_bbox, columns)
        
        return row_idx, col_idx
    
    def _find_closest_row(self, text_bbox: List[float], rows: List[Dict]) -> int:
        """Find the closest row to the text (by vertical distance)."""
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        min_dist = float('inf')
        best_idx = -1
        
        for i, row in enumerate(rows):
            row_bbox = row['bbox']
            row_center_y = (row_bbox[1] + row_bbox[3]) / 2
            dist = abs(text_center_y - row_center_y)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        return best_idx
    
    def _find_closest_column(self, text_bbox: List[float], columns: List[Dict]) -> int:
        """Find the closest column to the text (by horizontal distance)."""
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        min_dist = float('inf')
        best_idx = -1
        
        for j, col in enumerate(columns):
            col_bbox = col['bbox']
            col_center_x = (col_bbox[0] + col_bbox[2]) / 2
            dist = abs(text_center_x - col_center_x)
            if dist < min_dist:
                min_dist = dist
                best_idx = j
        
        return best_idx
    
    def _calculate_overlap_ratio(
        self,
        bbox1: List[float],
        bbox2: List[float],
        axis: str = 'both'
    ) -> float:
        """
        Calculate what fraction of bbox1 overlaps with bbox2.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            axis: 'x' for horizontal, 'y' for vertical, 'both' for area
            
        Returns:
            Overlap ratio (0 to 1)
        """
        if axis == 'x':
            # Horizontal overlap
            overlap_start = max(bbox1[0], bbox2[0])
            overlap_end = min(bbox1[2], bbox2[2])
            if overlap_end <= overlap_start:
                return 0.0
            text_width = bbox1[2] - bbox1[0]
            if text_width <= 0:
                return 0.0
            return (overlap_end - overlap_start) / text_width
        
        elif axis == 'y':
            # Vertical overlap
            overlap_start = max(bbox1[1], bbox2[1])
            overlap_end = min(bbox1[3], bbox2[3])
            if overlap_end <= overlap_start:
                return 0.0
            text_height = bbox1[3] - bbox1[1]
            if text_height <= 0:
                return 0.0
            return (overlap_end - overlap_start) / text_height
        
        else:
            # Area overlap
            intersection = self._calculate_overlap(bbox1, bbox2)
            text_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            if text_area <= 0:
                return 0.0
            return intersection / text_area
    
    def _calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate intersection area between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        return (x2 - x1) * (y2 - y1)
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score (0 to 1)
        """
        intersection = self._calculate_overlap(bbox1, bbox2)
        
        if intersection == 0:
            return 0.0
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union


def extract_table_text(
    image: Union[Image.Image, np.ndarray, str],
    structure: Dict,
    lang: str = 'en',
    align_method: str = 'hybrid'
) -> Tuple[List[List[str]], List[Dict]]:
    """
    Convenience function to extract text from a table image.
    
    Args:
        image: Table image
        structure: Structure dict from TableStructureRecognizer
        lang: OCR language
        align_method: Alignment method - 'iou', 'center', or 'hybrid' (default)
        
    Returns:
        Tuple of (grid_texts, raw_ocr_results)
        grid_texts: 2D list [row][col] of cell text
        raw_ocr_results: List of all OCR detections
    """
    ocr = TableOCR(lang=lang)
    
    # Extract all text
    ocr_results = ocr.extract_text(image)
    
    # Align to grid using specified method
    rows = structure.get('rows', [])
    columns = structure.get('columns', [])
    
    grid_texts = ocr.align_text_to_grid(ocr_results, rows, columns, method=align_method)
    
    return grid_texts, ocr_results


# Test function
def test_ocr():
    """Test OCR module with a sample image."""
    print("Testing PaddleOCR...")
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text
    draw.text((50, 50), "Revenue", fill='black')
    draw.text((200, 50), "2024", fill='black')
    draw.text((300, 50), "2023", fill='black')
    draw.text((50, 100), "Sales", fill='black')
    draw.text((200, 100), "$100M", fill='black')
    draw.text((300, 100), "$90M", fill='black')
    
    # Test OCR
    ocr = TableOCR(lang='en')
    results = ocr.extract_text(img)
    
    print(f"\nExtracted {len(results)} text regions:")
    for r in results:
        print(f"  '{r['text']}' (conf: {r['confidence']:.3f})")
    
    print("\nOCR test completed!")
    return results


if __name__ == '__main__':
    test_ocr()
