"""
Data loaders for all datasets:
- PubTabNet (JSONL)
- FinTabNet (XML)
- DocLayNet (COCO JSON)
- PubTables-1M (XML)
"""
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Generator, Optional, Tuple
from PIL import Image
import random


class PubTabNetLoader:
    """
    Loader for PubTabNet dataset.
    
    Format: JSONL with HTML structure annotations
    """
    
    def __init__(self, config: Dict):
        self.config = config['datasets']['pubtabnet']
        self.annotations_path = self.config['annotations']
        self.images_dir = {
            'train': self.config['images']['train'],
            'val': self.config['images']['val'],
            'test': self.config['images']['test']
        }
        self._data_cache = None
    
    def load_annotations(self, split: str = 'val', num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load annotations for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            num_samples: Number of samples to load (None for all)
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get('split') == split:
                    annotations.append(data)
                    if num_samples and len(annotations) >= num_samples:
                        break
        
        print(f"Loaded {len(annotations)} {split} samples from PubTabNet")
        return annotations
    
    def get_image_path(self, filename: str, split: str) -> str:
        """Get full path to image file."""
        return os.path.join(self.images_dir[split], filename)
    
    def load_sample(self, annotation: Dict, split: str = 'val') -> Tuple[Image.Image, Dict]:
        """
        Load a single sample (image + annotation).
        
        Returns:
            Tuple of (PIL Image, annotation dict)
        """
        image_path = self.get_image_path(annotation['filename'], split)
        image = Image.open(image_path).convert('RGB')
        return image, annotation
    
    def get_html_structure(self, annotation: Dict) -> str:
        """
        Get HTML structure from annotation.
        
        Returns:
            HTML string representing table structure
        """
        structure_tokens = annotation['html']['structure']['tokens']
        cells = annotation['html']['cells']
        
        # Build HTML with cell content
        html = '<table>'
        cell_idx = 0
        
        for token in structure_tokens:
            if token in ['<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>']:
                html += token
            elif token == '<td>':
                if cell_idx < len(cells):
                    cell_content = ''.join(cells[cell_idx].get('tokens', []))
                    html += f'<td>{cell_content}'
                    cell_idx += 1
                else:
                    html += '<td>'
            elif token == '</td>':
                html += '</td>'
            elif token.startswith('<td'):  # Handle colspan/rowspan
                html += token
                if cell_idx < len(cells):
                    cell_content = ''.join(cells[cell_idx].get('tokens', []))
                    html += cell_content
                    cell_idx += 1
        
        html += '</table>'
        return html


class FinTabNetLoader:
    """
    Loader for FinTabNet dataset.
    
    Format: XML (Pascal VOC style) with table structure annotations
    """
    
    def __init__(self, config: Dict):
        self.config = config['datasets']['fintabnet']
        self.images_dir = self.config['images']
        self.annotations_dir = {
            'train': self.config['annotations']['train'],
            'val': self.config['annotations']['val'],
            'test': self.config['annotations']['test']
        }
    
    def load_annotations(self, split: str = 'val', num_samples: Optional[int] = None) -> List[Dict]:
        """Load all XML annotations for a split."""
        annotations = []
        ann_dir = self.annotations_dir[split]
        
        xml_files = list(Path(ann_dir).glob('*.xml'))
        if num_samples:
            xml_files = xml_files[:num_samples]
        
        for xml_path in xml_files:
            ann = self.parse_xml(str(xml_path))
            if ann:
                annotations.append(ann)
        
        print(f"Loaded {len(annotations)} {split} samples from FinTabNet")
        return annotations
    
    def parse_xml(self, xml_path: str) -> Optional[Dict]:
        """Parse a single XML annotation file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotation = {
                'filename': root.find('filename').text,
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'objects': []
            }
            
            for obj in root.findall('object'):
                obj_data = {
                    'name': obj.find('name').text,
                    'bbox': [
                        float(obj.find('bndbox/xmin').text),
                        float(obj.find('bndbox/ymin').text),
                        float(obj.find('bndbox/xmax').text),
                        float(obj.find('bndbox/ymax').text)
                    ]
                }
                annotation['objects'].append(obj_data)
            
            return annotation
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def get_image_path(self, filename: str) -> str:
        """Get full path to image file."""
        return os.path.join(self.images_dir, filename)
    
    def load_sample(self, annotation: Dict) -> Tuple[Image.Image, Dict]:
        """Load a single sample."""
        image_path = self.get_image_path(annotation['filename'])
        image = Image.open(image_path).convert('RGB')
        return image, annotation
    
    def get_structure_elements(self, annotation: Dict) -> Dict[str, List]:
        """
        Extract structure elements from annotation.
        
        Returns:
            Dict with 'rows', 'columns', 'spanning_cells', 'cells'
        """
        elements = {
            'table': [],
            'rows': [],
            'columns': [],
            'spanning_cells': [],
            'cells': []
        }
        
        for obj in annotation['objects']:
            name = obj['name']
            if name == 'table':
                elements['table'].append(obj['bbox'])
            elif name == 'table row':
                elements['rows'].append(obj['bbox'])
            elif name == 'table column':
                elements['columns'].append(obj['bbox'])
            elif name == 'table spanning cell':
                elements['spanning_cells'].append(obj['bbox'])
            elif name == 'table cell':  # if exists
                elements['cells'].append(obj['bbox'])
        
        return elements


class DocLayNetLoader:
    """
    Loader for DocLayNet dataset.
    
    Format: COCO JSON with document layout annotations
    """
    
    # Category mapping
    CATEGORIES = {
        1: 'Caption',
        2: 'Footnote',
        3: 'Formula',
        4: 'List-item',
        5: 'Page-footer',
        6: 'Page-header',
        7: 'Picture',
        8: 'Section-header',
        9: 'Table',  # This is what we need!
        10: 'Text',
        11: 'Title'
    }
    
    def __init__(self, config: Dict):
        self.config = config['datasets']['doclaynet']
        self.images_dir = self.config['images']
        self.annotations_paths = {
            'train': self.config['annotations']['train'],
            'val': self.config['annotations']['val'],
            'test': self.config['annotations']['test']
        }
        self._coco_data = {}
    
    def load_coco_annotations(self, split: str = 'val') -> Dict:
        """Load COCO format annotations."""
        if split not in self._coco_data:
            with open(self.annotations_paths[split], 'r') as f:
                self._coco_data[split] = json.load(f)
        return self._coco_data[split]
    
    def get_table_annotations(self, split: str = 'val', num_samples: Optional[int] = None) -> List[Dict]:
        """
        Get only table annotations.
        
        Returns:
            List of dicts with image_id, filename, and table bboxes
        """
        coco_data = self.load_coco_annotations(split)
        
        # Build image_id to filename mapping
        id_to_image = {img['id']: img for img in coco_data['images']}
        
        # Get table annotations (category_id = 9)
        table_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] == 9]
        
        # Group by image
        image_tables = {}
        for ann in table_annotations:
            img_id = ann['image_id']
            if img_id not in image_tables:
                image_tables[img_id] = {
                    'image_id': img_id,
                    'filename': id_to_image[img_id]['file_name'],
                    'width': id_to_image[img_id]['width'],
                    'height': id_to_image[img_id]['height'],
                    'tables': []
                }
            # COCO bbox format: [x, y, width, height] -> convert to [x1, y1, x2, y2]
            bbox = ann['bbox']
            image_tables[img_id]['tables'].append({
                'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                'area': ann['area']
            })
        
        result = list(image_tables.values())
        if num_samples:
            result = result[:num_samples]
        
        print(f"Loaded {len(result)} images with tables from DocLayNet {split}")
        return result
    
    def get_image_path(self, filename: str) -> str:
        """Get full path to image file."""
        return os.path.join(self.images_dir, filename)
    
    def load_sample(self, annotation: Dict) -> Tuple[Image.Image, Dict]:
        """Load a single sample."""
        image_path = self.get_image_path(annotation['filename'])
        image = Image.open(image_path).convert('RGB')
        return image, annotation


class PubTables1MLoader:
    """
    Loader for PubTables-1M dataset.
    
    Format: XML (Pascal VOC style) with detailed structure annotations
    """
    
    def __init__(self, config: Dict):
        self.config = config['datasets']['pubtables1m']
        self.annotations_dir = {
            'train': self.config['structure']['train_annotations'],
            'val': self.config['structure']['val_annotations'],
            'test': self.config['structure']['test_annotations']
        }
        self.images_dir = self.config['structure']['test_images']
    
    def load_annotations(self, split: str = 'test', num_samples: Optional[int] = None) -> List[Dict]:
        """Load XML annotations."""
        annotations = []
        ann_dir = self.annotations_dir[split]
        
        if not os.path.exists(ann_dir):
            print(f"Warning: {ann_dir} does not exist")
            return annotations
        
        xml_files = list(Path(ann_dir).glob('*.xml'))
        if num_samples:
            xml_files = xml_files[:num_samples]
        
        for xml_path in xml_files:
            ann = self.parse_xml(str(xml_path))
            if ann:
                annotations.append(ann)
        
        print(f"Loaded {len(annotations)} {split} samples from PubTables-1M")
        return annotations
    
    def parse_xml(self, xml_path: str) -> Optional[Dict]:
        """Parse a single XML annotation file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotation = {
                'filename': root.find('filename').text,
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'objects': []
            }
            
            for obj in root.findall('object'):
                obj_data = {
                    'name': obj.find('name').text,
                    'bbox': [
                        float(obj.find('bndbox/xmin').text),
                        float(obj.find('bndbox/ymin').text),
                        float(obj.find('bndbox/xmax').text),
                        float(obj.find('bndbox/ymax').text)
                    ]
                }
                annotation['objects'].append(obj_data)
            
            return annotation
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def get_image_path(self, filename: str) -> str:
        """Get full path to image file."""
        return os.path.join(self.images_dir, filename)
    
    def load_sample(self, annotation: Dict) -> Tuple[Image.Image, Dict]:
        """Load a single sample."""
        image_path = self.get_image_path(annotation['filename'])
        image = Image.open(image_path).convert('RGB')
        return image, annotation


def get_data_loader(dataset_name: str, config: Dict):
    """
    Factory function to get the appropriate data loader.
    
    Args:
        dataset_name: One of 'pubtabnet', 'fintabnet', 'doclaynet', 'pubtables1m'
        config: Configuration dictionary
        
    Returns:
        Data loader instance
    """
    loaders = {
        'pubtabnet': PubTabNetLoader,
        'fintabnet': FinTabNetLoader,
        'doclaynet': DocLayNetLoader,
        'pubtables1m': PubTables1MLoader
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name](config)
