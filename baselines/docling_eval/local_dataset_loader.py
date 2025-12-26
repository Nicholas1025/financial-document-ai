"""
Local Dataset Loader for Baseline Comparison

Loads samples from locally downloaded datasets (FinTabNet, PubTabNet, etc.)
instead of downloading from HuggingFace.

Uses the existing data_loaders.py infrastructure.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.utils import load_config
from modules.data_loaders import (
    FinTabNetLoader,
    PubTabNetLoader,
    DocLayNetLoader,
    PubTables1MLoader,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Local Dataset Wrapper
# =============================================================================

class LocalDatasetLoader:
    """
    Unified loader for local datasets.
    
    Uses the project's existing data_loaders.py to load from local paths
    configured in configs/config.yaml.
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        self.config = load_config(config_path)
        self._loaders = {}
    
    def _get_loader(self, dataset_name: str):
        """Get or create a dataset loader."""
        if dataset_name not in self._loaders:
            # The loaders expect the full config, not just the dataset-specific part
            if dataset_name == 'fintabnet':
                self._loaders[dataset_name] = FinTabNetLoader(self.config)
            elif dataset_name == 'pubtabnet':
                self._loaders[dataset_name] = PubTabNetLoader(self.config)
            elif dataset_name == 'doclaynet':
                self._loaders[dataset_name] = DocLayNetLoader(self.config)
            elif dataset_name == 'pubtables1m':
                self._loaders[dataset_name] = PubTables1MLoader(self.config)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
        return self._loaders[dataset_name]
    
    def load_fintabnet(
        self,
        split: str = 'val',
        num_samples: Optional[int] = None,
        with_ground_truth: bool = True
    ) -> List[Tuple[str, Optional[List[List[str]]]]]:
        """
        Load FinTabNet samples from local disk.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_samples: Maximum samples to load
            with_ground_truth: Whether to include GT grid
            
        Returns:
            List of (image_path, ground_truth_grid) tuples
        """
        loader = self._get_loader('fintabnet')
        annotations = loader.load_annotations(split=split, num_samples=num_samples)
        
        samples = []
        for ann in annotations:
            image_path = loader.get_image_path(ann['filename'])
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Extract GT grid from annotation if available
            gt_grid = None
            if with_ground_truth:
                gt_grid = self._extract_fintabnet_gt_grid(ann, loader)
            
            samples.append((image_path, gt_grid))
        
        logger.info(f"Loaded {len(samples)} FinTabNet samples from local disk")
        return samples
    
    def _extract_fintabnet_gt_grid(
        self,
        annotation: Dict,
        loader: FinTabNetLoader
    ) -> Optional[List[List[str]]]:
        """
        Extract ground truth grid from FinTabNet annotation.
        
        FinTabNet provides bounding boxes for table cells.
        We can try to load word-level annotations if available.
        """
        # Try to load words file if available
        if loader.words_dir:
            words_file = Path(loader.words_dir) / annotation['filename'].replace('.png', '_words.json')
            if words_file.exists():
                try:
                    with open(words_file, 'r') as f:
                        words_data = json.load(f)
                    return self._words_to_grid(words_data, annotation)
                except Exception as e:
                    logger.debug(f"Could not load words file: {e}")
        
        # FinTabNet XML only has bboxes, not text content
        # Return None - we'll use structure-only evaluation
        return None
    
    def _words_to_grid(
        self,
        words_data: Dict,
        annotation: Dict
    ) -> List[List[str]]:
        """Convert words data to grid format."""
        # This depends on the specific words file format
        # Placeholder - implement based on actual format
        return []
    
    def load_pubtabnet(
        self,
        split: str = 'val',
        num_samples: Optional[int] = None,
        with_ground_truth: bool = True
    ) -> List[Tuple[str, Optional[List[List[str]]]]]:
        """
        Load PubTabNet samples from local disk.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_samples: Maximum samples to load
            with_ground_truth: Whether to include GT grid
            
        Returns:
            List of (image_path, ground_truth_grid) tuples
        """
        loader = self._get_loader('pubtabnet')
        annotations = loader.load_annotations(split=split, num_samples=num_samples)
        
        samples = []
        for ann in annotations:
            image_path = loader.get_image_path(split, ann['filename'])
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Extract GT grid from HTML if available
            gt_grid = None
            if with_ground_truth and 'html' in ann:
                gt_grid = self._html_to_grid(ann['html'])
            
            samples.append((image_path, gt_grid))
        
        logger.info(f"Loaded {len(samples)} PubTabNet samples from local disk")
        return samples
    
    def _html_to_grid(self, html_content: Dict) -> Optional[List[List[str]]]:
        """Convert PubTabNet HTML structure to grid."""
        try:
            # PubTabNet stores HTML as {'structure': {'tokens': [...]}, 'cells': [...]}
            cells = html_content.get('cells', [])
            structure = html_content.get('structure', {})
            tokens = structure.get('tokens', [])
            
            if not cells:
                return None
            
            # Parse tokens to determine grid structure
            num_rows = tokens.count('</tr>')
            num_cols = 0
            
            # Build grid from cells
            grid = []
            current_row = []
            cell_idx = 0
            
            for token in tokens:
                if token in ['<td>', '<td']:
                    # Get cell content
                    if cell_idx < len(cells):
                        cell_tokens = cells[cell_idx].get('tokens', [])
                        cell_text = ' '.join(cell_tokens)
                        current_row.append(cell_text)
                        cell_idx += 1
                elif token == '</tr>':
                    if current_row:
                        grid.append(current_row)
                        num_cols = max(num_cols, len(current_row))
                        current_row = []
            
            # Pad rows to same length
            for row in grid:
                while len(row) < num_cols:
                    row.append('')
            
            return grid if grid else None
            
        except Exception as e:
            logger.debug(f"Could not parse HTML to grid: {e}")
            return None
    
    def load_pubtables1m(
        self,
        split: str = 'val',
        num_samples: Optional[int] = None,
        with_ground_truth: bool = True
    ) -> List[Tuple[str, Optional[List[List[str]]]]]:
        """Load PubTables-1M samples from local disk."""
        loader = self._get_loader('pubtables1m')
        annotations = loader.load_structure_annotations(split=split, num_samples=num_samples)
        
        samples = []
        for ann in annotations:
            image_path = loader.get_image_path(ann['filename'], split)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # PubTables-1M has detailed structure annotations
            gt_grid = None
            if with_ground_truth:
                gt_grid = self._pubtables1m_to_grid(ann, loader)
            
            samples.append((image_path, gt_grid))
        
        logger.info(f"Loaded {len(samples)} PubTables-1M samples from local disk")
        return samples
    
    def _pubtables1m_to_grid(
        self,
        annotation: Dict,
        loader: PubTables1MLoader
    ) -> Optional[List[List[str]]]:
        """Convert PubTables-1M annotation to grid with text."""
        try:
            # Load words if available
            words = loader.load_words(annotation['filename'])
            if not words:
                return None
            
            # Get row/column structure
            row_headers = annotation.get('row_headers', [])
            col_headers = annotation.get('col_headers', [])
            cells = annotation.get('cells', [])
            
            # Build grid
            # ... implementation depends on exact format
            return None
            
        except Exception as e:
            logger.debug(f"Could not build grid: {e}")
            return None
    
    def load_local_samples(
        self,
        samples_dir: str = 'data/samples',
        num_samples: Optional[int] = None
    ) -> List[Tuple[str, Optional[List[List[str]]]]]:
        """
        Load samples from local samples directory.
        
        Args:
            samples_dir: Path to samples directory
            num_samples: Maximum samples to load
            
        Returns:
            List of (image_path, None) tuples (no GT for local samples)
        """
        samples_path = Path(samples_dir)
        if not samples_path.is_absolute():
            samples_path = project_root / samples_path
        
        if not samples_path.exists():
            logger.error(f"Samples directory not found: {samples_path}")
            return []
        
        samples = []
        extensions = ['*.png', '*.jpg', '*.jpeg']
        
        for ext in extensions:
            for img_path in samples_path.glob(ext):
                samples.append((str(img_path), None))
        
        if num_samples:
            samples = samples[:num_samples]
        
        logger.info(f"Loaded {len(samples)} samples from {samples_path}")
        return samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available datasets."""
        info = {}
        
        # Check FinTabNet
        fintabnet_cfg = self.config['datasets'].get('fintabnet', {})
        fintabnet_root = fintabnet_cfg.get('root', '')
        info['fintabnet'] = {
            'configured': bool(fintabnet_root),
            'path': fintabnet_root,
            'exists': os.path.exists(fintabnet_root) if fintabnet_root else False
        }
        
        # Check PubTabNet
        pubtabnet_cfg = self.config['datasets'].get('pubtabnet', {})
        pubtabnet_root = pubtabnet_cfg.get('root', '')
        info['pubtabnet'] = {
            'configured': bool(pubtabnet_root),
            'path': pubtabnet_root,
            'exists': os.path.exists(pubtabnet_root) if pubtabnet_root else False
        }
        
        # Check PubTables-1M
        pubtables_cfg = self.config['datasets'].get('pubtables1m', {})
        pubtables_root = pubtables_cfg.get('root', '')
        info['pubtables1m'] = {
            'configured': bool(pubtables_root),
            'path': pubtables_root,
            'exists': os.path.exists(pubtables_root) if pubtables_root else False
        }
        
        # Check local samples
        local_samples = project_root / 'data' / 'samples'
        info['local_samples'] = {
            'path': str(local_samples),
            'exists': local_samples.exists(),
            'count': len(list(local_samples.glob('*.png'))) + len(list(local_samples.glob('*.jpg'))) if local_samples.exists() else 0
        }
        
        return info


# =============================================================================
# Convenience Functions
# =============================================================================

def load_fintabnet_local(
    split: str = 'val',
    num_samples: int = 100,
    config_path: str = None
) -> List[Tuple[str, Optional[List[List[str]]]]]:
    """Load FinTabNet samples from local disk."""
    if config_path is None:
        config_path = str(project_root / 'configs' / 'config.yaml')
    loader = LocalDatasetLoader(config_path)
    return loader.load_fintabnet(split, num_samples)


def load_pubtabnet_local(
    split: str = 'val',
    num_samples: int = 100,
    config_path: str = None
) -> List[Tuple[str, Optional[List[List[str]]]]]:
    """Load PubTabNet samples from local disk."""
    if config_path is None:
        config_path = str(project_root / 'configs' / 'config.yaml')
    loader = LocalDatasetLoader(config_path)
    return loader.load_pubtabnet(split, num_samples)


def load_samples_local(
    samples_dir: str = 'data/samples',
    num_samples: Optional[int] = None,
    config_path: str = None
) -> List[Tuple[str, Optional[List[List[str]]]]]:
    """Load samples from local samples directory."""
    if config_path is None:
        config_path = str(project_root / 'configs' / 'config.yaml')
    loader = LocalDatasetLoader(config_path)
    return loader.load_local_samples(samples_dir, num_samples)


def print_dataset_info(config_path: str = 'configs/config.yaml'):
    """Print information about available datasets."""
    loader = LocalDatasetLoader(config_path)
    info = loader.get_dataset_info()
    
    print("\n" + "=" * 60)
    print("Available Datasets")
    print("=" * 60)
    
    for name, details in info.items():
        status = "✅" if details.get('exists') else "❌"
        print(f"\n{status} {name}")
        print(f"   Path: {details.get('path', 'N/A')}")
        if 'count' in details:
            print(f"   Files: {details['count']}")
    
    print("\n" + "=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check local dataset availability')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--test-load', choices=['fintabnet', 'pubtabnet', 'local'], help='Test loading a dataset')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to test')
    
    args = parser.parse_args()
    
    print_dataset_info(args.config)
    
    if args.test_load:
        loader = LocalDatasetLoader(args.config)
        
        if args.test_load == 'fintabnet':
            samples = loader.load_fintabnet(num_samples=args.num_samples)
        elif args.test_load == 'pubtabnet':
            samples = loader.load_pubtabnet(num_samples=args.num_samples)
        else:
            samples = loader.load_local_samples(num_samples=args.num_samples)
        
        print(f"\nLoaded {len(samples)} samples:")
        for path, gt in samples[:5]:
            print(f"  - {Path(path).name}")
            if gt:
                print(f"    GT grid: {len(gt)} rows x {len(gt[0]) if gt else 0} cols")
