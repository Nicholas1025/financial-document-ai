"""
FinTabNet OTSL Local Loader

Load FinTabNet_OTSL dataset from local parquet files.
Provides images and HTML ground truth for TEDS evaluation.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import io

logger = logging.getLogger(__name__)


def load_fintabnet_otsl(
    data_dir: str = "D:/datasets/FinTabNet_OTSL/data",
    split: str = "test",
    num_samples: int = 100
) -> List[Tuple[str, Optional[str]]]:
    """
    Load FinTabNet_OTSL samples from local parquet files.
    
    Args:
        data_dir: Path to the data directory containing parquet files
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to load
        
    Returns:
        List of (image_path, html_ground_truth) tuples
    """
    try:
        import pandas as pd
        from PIL import Image
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    # Find parquet files for the split
    parquet_files = sorted(data_path.glob(f"{split}-*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found for split '{split}' in {data_dir}")
        return []
    
    logger.info(f"Found {len(parquet_files)} parquet files for split '{split}'")
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp(prefix="fintabnet_otsl_")
    logger.info(f"Saving images to temp directory: {temp_dir}")
    
    samples = []
    count = 0
    
    for parquet_file in parquet_files:
        if count >= num_samples:
            break
            
        logger.info(f"Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        
        for idx, row in df.iterrows():
            if count >= num_samples:
                break
            
            try:
                # Extract image
                image_data = row.get('image')
                if image_data is None:
                    continue
                
                # Handle different image formats
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                    image = Image.open(io.BytesIO(image_bytes))
                elif hasattr(image_data, 'tobytes'):
                    image = image_data
                else:
                    logger.warning(f"Unknown image format at index {idx}")
                    continue
                
                # Save image to temp file
                filename = row.get('filename', f'sample_{count}.png')
                if not filename.endswith(('.png', '.jpg', '.jpeg')):
                    filename = f"{filename}.png"
                image_path = os.path.join(temp_dir, filename)
                image.save(image_path)
                
                # Extract HTML ground truth
                html_gt = row.get('html')
                if html_gt is None:
                    html_gt = row.get('html_restored')
                
                samples.append((image_path, html_gt))
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} samples from FinTabNet_OTSL ({split})")
    return samples


def html_to_grid(html: str) -> Optional[List[List[str]]]:
    """Convert HTML table to grid format."""
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table:
            # Try parsing the whole thing as a table
            table = soup
        
        grid = []
        for tr in table.find_all('tr'):
            row = []
            for cell in tr.find_all(['td', 'th']):
                text = cell.get_text(strip=True)
                row.append(text)
            if row:
                grid.append(row)
        
        return grid if grid else None
        
    except Exception as e:
        logger.debug(f"Failed to parse HTML: {e}")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FinTabNet_OTSL loader')
    parser.add_argument('--data-dir', default='D:/datasets/FinTabNet_OTSL/data')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num-samples', type=int, default=5)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    samples = load_fintabnet_otsl(args.data_dir, args.split, args.num_samples)
    
    print(f"\nLoaded {len(samples)} samples:")
    for i, (img_path, html_gt) in enumerate(samples[:3]):
        print(f"\n{i+1}. {Path(img_path).name}")
        if html_gt is not None:
            # Handle if html_gt is a list/array
            html_str = html_gt[0] if isinstance(html_gt, (list, tuple)) else str(html_gt)
            grid = html_to_grid(html_str)
            if grid:
                print(f"   Grid: {len(grid)} rows x {len(grid[0]) if grid else 0} cols")
