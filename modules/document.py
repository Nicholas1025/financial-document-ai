"""
Document Processing Module

Handles the initial processing of financial documents:
1. PDF to Image conversion
2. Image preprocessing (enhancement, normalization)
3. Document classification and page filtering

This module serves as the entry point for the financial document processing pipeline.
"""
import os
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path


class PDFLoader:
    """
    Converts PDF documents to images for downstream processing.
    
    Uses pdf2image library (requires poppler installation).
    """
    
    def __init__(self, dpi: int = 200, output_format: str = 'PNG'):
        """
        Initialize PDF loader.
        
        Args:
            dpi: Resolution for PDF rendering (default 200 for balance of quality/speed)
            output_format: Output image format ('PNG' or 'JPEG')
        """
        self.dpi = dpi
        self.output_format = output_format
        self._pdf2image_available = self._check_pdf2image()
    
    def _check_pdf2image(self) -> bool:
        """Check if pdf2image is available."""
        try:
            import pdf2image
            return True
        except ImportError:
            print("Warning: pdf2image not installed. Install with: pip install pdf2image")
            print("Also requires poppler: https://github.com/osber/poppler-windows/releases")
            return False
    
    def load(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to convert (1-indexed). None for all pages.
            
        Returns:
            List of PIL Images, one per page
        """
        if not self._pdf2image_available:
            raise RuntimeError("pdf2image is not available. Please install it first.")
        
        from pdf2image import convert_from_path
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Convert pages to 0-indexed if specified
        first_page = None
        last_page = None
        if pages:
            first_page = min(pages)
            last_page = max(pages)
        
        images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            first_page=first_page,
            last_page=last_page,
            fmt=self.output_format.lower()
        )
        
        # Filter to requested pages if specific pages requested
        if pages and len(pages) < len(images):
            # pages is 1-indexed, adjust for filtering
            page_set = set(p - (first_page or 1) for p in pages)
            images = [img for i, img in enumerate(images) if i in page_set]
        
        print(f"Loaded {len(images)} pages from {pdf_path}")
        return images
    
    def load_page(self, pdf_path: str, page_number: int) -> Image.Image:
        """
        Load a single page from PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            PIL Image of the specified page
        """
        images = self.load(pdf_path, pages=[page_number])
        if not images:
            raise ValueError(f"Could not load page {page_number} from {pdf_path}")
        return images[0]
    
    def save_pages(self, images: List[Image.Image], output_dir: str, 
                   prefix: str = 'page') -> List[str]:
        """
        Save converted pages to disk.
        
        Args:
            images: List of PIL Images
            output_dir: Directory to save images
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        ext = self.output_format.lower()
        
        for i, img in enumerate(images, start=1):
            filename = f"{prefix}_{i:03d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            saved_paths.append(filepath)
        
        print(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths


class ImagePreprocessor:
    """
    Preprocesses document images for optimal OCR and table detection.
    
    Provides various enhancement techniques commonly used in document processing.
    """
    
    def __init__(self, target_dpi: int = 300):
        """
        Initialize image preprocessor.
        
        Args:
            target_dpi: Target DPI for processing (higher = better quality but slower)
        """
        self.target_dpi = target_dpi
    
    def preprocess(self, image: Image.Image, 
                   enhance_contrast: bool = True,
                   denoise: bool = False,
                   deskew: bool = False,
                   binarize: bool = False) -> Image.Image:
        """
        Apply preprocessing pipeline to image.
        
        Args:
            image: Input PIL Image
            enhance_contrast: Whether to enhance contrast
            denoise: Whether to apply denoising
            deskew: Whether to correct skew (requires additional processing)
            binarize: Whether to convert to binary (black/white)
            
        Returns:
            Preprocessed PIL Image
        """
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply enhancements in order
        if enhance_contrast:
            image = self.enhance_contrast(image)
        
        if denoise:
            image = self.denoise(image)
        
        if deskew:
            image = self.deskew(image)
        
        if binarize:
            image = self.binarize(image)
        
        return image
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.3) -> Image.Image:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            factor: Contrast enhancement factor (1.0 = no change, >1 = more contrast)
            
        Returns:
            Contrast-enhanced image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_sharpness(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance image sharpness.
        
        Args:
            image: Input image
            factor: Sharpness enhancement factor
            
        Returns:
            Sharpened image
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def denoise(self, image: Image.Image) -> Image.Image:
        """
        Apply denoising filter.
        
        Uses median filter which is effective for salt-and-pepper noise
        common in scanned documents.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def binarize(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Convert image to binary (black and white).
        
        Useful for scanned documents with clear text.
        
        Args:
            image: Input image
            threshold: Pixel value threshold (0-255)
            
        Returns:
            Binary image
        """
        grayscale = image.convert('L')
        return grayscale.point(lambda x: 255 if x > threshold else 0, mode='1')
    
    def deskew(self, image: Image.Image) -> Image.Image:
        """
        Correct image skew (rotation).
        
        Uses a simple approach based on detecting dominant angles.
        For production use, consider using more sophisticated methods.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale for analysis
        gray = image.convert('L')
        gray_array = np.array(gray)
        
        # Simple edge-based skew detection
        # This is a basic implementation; for better results use OpenCV's HoughLines
        try:
            # Detect edges
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            
            # Find non-zero points
            coords = np.column_stack(np.where(edges_array > 50))
            
            if len(coords) < 100:
                return image  # Not enough edges to detect skew
            
            # Fit a line to estimate angle (simplified approach)
            # For better accuracy, use Hough transform
            angle = self._estimate_skew_angle(coords)
            
            if abs(angle) < 0.5:  # Less than 0.5 degrees, don't rotate
                return image
            
            if abs(angle) > 10:  # Too much skew, probably wrong detection
                return image
            
            # Rotate image
            return image.rotate(angle, expand=True, fillcolor='white')
            
        except Exception:
            # If skew detection fails, return original
            return image
    
    def _estimate_skew_angle(self, coords: np.ndarray) -> float:
        """Estimate skew angle from edge coordinates."""
        # Use PCA to find dominant direction
        coords = coords.astype(np.float64)
        coords -= coords.mean(axis=0)
        
        # Covariance matrix
        cov = np.cov(coords.T)
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Get angle of dominant eigenvector
        idx = np.argmax(eigenvalues)
        angle = np.degrees(np.arctan2(eigenvectors[1, idx], eigenvectors[0, idx]))
        
        # Normalize to small rotation
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
            
        return angle
    
    def normalize_size(self, image: Image.Image, 
                       max_width: int = 2000, 
                       max_height: int = 2800) -> Image.Image:
        """
        Normalize image size while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image (if larger than max dimensions)
        """
        width, height = image.size
        
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate scale factor
        scale = min(max_width / width, max_height / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def auto_crop(self, image: Image.Image, padding: int = 10) -> Image.Image:
        """
        Automatically crop whitespace/margins from image.
        
        Args:
            image: Input image
            padding: Padding to keep around content
            
        Returns:
            Cropped image
        """
        # Convert to grayscale
        gray = image.convert('L')
        
        # Invert so content is white
        inverted = ImageOps.invert(gray)
        
        # Get bounding box of non-zero regions
        bbox = inverted.getbbox()
        
        if bbox:
            # Add padding
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(image.width, bbox[2] + padding)
            bottom = min(image.height, bbox[3] + padding)
            
            return image.crop((left, top, right, bottom))
        
        return image


class DocumentProcessor:
    """
    High-level document processing class that orchestrates the full pipeline.
    
    Combines PDF loading, preprocessing, and prepares images for table extraction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.pdf_loader = PDFLoader(
            dpi=self.config.get('pdf_dpi', 200)
        )
        self.preprocessor = ImagePreprocessor(
            target_dpi=self.config.get('target_dpi', 300)
        )
    
    def process_pdf(self, pdf_path: str, 
                    pages: Optional[List[int]] = None,
                    preprocess: bool = True) -> List[Dict[str, Any]]:
        """
        Process a PDF document and prepare pages for table extraction.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to process (1-indexed), None for all
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of dictionaries containing processed page information
        """
        results = []
        
        # Load PDF pages as images
        images = self.pdf_loader.load(pdf_path, pages=pages)
        
        for i, image in enumerate(images, start=1):
            page_num = pages[i-1] if pages else i
            
            # Apply preprocessing if requested
            if preprocess:
                processed_image = self.preprocessor.preprocess(
                    image,
                    enhance_contrast=True,
                    denoise=False,  # Only if needed for scanned docs
                    deskew=False,   # Only if needed
                    binarize=False  # Keep RGB for table detection
                )
            else:
                processed_image = image
            
            results.append({
                'page_number': page_num,
                'original_image': image,
                'processed_image': processed_image,
                'width': processed_image.width,
                'height': processed_image.height,
                'source': pdf_path
            })
        
        print(f"Processed {len(results)} pages from {pdf_path}")
        return results
    
    def process_image(self, image_path: str, 
                      preprocess: bool = True) -> Dict[str, Any]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to image file
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing processed image information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        if preprocess:
            processed_image = self.preprocessor.preprocess(
                image,
                enhance_contrast=True,
                denoise=False,
                deskew=False,
                binarize=False
            )
        else:
            processed_image = image
        
        return {
            'page_number': 1,
            'original_image': image,
            'processed_image': processed_image,
            'width': processed_image.width,
            'height': processed_image.height,
            'source': image_path
        }
    
    def process_batch(self, paths: List[str], 
                      preprocess: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents (PDFs or images).
        
        Args:
            paths: List of file paths
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of processed document results
        """
        all_results = []
        
        for path in paths:
            ext = Path(path).suffix.lower()
            
            if ext == '.pdf':
                results = self.process_pdf(path, preprocess=preprocess)
                all_results.extend(results)
            elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                result = self.process_image(path, preprocess=preprocess)
                all_results.append(result)
            else:
                print(f"Unsupported file format: {path}")
        
        return all_results
    
    def save_processed(self, results: List[Dict[str, Any]], 
                       output_dir: str,
                       save_original: bool = False) -> List[str]:
        """
        Save processed images to disk.
        
        Args:
            results: List of processing results
            output_dir: Output directory
            save_original: Whether to also save original images
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for result in results:
            source_name = Path(result['source']).stem
            page_num = result['page_number']
            
            # Save processed image
            processed_path = os.path.join(
                output_dir, 
                f"{source_name}_page{page_num:03d}_processed.png"
            )
            result['processed_image'].save(processed_path)
            saved_paths.append(processed_path)
            
            # Optionally save original
            if save_original:
                original_path = os.path.join(
                    output_dir,
                    f"{source_name}_page{page_num:03d}_original.png"
                )
                result['original_image'].save(original_path)
                saved_paths.append(original_path)
        
        print(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths


# Convenience functions for quick usage
def load_pdf(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """Quick function to load PDF pages as images."""
    loader = PDFLoader(dpi=dpi)
    return loader.load(pdf_path)


def preprocess_image(image: Image.Image, enhance: bool = True) -> Image.Image:
    """Quick function to preprocess an image."""
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess(image, enhance_contrast=enhance)
