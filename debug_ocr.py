import os
import sys
from PIL import Image, ImageOps
import numpy as np

# Add workspace root to path
sys.path.append(r'c:\Users\User\Documents\financial-document-ai')

from modules.ocr import TableOCR

def main():
    ocr = TableOCR(lang='en')
    image_path = r'c:\Users\User\Documents\financial-document-ai\data\samples\sample_3.png'
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"Processing {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # 1. Run OCR on original image
    print("--- Original Image OCR ---")
    results = ocr.extract_text(image)
    for item in results:
        # Check for the problematic words
        text = item['text']
        if any(x in text for x in ['evenue', 'ncome', 'ost', 'Revenue', 'Income', 'Cost']):
            print(f"Text: '{text}', BBox: {item['bbox']}, Conf: {item['confidence']:.4f}")

    # 2. Run OCR on padded image
    print("\n--- Padded Image OCR ---")
    # Add padding (white border)
    padding = 50
    padded_image = ImageOps.expand(image, border=padding, fill='white')
    
    results_padded = ocr.extract_text(padded_image)
    for item in results_padded:
        text = item['text']
        if any(x in text for x in ['evenue', 'ncome', 'ost', 'Revenue', 'Income', 'Cost']):
             print(f"Text: '{text}', Conf: {item['confidence']:.4f}")

if __name__ == '__main__':
    main()
