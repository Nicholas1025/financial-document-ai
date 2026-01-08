"""
Test Mistral OCR on financial table image using official Mistral API

Usage:
    1. Get API key from https://console.mistral.ai/
    2. Set environment variable: $env:MISTRAL_API_KEY = "your_key"
    3. Run: python test_mistral_ocr_official.py
"""

import os
import json
import base64
import time
from pathlib import Path

from mistralai import Mistral

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime_type(image_path: str) -> str:
    """Get MIME type from file extension"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.pdf': 'application/pdf'
    }
    return mime_types.get(ext, 'image/png')

def test_mistral_ocr(image_path: str, api_key: str = None):
    """
    Test Mistral OCR on an image
    
    Args:
        image_path: Path to image file
        api_key: Mistral API key (or set MISTRAL_API_KEY env var)
    """
    print(f"\n{'='*70}")
    print(f"  MISTRAL OCR TEST (Official API)")
    print(f"{'='*70}")
    
    # Get API key
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("\nERROR: No API key found!")
        print("Please set your Mistral API key:")
        print("  PowerShell: $env:MISTRAL_API_KEY = 'your_key_here'")
        print("  Or pass api_key parameter to this function")
        print("\nGet your API key from: https://console.mistral.ai/")
        return None
    
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Image: {image_path}")
    
    # Check file exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Initialize client
    client = Mistral(api_key=api_key)
    
    # Encode image
    print("\nEncoding image to base64...")
    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    print(f"Base64 length: {len(image_base64)}")
    print(f"MIME type: {mime_type}")
    
    # Call OCR API
    print("\nCalling Mistral OCR API...")
    start_time = time.time()
    
    try:
        # Use the OCR endpoint
        ocr_response = client.ocr.process(
            model="mistral-ocr-2505",
            document={
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_base64}"
            },
            include_image_base64=False  # Don't include images in response to save bandwidth
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"Response received in {elapsed_time:.0f}ms")
        
        return ocr_response, elapsed_time
        
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        print(f"Error calling OCR API: {e}")
        return None, elapsed_time

def parse_and_display_result(result, elapsed_time: float):
    """Parse and display OCR result"""
    if not result:
        return
    
    print(f"\n{'='*70}")
    print("  OCR RESULT")
    print(f"{'='*70}")
    
    # Convert to dict if it's a Pydantic model
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump()
    elif hasattr(result, '__dict__'):
        result_dict = result.__dict__
    else:
        result_dict = result
    
    # Check for pages
    pages = result_dict.get('pages', [])
    if not pages and hasattr(result, 'pages'):
        pages = result.pages
    
    print(f"Total pages: {len(pages)}")
    
    full_markdown = ""
    
    for page in pages:
        # Handle both dict and object
        if hasattr(page, 'index'):
            page_idx = page.index
            markdown = page.markdown
            images = getattr(page, 'images', [])
            dimensions = getattr(page, 'dimensions', {})
        else:
            page_idx = page.get("index", 0)
            markdown = page.get("markdown", "")
            images = page.get("images", [])
            dimensions = page.get("dimensions", {})
        
        print(f"\n--- Page {page_idx} ---")
        if dimensions:
            if hasattr(dimensions, 'dpi'):
                print(f"Dimensions: {dimensions.width}x{dimensions.height} @ {dimensions.dpi}dpi")
            else:
                print(f"Dimensions: {dimensions}")
        print(f"Images in page: {len(images) if images else 0}")
        
        print(f"\nMarkdown content ({len(markdown)} chars):")
        print("-" * 50)
        # Print markdown (truncate if too long)
        if len(markdown) > 3000:
            print(markdown[:3000])
            print(f"\n... (truncated, showing 3000/{len(markdown)} chars)")
        else:
            print(markdown)
        print("-" * 50)
        
        full_markdown += markdown + "\n"
    
    return full_markdown

def extract_table_from_markdown(markdown: str):
    """Extract table data from Mistral OCR markdown output"""
    lines = markdown.split('\n')
    tables = []
    current_table = []
    
    for line in lines:
        line = line.strip()
        
        # Detect markdown table rows (lines starting and ending with |)
        if line.startswith('|') and line.endswith('|'):
            # Skip separator rows (|---|---|)
            if set(line.replace('|', '').replace('-', '').replace(':', '').strip()) == set():
                continue
            
            # Parse cells
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # Remove empty strings
            
            if cells:
                current_table.append(cells)
        else:
            # End of table
            if current_table:
                tables.append(current_table)
                current_table = []
    
    # Don't forget last table
    if current_table:
        tables.append(current_table)
    
    return tables

def compare_with_ground_truth(markdown: str, gt_path: str):
    """Compare OCR output with ground truth"""
    print(f"\n{'='*70}")
    print("  COMPARISON WITH GROUND TRUTH")
    print(f"{'='*70}")
    
    # Load ground truth
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    
    # Extract tables from markdown
    tables = extract_table_from_markdown(markdown)
    
    print(f"Tables found in OCR: {len(tables)}")
    
    if not tables:
        print("No tables found in OCR output!")
        return
    
    # Use first table
    ocr_table = tables[0]
    print(f"First table rows: {len(ocr_table)}")
    
    # Show first few rows
    print("\nOCR Table Preview:")
    for i, row in enumerate(ocr_table[:5]):
        print(f"  Row {i}: {row[:4]}..." if len(row) > 4 else f"  Row {i}: {row}")
    
    # Try to answer GT questions
    qa_questions = gt.get("qa_questions", [])
    print(f"\nTesting {len(qa_questions)} QA questions:")
    
    correct = 0
    for q in qa_questions:
        row_key = q.get("row_key", "")
        col_key = q.get("col_key", "")
        expected = q.get("answer", "")
        
        # Simple lookup (fuzzy match would be better)
        predicted = None
        for row in ocr_table:
            if len(row) > 0 and row_key.lower() in row[0].lower():
                # Found row, now find column
                # This is simplified - real implementation needs header matching
                if len(row) > 1:
                    predicted = row[1]  # Just take second column as demo
                break
        
        is_correct = predicted and expected in str(predicted)
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} {row_key[:30]}: expected='{expected}', got='{predicted}'")
    
    accuracy = (correct / len(qa_questions) * 100) if qa_questions else 0
    print(f"\nQA Accuracy: {accuracy:.1f}% ({correct}/{len(qa_questions)})")

def main():
    """Main function"""
    # Test image path
    image_path = "tests/real_table_pipeline_test/nanyang_sample1.png"
    gt_path = "tests/real_table_pipeline_test/ground_truth.json"
    output_dir = "tests/real_table_pipeline_test/output_fair"
    
    # Run OCR
    result, elapsed_time = test_mistral_ocr(image_path)
    
    if result:
        # Parse and display
        markdown = parse_and_display_result(result, elapsed_time)
        
        # Save result
        output_path = os.path.join(output_dir, "mistral_ocr_result.json")
        
        # Convert result to dict for JSON serialization
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, '__dict__'):
            result_dict = dict(result.__dict__)
        else:
            result_dict = result
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResult saved to: {output_path}")
        
        # Save markdown
        md_path = os.path.join(output_dir, "mistral_ocr_output.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Mistral OCR Output\n\n")
            f.write(f"Processing time: {elapsed_time:.0f}ms\n\n")
            f.write(markdown)
        print(f"Markdown saved to: {md_path}")
        
        # Extract and show tables
        tables = extract_table_from_markdown(markdown)
        if tables:
            print(f"\n{'='*70}")
            print(f"  EXTRACTED TABLES")
            print(f"{'='*70}")
            
            for i, table in enumerate(tables):
                print(f"\nTable {i+1} ({len(table)} rows):")
                for j, row in enumerate(table[:8]):  # Show first 8 rows
                    print(f"  {j}: {row}")
                if len(table) > 8:
                    print(f"  ... ({len(table) - 8} more rows)")
        
        # Compare with ground truth if available
        if os.path.exists(gt_path):
            compare_with_ground_truth(markdown, gt_path)
        
        # Summary
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"Model: Mistral OCR (mistral-ocr-2505)")
        print(f"Processing time: {elapsed_time:.0f}ms")
        print(f"Tables found: {len(tables)}")
        if tables:
            total_rows = sum(len(t) for t in tables)
            print(f"Total rows: {total_rows}")
        print(f"Status: SUCCESS")
    else:
        print("\nOCR failed!")

if __name__ == "__main__":
    main()
