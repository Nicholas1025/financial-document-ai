"""
Test Mistral OCR on financial table image via Google Cloud Vertex AI
"""

import os
import json
import base64
import subprocess
import requests
import time
from pathlib import Path

# Configuration
MODEL_ID = "mistral-ocr-2505"
PROJECT_ID = "fyp-482818"  # From your URL
REGION = "us-central1"  # Common region, adjust if needed

def get_access_token():
    """Get Google Cloud access token"""
    process = subprocess.Popen(
        "gcloud auth print-access-token",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    (access_token_bytes, err) = process.communicate()
    
    if err:
        print(f"Warning: {err.decode('utf-8')}")
    
    access_token = access_token_bytes.decode("utf-8").strip()
    return access_token

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_mistral_ocr(image_path: str, include_images: bool = True):
    """
    Call Mistral OCR API via Vertex AI
    
    Args:
        image_path: Path to image file (PNG, JPEG, or PDF)
        include_images: Whether to include base64 images in output
    
    Returns:
        OCR result dictionary
    """
    print(f"\n{'='*60}")
    print(f"  MISTRAL OCR TEST")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Image: {image_path}")
    
    # Get access token
    print("\nGetting access token...")
    access_token = get_access_token()
    if not access_token:
        print("ERROR: Failed to get access token. Make sure you're logged in with 'gcloud auth login'")
        return None
    print(f"Token obtained (length: {len(access_token)})")
    
    # Encode image
    print("\nEncoding image to base64...")
    image_base64 = encode_image_to_base64(image_path)
    print(f"Base64 length: {len(image_base64)}")
    
    # Determine mime type
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.pdf': 'application/pdf'
    }
    mime_type = mime_types.get(ext, 'image/png')
    print(f"MIME type: {mime_type}")
    
    # Build URL
    url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/mistralai/models/{MODEL_ID}:rawPredict"
    print(f"\nAPI URL: {url}")
    
    # Build payload
    payload = {
        "model": MODEL_ID,
        "document": {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{image_base64}"
        },
        "include_image_base64": include_images
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Make request
    print("\nSending request to Mistral OCR...")
    start_time = time.time()
    
    try:
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=120  # 2 minute timeout
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"Response received in {elapsed_time:.0f}ms")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                return result, elapsed_time
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Raw response: {response.text[:500]}...")
                return None, elapsed_time
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response: {response.text[:1000]}")
            return None, elapsed_time
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None, 120000
    except Exception as e:
        print(f"Error: {e}")
        return None, 0

def parse_ocr_result(result: dict):
    """Parse and display OCR result"""
    if not result:
        return
    
    print(f"\n{'='*60}")
    print("  OCR RESULT")
    print(f"{'='*60}")
    
    if "pages" in result:
        pages = result["pages"]
        print(f"Total pages: {len(pages)}")
        
        for page in pages:
            page_idx = page.get("index", 0)
            markdown = page.get("markdown", "")
            images = page.get("images", [])
            dimensions = page.get("dimensions", {})
            
            print(f"\n--- Page {page_idx} ---")
            print(f"Dimensions: {dimensions}")
            print(f"Images: {len(images)}")
            print(f"\nMarkdown content:")
            print("-" * 40)
            # Print first 2000 chars of markdown
            if len(markdown) > 2000:
                print(markdown[:2000])
                print(f"\n... (truncated, total {len(markdown)} chars)")
            else:
                print(markdown)
            print("-" * 40)
    else:
        print("Unexpected result format:")
        print(json.dumps(result, indent=2)[:2000])

def extract_table_from_markdown(markdown: str):
    """Extract table data from markdown"""
    lines = markdown.split('\n')
    table_lines = []
    in_table = False
    
    for line in lines:
        # Detect table rows (lines with |)
        if '|' in line:
            in_table = True
            # Parse table row
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells and not all(c.replace('-', '').strip() == '' for c in cells):
                table_lines.append(cells)
        elif in_table and line.strip() == '':
            # End of table
            break
    
    return table_lines

def main():
    # Test image path
    image_path = "tests/real_table_pipeline_test/nanyang_sample1.png"
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Call Mistral OCR
    result, elapsed_time = call_mistral_ocr(image_path, include_images=False)
    
    if result:
        # Parse and display result
        parse_ocr_result(result)
        
        # Save full result
        output_path = "tests/real_table_pipeline_test/output_fair/mistral_ocr_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull result saved to: {output_path}")
        
        # Extract table data
        if "pages" in result and result["pages"]:
            markdown = result["pages"][0].get("markdown", "")
            table_data = extract_table_from_markdown(markdown)
            
            if table_data:
                print(f"\n{'='*60}")
                print("  EXTRACTED TABLE DATA")
                print(f"{'='*60}")
                print(f"Rows found: {len(table_data)}")
                for i, row in enumerate(table_data[:10]):  # First 10 rows
                    print(f"Row {i}: {row}")
                if len(table_data) > 10:
                    print(f"... ({len(table_data) - 10} more rows)")
        
        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"Processing time: {elapsed_time:.0f}ms")
        print(f"Model: Mistral OCR ({MODEL_ID})")
        print(f"Status: SUCCESS")
    else:
        print("\nOCR failed. Please check:")
        print("1. gcloud CLI is installed and authenticated")
        print("2. Vertex AI API is enabled in your project")
        print("3. The region is correct")
        print("4. You have permissions to access the model")

if __name__ == "__main__":
    main()
