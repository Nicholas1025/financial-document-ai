"""
Test Mistral OCR via Google Cloud Vertex AI
Uses Service Account authentication
"""

import os
import json
import base64
import time
import requests
from pathlib import Path

from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Configuration
MODEL_ID = "mistral-ocr-2505"
PROJECT_ID = "fyp-482818"
REGION = "us-central1"  # Try this first, or "europe-west4"

# Service account JSON path
SERVICE_ACCOUNT_FILE = r"C:\Users\User\Downloads\fyp-482818-196822ca9b0e.json"

def get_access_token():
    """Get access token using service account"""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    credentials.refresh(Request())
    return credentials.token

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

def call_mistral_ocr(image_path: str):
    """
    Call Mistral OCR API via Vertex AI
    """
    print(f"\n{'='*70}")
    print(f"  MISTRAL OCR via Google Cloud Vertex AI")
    print(f"{'='*70}")
    print(f"Model: {MODEL_ID}")
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Image: {image_path}")
    
    # Get access token
    print("\nGetting access token from service account...")
    try:
        access_token = get_access_token()
        print(f"Token obtained: {access_token[:20]}...{access_token[-10:]}")
    except Exception as e:
        print(f"ERROR getting token: {e}")
        return None, 0
    
    # Encode image
    print("\nEncoding image to base64...")
    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    print(f"Base64 length: {len(image_base64)}")
    print(f"MIME type: {mime_type}")
    
    # Build URL
    url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/mistralai/models/{MODEL_ID}:rawPredict"
    print(f"\nAPI URL: {url}")
    
    # Build payload - for image, use image_url type
    payload = {
        "model": MODEL_ID,
        "document": {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{image_base64}"
        },
        "include_image_base64": False
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    # Make request
    print("\nSending request to Mistral OCR...")
    start_time = time.time()
    
    try:
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=180  # 3 minute timeout for large images
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
                print(f"Raw response: {response.text[:1000]}...")
                return None, elapsed_time
        else:
            print(f"Request failed!")
            print(f"Response: {response.text[:2000]}")
            
            # Try different region if us-central1 fails
            if REGION == "us-central1":
                print("\nTrying europe-west4...")
                return try_alternate_region(image_path, access_token, image_base64, mime_type)
            
            return None, elapsed_time
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None, 180000
    except Exception as e:
        print(f"Error: {e}")
        return None, 0

def try_alternate_region(image_path, access_token, image_base64, mime_type):
    """Try alternate region"""
    alt_region = "europe-west4"
    url = f"https://{alt_region}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{alt_region}/publishers/mistralai/models/{MODEL_ID}:rawPredict"
    
    payload = {
        "model": MODEL_ID,
        "document": {
            "type": "image_url", 
            "image_url": f"data:{mime_type};base64,{image_base64}"
        },
        "include_image_base64": False
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    start_time = time.time()
    response = requests.post(url=url, headers=headers, json=payload, timeout=180)
    elapsed_time = (time.time() - start_time) * 1000
    
    print(f"Alternate region status: {response.status_code}")
    
    if response.status_code == 200:
        return response.json(), elapsed_time
    else:
        print(f"Alternate region also failed: {response.text[:1000]}")
        return None, elapsed_time

def parse_result(result):
    """Parse and display OCR result"""
    if not result:
        return ""
    
    print(f"\n{'='*70}")
    print("  OCR RESULT")
    print(f"{'='*70}")
    
    pages = result.get("pages", [])
    print(f"Total pages: {len(pages)}")
    
    full_markdown = ""
    
    for page in pages:
        page_idx = page.get("index", 0)
        markdown = page.get("markdown", "")
        images = page.get("images", [])
        dimensions = page.get("dimensions", {})
        
        print(f"\n--- Page {page_idx} ---")
        if dimensions:
            print(f"Dimensions: {dimensions.get('width', '?')}x{dimensions.get('height', '?')} @ {dimensions.get('dpi', '?')}dpi")
        print(f"Images: {len(images)}")
        print(f"Markdown length: {len(markdown)} chars")
        
        print(f"\nContent preview:")
        print("-" * 50)
        if len(markdown) > 4000:
            print(markdown[:4000])
            print(f"\n... (truncated, showing 4000/{len(markdown)} chars)")
        else:
            print(markdown)
        print("-" * 50)
        
        full_markdown += markdown + "\n"
    
    return full_markdown

def extract_tables(markdown: str):
    """Extract table data from markdown"""
    lines = markdown.split('\n')
    tables = []
    current_table = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('|') and line.endswith('|'):
            # Skip separator rows
            inner = line[1:-1].replace('-', '').replace(':', '').replace('|', '').strip()
            if not inner:
                continue
            
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]
            
            if cells:
                current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []
    
    if current_table:
        tables.append(current_table)
    
    return tables

def main():
    """Main function"""
    image_path = "tests/real_table_pipeline_test/nanyang_sample1.png"
    output_dir = "tests/real_table_pipeline_test/output_fair"
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Check service account file
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"ERROR: Service account file not found: {SERVICE_ACCOUNT_FILE}")
        return
    
    # Call OCR
    result, elapsed_time = call_mistral_ocr(image_path)
    
    if result:
        # Parse result
        markdown = parse_result(result)
        
        # Save JSON result
        json_path = os.path.join(output_dir, "mistral_ocr_vertexai_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved: {json_path}")
        
        # Save markdown
        md_path = os.path.join(output_dir, "mistral_ocr_output.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Mistral OCR Output (Vertex AI)\n\n")
            f.write(f"**Processing time:** {elapsed_time:.0f}ms\n\n")
            f.write("---\n\n")
            f.write(markdown)
        print(f"Markdown saved: {md_path}")
        
        # Extract tables
        tables = extract_tables(markdown)
        if tables:
            print(f"\n{'='*70}")
            print(f"  EXTRACTED TABLES: {len(tables)}")
            print(f"{'='*70}")
            
            for i, table in enumerate(tables):
                print(f"\nTable {i+1} ({len(table)} rows):")
                for j, row in enumerate(table[:10]):
                    print(f"  {j}: {row}")
                if len(table) > 10:
                    print(f"  ... ({len(table)-10} more rows)")
        
        # Summary
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"Model: Mistral OCR ({MODEL_ID})")
        print(f"Processing time: {elapsed_time:.0f}ms")
        print(f"Tables found: {len(tables)}")
        print(f"Status: SUCCESS")
        
    else:
        print("\n" + "="*70)
        print("  OCR FAILED")
        print("="*70)
        print("Possible issues:")
        print("1. Model not enabled in your project")
        print("2. Region not supported")
        print("3. Service account lacks permissions")
        print("\nTo enable Mistral OCR:")
        print(f"  Visit: https://console.cloud.google.com/vertex-ai/publishers/mistralai/model-garden/mistral-ocr-2505")

if __name__ == "__main__":
    main()
