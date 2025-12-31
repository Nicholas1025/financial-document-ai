"""
Step 6: LLM-based Validation using Gemini 2.5 Flash
=================================================
Uses Google's Gemini 2.5 Flash (FREE tier) for Table QA validation.

Based on SynFinTabs paper approach:
- Send table image + structured QA prompts
- Compare LLM answers against ground truth
- Calculate accuracy metrics
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Try new google-genai first, fallback to deprecated package
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_SDK = False

# Configure Gemini API - supports both API key and Service Account
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path(__file__).parent.parent / "configs" / "gemini_service_account.json")
)


class GeminiTableValidator:
    """
    LLM-based Table QA Validator using Gemini 2.5 Flash.
    
    Implements validation similar to SynFinTabs paper:
    - Cell Value QA: "What is the value in row X, column Y?"
    - Aggregation QA: "What is the total of column X?"
    - Lookup QA: "What is the value for item X in year Y?"
    """
    
    def __init__(self, api_key: str = None, service_account_file: str = None,
                 model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini validator.
        
        Args:
            api_key: Gemini API key (optional)
            service_account_file: Path to service account JSON (optional)
            model: Model name (gemini-2.5-flash is FREE)
        """
        self.model_name = model
        
        # Initialize client based on available credentials
        if USE_NEW_SDK:
            # Use new google-genai SDK
            sa_file = service_account_file or SERVICE_ACCOUNT_FILE
            if sa_file and Path(sa_file).exists():
                # Service Account authentication
                self.client = genai.Client(
                    vertexai=True,
                    project="fyp-482818",
                    location="us-central1"
                )
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_file
                print(f"Using Service Account: {sa_file}")
            else:
                # API Key authentication
                key = api_key or GEMINI_API_KEY
                self.client = genai.Client(api_key=key)
                print("Using API Key authentication")
            self.model = None  # Will use client.models.generate_content
        else:
            # Use deprecated google.generativeai SDK
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(model)
            self.client = None
            print("Using deprecated SDK with API Key")
        
        # Rate limiting for free tier (5 RPM for gemini-2.5-flash)
        self.request_count = 0
        self.last_request_time = None
        self.min_request_interval = 13.0  # seconds between requests (5 RPM = 12s + buffer)
        
        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        
    def _rate_limit(self):
        """Apply rate limiting for free tier (5 RPM)."""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed
                print(f"    [Rate limit: waiting {wait_time:.1f}s...]")
                time.sleep(wait_time)
        self.last_request_time = time.time()
        self.request_count += 1
        
    def _load_image(self, image_path: str) -> dict:
        """Load image and convert to base64 for Gemini."""
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        if USE_NEW_SDK:
            return types.Part.from_bytes(data=image_data, mime_type="image/png")
        else:
            return {
                "mime_type": "image/png",
                "data": base64.b64encode(image_data).decode("utf-8")
            }
    
    def _generate(self, prompt: str, image) -> Tuple[str, Dict]:
        """Generate content using the appropriate SDK.
        
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        token_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            if USE_NEW_SDK and self.client:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image]
                )
                # Extract token usage from response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    token_usage = {
                        "prompt_tokens": getattr(usage, 'prompt_token_count', 0) or 0,
                        "output_tokens": getattr(usage, 'candidates_token_count', 0) or 0,
                        "total_tokens": getattr(usage, 'total_token_count', 0) or 0
                    }
                # Update totals
                self.total_prompt_tokens += token_usage["prompt_tokens"]
                self.total_output_tokens += token_usage["output_tokens"]
                self.total_tokens += token_usage["total_tokens"]
                return response.text.strip(), token_usage
            else:
                response = self.model.generate_content([prompt, image])
                return response.text.strip(), token_usage
        except Exception as e:
            return f"ERROR: {str(e)}", token_usage
    
    def get_token_usage(self) -> Dict:
        """Get cumulative token usage."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count
        }
    
    def _extract_numeric(self, text: str) -> Optional[float]:
        """Extract numeric value from text response."""
        if not text:
            return None
        # Remove common formatting
        cleaned = text.replace(",", "").replace("'", "").replace(" ", "")
        cleaned = cleaned.replace("RM", "").replace("$", "").replace("USD", "")
        cleaned = cleaned.strip()
        
        # Try to parse as number
        try:
            # Handle parentheses for negative numbers
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            return float(cleaned)
        except ValueError:
            return None
    
    def ask_cell_value(self, image_path: str, row_label: str, col_label: str) -> str:
        """
        Ask Gemini for a specific cell value.
        
        Args:
            image_path: Path to table image
            row_label: Row identifier (e.g., "Cash and short-term funds")
            col_label: Column identifier (e.g., "2024 RM'000")
            
        Returns:
            LLM's answer as string
        """
        self._rate_limit()
        
        image = self._load_image(image_path)
        prompt = f"""Look at this financial table image.

Question: What is the value in the row "{row_label}" under the column "{col_label}"?

Instructions:
- Return ONLY the numeric value as shown in the table
- Include commas if present (e.g., "29,608,638")
- If the cell is empty, return "EMPTY"
- If you cannot find the value, return "NOT_FOUND"
- Do not add any explanation, just the value"""

        text, token_usage = self._generate(prompt, image)
        return text
    
    def ask_total(self, image_path: str, col_label: str, section: str = None) -> str:
        """
        Ask Gemini for a column total or subtotal.
        
        Args:
            image_path: Path to table image
            col_label: Column to sum
            section: Optional section name (e.g., "ASSETS", "LIABILITIES")
            
        Returns:
            LLM's answer as string
        """
        self._rate_limit()
        
        image = self._load_image(image_path)
        
        if section:
            prompt = f"""Look at this financial table image.

Question: What is the TOTAL value for the "{section}" section under the column "{col_label}"?

Instructions:
- Look for a row labeled "Total {section}" or "TOTAL {section}" 
- Return ONLY the numeric value
- Include commas if present
- If you cannot find it, return "NOT_FOUND"
- Do not calculate, just read the value from the table"""
        else:
            prompt = f"""Look at this financial table image.

Question: What is the TOTAL value at the bottom of the column "{col_label}"?

Instructions:
- Return ONLY the numeric value
- Include commas if present
- If you cannot find it, return "NOT_FOUND"
- Do not calculate, just read the value from the table"""

        text, token_usage = self._generate(prompt, image)
        return text
    
    def ask_year_comparison(self, image_path: str, row_label: str, 
                           year1: str, year2: str) -> str:
        """
        Ask Gemini to compare values between two years.
        
        Args:
            image_path: Path to table image
            row_label: Row identifier
            year1, year2: Years to compare
            
        Returns:
            LLM's answer (e.g., "increased", "decreased", "unchanged")
        """
        self._rate_limit()
        
        image = self._load_image(image_path)
        prompt = f"""Look at this financial table image.

Question: For the row "{row_label}", did the value INCREASE or DECREASE from {year1} to {year2}?

Instructions:
- Compare the numeric values between the two years
- Return ONLY one of: "INCREASED", "DECREASED", or "UNCHANGED"
- Do not add any explanation"""

        answer, token_usage = self._generate(prompt, image)
        if "INCREASE" in answer.upper():
            return "INCREASED"
        elif "DECREASE" in answer.upper():
            return "DECREASED"
        else:
            return "UNCHANGED"
    
    def extract_full_table(self, image_path: str) -> Dict:
        """
        Ask Gemini to extract the full table structure.
        
        Args:
            image_path: Path to table image
            
        Returns:
            Dictionary with table data
        """
        self._rate_limit()
        
        image = self._load_image(image_path)
        prompt = """Look at this financial table image and extract ALL data.

Return the table as JSON with this structure:
{
  "headers": ["Column1", "Column2", ...],
  "rows": [
    {"row_header": "Row Name", "values": ["val1", "val2", ...]},
    ...
  ]
}

Instructions:
- Include ALL rows and columns
- Preserve exact text as shown (including numbers with commas)
- Keep empty cells as empty strings ""
- Return ONLY the JSON, no other text"""

        text, token_usage = self._generate(prompt, image)
        try:
            # Remove markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": text}
        except Exception as e:
            return {"error": str(e)}


class TableQAEvaluator:
    """
    Evaluates LLM Table QA accuracy against ground truth.
    Similar to SynFinTabs paper evaluation methodology.
    """
    
    def __init__(self, validator: GeminiTableValidator):
        """
        Initialize evaluator.
        
        Args:
            validator: GeminiTableValidator instance
        """
        self.validator = validator
        self.results = []
        
    def generate_qa_pairs(self, gt_data: Dict) -> List[Dict]:
        """
        Generate QA pairs from ground truth data.
        
        Args:
            gt_data: Ground truth JSON data
            
        Returns:
            List of QA pairs with expected answers
        """
        qa_pairs = []
        cells = gt_data.get("cells", [])
        
        # Build lookup structures
        headers = {}  # col -> header text
        row_labels = {}  # row -> label text
        data_cells = []
        
        for cell in cells:
            row, col = cell["row"], cell["col"]
            text = cell.get("text", "")
            semantic = cell.get("semantic_type", "data")
            
            if semantic == "column_header" and row == 0:
                headers[col] = text
            elif semantic == "row_header" or col == 0:
                row_labels[row] = text
            
            if cell.get("numeric_value") is not None and semantic == "data":
                data_cells.append(cell)
        
        # Generate Cell Value QA pairs
        for cell in data_cells[:10]:  # Limit to 10 questions per table
            row, col = cell["row"], cell["col"]
            if row in row_labels and col in headers:
                qa_pairs.append({
                    "type": "cell_value",
                    "question": f"What is the value for '{row_labels[row]}' in column '{headers[col]}'?",
                    "row_label": row_labels[row],
                    "col_label": headers[col],
                    "expected_text": cell.get("text", ""),
                    "expected_numeric": cell.get("numeric_value")
                })
        
        return qa_pairs
    
    def evaluate_sample(self, image_path: str, gt_path: str) -> Dict:
        """
        Evaluate LLM on a single sample.
        
        Args:
            image_path: Path to table image
            gt_path: Path to ground truth JSON
            
        Returns:
            Evaluation results
        """
        # Load ground truth
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        
        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(gt_data)
        
        results = {
            "image": str(image_path),
            "ground_truth": str(gt_path),
            "total_questions": len(qa_pairs),
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "details": []
        }
        
        for qa in qa_pairs:
            print(f"  Q: {qa['row_label']} / {qa['col_label']}")
            
            # Ask LLM
            answer = self.validator.ask_cell_value(
                image_path, 
                qa["row_label"], 
                qa["col_label"]
            )
            
            # Evaluate answer
            is_correct = False
            expected = qa["expected_text"]
            
            # Check text match
            if answer.replace(",", "") == expected.replace(",", ""):
                is_correct = True
            # Check numeric match
            elif qa["expected_numeric"] is not None:
                answer_num = self.validator._extract_numeric(answer)
                if answer_num is not None:
                    is_correct = abs(answer_num - qa["expected_numeric"]) < 0.01
            
            status = "correct" if is_correct else "incorrect"
            if "ERROR" in answer:
                status = "error"
                results["errors"] += 1
            elif is_correct:
                results["correct"] += 1
            else:
                results["incorrect"] += 1
            
            results["details"].append({
                "question": qa["question"],
                "expected": expected,
                "got": answer,
                "status": status
            })
            
            print(f"    Expected: {expected}, Got: {answer} [{status}]")
        
        # Calculate accuracy
        answered = results["total_questions"] - results["errors"]
        results["accuracy"] = results["correct"] / answered if answered > 0 else 0.0
        
        return results
    
    def evaluate_batch(self, samples: List[Tuple[str, str]], 
                       output_dir: str = None) -> Dict:
        """
        Evaluate LLM on multiple samples.
        
        Args:
            samples: List of (image_path, gt_path) tuples
            output_dir: Optional directory to save results
            
        Returns:
            Aggregated evaluation results
        """
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.validator.model_name,
            "total_samples": len(samples),
            "total_questions": 0,
            "total_correct": 0,
            "total_incorrect": 0,
            "total_errors": 0,
            "token_usage": {},
            "samples": []
        }
        
        for i, (img_path, gt_path) in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Evaluating: {Path(img_path).name}")
            
            sample_result = self.evaluate_sample(img_path, gt_path)
            all_results["samples"].append(sample_result)
            all_results["total_questions"] += sample_result["total_questions"]
            all_results["total_correct"] += sample_result["correct"]
            all_results["total_incorrect"] += sample_result["incorrect"]
            all_results["total_errors"] += sample_result["errors"]
        
        # Calculate overall accuracy
        answered = all_results["total_questions"] - all_results["total_errors"]
        all_results["overall_accuracy"] = (
            all_results["total_correct"] / answered if answered > 0 else 0.0
        )
        
        # Add token usage summary
        all_results["token_usage"] = self.validator.get_token_usage()
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir) / f"gemini_validation_{timestamp}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        
        return all_results


def run_demo(num_questions: int = 20):
    """Run validation demo on CIMB Bank sample.
    
    Args:
        num_questions: Number of QA questions to test (default 20)
    """
    print("=" * 60)
    print("Step 6: LLM-based Validation using Gemini 2.5 Flash")
    print("=" * 60)
    
    # Setup paths
    data_dir = Path(__file__).parent.parent / "data" / "samples"
    output_dir = Path(__file__).parent.parent / "outputs" / "validation"
    
    # Initialize validator
    print("\nInitializing Gemini 2.5 Flash validator...")
    validator = GeminiTableValidator()
    evaluator = TableQAEvaluator(validator)
    
    # Override QA pair generation to use more questions
    original_generate = evaluator.generate_qa_pairs
    def generate_more_qa(gt_data):
        qa_pairs = []
        cells = gt_data.get("cells", [])
        headers = {}
        row_labels = {}
        data_cells = []
        
        for cell in cells:
            row, col = cell["row"], cell["col"]
            text = cell.get("text", "")
            semantic = cell.get("semantic_type", "data")
            
            if semantic == "column_header" and row == 0:
                headers[col] = text
            elif semantic == "row_header" or col == 0:
                row_labels[row] = text
            
            if cell.get("numeric_value") is not None and semantic == "data":
                data_cells.append(cell)
        
        # Generate more QA pairs (up to num_questions)
        for cell in data_cells[:num_questions]:
            row, col = cell["row"], cell["col"]
            if row in row_labels and col in headers:
                qa_pairs.append({
                    "type": "cell_value",
                    "question": f"What is the value for '{row_labels[row]}' in column '{headers[col]}'?",
                    "row_label": row_labels[row],
                    "col_label": headers[col],
                    "expected_text": cell.get("text", ""),
                    "expected_numeric": cell.get("numeric_value")
                })
        return qa_pairs
    
    evaluator.generate_qa_pairs = generate_more_qa
    
    # Define samples
    samples = [
        (
            str(data_dir / "CIMB_BANK-SAMPLE1.png"),
            str(data_dir / "CIMB_BANK-SAMPLE1_gt.json")
        )
    ]
    
    # Check if files exist
    for img, gt in samples:
        if not Path(img).exists():
            print(f"ERROR: Image not found: {img}")
            return
        if not Path(gt).exists():
            print(f"ERROR: Ground truth not found: {gt}")
            return
    
    # Run evaluation
    print(f"\nRunning Table QA evaluation ({num_questions} questions)...")
    results = evaluator.evaluate_batch(samples, str(output_dir))
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct: {results['total_correct']}")
    print(f"Incorrect: {results['total_incorrect']}")
    print(f"Errors: {results['total_errors']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    print("-" * 60)
    print("TOKEN USAGE:")
    token_usage = results.get('token_usage', {})
    print(f"  Prompt Tokens:  {token_usage.get('total_prompt_tokens', 0):,}")
    print(f"  Output Tokens:  {token_usage.get('total_output_tokens', 0):,}")
    print(f"  Total Tokens:   {token_usage.get('total_tokens', 0):,}")
    print(f"  API Requests:   {token_usage.get('request_count', 0)}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import sys
    num_q = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_demo(num_questions=num_q)
