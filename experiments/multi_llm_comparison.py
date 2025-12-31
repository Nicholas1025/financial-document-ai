"""
Multi-LLM Comparison on SynFinTabs
==================================

Compare different VLM/LLM APIs on the same SynFinTabs test samples:
1. Gemini 2.5 Flash (Google)
2. Claude 3.5 Haiku (Anthropic)  
3. Llama 3.3 70B (Together AI / Groq)

Task: Given table image + (row_key, col_key), extract cell value.
Metric: Exact Match accuracy.
"""

import os
import sys
import json
import time
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    text = str(answer).strip()
    text = text.replace(" ", "")
    text = text.replace("'", "")
    text = text.replace("'", "")
    text = text.replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return text.lower()


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


# =============================================================================
# LLM Backends
# =============================================================================

class BaseLLM:
    """Base class for LLM backends."""
    
    def __init__(self, name: str):
        self.name = name
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None
        self.min_interval = 0.5  # Default rate limit
    
    def _rate_limit(self):
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def ask_cell_value(self, image_path: str, row_label: str, col_label: str) -> str:
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    """Gemini 2.5 Flash backend."""
    
    def __init__(self):
        super().__init__("Gemini 2.5 Flash")
        self.min_interval = 13.0  # 5 RPM limit
        
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types
        
        # Use service account
        sa_file = Path(__file__).parent.parent / "configs" / "gemini_service_account.json"
        if sa_file.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(sa_file)
            self.client = genai.Client(
                vertexai=True,
                project="fyp-482818",
                location="us-central1"
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY", "")
            self.client = genai.Client(api_key=api_key)
    
    def ask_cell_value(self, image_path: str, row_label: str, col_label: str) -> str:
        self._rate_limit()
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        image_part = self.types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        prompt = f"""Look at this financial table image.

Question: What is the value in the row "{row_label}" under the column "{col_label}"?

Instructions:
- Return ONLY the numeric value as shown in the table
- Include commas if present (e.g., "29,608,638")
- Include parentheses for negative numbers (e.g., "(1,234)")
- If the cell is empty, return "EMPTY"
- If you cannot find the value, return "NOT_FOUND"
- Do not add any explanation, just the value"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image_part]
            )
            return response.text.strip()
        except Exception as e:
            return f"ERROR: {e}"


class ClaudeLLM(BaseLLM):
    """Claude 3.5 Haiku backend."""
    
    def __init__(self):
        super().__init__("Claude 3.5 Haiku")
        self.min_interval = 1.0  # Faster rate limit
        
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = httpx.Client(timeout=60.0)
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    def ask_cell_value(self, image_path: str, row_label: str, col_label: str) -> str:
        self._rate_limit()
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = f"""Look at this financial table image.

Question: What is the value in the row "{row_label}" under the column "{col_label}"?

Instructions:
- Return ONLY the numeric value as shown in the table
- Include commas if present (e.g., "29,608,638")
- Include parentheses for negative numbers (e.g., "(1,234)")
- If the cell is empty, return "EMPTY"
- If you cannot find the value, return "NOT_FOUND"
- Do not add any explanation, just the value"""

        try:
            response = self.client.post(
                self.api_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"].strip()
            else:
                return f"ERROR: {response.status_code} - {response.text}"
        except Exception as e:
            return f"ERROR: {e}"


class LlamaLLM(BaseLLM):
    """Llama 3.3 70B via Together AI."""
    
    def __init__(self):
        super().__init__("Llama 3.3 70B")
        self.min_interval = 0.5
        
        self.api_key = os.getenv("TOGETHER_API_KEY", "")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set")
        
        self.client = httpx.Client(timeout=60.0)
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        # Llama 3.2 Vision model (Llama 3.3 is text-only, use 3.2 11B Vision)
        self.model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    
    def ask_cell_value(self, image_path: str, row_label: str, col_label: str) -> str:
        self._rate_limit()
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = f"""Look at this financial table image.

Question: What is the value in the row "{row_label}" under the column "{col_label}"?

Instructions:
- Return ONLY the numeric value as shown in the table
- Include commas if present (e.g., "29,608,638")
- Include parentheses for negative numbers (e.g., "(1,234)")
- If the cell is empty, return "EMPTY"
- If you cannot find the value, return "NOT_FOUND"
- Do not add any explanation, just the value"""

        try:
            response = self.client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return f"ERROR: {response.status_code} - {response.text}"
        except Exception as e:
            return f"ERROR: {e}"


# =============================================================================
# Evaluator
# =============================================================================

class MultiLLMEvaluator:
    """Evaluate multiple LLMs on the same samples."""
    
    def __init__(self, data_path: str, output_dir: str = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/multi_llm_eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = self.output_dir / "temp_images"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize LLMs
        self.llms = {}
    
    def add_llm(self, name: str, llm: BaseLLM):
        self.llms[name] = llm
        print(f"Added LLM: {name}")
    
    def load_test_data(self, num_samples: int) -> pd.DataFrame:
        test_files = sorted(self.data_path.glob("test-*.parquet"))
        df = pd.read_parquet(test_files[0])
        return df.head(num_samples)
    
    def save_image(self, image_data: dict, sample_id: str) -> str:
        image_path = self.temp_dir / f"{sample_id}.png"
        if not image_path.exists():
            if isinstance(image_data, dict) and 'bytes' in image_data:
                img_bytes = image_data['bytes']
            else:
                img_bytes = image_data
            with open(image_path, 'wb') as f:
                f.write(img_bytes)
        return str(image_path)
    
    def run_evaluation(self, num_samples: int = 10, questions_per_sample: int = 3) -> Dict:
        """Run evaluation on all LLMs."""
        print(f"\n{'='*70}")
        print("Multi-LLM Comparison on SynFinTabs")
        print(f"LLMs: {', '.join(self.llms.keys())}")
        print(f"{'='*70}")
        
        df = self.load_test_data(num_samples)
        print(f"Loaded {len(df)} samples")
        
        # Results per LLM
        results = {name: {'correct': 0, 'total': 0, 'details': []} for name in self.llms}
        
        start_time = time.time()
        
        for idx, (_, sample) in enumerate(df.iterrows()):
            sample_id = sample['id']
            print(f"\n[{idx+1}/{len(df)}] Sample: {sample_id[:20]}...")
            
            # Save image
            image_path = self.save_image(sample['image'], sample_id)
            
            # Get questions
            questions = sample['questions']
            if questions_per_sample and len(questions) > questions_per_sample:
                step = len(questions) // questions_per_sample
                questions = questions[::step][:questions_per_sample]
            
            # Evaluate each question with each LLM
            for q in questions:
                gt_answer = q['answer']
                answer_keys = q.get('answer_keys', {})
                row_key = answer_keys.get('row', '')
                col_key = answer_keys.get('col', '')
                
                print(f"  Q: [{row_key[:25]}... x {col_key}] GT={gt_answer}")
                
                for llm_name, llm in self.llms.items():
                    try:
                        pred = llm.ask_cell_value(image_path, row_key, col_key)
                        is_correct = exact_match(pred, gt_answer)
                        
                        results[llm_name]['total'] += 1
                        if is_correct:
                            results[llm_name]['correct'] += 1
                        
                        status = "✓" if is_correct else "✗"
                        print(f"    {llm_name}: {status} Pred={pred}")
                        
                        results[llm_name]['details'].append({
                            'sample_id': sample_id,
                            'row_key': row_key,
                            'col_key': col_key,
                            'gt': gt_answer,
                            'pred': pred,
                            'correct': is_correct
                        })
                    except Exception as e:
                        print(f"    {llm_name}: ERROR - {e}")
                        results[llm_name]['total'] += 1
                        results[llm_name]['details'].append({
                            'sample_id': sample_id,
                            'error': str(e)
                        })
        
        elapsed = time.time() - start_time
        
        # Calculate accuracies
        for name in results:
            total = results[name]['total']
            correct = results[name]['correct']
            results[name]['accuracy'] = correct / total if total > 0 else 0
        
        # Print summary
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}")
        print(f"{'LLM':<25} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
        print("-" * 60)
        
        for name, r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
            print(f"{name:<25} {r['correct']:>10} {r['total']:>10} {r['accuracy']:>11.2%}")
        
        print(f"\nTotal time: {elapsed:.1f}s")
        
        # Save results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples,
            'questions_per_sample': questions_per_sample,
            'elapsed_time_sec': elapsed,
            'results': {name: {
                'correct': r['correct'],
                'total': r['total'],
                'accuracy': r['accuracy']
            } for name, r in results.items()},
            'details': {name: r['details'] for name, r in results.items()}
        }
        
        output_file = self.output_dir / f"multi_llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved to: {output_file}")
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-LLM Comparison on SynFinTabs")
    parser.add_argument("--data-path", type=str, default="D:/datasets/synfintabs/data")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--questions-per-sample", type=int, default=3)
    parser.add_argument("--llms", type=str, default="gemini,claude,llama",
                        help="Comma-separated list of LLMs to test")
    parser.add_argument("--output-dir", type=str, default="outputs/multi_llm_eval")
    
    args = parser.parse_args()
    
    evaluator = MultiLLMEvaluator(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Add requested LLMs
    llm_list = [x.strip().lower() for x in args.llms.split(",")]
    
    if "gemini" in llm_list:
        try:
            evaluator.add_llm("Gemini 2.5 Flash", GeminiLLM())
        except Exception as e:
            print(f"Failed to init Gemini: {e}")
    
    if "claude" in llm_list:
        try:
            evaluator.add_llm("Claude 3.5 Haiku", ClaudeLLM())
        except Exception as e:
            print(f"Failed to init Claude: {e}")
    
    if "llama" in llm_list:
        try:
            evaluator.add_llm("Llama 3.2 11B Vision", LlamaLLM())
        except Exception as e:
            print(f"Failed to init Llama: {e}")
    
    if not evaluator.llms:
        print("No LLMs available!")
        return
    
    results = evaluator.run_evaluation(
        num_samples=args.num_samples,
        questions_per_sample=args.questions_per_sample
    )
    
    return results


if __name__ == "__main__":
    main()
