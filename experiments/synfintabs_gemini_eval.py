"""
SynFinTabs Dataset Evaluation with Gemini 2.5 Flash (Stage 2)
=============================================================

Evaluates the Gemini LLM validation module on SynFinTabs test set.

Task: Given (row_header, col_header) from GT questions, extract cell value from table image.
Metric: Exact Match (EM) accuracy.

Reference:
- SynFinTabs Paper: FinTabQA achieves 95.87% EM on test set
- GPT-4V achieves 94% on real-world data
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.gemini_validation import GeminiTableValidator


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    - Remove whitespace
    - Standardize formatting
    """
    if not answer:
        return ""
    
    # Strip and lowercase
    text = str(answer).strip()
    
    # Normalize common variations
    text = text.replace(" ", "")  # Remove spaces
    text = text.replace("'", "")  # Remove apostrophes in £'000
    text = text.replace("'", "")  # Remove curly apostrophes
    text = text.replace(",", "")  # Remove commas for comparison
    
    # Normalize negative numbers: (xxx) -> -xxx
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    
    return text.lower()


def exact_match(pred: str, gt: str) -> bool:
    """
    Check if prediction exactly matches ground truth.
    Uses normalized comparison.
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    return pred_norm == gt_norm


def fuzzy_match(pred: str, gt: str, tolerance: float = 0.001) -> bool:
    """
    Check if prediction matches ground truth numerically.
    Allows small tolerance for floating point differences.
    """
    if exact_match(pred, gt):
        return True
    
    # Try numeric comparison
    try:
        pred_val = float(normalize_answer(pred))
        gt_val = float(normalize_answer(gt))
        
        if gt_val == 0:
            return abs(pred_val) < tolerance
        
        rel_diff = abs(pred_val - gt_val) / abs(gt_val)
        return rel_diff < tolerance
    except (ValueError, TypeError):
        return False


class SynFinTabsEvaluator:
    """
    Evaluator for SynFinTabs dataset using Gemini.
    """
    
    def __init__(self, data_path: str, output_dir: str = None):
        """
        Initialize evaluator.
        
        Args:
            data_path: Path to SynFinTabs parquet files
            output_dir: Output directory for results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/synfintabs_eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Gemini validator
        print("Initializing Gemini validator...")
        self.validator = GeminiTableValidator(model="gemini-2.5-flash")
        
        # Temporary image directory
        self.temp_dir = self.output_dir / "temp_images"
        self.temp_dir.mkdir(exist_ok=True)
        
    def load_test_data(self, num_samples: int = None) -> pd.DataFrame:
        """
        Load test data from parquet files.
        
        Args:
            num_samples: Number of samples to load (None for all)
        """
        test_files = sorted(self.data_path.glob("test-*.parquet"))
        if not test_files:
            raise FileNotFoundError(f"No test parquet files found in {self.data_path}")
        
        print(f"Found {len(test_files)} test files")
        
        # Load first file for now (each file has ~5000 samples)
        df = pd.read_parquet(test_files[0])
        print(f"Loaded {len(df)} samples from {test_files[0].name}")
        
        if num_samples:
            df = df.head(num_samples)
            print(f"Using first {num_samples} samples")
        
        return df
    
    def save_image(self, image_data: dict, sample_id: str) -> str:
        """
        Save image from parquet to disk.
        
        Args:
            image_data: Image dict with 'bytes' and 'path' keys
            sample_id: Sample identifier
            
        Returns:
            Path to saved image
        """
        image_path = self.temp_dir / f"{sample_id}.png"
        
        if not image_path.exists():
            # Extract bytes from the image dict
            if isinstance(image_data, dict) and 'bytes' in image_data:
                img_bytes = image_data['bytes']
            else:
                img_bytes = image_data
            
            # Save to file
            with open(image_path, 'wb') as f:
                f.write(img_bytes)
        
        return str(image_path)
    
    def evaluate_sample(self, sample: pd.Series) -> Dict:
        """
        Evaluate a single sample.
        
        Args:
            sample: Row from the dataframe
            
        Returns:
            Evaluation results dict
        """
        sample_id = sample['id']
        questions = sample['questions']
        
        # Save image
        try:
            image_path = self.save_image(sample['image'], sample_id)
        except Exception as e:
            print(f"  Error saving image: {e}")
            return {
                'sample_id': sample_id,
                'error': str(e),
                'questions_evaluated': 0,
                'correct': 0
            }
        
        results = {
            'sample_id': sample_id,
            'theme': sample.get('theme', -1),
            'num_questions': len(questions),
            'qa_results': [],
            'correct': 0,
            'fuzzy_correct': 0
        }
        
        # Evaluate each question
        for q in questions:
            q_result = self.evaluate_question(image_path, q)
            results['qa_results'].append(q_result)
            
            if q_result['exact_match']:
                results['correct'] += 1
            if q_result.get('fuzzy_match', False):
                results['fuzzy_correct'] += 1
        
        results['accuracy'] = results['correct'] / len(questions) if len(questions) > 0 else 0
        results['fuzzy_accuracy'] = results['fuzzy_correct'] / len(questions) if len(questions) > 0 else 0
        
        return results
    
    def evaluate_question(self, image_path: str, question: dict) -> Dict:
        """
        Evaluate a single QA question using Gemini.
        
        Args:
            image_path: Path to table image
            question: Question dict with keys: question, answer, answer_keys
            
        Returns:
            QA result dict
        """
        gt_answer = question['answer']
        answer_keys = question.get('answer_keys', {})
        row_key = answer_keys.get('row', '')
        col_key = answer_keys.get('col', '')
        
        # Use Gemini to answer the question
        try:
            pred_answer = self.validator.ask_cell_value(
                image_path=image_path,
                row_label=row_key,
                col_label=col_key
            )
        except Exception as e:
            pred_answer = f"ERROR: {str(e)}"
        
        # Evaluate
        is_exact = exact_match(pred_answer, gt_answer)
        is_fuzzy = fuzzy_match(pred_answer, gt_answer)
        
        return {
            'question_id': question.get('id', ''),
            'question': question.get('question', ''),
            'row_key': row_key,
            'col_key': col_key,
            'gt_answer': gt_answer,
            'pred_answer': pred_answer,
            'exact_match': is_exact,
            'fuzzy_match': is_fuzzy
        }
    
    def run_evaluation(self, num_samples: int = 50, questions_per_sample: int = 5) -> Dict:
        """
        Run evaluation on test set.
        
        Args:
            num_samples: Number of table samples to evaluate
            questions_per_sample: Max questions per sample (to control API calls)
            
        Returns:
            Evaluation summary
        """
        print(f"\n{'='*60}")
        print("SynFinTabs Evaluation with Gemini 2.5 Flash")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_test_data(num_samples)
        
        all_results = []
        total_questions = 0
        total_correct = 0
        total_fuzzy_correct = 0
        
        start_time = time.time()
        
        for idx, (_, sample) in enumerate(df.iterrows()):
            print(f"\n[{idx+1}/{len(df)}] Sample: {sample['id']}")
            
            # Limit questions per sample
            questions = sample['questions']
            if questions_per_sample and len(questions) > questions_per_sample:
                # Sample evenly across questions
                step = len(questions) // questions_per_sample
                sample_copy = sample.copy()
                sample_copy['questions'] = questions[::step][:questions_per_sample]
            else:
                sample_copy = sample
            
            # Evaluate
            result = self.evaluate_sample(sample_copy)
            all_results.append(result)
            
            # Update totals
            total_questions += len(result.get('qa_results', []))
            total_correct += result.get('correct', 0)
            total_fuzzy_correct += result.get('fuzzy_correct', 0)
            
            # Print progress
            if result.get('qa_results'):
                print(f"  Questions: {len(result['qa_results'])}, "
                      f"Correct: {result['correct']}, "
                      f"Accuracy: {result['accuracy']:.1%}")
                
                # Show examples
                for qr in result['qa_results'][:2]:
                    match_str = "✓" if qr['exact_match'] else "✗"
                    print(f"    {match_str} [{qr['row_key']} x {qr['col_key']}]: "
                          f"GT={qr['gt_answer']}, Pred={qr['pred_answer']}")
        
        elapsed_time = time.time() - start_time
        
        # Calculate final metrics
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        overall_fuzzy = total_fuzzy_correct / total_questions if total_questions > 0 else 0
        
        # Get token usage
        token_usage = self.validator.get_token_usage()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(df),
            'total_questions': total_questions,
            'total_correct': total_correct,
            'total_fuzzy_correct': total_fuzzy_correct,
            'exact_match_accuracy': overall_accuracy,
            'fuzzy_match_accuracy': overall_fuzzy,
            'elapsed_time_sec': elapsed_time,
            'avg_time_per_sample': elapsed_time / len(df) if df.size > 0 else 0,
            'token_usage': token_usage,
            'config': {
                'model': self.validator.model_name,
                'questions_per_sample': questions_per_sample
            },
            'results_by_sample': all_results
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Samples evaluated: {len(df)}")
        print(f"Total questions: {total_questions}")
        print(f"Exact Match accuracy: {overall_accuracy:.2%}")
        print(f"Fuzzy Match accuracy: {overall_fuzzy:.2%}")
        print(f"Time elapsed: {elapsed_time:.1f}s")
        print(f"API requests: {token_usage['request_count']}")
        print(f"Total tokens: {token_usage['total_tokens']:,}")
        
        print(f"\n--- Comparison with FinTabQA Paper ---")
        print(f"FinTabQA (LayoutLM): 95.87% EM")
        print(f"This run (Gemini):   {overall_accuracy:.2%} EM")
        print(f"Difference: {overall_accuracy - 0.9587:+.2%}")
        
        # Save results
        output_file = self.output_dir / f"synfintabs_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {output_file}")
        
        return summary


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Gemini on SynFinTabs")
    parser.add_argument("--data-path", type=str, 
                        default="D:/datasets/synfintabs/data",
                        help="Path to SynFinTabs parquet files")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of samples to evaluate")
    parser.add_argument("--questions-per-sample", type=int, default=3,
                        help="Max questions per sample")
    parser.add_argument("--output-dir", type=str, 
                        default="outputs/synfintabs_eval",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = SynFinTabsEvaluator(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_evaluation(
        num_samples=args.num_samples,
        questions_per_sample=args.questions_per_sample
    )
    
    return results


if __name__ == "__main__":
    main()
