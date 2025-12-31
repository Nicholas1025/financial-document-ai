"""
SynFinTabs Full Pipeline Evaluation
====================================

Tests the complete pipeline on SynFinTabs:
1. Use GT table structure (rows/cells data from parquet)
2. Build grid from GT annotations
3. Apply Numeric Normalization
4. Apply Semantic Mapping
5. Use LLM (Gemini) only for discrepancy resolution

This evaluates our Stage 2-4 modules without OCR/Structure detection overhead.

Flow:
  GT Structure → Grid Build → Numeric → Semantic → Cell Lookup → Compare with GT Answer
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

from modules.numeric import normalize_numeric, detect_cell_type, _detect_currency, _detect_unit
from modules.semantic import map_alias, correct_ocr_text


def fuzzy_match_term(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Simple fuzzy matching based on character overlap."""
    if not text1 or not text2:
        return False
    
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    
    # Exact match
    if t1 == t2:
        return True
    
    # One is substring of other
    if t1 in t2 or t2 in t1:
        return True
    
    # Character-level similarity (Jaccard)
    set1 = set(t1)
    set2 = set(t2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union > 0:
        similarity = intersection / union
        return similarity >= threshold
    
    return False


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
    """Check exact match after normalization."""
    return normalize_answer(pred) == normalize_answer(gt)


class SynFinTabsGridBuilder:
    """
    Build a structured grid from SynFinTabs GT annotations.
    
    This simulates what our pipeline would produce if OCR and Structure Detection were perfect.
    """
    
    def __init__(self):
        pass
    
    def build_grid_from_sample(self, sample: pd.Series) -> Tuple[List[List[str]], List[str], List[str]]:
        """
        Build grid from GT row/cell annotations.
        
        Args:
            sample: Row from SynFinTabs dataframe
            
        Returns:
            Tuple of (grid, row_labels, col_headers)
        """
        rows_data = sample['rows']
        
        if rows_data is None or (hasattr(rows_data, '__len__') and len(rows_data) == 0):
            return [], [], []
        
        # Extract column headers from first row
        first_row = rows_data[0]
        cells = first_row.get('cells', [])
        col_headers = []
        for cell in cells:
            text = cell.get('text', '')
            col_headers.append(text)
        
        # Build grid
        grid = []
        row_labels = []
        
        for row in rows_data:
            cells = row.get('cells', [])
            row_data = []
            for idx, cell in enumerate(cells):
                text = cell.get('text', '')
                row_data.append(text)
            
            if row_data:
                grid.append(row_data)
                row_labels.append(row_data[0] if row_data else '')
        
        return grid, row_labels, col_headers
    
    def apply_ocr_simulation(self, grid: List[List[str]], ocr_results: dict) -> List[List[str]]:
        """
        Optionally replace GT text with OCR results to simulate real pipeline.
        
        Args:
            grid: GT grid
            ocr_results: OCR results from parquet (with OCR errors)
            
        Returns:
            Grid with OCR text (may contain errors)
        """
        # For now, return GT grid as-is
        # TODO: Map OCR words to cells based on bbox overlap
        return grid


class PipelineCellLookup:
    """
    Cell lookup using Numeric and Semantic modules.
    
    Given a grid and (row_key, col_key), find the cell value.
    """
    
    def __init__(self, use_semantic: bool = True, use_numeric: bool = True):
        self.use_semantic = use_semantic
        self.use_numeric = use_numeric
    
    def find_cell(self, grid: List[List[str]], row_labels: List[str], 
                  col_headers: List[str], row_key: str, col_key: str) -> Tuple[str, Dict]:
        """
        Find cell value by row and column key.
        
        Args:
            grid: Table grid
            row_labels: Row labels
            col_headers: Column headers  
            row_key: Row identifier to search for
            col_key: Column identifier to search for
            
        Returns:
            Tuple of (cell_value, debug_info)
        """
        debug = {
            'row_key': row_key,
            'col_key': col_key,
            'row_idx': None,
            'col_idx': None,
            'match_method': None
        }
        
        # Step 1: Find row index
        row_idx = self._find_row(row_labels, row_key)
        debug['row_idx'] = row_idx
        
        # Step 2: Find column index
        col_idx = self._find_column(col_headers, col_key)
        debug['col_idx'] = col_idx
        
        if row_idx is None or col_idx is None:
            debug['error'] = f"Row found: {row_idx is not None}, Col found: {col_idx is not None}"
            return "NOT_FOUND", debug
        
        # Step 3: Get cell value
        if row_idx < len(grid) and col_idx < len(grid[row_idx]):
            cell_value = grid[row_idx][col_idx]
            
            # Step 4: Apply numeric normalization if needed
            if self.use_numeric and cell_value:
                normalized = normalize_numeric(cell_value)
                debug['raw_value'] = cell_value
                debug['normalized'] = normalized
                # For comparison, we use the raw string value
            
            return cell_value, debug
        
        debug['error'] = "Index out of range"
        return "NOT_FOUND", debug
    
    def _find_row(self, row_labels: List[str], row_key: str) -> Optional[int]:
        """Find row index by label matching."""
        row_key_norm = row_key.lower().strip()
        
        # Exact match first
        for idx, label in enumerate(row_labels):
            if label.lower().strip() == row_key_norm:
                return idx
        
        # Semantic match (with OCR correction)
        if self.use_semantic:
            for idx, label in enumerate(row_labels):
                corrected = correct_ocr_text(label)
                if corrected.lower().strip() == row_key_norm:
                    return idx
                
                # Fuzzy match
                if fuzzy_match_term(label, row_key, threshold=0.85):
                    return idx
        
        # Substring match
        for idx, label in enumerate(row_labels):
            if row_key_norm in label.lower() or label.lower() in row_key_norm:
                return idx
        
        return None
    
    def _find_column(self, col_headers: List[str], col_key: str) -> Optional[int]:
        """Find column index by header matching."""
        col_key_norm = col_key.lower().strip()
        
        # Exact match
        for idx, header in enumerate(col_headers):
            if header.lower().strip() == col_key_norm:
                return idx
        
        # Substring match (for year columns like "2001" in "31.12.2001")
        for idx, header in enumerate(col_headers):
            if col_key_norm in header.lower():
                return idx
        
        return None


class SynFinTabsFullPipelineEvaluator:
    """
    Evaluate full pipeline on SynFinTabs.
    
    Modes:
    1. GT-only: Use GT structure and text → Test pure lookup logic
    2. GT+OCR: Use GT structure but OCR text → Test with OCR errors
    3. Full Pipeline: Run actual pipeline → Test end-to-end (slow)
    """
    
    def __init__(self, data_path: str, output_dir: str = None, use_llm_fallback: bool = False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/synfintabs_pipeline_eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.grid_builder = SynFinTabsGridBuilder()
        self.cell_lookup = PipelineCellLookup(use_semantic=True, use_numeric=True)
        
        self.use_llm_fallback = use_llm_fallback
        self.llm_validator = None
        
        if use_llm_fallback:
            from modules.gemini_validation import GeminiTableValidator
            self.llm_validator = GeminiTableValidator(model="gemini-2.5-flash")
        
        # Temp image dir for LLM
        self.temp_dir = self.output_dir / "temp_images"
        self.temp_dir.mkdir(exist_ok=True)
    
    def load_test_data(self, num_samples: int = None) -> pd.DataFrame:
        """Load test data."""
        test_files = sorted(self.data_path.glob("test-*.parquet"))
        if not test_files:
            raise FileNotFoundError(f"No test files in {self.data_path}")
        
        df = pd.read_parquet(test_files[0])
        print(f"Loaded {len(df)} samples")
        
        if num_samples:
            df = df.head(num_samples)
        
        return df
    
    def save_image(self, image_data: dict, sample_id: str) -> str:
        """Save image for LLM fallback."""
        image_path = self.temp_dir / f"{sample_id}.png"
        if not image_path.exists():
            if isinstance(image_data, dict) and 'bytes' in image_data:
                img_bytes = image_data['bytes']
            else:
                img_bytes = image_data
            with open(image_path, 'wb') as f:
                f.write(img_bytes)
        return str(image_path)
    
    def evaluate_sample(self, sample: pd.Series, questions_limit: int = None) -> Dict:
        """Evaluate a single sample."""
        sample_id = sample['id']
        questions = sample['questions']
        
        # Build grid from GT
        grid, row_labels, col_headers = self.grid_builder.build_grid_from_sample(sample)
        
        if not grid:
            return {
                'sample_id': sample_id,
                'error': 'Failed to build grid',
                'correct': 0,
                'total': 0
            }
        
        # Limit questions
        if questions_limit and len(questions) > questions_limit:
            step = len(questions) // questions_limit
            questions = questions[::step][:questions_limit]
        
        results = {
            'sample_id': sample_id,
            'theme': sample.get('theme', -1),
            'grid_size': f"{len(grid)}x{len(grid[0]) if grid else 0}",
            'num_questions': len(questions),
            'qa_results': [],
            'correct': 0,
            'llm_fallback_used': 0,
            'llm_correct': 0
        }
        
        for q in questions:
            gt_answer = q['answer']
            answer_keys = q.get('answer_keys', {})
            row_key = answer_keys.get('row', '')
            col_key = answer_keys.get('col', '')
            
            # Step 1: Try pipeline lookup (Numeric + Semantic)
            pred_value, debug_info = self.cell_lookup.find_cell(
                grid, row_labels, col_headers, row_key, col_key
            )
            
            is_correct = exact_match(pred_value, gt_answer)
            used_llm = False
            llm_answer = None
            
            # Step 2: LLM fallback if pipeline fails
            if not is_correct and self.use_llm_fallback and self.llm_validator:
                image_path = self.save_image(sample['image'], sample_id)
                try:
                    llm_answer = self.llm_validator.ask_cell_value(
                        image_path=image_path,
                        row_label=row_key,
                        col_label=col_key
                    )
                    used_llm = True
                    results['llm_fallback_used'] += 1
                    
                    # Check if LLM got it right
                    if exact_match(llm_answer, gt_answer):
                        is_correct = True
                        results['llm_correct'] += 1
                        pred_value = llm_answer
                except Exception as e:
                    llm_answer = f"ERROR: {e}"
            
            if is_correct:
                results['correct'] += 1
            
            results['qa_results'].append({
                'question_id': q.get('id', ''),
                'row_key': row_key,
                'col_key': col_key,
                'gt_answer': gt_answer,
                'pipeline_answer': pred_value,
                'llm_answer': llm_answer,
                'used_llm_fallback': used_llm,
                'correct': is_correct,
                'debug': debug_info
            })
        
        results['total'] = len(questions)
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
        
        return results
    
    def run_evaluation(self, num_samples: int = 50, questions_per_sample: int = 5) -> Dict:
        """Run full evaluation."""
        print(f"\n{'='*60}")
        print("SynFinTabs Full Pipeline Evaluation")
        print(f"Mode: Numeric + Semantic" + (" + LLM Fallback" if self.use_llm_fallback else ""))
        print(f"{'='*60}")
        
        df = self.load_test_data(num_samples)
        
        all_results = []
        total_questions = 0
        total_correct = 0
        total_pipeline_only_correct = 0
        total_llm_fallback = 0
        total_llm_correct = 0
        
        start_time = time.time()
        
        for idx, (_, sample) in enumerate(df.iterrows()):
            print(f"\n[{idx+1}/{len(df)}] Sample: {sample['id'][:20]}...")
            
            result = self.evaluate_sample(sample, questions_limit=questions_per_sample)
            all_results.append(result)
            
            total_questions += result.get('total', 0)
            total_correct += result.get('correct', 0)
            total_llm_fallback += result.get('llm_fallback_used', 0)
            total_llm_correct += result.get('llm_correct', 0)
            
            # Pipeline-only correct (without LLM help)
            pipeline_correct = result.get('correct', 0) - result.get('llm_correct', 0)
            total_pipeline_only_correct += pipeline_correct
            
            # Show progress
            print(f"  Grid: {result.get('grid_size', 'N/A')}, "
                  f"Questions: {result.get('total', 0)}, "
                  f"Pipeline: {pipeline_correct}, "
                  f"LLM Fallback: {result.get('llm_correct', 0)}, "
                  f"Total Correct: {result.get('correct', 0)}")
            
            # Show first few results
            for qr in result.get('qa_results', [])[:2]:
                status = "✓" if qr['correct'] else "✗"
                src = " (LLM)" if qr.get('used_llm_fallback') and qr['correct'] else ""
                print(f"    {status}{src} [{qr['row_key'][:20]} x {qr['col_key']}]: "
                      f"GT={qr['gt_answer']}, Pred={qr['pipeline_answer']}")
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        pipeline_accuracy = total_pipeline_only_correct / total_questions if total_questions > 0 else 0
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        llm_recovery_rate = total_llm_correct / total_llm_fallback if total_llm_fallback > 0 else 0
        
        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(df),
            'total_questions': total_questions,
            'metrics': {
                'pipeline_only_correct': total_pipeline_only_correct,
                'pipeline_accuracy': pipeline_accuracy,
                'llm_fallback_calls': total_llm_fallback,
                'llm_recovered_correct': total_llm_correct,
                'llm_recovery_rate': llm_recovery_rate,
                'overall_correct': total_correct,
                'overall_accuracy': overall_accuracy,
            },
            'config': {
                'use_semantic': True,
                'use_numeric': True,
                'use_llm_fallback': self.use_llm_fallback
            },
            'elapsed_time_sec': elapsed,
            'results_by_sample': all_results
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Samples: {len(df)}")
        print(f"Total questions: {total_questions}")
        print(f"\n--- Pipeline (Numeric + Semantic) ---")
        print(f"Correct: {total_pipeline_only_correct}/{total_questions}")
        print(f"Accuracy: {pipeline_accuracy:.2%}")
        
        if self.use_llm_fallback:
            print(f"\n--- LLM Fallback (Gemini) ---")
            print(f"Fallback calls: {total_llm_fallback}")
            print(f"Recovered: {total_llm_correct}")
            print(f"Recovery rate: {llm_recovery_rate:.2%}")
            print(f"\n--- Overall (Pipeline + LLM) ---")
            print(f"Correct: {total_correct}/{total_questions}")
            print(f"Accuracy: {overall_accuracy:.2%}")
        
        print(f"\nTime: {elapsed:.1f}s")
        
        # Comparison
        print(f"\n--- Comparison ---")
        print(f"FinTabQA (LayoutLM): 95.87%")
        print(f"This run (Pipeline): {pipeline_accuracy:.2%}")
        if self.use_llm_fallback:
            print(f"This run (+ LLM):    {overall_accuracy:.2%}")
        
        # Save
        output_file = self.output_dir / f"pipeline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved to: {output_file}")
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Pipeline Evaluation on SynFinTabs")
    parser.add_argument("--data-path", type=str, default="D:/datasets/synfintabs/data")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--questions-per-sample", type=int, default=5)
    parser.add_argument("--use-llm-fallback", action="store_true", help="Use Gemini as fallback")
    parser.add_argument("--output-dir", type=str, default="outputs/synfintabs_pipeline_eval")
    
    args = parser.parse_args()
    
    evaluator = SynFinTabsFullPipelineEvaluator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_llm_fallback=args.use_llm_fallback
    )
    
    results = evaluator.run_evaluation(
        num_samples=args.num_samples,
        questions_per_sample=args.questions_per_sample
    )
    
    return results


if __name__ == "__main__":
    main()
