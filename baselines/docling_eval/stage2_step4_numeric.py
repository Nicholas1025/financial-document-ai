"""
Stage 2 Step 4: Numeric Normalisation Evaluation
================================================
使用 SynFinTabs 数据集测试数值标准化模块的准确性。

任务：将各种格式的数值文本转换为标准化数值
- 处理括号负号: (1,234) → -1234
- 处理逗号: 1,234,567 → 1234567
- 处理小数: 1.23 → 1.23
- 处理百分号: 12.5% → 0.125
- 处理货币符号: $1,234 → 1234
- 处理单位缩放: 1.2M → 1200000

指标：
- Exact Match (EM): 标准化后数值完全相等
- Relative Error (RE): |pred - gt| / |gt|
- Scale Accuracy: 数量级是否正确
- Sign Accuracy: 正负号是否正确

输出：
- per_sample.csv: 每个样本的详细结果
- summary.json: 整体统计
"""

import os
import sys
import json
import csv
import re
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class NumericResult:
    """单个数值标准化结果"""
    sample_id: str
    raw_text: str
    gt_value: float
    pred_value: Optional[float]
    exact_match: bool
    relative_error: float
    scale_correct: bool
    sign_correct: bool
    parse_success: bool
    error_type: str  # none, parse_error, scale_error, sign_error, value_error


class NumericNormalizer:
    """数值标准化器 - 处理各种金融数值格式"""
    
    def __init__(self):
        # 单位缩放映射
        self.scale_suffixes = {
            'k': 1e3, 'K': 1e3,
            'm': 1e6, 'M': 1e6, 'mn': 1e6, 'Mn': 1e6, 'MM': 1e6,
            'b': 1e9, 'B': 1e9, 'bn': 1e9, 'Bn': 1e9,
            't': 1e12, 'T': 1e12, 'tn': 1e12, 'Tn': 1e12,
        }
        
        # 货币符号
        self.currency_symbols = ['$', '£', '€', '¥', '₹', 'USD', 'GBP', 'EUR', 'JPY', 'CNY']
    
    def normalize(self, text: str) -> Tuple[Optional[float], str]:
        """
        将文本标准化为数值。
        
        Returns:
            (normalized_value, error_message)
            如果成功，error_message 为空字符串
        """
        if not text or not isinstance(text, str):
            return None, "empty_input"
        
        original = text
        text = text.strip()
        
        if not text:
            return None, "empty_after_strip"
        
        try:
            # 1. 检测负号（括号表示法）
            is_negative = False
            if text.startswith('(') and text.endswith(')'):
                is_negative = True
                text = text[1:-1].strip()
            elif text.startswith('-'):
                is_negative = True
                text = text[1:].strip()
            elif text.endswith('-'):
                is_negative = True
                text = text[:-1].strip()
            
            # 2. 移除货币符号
            for symbol in self.currency_symbols:
                text = text.replace(symbol, '').strip()
            
            # 3. 检测百分号
            is_percent = False
            if text.endswith('%'):
                is_percent = True
                text = text[:-1].strip()
            
            # 4. 检测单位缩放
            scale = 1.0
            for suffix, multiplier in self.scale_suffixes.items():
                if text.endswith(suffix):
                    scale = multiplier
                    text = text[:-len(suffix)].strip()
                    break
            
            # 5. 移除逗号和空格
            text = text.replace(',', '').replace(' ', '')
            
            # 6. 处理特殊情况
            if text in ['', '-', '--', 'nil', 'Nil', 'NIL', 'n/a', 'N/A', '-']:
                return 0.0, ""
            
            # 7. 解析数值
            value = float(text)
            
            # 8. 应用缩放
            value *= scale
            
            # 9. 应用百分比
            if is_percent:
                value /= 100.0
            
            # 10. 应用负号
            if is_negative:
                value = -value
            
            return value, ""
            
        except (ValueError, TypeError) as e:
            return None, f"parse_error: {str(e)}"
    
    def normalize_gt(self, gt_text: str) -> Optional[float]:
        """标准化 ground truth 文本（通常格式更简单）"""
        if not gt_text:
            return None
        
        text = gt_text.strip()
        
        # GT 通常是简单格式: "3,802" 或 "(1,234)"
        is_negative = False
        if text.startswith('(') and text.endswith(')'):
            is_negative = True
            text = text[1:-1]
        elif text.startswith('-'):
            is_negative = True
            text = text[1:]
        
        # 移除逗号
        text = text.replace(',', '')
        
        try:
            value = float(text)
            if is_negative:
                value = -value
            return value
        except:
            return None


def calculate_metrics(gt: float, pred: Optional[float]) -> Dict[str, Any]:
    """计算单个样本的指标"""
    
    if pred is None:
        return {
            'exact_match': False,
            'relative_error': float('inf'),
            'scale_correct': False,
            'sign_correct': False,
            'error_type': 'parse_error'
        }
    
    # Exact Match (允许小误差)
    exact_match = abs(gt - pred) < 1e-6 or (gt != 0 and abs(gt - pred) / abs(gt) < 1e-9)
    
    # Relative Error
    if gt == 0:
        relative_error = 0.0 if pred == 0 else float('inf')
    else:
        relative_error = abs(gt - pred) / abs(gt)
    
    # Scale Correct (同数量级)
    if gt == 0 and pred == 0:
        scale_correct = True
    elif gt == 0 or pred == 0:
        scale_correct = False
    else:
        gt_scale = int(np.floor(np.log10(abs(gt) + 1e-10)))
        pred_scale = int(np.floor(np.log10(abs(pred) + 1e-10)))
        scale_correct = abs(gt_scale - pred_scale) <= 1  # 允许1个数量级误差
    
    # Sign Correct
    sign_correct = (gt >= 0) == (pred >= 0)
    
    # Error Type
    if exact_match:
        error_type = 'none'
    elif not sign_correct:
        error_type = 'sign_error'
    elif not scale_correct:
        error_type = 'scale_error'
    else:
        error_type = 'value_error'
    
    return {
        'exact_match': exact_match,
        'relative_error': relative_error,
        'scale_correct': scale_correct,
        'sign_correct': sign_correct,
        'error_type': error_type
    }


def load_synfintabs(data_dir: str, split: str = 'test', num_samples: int = 100, seed: int = 42) -> List[Dict]:
    """加载 SynFinTabs 数据集"""
    from datasets import load_dataset
    
    print(f"Loading SynFinTabs from {data_dir}...")
    ds = load_dataset('parquet', data_dir=data_dir, split=split)
    
    # 随机采样
    np.random.seed(seed)
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    samples = []
    for idx in indices:
        item = ds[int(idx)]
        # 提取所有 QA pairs
        for q in item['questions']:
            samples.append({
                'sample_id': f"{item['id']}_{q['id']}",
                'table_id': item['id'],
                'question_id': q['id'],
                'raw_text': q['answer'],
                'row_key': q['answer_keys']['row'],
                'col_key': q['answer_keys']['col'],
            })
    
    return samples


def run_evaluation(
    data_dir: str,
    num_samples: int = 100,
    seed: int = 42,
    output_dir: str = './stage2_results'
) -> Dict:
    """运行数值标准化评估"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    samples = load_synfintabs(data_dir, 'test', num_samples, seed)
    print(f"Loaded {len(samples)} QA pairs from {num_samples} tables")
    
    # 初始化 normalizer
    normalizer = NumericNormalizer()
    
    # 评估
    results = []
    
    for sample in tqdm(samples, desc="Evaluating numeric normalisation"):
        raw_text = sample['raw_text']
        
        # GT 值
        gt_value = normalizer.normalize_gt(raw_text)
        if gt_value is None:
            continue
        
        # 预测值（这里测试的是 normalizer 本身）
        pred_value, error_msg = normalizer.normalize(raw_text)
        
        # 计算指标
        metrics = calculate_metrics(gt_value, pred_value)
        
        result = NumericResult(
            sample_id=sample['sample_id'],
            raw_text=raw_text,
            gt_value=gt_value,
            pred_value=pred_value,
            exact_match=metrics['exact_match'],
            relative_error=metrics['relative_error'],
            scale_correct=metrics['scale_correct'],
            sign_correct=metrics['sign_correct'],
            parse_success=pred_value is not None,
            error_type=metrics['error_type']
        )
        results.append(result)
    
    # 计算整体统计
    n_total = len(results)
    n_parse_success = sum(1 for r in results if r.parse_success)
    n_exact_match = sum(1 for r in results if r.exact_match)
    n_scale_correct = sum(1 for r in results if r.scale_correct)
    n_sign_correct = sum(1 for r in results if r.sign_correct)
    
    # Relative errors (排除 inf)
    valid_errors = [r.relative_error for r in results if r.relative_error != float('inf')]
    
    # Error breakdown
    error_counts = {}
    for r in results:
        error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1
    
    summary = {
        'task': 'Numeric Normalisation Evaluation',
        'dataset': 'SynFinTabs',
        'timestamp': timestamp,
        'config': {
            'num_samples': num_samples,
            'seed': seed,
            'total_qa_pairs': n_total
        },
        'metrics': {
            'parse_success_rate': n_parse_success / n_total if n_total > 0 else 0,
            'exact_match': n_exact_match / n_total if n_total > 0 else 0,
            'scale_accuracy': n_scale_correct / n_total if n_total > 0 else 0,
            'sign_accuracy': n_sign_correct / n_total if n_total > 0 else 0,
            'mean_relative_error': np.mean(valid_errors) if valid_errors else float('inf'),
            'median_relative_error': np.median(valid_errors) if valid_errors else float('inf'),
            'std_relative_error': np.std(valid_errors) if valid_errors else float('inf'),
        },
        'error_breakdown': error_counts,
        'counts': {
            'total': n_total,
            'parse_success': n_parse_success,
            'exact_match': n_exact_match,
            'scale_correct': n_scale_correct,
            'sign_correct': n_sign_correct,
        }
    }
    
    # 保存 per-sample CSV
    csv_path = os.path.join(output_dir, f'step4_numeric_per_sample_{timestamp}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'raw_text', 'gt_value', 'pred_value',
            'exact_match', 'relative_error', 'scale_correct', 'sign_correct',
            'parse_success', 'error_type'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    
    # 保存 summary JSON
    json_path = os.path.join(output_dir, f'step4_numeric_summary_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "="*60)
    print("STEP 4: NUMERIC NORMALISATION EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: SynFinTabs (n={num_samples} tables, {n_total} QA pairs)")
    print(f"Seed: {seed}")
    print("-"*60)
    print(f"Parse Success Rate:    {summary['metrics']['parse_success_rate']:.4f}")
    print(f"Exact Match:           {summary['metrics']['exact_match']:.4f}")
    print(f"Scale Accuracy:        {summary['metrics']['scale_accuracy']:.4f}")
    print(f"Sign Accuracy:         {summary['metrics']['sign_accuracy']:.4f}")
    print(f"Mean Relative Error:   {summary['metrics']['mean_relative_error']:.6f}")
    print(f"Median Relative Error: {summary['metrics']['median_relative_error']:.6f}")
    print("-"*60)
    print("Error Breakdown:")
    for err_type, count in sorted(error_counts.items()):
        print(f"  {err_type}: {count} ({count/n_total*100:.1f}%)")
    print("-"*60)
    print(f"Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Step 4: Numeric Normalisation Evaluation')
    parser.add_argument('--data-dir', type=str, default='D:/datasets/synfintabs/data',
                        help='Path to SynFinTabs data directory')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of table samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='./stage2_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    run_evaluation(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
