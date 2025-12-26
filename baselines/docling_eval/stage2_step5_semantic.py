"""
Stage 2 Step 5: Semantic Cell Type Classification Evaluation
============================================================
使用 SynFinTabs 数据集测试 Cell 语义分类的准确性。

任务：预测 cell 的语义类型
- section_title: 章节标题
- column_header: 列头（通常是日期/年份）
- row_header: 行头（通常是账目名称）
- data: 数据单元格（数值）
- currency_unit: 货币单位

方法：
1. Heuristic Baseline: 基于规则的分类
2. Position-based: 基于位置的分类
3. Content-based: 基于内容的分类

指标：
- Accuracy
- Macro F1
- Per-class Precision/Recall/F1
- Confusion Matrix

输出：
- per_cell.csv: 每个 cell 的详细结果
- confusion_matrix.csv: 混淆矩阵
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
from collections import Counter
import numpy as np
from tqdm import tqdm

# Add parent path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Cell type labels
CELL_TYPES = ['section_title', 'column_header', 'row_header', 'data', 'currency_unit']


@dataclass
class CellResult:
    """单个 cell 分类结果"""
    sample_id: str
    cell_id: str
    row_idx: int
    col_idx: int
    text: str
    gt_label: str
    pred_label: str
    correct: bool


class HeuristicCellClassifier:
    """基于规则的 Cell 分类器"""
    
    def __init__(self):
        # 货币单位模式
        self.currency_patterns = [
            r'^[\$£€¥₹]$',
            r'^[\$£€¥₹]\s*\'?000s?$',
            r'^\'?000s?$',
            r'^[Mm]illions?$',
            r'^[Bb]illions?$',
            r'^[Tt]housands?$',
            r'^USD$', r'^GBP$', r'^EUR$',
        ]
        
        # 年份/日期模式（column header）
        self.date_patterns = [
            r'^\d{4}$',  # 2001
            r'^\d{1,2}\.\d{1,2}\.\d{2,4}$',  # 31.12.2001
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # 31/12/2001
            r'^[A-Za-z]{3,9}\s+\d{4}$',  # December 2001
            r'^\d{4}-\d{2}-\d{2}$',  # 2001-12-31
        ]
        
        # 数值模式（data）
        self.number_patterns = [
            r'^-?\d+$',  # 123
            r'^-?\d{1,3}(,\d{3})*$',  # 1,234,567
            r'^-?\d+\.\d+$',  # 123.45
            r'^-?\d{1,3}(,\d{3})*\.\d+$',  # 1,234.56
            r'^\(-?\d{1,3}(,\d{3})*\)$',  # (1,234)
            r'^\(-?\d{1,3}(,\d{3})*\.\d+\)$',  # (1,234.56)
            r'^[\$£€¥₹]?\s*-?\d',  # $123
            r'^-$',  # dash for nil
            r'^nil$',
            r'^—$',  # em dash
        ]
        
        # Section title 关键词
        self.section_keywords = [
            'total', 'subtotal', 'net', 'gross', 'assets', 'liabilities',
            'equity', 'revenue', 'expenses', 'profit', 'loss', 'income',
            'creditors', 'debtors', 'capital', 'reserves', 'provisions',
            'amounts falling due', 'amounts owed', 'called up', 'share capital'
        ]
    
    def classify(self, text: str, row_idx: int, col_idx: int, 
                 num_rows: int, num_cols: int, row_cells: List[Dict]) -> str:
        """
        分类单个 cell。
        
        Args:
            text: cell 文本
            row_idx: 行索引
            col_idx: 列索引
            num_rows: 总行数
            num_cols: 总列数
            row_cells: 同行所有 cells
        
        Returns:
            预测的 cell 类型
        """
        text = text.strip() if text else ""
        text_lower = text.lower()
        
        # 空 cell
        if not text:
            return 'data'  # 空 cell 默认为 data
        
        # 1. 检查是否为货币单位
        for pattern in self.currency_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return 'currency_unit'
        
        # 2. 第一行通常是 column headers
        if row_idx == 0:
            # 检查是否为日期/年份
            for pattern in self.date_patterns:
                if re.match(pattern, text):
                    return 'column_header'
            # 第一行第一列可能是空的或标题
            if col_idx == 0:
                if not text or text_lower in ['', 'note', 'notes']:
                    return 'data'
                return 'column_header'
            return 'column_header'
        
        # 3. 检查是否为数值（data）
        for pattern in self.number_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return 'data'
        
        # 4. 第一列通常是 row headers 或 section titles
        if col_idx == 0:
            # 检查是否包含 section 关键词
            for keyword in self.section_keywords:
                if keyword in text_lower:
                    return 'section_title'
            return 'row_header'
        
        # 5. 如果是文本且不在第一列，可能是 data 的文本描述
        # 或者是 header 的延续
        
        # 检查同行第一列是否为空或 section title
        if row_cells and len(row_cells) > 0:
            first_cell_text = row_cells[0].get('text', '').lower()
            for keyword in self.section_keywords:
                if keyword in first_cell_text:
                    return 'data'
        
        # 默认为 data
        return 'data'


class PositionBasedClassifier:
    """基于位置的分类器（更简单的 baseline）"""
    
    def classify(self, text: str, row_idx: int, col_idx: int,
                 num_rows: int, num_cols: int, row_cells: List[Dict]) -> str:
        """纯基于位置的分类"""
        
        # 第一行 = column_header
        if row_idx == 0:
            return 'column_header'
        
        # 第一列 = row_header
        if col_idx == 0:
            return 'row_header'
        
        # 其他 = data
        return 'data'


def calculate_metrics(results: List[CellResult]) -> Dict:
    """计算分类指标"""
    
    # 基础统计
    n_total = len(results)
    n_correct = sum(1 for r in results if r.correct)
    accuracy = n_correct / n_total if n_total > 0 else 0
    
    # Per-class metrics
    per_class = {}
    for label in CELL_TYPES:
        tp = sum(1 for r in results if r.gt_label == label and r.pred_label == label)
        fp = sum(1 for r in results if r.gt_label != label and r.pred_label == label)
        fn = sum(1 for r in results if r.gt_label == label and r.pred_label != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for r in results if r.gt_label == label)
        }
    
    # Macro F1
    macro_f1 = np.mean([m['f1'] for m in per_class.values()])
    
    # Weighted F1
    total_support = sum(m['support'] for m in per_class.values())
    weighted_f1 = sum(m['f1'] * m['support'] for m in per_class.values()) / total_support if total_support > 0 else 0
    
    # Confusion matrix
    confusion = {gt: {pred: 0 for pred in CELL_TYPES} for gt in CELL_TYPES}
    for r in results:
        if r.gt_label in confusion and r.pred_label in confusion[r.gt_label]:
            confusion[r.gt_label][r.pred_label] += 1
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': per_class,
        'confusion_matrix': confusion,
        'counts': {
            'total': n_total,
            'correct': n_correct
        }
    }


def load_synfintabs_cells(data_dir: str, split: str = 'test', 
                          num_samples: int = 100, seed: int = 42) -> List[Dict]:
    """加载 SynFinTabs 并提取所有 cells"""
    from datasets import load_dataset
    
    print(f"Loading SynFinTabs from {data_dir}...")
    ds = load_dataset('parquet', data_dir=data_dir, split=split)
    
    # 随机采样
    np.random.seed(seed)
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    all_cells = []
    for idx in tqdm(indices, desc="Extracting cells"):
        item = ds[int(idx)]
        table_id = item['id']
        rows = item['rows']
        num_rows = len(rows)
        
        for row_idx, row in enumerate(rows):
            cells = row.get('cells', [])
            num_cols = len(cells)
            
            for col_idx, cell in enumerate(cells):
                all_cells.append({
                    'sample_id': table_id,
                    'cell_id': f"{table_id}_r{row_idx}_c{col_idx}",
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'num_rows': num_rows,
                    'num_cols': num_cols,
                    'text': cell.get('text', ''),
                    'gt_label': cell.get('label', 'data'),
                    'row_cells': cells,  # 同行所有 cells
                })
    
    return all_cells


def run_evaluation(
    data_dir: str,
    num_samples: int = 100,
    seed: int = 42,
    output_dir: str = './stage2_results'
) -> Dict:
    """运行 Cell 分类评估"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    cells = load_synfintabs_cells(data_dir, 'test', num_samples, seed)
    print(f"Loaded {len(cells)} cells from {num_samples} tables")
    
    # GT label 分布
    gt_dist = Counter(c['gt_label'] for c in cells)
    print(f"GT Label Distribution: {dict(gt_dist)}")
    
    # 初始化分类器
    classifiers = {
        'heuristic': HeuristicCellClassifier(),
        'position_only': PositionBasedClassifier(),
    }
    
    all_results = {}
    
    for clf_name, classifier in classifiers.items():
        print(f"\nEvaluating {clf_name} classifier...")
        
        results = []
        for cell in tqdm(cells, desc=f"Classifying ({clf_name})"):
            pred_label = classifier.classify(
                text=cell['text'],
                row_idx=cell['row_idx'],
                col_idx=cell['col_idx'],
                num_rows=cell['num_rows'],
                num_cols=cell['num_cols'],
                row_cells=cell['row_cells']
            )
            
            result = CellResult(
                sample_id=cell['sample_id'],
                cell_id=cell['cell_id'],
                row_idx=cell['row_idx'],
                col_idx=cell['col_idx'],
                text=cell['text'][:100],  # 截断长文本
                gt_label=cell['gt_label'],
                pred_label=pred_label,
                correct=cell['gt_label'] == pred_label
            )
            results.append(result)
        
        # 计算指标
        metrics = calculate_metrics(results)
        all_results[clf_name] = {
            'results': results,
            'metrics': metrics
        }
    
    # 汇总
    summary = {
        'task': 'Semantic Cell Type Classification',
        'dataset': 'SynFinTabs',
        'timestamp': timestamp,
        'config': {
            'num_samples': num_samples,
            'seed': seed,
            'total_cells': len(cells),
            'gt_distribution': dict(gt_dist)
        },
        'classifiers': {}
    }
    
    for clf_name, data in all_results.items():
        summary['classifiers'][clf_name] = {
            'accuracy': data['metrics']['accuracy'],
            'macro_f1': data['metrics']['macro_f1'],
            'weighted_f1': data['metrics']['weighted_f1'],
            'per_class': data['metrics']['per_class'],
            'counts': data['metrics']['counts']
        }
    
    # 保存 per-cell CSV (只保存 heuristic 的详细结果)
    heuristic_results = all_results['heuristic']['results']
    csv_path = os.path.join(output_dir, f'step5_semantic_per_cell_{timestamp}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'cell_id', 'row_idx', 'col_idx', 'text',
            'gt_label', 'pred_label', 'correct'
        ])
        writer.writeheader()
        for r in heuristic_results:
            writer.writerow(asdict(r))
    
    # 保存 confusion matrix CSV
    confusion_path = os.path.join(output_dir, f'step5_confusion_matrix_{timestamp}.csv')
    with open(confusion_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['GT \\ Pred'] + CELL_TYPES)
        # Data
        confusion = all_results['heuristic']['metrics']['confusion_matrix']
        for gt_label in CELL_TYPES:
            row = [gt_label] + [confusion[gt_label][pred] for pred in CELL_TYPES]
            writer.writerow(row)
    
    # 保存 summary JSON
    json_path = os.path.join(output_dir, f'step5_semantic_summary_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "="*70)
    print("STEP 5: SEMANTIC CELL TYPE CLASSIFICATION RESULTS")
    print("="*70)
    print(f"Dataset: SynFinTabs (n={num_samples} tables, {len(cells)} cells)")
    print(f"Seed: {seed}")
    print("-"*70)
    
    print(f"\n{'Classifier':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-"*70)
    for clf_name in classifiers.keys():
        m = summary['classifiers'][clf_name]
        print(f"{clf_name:<20} {m['accuracy']:.4f}       {m['macro_f1']:.4f}       {m['weighted_f1']:.4f}")
    
    # Heuristic per-class
    print("\n" + "-"*70)
    print("Heuristic Classifier - Per-Class Metrics:")
    print(f"{'Label':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-"*70)
    heuristic_metrics = summary['classifiers']['heuristic']['per_class']
    for label in CELL_TYPES:
        m = heuristic_metrics[label]
        print(f"{label:<20} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['support']}")
    
    # Confusion matrix
    print("\n" + "-"*70)
    print("Confusion Matrix (Heuristic):")
    header = "GT \\ Pred"
    print(f"{header:<15}", end='')
    for pred in CELL_TYPES:
        print(f"{pred[:8]:<10}", end='')
    print()
    confusion = all_results['heuristic']['metrics']['confusion_matrix']
    for gt_label in CELL_TYPES:
        print(f"{gt_label:<15}", end='')
        for pred in CELL_TYPES:
            print(f"{confusion[gt_label][pred]:<10}", end='')
        print()
    
    print("-"*70)
    print(f"Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {confusion_path}")
    print(f"  - {json_path}")
    print("="*70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Step 5: Semantic Cell Type Classification')
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
