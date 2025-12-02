"""
Evaluation Metrics for Table Understanding

Includes:
- TEDS (Tree-Edit-Distance-based Similarity) for structure evaluation
- Detection metrics (Precision, Recall, F1, mAP)
- Structure metrics (Row/Column accuracy)
"""
from typing import Dict, List, Tuple, Optional
from collections import deque
import re
from html.parser import HTMLParser


class HTMLTableParser(HTMLParser):
    """Parse HTML table into tree structure for TEDS."""
    
    def __init__(self):
        super().__init__()
        self.tree = []
        self.current_path = []
        self.current_text = ""
    
    def handle_starttag(self, tag, attrs):
        if tag in ['table', 'thead', 'tbody', 'tr', 'td', 'th']:
            node = {'tag': tag, 'children': [], 'text': ''}
            if self.current_path:
                self.current_path[-1]['children'].append(node)
            else:
                self.tree.append(node)
            self.current_path.append(node)
    
    def handle_endtag(self, tag):
        if tag in ['table', 'thead', 'tbody', 'tr', 'td', 'th']:
            if self.current_path:
                self.current_path[-1]['text'] = self.current_text.strip()
                self.current_text = ""
                self.current_path.pop()
    
    def handle_data(self, data):
        self.current_text += data


def parse_html_to_tree(html: str) -> Dict:
    """Parse HTML string to tree structure."""
    parser = HTMLTableParser()
    # Clean HTML
    html = re.sub(r'\s+', ' ', html)
    parser.feed(html)
    
    if parser.tree:
        return parser.tree[0]
    return {'tag': 'table', 'children': [], 'text': ''}


def tree_edit_distance(tree1: Dict, tree2: Dict, structure_only: bool = False) -> int:
    """
    Calculate tree edit distance between two trees.
    
    Uses Zhang-Shasha algorithm (simplified).
    
    Args:
        tree1, tree2: Tree structures
        structure_only: If True, ignore text content
        
    Returns:
        Edit distance (number of operations)
    """
    def get_nodes(tree: Dict) -> List[Dict]:
        """Get all nodes in post-order."""
        nodes = []
        
        def traverse(node):
            for child in node.get('children', []):
                traverse(child)
            nodes.append(node)
        
        traverse(tree)
        return nodes
    
    def node_equal(n1: Dict, n2: Dict) -> bool:
        """Check if two nodes are equal."""
        if n1['tag'] != n2['tag']:
            return False
        if not structure_only:
            if n1.get('text', '') != n2.get('text', ''):
                return False
        return True
    
    nodes1 = get_nodes(tree1)
    nodes2 = get_nodes(tree2)
    
    m, n = len(nodes1), len(nodes2)
    
    # Dynamic programming table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if node_equal(nodes1[i-1], nodes2[j-1]):
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # Delete
                    dp[i][j-1] + 1,      # Insert
                    dp[i-1][j-1] + 1     # Replace
                )
    
    return dp[m][n]


def calculate_teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """
    Calculate TEDS (Tree-Edit-Distance-based Similarity).
    
    TEDS = 1 - (edit_distance / max(|T1|, |T2|))
    
    Args:
        pred_html: Predicted HTML table
        gt_html: Ground truth HTML table
        structure_only: If True, ignore cell content
        
    Returns:
        TEDS score (0-1, higher is better)
    """
    pred_tree = parse_html_to_tree(pred_html)
    gt_tree = parse_html_to_tree(gt_html)
    
    def count_nodes(tree: Dict) -> int:
        count = 1
        for child in tree.get('children', []):
            count += count_nodes(child)
        return count
    
    pred_size = count_nodes(pred_tree)
    gt_size = count_nodes(gt_tree)
    max_size = max(pred_size, gt_size)
    
    if max_size == 0:
        return 1.0
    
    edit_dist = tree_edit_distance(pred_tree, gt_tree, structure_only)
    teds = 1 - (edit_dist / max_size)
    
    return max(0.0, teds)  # Ensure non-negative


class TEDSEvaluator:
    """
    Evaluator for TEDS metric.
    
    Accumulates TEDS scores over multiple samples.
    """
    
    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only
        self.reset()
    
    def reset(self):
        self.scores = []
    
    def update(self, pred_html: str, gt_html: str):
        """Add a prediction-ground truth pair."""
        score = calculate_teds(pred_html, gt_html, self.structure_only)
        self.scores.append(score)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if not self.scores:
            return {'teds_mean': 0.0, 'teds_std': 0.0, 'num_samples': 0}
        
        import numpy as np
        scores_arr = np.array(self.scores)
        
        return {
            'teds_mean': float(np.mean(scores_arr)),
            'teds_std': float(np.std(scores_arr)),
            'teds_median': float(np.median(scores_arr)),
            'teds_min': float(np.min(scores_arr)),
            'teds_max': float(np.max(scores_arr)),
            'num_samples': len(self.scores)
        }


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Dict with precision, recall, f1
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Calculate Average Precision (AP).
    
    Uses 11-point interpolation.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        
    Returns:
        AP score
    """
    if not precisions or not recalls:
        return 0.0
    
    # Sort by recall
    sorted_pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    recalls_sorted = [p[0] for p in sorted_pairs]
    precisions_sorted = [p[1] for p in sorted_pairs]
    
    # 11-point interpolation
    ap = 0.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_interp = 0.0
        for r, p in zip(recalls_sorted, precisions_sorted):
            if r >= t:
                p_interp = max(p_interp, p)
        ap += p_interp / 11
    
    return ap
