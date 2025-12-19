"""
Rule-based financial validation utilities.

Provides validation rules for common financial statement equations:
- Balance Sheet: Assets = Liabilities + Equity
- Income Statement: Revenue - Expenses = Net Profit
- Column Sum: Components should sum to Total row
- Cash Flow: Opening + Changes = Closing balance

Enhanced Features:
- Automatic subtotal row detection (even without labels)
- Special row classification (subtotal, additional items, total)
- Column metadata awareness
"""
import re
from typing import Dict, List, Any, Optional, Tuple


# =============================================================================
# Row Classification
# =============================================================================

class RowClassification:
    """Classification types for table rows."""
    HEADER = 'header'
    LABEL = 'label'
    DATA = 'data'
    SUBTOTAL = 'subtotal'
    ADDITIONAL = 'additional'  # Items between subtotal and total
    TOTAL = 'total'
    EMPTY = 'empty'
    SECTION_HEADER = 'section_header'  # e.g. "Non-current assets", "Current assets"


# Section header patterns (rows that mark section boundaries)
# NOTE: For cash flow statements, section headers like 'Operating activities' 
# often have data in them, so we should NOT treat them as section headers.
# Only truly empty-value rows should be section headers.
_SECTION_HEADER_KEYWORDS = [
    'non-current assets', 'current assets',
    'non-current liabilities', 'current liabilities',
    'shareholders equity', 'stockholders equity',
    'equity attributable',
    # 'operating activities', 'investing activities', 'financing activities',  # These have values in cash flow
    'revenue', 'cost of sales', 'operating expenses',
    'changes in working capital',  # Often a header with no values
]


def classify_row(label: str, row_values: List[Optional[float]], 
                 position: int, total_rows: int) -> str:
    """
    Classify a row based on its label, values, and position.
    
    Args:
        label: Row label text
        row_values: Numeric values in the row
        position: Row index (0-based)
        total_rows: Total number of rows in table
        
    Returns:
        RowClassification type string
    """
    if not label and not any(v is not None for v in row_values):
        return RowClassification.EMPTY
    
    label_lower = (label or '').lower().strip()
    
    # Header detection (first few rows)
    if position < 2:
        return RowClassification.HEADER
    
    # Total row keywords
    total_keywords = ['total assets', 'total liabilities', 'total equity', 
                      'total revenue', 'grand total', 'net income', 'net profit',
                      # Cash flow statement endings
                      'cash and cash equivalents at the end',
                      'cash at end', 'closing cash',
                      'at the end of the year', 'at end of year']
    for kw in total_keywords:
        if kw in label_lower:
            return RowClassification.TOTAL
    
    # Cash flow statement subtotal patterns (check before generic subtotal)
    # Use regex-like matching to handle OCR errors (case variations)
    import re
    cashflow_subtotal_patterns = [
        r'net\s*cash\s*flow.*?(from|used)',  # Net cash flow ... from/used
        r'cash\s*flow\s*(from|used)',
        r'net\s*(increase|decrease).*cash',
        r'increase.*decrease.*cash',
    ]
    for pattern in cashflow_subtotal_patterns:
        if re.search(pattern, label_lower):
            return RowClassification.SUBTOTAL
    
    # Subtotal keywords
    subtotal_keywords = ['subtotal', 'sub-total', 'sub total']
    for kw in subtotal_keywords:
        if kw in label_lower:
            return RowClassification.SUBTOTAL
    
    # Generic total (check if near end)
    if 'total' in label_lower and position > total_rows * 0.7:
        return RowClassification.TOTAL
    elif 'total' in label_lower:
        return RowClassification.SUBTOTAL
    
    # Empty label but has values:
    # - If near end (last 15% of rows), likely subtotal
    # - Otherwise, treat as DATA (OCR may have missed the label)
    if not label and any(v is not None for v in row_values):
        # Only classify as subtotal if in the last portion of the table
        if position > total_rows * 0.85:
            return RowClassification.SUBTOTAL
        # Otherwise, it's likely a data row with missing label (OCR issue)
        return RowClassification.DATA
    
    # Section header detection:
    # Has a label but ALL numeric columns are empty → likely a section header
    # e.g. "Non-current assets", "Current assets", "LIABILITIES"
    if label and not any(v is not None for v in row_values):
        # Check if it matches known section header patterns
        for kw in _SECTION_HEADER_KEYWORDS:
            if kw in label_lower:
                return RowClassification.SECTION_HEADER
        # Also treat short uppercase labels with no numbers as section headers
        # e.g. "ASSETS", "LIABILITIES"
        if label.isupper() and len(label) < 30:
            return RowClassification.SECTION_HEADER
    
    return RowClassification.DATA


def classify_all_rows(labels: List[str], 
                      value_grid: List[List[Optional[float]]],
                      detect_implicit_subtotals: bool = True) -> List[str]:
    """
    Classify all rows in a table.
    
    Args:
        labels: Row labels
        value_grid: Grid of numeric values
        detect_implicit_subtotals: If True, detect rows with empty labels 
                                   that equal the sum of preceding rows
    
    Returns list of RowClassification types.
    """
    classifications = []
    total_rows = len(labels)
    
    for i, label in enumerate(labels):
        row_values = value_grid[i] if i < len(value_grid) else []
        classification = classify_row(label, row_values, i, total_rows)
        classifications.append(classification)
    
    # Post-processing: detect implicit subtotals
    # An implicit subtotal is a row with empty label whose value equals
    # the sum of preceding DATA rows (since last subtotal/total/section_header)
    if detect_implicit_subtotals:
        for i in range(len(classifications)):
            # Only check DATA rows with empty labels
            if classifications[i] != RowClassification.DATA:
                continue
            if labels[i] and labels[i].strip():
                continue  # Has a label, not implicit
            
            # Find start of component range (after last subtotal/total/section_header)
            component_start = 0
            for j in range(i - 1, -1, -1):
                if classifications[j] in (RowClassification.SUBTOTAL, 
                                          RowClassification.TOTAL,
                                          RowClassification.SECTION_HEADER):
                    component_start = j + 1
                    break
            
            # Check if this row's value equals sum of preceding DATA rows
            # for at least one numeric column
            if component_start < i and len(value_grid) > i:
                for col in range(len(value_grid[i])):
                    row_val = value_grid[i][col] if col < len(value_grid[i]) else None
                    if row_val is None:
                        continue
                    
                    # Sum preceding DATA rows
                    component_sum = 0.0
                    component_count = 0
                    for j in range(component_start, i):
                        if classifications[j] == RowClassification.DATA:
                            val = value_grid[j][col] if j < len(value_grid) and col < len(value_grid[j]) else None
                            if val is not None:
                                component_sum += val
                                component_count += 1
                    
                    # If sum matches (within 1% tolerance) and we have at least 2 components
                    if component_count >= 2 and abs(row_val) > 0:
                        tolerance = 0.01 * abs(row_val)
                        if abs(component_sum - row_val) <= tolerance:
                            classifications[i] = RowClassification.SUBTOTAL
                            break  # Found match, no need to check other columns
    
    return classifications


# =============================================================================
# Core Validation Functions
# =============================================================================

def check_equity_balance(assets: float, liabilities: float, equity: float, 
                         tolerance: float = 0.01) -> Dict:
    """
    Check Assets ≈ Liabilities + Equity (Balance Sheet equation).
    
    Args:
        assets: Total assets value
        liabilities: Total liabilities value
        equity: Total equity value
        tolerance: Relative tolerance (e.g., 0.01 = 1%)
        
    Returns:
        Dict with 'passed', 'diff', 'tolerance', 'message'
    """
    diff = assets - (liabilities + equity)
    denom = max(abs(assets), 1.0)
    passed = abs(diff) <= tolerance * denom
    return {
        'rule': 'equity_balance',
        'passed': passed,
        'expected': liabilities + equity,
        'actual': assets,
        'diff': diff,
        'tolerance': tolerance,
        'message': 'OK' if passed else 'Assets != Liabilities + Equity'
    }


def check_profit_equation(revenue: float, expenses: float, net_profit: float, 
                          tolerance: float = 0.01) -> Dict:
    """
    Check Revenue - Expenses ≈ Net Profit (Income Statement equation).
    
    Args:
        revenue: Total revenue
        expenses: Total expenses
        net_profit: Net profit/income
        tolerance: Relative tolerance
        
    Returns:
        Dict with validation result
    """
    expected = revenue - expenses
    diff = expected - net_profit
    denom = max(abs(revenue), 1.0)
    passed = abs(diff) <= tolerance * denom
    return {
        'rule': 'profit_equation',
        'passed': passed,
        'expected': expected,
        'actual': net_profit,
        'diff': diff,
        'tolerance': tolerance,
        'message': 'OK' if passed else 'Revenue - Expenses != Net Profit'
    }


def check_column_sum(values: List[float], total: float, 
                     tolerance: float = 0.01) -> Dict:
    """
    Check if a list of values sums to a total (Column Sum validation).
    
    This is the most common validation in financial tables:
    - Line items should sum to subtotal/total rows
    - Components should add up to aggregate values
    
    Args:
        values: List of numeric values to sum
        total: Expected total value
        tolerance: Relative tolerance
        
    Returns:
        Dict with validation result
    """
    computed_sum = sum(v for v in values if v is not None)
    diff = computed_sum - total
    denom = max(abs(total), 1.0)
    passed = abs(diff) <= tolerance * denom
    
    return {
        'rule': 'column_sum',
        'passed': passed,
        'expected': total,
        'actual': computed_sum,
        'diff': diff,
        'num_components': len(values),
        'tolerance': tolerance,
        'message': 'OK' if passed else f'Sum({computed_sum:.2f}) != Total({total:.2f})'
    }


def check_cash_flow_balance(opening: float, changes: float, closing: float,
                            tolerance: float = 0.01) -> Dict:
    """
    Check Opening Balance + Net Changes ≈ Closing Balance (Cash Flow equation).
    
    Args:
        opening: Opening cash balance
        changes: Net cash changes during period
        closing: Closing cash balance
        tolerance: Relative tolerance
        
    Returns:
        Dict with validation result
    """
    expected = opening + changes
    diff = expected - closing
    denom = max(abs(closing), 1.0)
    passed = abs(diff) <= tolerance * denom
    
    return {
        'rule': 'cash_flow_balance',
        'passed': passed,
        'expected': expected,
        'actual': closing,
        'diff': diff,
        'tolerance': tolerance,
        'message': 'OK' if passed else 'Opening + Changes != Closing'
    }


def check_percentage_consistency(value: float, base: float, percentage: float,
                                 tolerance: float = 0.02) -> Dict:
    """
    Check if percentage calculation is correct: value/base ≈ percentage.
    
    Common in financial reports: margins, ratios, growth rates.
    
    Args:
        value: The numerator value
        base: The denominator/base value
        percentage: The reported percentage (as decimal, e.g., 0.25 for 25%)
        tolerance: Absolute tolerance for percentage difference
        
    Returns:
        Dict with validation result
    """
    if abs(base) < 0.001:
        return {
            'rule': 'percentage_consistency',
            'passed': False,
            'message': 'Base value too small for percentage calculation'
        }
    
    computed_pct = value / base
    diff = abs(computed_pct - percentage)
    passed = diff <= tolerance
    
    return {
        'rule': 'percentage_consistency',
        'passed': passed,
        'expected': percentage,
        'actual': computed_pct,
        'diff': diff,
        'tolerance': tolerance,
        'message': 'OK' if passed else f'Computed {computed_pct:.2%} != Reported {percentage:.2%}'
    }


# =============================================================================
# Grid-based Validation (for extracted table data)
# =============================================================================

class TableValidator:
    """
    Validates extracted table data using financial rules.
    
    Works with grid data structure from the pipeline.
    Enhanced with:
    - Automatic subtotal detection (even for rows without labels)
    - Row classification
    - Special handling for additional items between subtotal and total
    """
    
    # Keywords that indicate total/summary rows
    TOTAL_KEYWORDS = [
        'total', 'net', 'sum', 'subtotal', 'grand total',
        'balance', 'aggregate', 'overall'
    ]
    
    # Keywords for specific financial statement rows
    ASSETS_KEYWORDS = ['total assets', 'assets']
    LIABILITIES_KEYWORDS = ['total liabilities', 'liabilities']
    EQUITY_KEYWORDS = ['total equity', 'equity', 'shareholders equity', 'stockholders equity']
    REVENUE_KEYWORDS = ['revenue', 'total revenue', 'net revenue', 'sales', 'turnover']
    EXPENSE_KEYWORDS = ['total expenses', 'operating expenses', 'expenses']
    PROFIT_KEYWORDS = ['net income', 'net profit', 'profit', 'net earnings', 'net periodic benefit cost']
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize validator.
        
        Args:
            tolerance: Default tolerance for validations
        """
        self.tolerance = tolerance
    
    def validate_grid(self, grid: List[List[Any]], 
                      labels: List[str],
                      normalized_grid: Optional[List[List[Dict]]] = None) -> List[Dict]:
        """
        Run all applicable validations on a table grid.
        
        Args:
            grid: 2D grid of cell values (raw strings)
            labels: Row labels (first column)
            normalized_grid: Optional normalized grid with parsed values
            
        Returns:
            List of validation results
        """
        results = []
        
        # Use normalized grid if available, otherwise try to parse raw values
        if normalized_grid:
            value_grid = self._extract_values_from_normalized(normalized_grid)
        else:
            value_grid = self._parse_grid_values(grid)
        
        # Classify all rows
        row_classes = classify_all_rows(labels, value_grid)
        
        # Run enhanced column sum validations with row classification
        column_results = self.validate_column_sums_enhanced(value_grid, labels, row_classes)
        results.extend(column_results)
        
        # Run balance sheet validation if applicable
        balance_result = self.validate_balance_sheet(value_grid, labels)
        if balance_result:
            results.append(balance_result)
        
        # Run income statement validation if applicable
        income_result = self.validate_income_statement(value_grid, labels)
        if income_result:
            results.append(income_result)
        
        return results
    
    def validate_column_sums_enhanced(self, value_grid: List[List[Optional[float]]], 
                                      labels: List[str],
                                      row_classes: List[str],
                                      include_empty_label_rows: bool = True,
                                      debug: bool = False) -> List[Dict]:
        """
        Enhanced column sum validation using row classification.
        
        Detects subtotal rows even without explicit labels.
        Handles patterns like: [data rows] -> [subtotal] -> [additional items] -> [total]
        
        Args:
            value_grid: Grid of numeric values
            labels: Row labels
            row_classes: Row classifications
            include_empty_label_rows: If True, include rows with empty labels but valid numbers
            debug: If True, include debug info about row inclusion decisions
            
        Returns:
            List of validation results
        """
        results = []
        debug_info = [] if debug else None
        
        if not value_grid or not labels:
            return results
        
        num_cols = len(value_grid[0]) if value_grid else 0
        
        # Find subtotal and total rows
        subtotal_indices = []
        total_indices = []
        
        for i, cls in enumerate(row_classes):
            if cls == RowClassification.SUBTOTAL:
                subtotal_indices.append(i)
            elif cls == RowClassification.TOTAL:
                total_indices.append(i)
        
        # Validate subtotals (components -> subtotal)
        for subtotal_idx in subtotal_indices:
            # Find component rows (DATA rows before subtotal)
            # Start from previous subtotal/total, but also respect section headers
            component_start = 0
            for prev_idx in subtotal_indices + total_indices:
                if prev_idx < subtotal_idx and prev_idx > component_start:
                    component_start = prev_idx + 1
            
            # CRITICAL FIX: Also look for section headers between component_start and subtotal_idx
            # If there's a section header, start counting from AFTER that header
            # This prevents mixing rows from different sections
            for i in range(component_start, subtotal_idx):
                if row_classes[i] == RowClassification.SECTION_HEADER:
                    # Found a section header - start counting from after it
                    component_start = i + 1
            
            # Include DATA rows; also include rows with empty labels if they have numeric values
            component_indices = []
            for i in range(component_start, subtotal_idx):
                row_class = row_classes[i]
                label = labels[i] if i < len(labels) else ''
                has_numeric = any(self._get_value(value_grid, i, c) is not None 
                                 for c in range(1, num_cols))
                
                # Skip section headers - they mark boundaries, not data
                if row_class == RowClassification.SECTION_HEADER:
                    if debug_info is not None:
                        debug_info.append({
                            'row_idx': i, 'label': label[:30] if label else '(empty)',
                            'row_class': row_class, 'has_numeric': has_numeric,
                            'included': False, 'reason': 'SECTION_HEADER (boundary)'
                        })
                    continue
                
                # Include if: DATA row, OR empty-label row with numeric values (if flag set)
                include = False
                reason = ''
                if row_class == RowClassification.DATA:
                    include = True
                    reason = 'DATA row'
                elif row_class == RowClassification.HEADER and has_numeric:
                    # Some valid data rows get misclassified as HEADER
                    include = True
                    reason = 'HEADER with numeric (reclassified as DATA)'
                elif include_empty_label_rows and not label and has_numeric:
                    include = True
                    reason = 'Empty label with numeric values'
                else:
                    reason = f'Excluded: class={row_class}, label={bool(label)}, numeric={has_numeric}'
                
                if include:
                    component_indices.append(i)
                
                if debug_info is not None:
                    debug_info.append({
                        'row_idx': i,
                        'label': label[:30] if label else '(empty)',
                        'row_class': row_class,
                        'has_numeric': has_numeric,
                        'included': include,
                        'reason': reason
                    })
            
            if not component_indices:
                continue
            
            # Validate each column
            for col in range(1, num_cols):
                component_values = [value_grid[i][col] for i in component_indices 
                                   if i < len(value_grid) and col < len(value_grid[i]) 
                                   and value_grid[i][col] is not None]
                
                if not component_values:
                    continue
                
                subtotal_value = self._get_value(value_grid, subtotal_idx, col)
                if subtotal_value is not None:
                    result = check_column_sum(component_values, subtotal_value, self.tolerance)
                    result['column'] = col
                    result['row_type'] = 'subtotal'
                    result['total_row'] = labels[subtotal_idx] if labels[subtotal_idx] else f'Subtotal (Row {subtotal_idx})'
                    result['total_row_index'] = subtotal_idx
                    result['component_rows'] = [labels[i] if labels[i] else f'Row {i}' for i in component_indices]
                    result['component_count'] = len(component_values)
                    if debug_info is not None:
                        result['debug_row_selection'] = [d for d in debug_info if d['row_idx'] in range(component_start, subtotal_idx)]
                    results.append(result)
        
        # Validate totals with additional items (subtotal + additional -> total)
        # Also handle grand totals: Total assets = Total non-current + Total current
        for total_idx in total_indices:
            # Find ALL preceding subtotals (not just the last one)
            # This handles: Total assets = Total non-current assets + Total current assets
            preceding_subtotals = [st_idx for st_idx in subtotal_indices if st_idx < total_idx]
            
            # Find the nearest boundary (previous total or start)
            boundary_start = 0
            for prev_total in total_indices:
                if prev_total < total_idx:
                    boundary_start = prev_total + 1
            
            # Filter subtotals to only those after the boundary
            relevant_subtotals = [st for st in preceding_subtotals if st >= boundary_start]
            
            if not relevant_subtotals:
                # No subtotal, sum all data rows to total
                component_indices = [i for i in range(boundary_start, total_idx)
                                    if row_classes[i] == RowClassification.DATA]
                
                # Simple validation: sum data rows to total
                for col in range(1, num_cols):
                    component_values = [value_grid[i][col] for i in component_indices
                                       if i < len(value_grid) and col < len(value_grid[i])
                                       and value_grid[i][col] is not None]
                    
                    if not component_values:
                        continue
                    
                    total_value = self._get_value(value_grid, total_idx, col)
                    if total_value is not None:
                        result = check_column_sum(component_values, total_value, self.tolerance)
                        result['column'] = col
                        result['row_type'] = 'total'
                        result['total_row'] = labels[total_idx] if labels[total_idx] else f'Total (Row {total_idx})'
                        result['total_row_index'] = total_idx
                        result['component_rows'] = [labels[i] if labels[i] else f'Row {i}' for i in component_indices]
                        result['component_count'] = len(component_values)
                        results.append(result)
            
            elif len(relevant_subtotals) == 1:
                # Single subtotal: Total = subtotal + additional items
                preceding_subtotal = relevant_subtotals[0]
                additional_indices = [i for i in range(preceding_subtotal + 1, total_idx)
                                     if row_classes[i] in (RowClassification.DATA, RowClassification.ADDITIONAL)]
                
                for col in range(1, num_cols):
                    subtotal_val = self._get_value(value_grid, preceding_subtotal, col)
                    additional_vals = [value_grid[i][col] for i in additional_indices
                                      if i < len(value_grid) and col < len(value_grid[i])
                                      and value_grid[i][col] is not None]
                    total_val = self._get_value(value_grid, total_idx, col)
                    
                    if subtotal_val is not None and total_val is not None:
                        result = check_column_sum([subtotal_val] + additional_vals, total_val, self.tolerance)
                        result['column'] = col
                        result['row_type'] = 'total_with_additions'
                        result['total_row'] = labels[total_idx] if labels[total_idx] else f'Total (Row {total_idx})'
                        result['total_row_index'] = total_idx
                        result['subtotal_row'] = labels[preceding_subtotal] if labels[preceding_subtotal] else f'Subtotal (Row {preceding_subtotal})'
                        result['subtotal_value'] = subtotal_val
                        result['additional_items'] = len(additional_vals)
                        results.append(result)
            
            else:
                # Multiple subtotals: Grand Total = sum of subtotals
                # e.g. Total assets = Total non-current assets + Total current assets
                for col in range(1, num_cols):
                    subtotal_vals = []
                    subtotal_labels = []
                    for st_idx in relevant_subtotals:
                        val = self._get_value(value_grid, st_idx, col)
                        if val is not None:
                            subtotal_vals.append(val)
                            subtotal_labels.append(labels[st_idx] if labels[st_idx] else f'Subtotal (Row {st_idx})')
                    
                    total_val = self._get_value(value_grid, total_idx, col)
                    
                    if subtotal_vals and total_val is not None:
                        result = check_column_sum(subtotal_vals, total_val, self.tolerance)
                        result['column'] = col
                        result['row_type'] = 'grand_total'
                        result['total_row'] = labels[total_idx] if labels[total_idx] else f'Total (Row {total_idx})'
                        result['total_row_index'] = total_idx
                        result['subtotal_rows'] = subtotal_labels
                        result['subtotal_count'] = len(subtotal_vals)
                        results.append(result)
        
        return results
    
    def validate_column_sums(self, value_grid: List[List[Optional[float]]], 
                             labels: List[str]) -> List[Dict]:
        """
        Legacy column sum validation (uses keywords only).
        
        For backward compatibility. Prefer validate_column_sums_enhanced.
        """
        results = []
        
        if not value_grid or not labels:
            return results
        
        num_cols = len(value_grid[0]) if value_grid else 0
        
        # Find total rows
        total_row_indices = self._find_total_rows(labels)
        
        for total_idx in total_row_indices:
            # Find component rows (rows between previous total and this total)
            prev_total = -1
            for idx in total_row_indices:
                if idx < total_idx:
                    prev_total = idx
                else:
                    break
            
            component_indices = list(range(prev_total + 1, total_idx))
            
            if not component_indices:
                continue
            
            # Validate each column
            for col in range(1, num_cols):  # Skip label column
                component_values = []
                for row_idx in component_indices:
                    if row_idx < len(value_grid) and col < len(value_grid[row_idx]):
                        val = value_grid[row_idx][col]
                        if val is not None:
                            component_values.append(val)
                
                if not component_values:
                    continue
                
                # Get total value
                if total_idx < len(value_grid) and col < len(value_grid[total_idx]):
                    total_value = value_grid[total_idx][col]
                    
                    if total_value is not None:
                        result = check_column_sum(component_values, total_value, self.tolerance)
                        result['column'] = col
                        result['total_row'] = labels[total_idx] if total_idx < len(labels) else f'Row {total_idx}'
                        result['component_rows'] = [labels[i] for i in component_indices if i < len(labels)]
                        results.append(result)
        
        return results
        
        return results
    
    def validate_balance_sheet(self, value_grid: List[List[Optional[float]]], 
                               labels: List[str]) -> Optional[Dict]:
        """
        Validate Assets = Liabilities + Equity.
        
        Args:
            value_grid: Grid of numeric values
            labels: Row labels
            
        Returns:
            Validation result or None if not applicable
        """
        assets_idx = self._find_row_by_keywords(labels, self.ASSETS_KEYWORDS)
        liab_idx = self._find_row_by_keywords(labels, self.LIABILITIES_KEYWORDS)
        equity_idx = self._find_row_by_keywords(labels, self.EQUITY_KEYWORDS)
        
        if assets_idx is None or liab_idx is None or equity_idx is None:
            return None
        
        # Validate for each numeric column
        results = []
        num_cols = len(value_grid[0]) if value_grid else 0
        
        for col in range(1, num_cols):
            assets = self._get_value(value_grid, assets_idx, col)
            liab = self._get_value(value_grid, liab_idx, col)
            equity = self._get_value(value_grid, equity_idx, col)
            
            if assets is not None and liab is not None and equity is not None:
                result = check_equity_balance(assets, liab, equity, self.tolerance)
                result['column'] = col
                results.append(result)
        
        if results:
            # Return summary of all column validations
            all_passed = all(r['passed'] for r in results)
            return {
                'rule': 'balance_sheet',
                'passed': all_passed,
                'column_results': results,
                'message': 'All columns pass' if all_passed else 'Some columns failed balance sheet check'
            }
        
        return None
    
    def validate_income_statement(self, value_grid: List[List[Optional[float]]], 
                                  labels: List[str]) -> Optional[Dict]:
        """
        Validate Income Statement equations.
        
        Args:
            value_grid: Grid of numeric values
            labels: Row labels
            
        Returns:
            Validation result or None if not applicable
        """
        revenue_idx = self._find_row_by_keywords(labels, self.REVENUE_KEYWORDS)
        expense_idx = self._find_row_by_keywords(labels, self.EXPENSE_KEYWORDS)
        profit_idx = self._find_row_by_keywords(labels, self.PROFIT_KEYWORDS)
        
        if revenue_idx is None or profit_idx is None:
            return None
        
        # Validate for each numeric column
        results = []
        num_cols = len(value_grid[0]) if value_grid else 0
        
        for col in range(1, num_cols):
            revenue = self._get_value(value_grid, revenue_idx, col)
            profit = self._get_value(value_grid, profit_idx, col)
            
            if revenue is not None and profit is not None:
                # If we have expenses, use full equation
                if expense_idx is not None:
                    expense = self._get_value(value_grid, expense_idx, col)
                    if expense is not None:
                        result = check_profit_equation(revenue, expense, profit, self.tolerance)
                        result['column'] = col
                        results.append(result)
        
        if results:
            all_passed = all(r['passed'] for r in results)
            return {
                'rule': 'income_statement',
                'passed': all_passed,
                'column_results': results,
                'message': 'All columns pass' if all_passed else 'Some columns failed income statement check'
            }
        
        return None
    
    def _find_total_rows(self, labels: List[str]) -> List[int]:
        """Find indices of rows that appear to be totals."""
        total_indices = []
        for i, label in enumerate(labels):
            if label:
                label_lower = label.lower().strip()
                for keyword in self.TOTAL_KEYWORDS:
                    if keyword in label_lower:
                        total_indices.append(i)
                        break
        return total_indices
    
    def _find_row_by_keywords(self, labels: List[str], keywords: List[str]) -> Optional[int]:
        """Find first row matching any of the keywords."""
        for i, label in enumerate(labels):
            if label:
                label_lower = label.lower().strip()
                for keyword in keywords:
                    if keyword in label_lower:
                        return i
        return None
    
    def _get_value(self, grid: List[List[Optional[float]]], 
                   row: int, col: int) -> Optional[float]:
        """Safely get value from grid."""
        if row < len(grid) and col < len(grid[row]):
            return grid[row][col]
        return None
    
    def _extract_values_from_normalized(self, normalized_grid: List[List[Dict]]) -> List[List[Optional[float]]]:
        """Extract numeric values from normalized grid."""
        value_grid = []
        for row in normalized_grid:
            value_row = []
            for cell in row:
                if isinstance(cell, dict):
                    value_row.append(cell.get('value'))
                else:
                    value_row.append(None)
            value_grid.append(value_row)
        return value_grid
    
    def _parse_grid_values(self, grid: List[List[Any]]) -> List[List[Optional[float]]]:
        """Parse numeric values from raw grid strings."""
        value_grid = []
        for row in grid:
            value_row = []
            for cell in row:
                value_row.append(self._parse_number(cell))
            value_grid.append(value_row)
        return value_grid
    
    def _parse_number(self, text: Any) -> Optional[float]:
        """Parse a number from text."""
        if text is None:
            return None
        
        text = str(text).strip()
        if not text or text == '-' or text == '—':
            return None
        
        # Handle parentheses as negative
        is_negative = False
        if '(' in text and ')' in text:
            is_negative = True
            text = text.replace('(', '').replace(')', '')
        
        # Remove currency symbols and commas
        text = re.sub(r'[$€£¥₹,]', '', text)
        
        # Extract number
        match = re.search(r'[-+]?\d+\.?\d*', text)
        if match:
            try:
                value = float(match.group())
                return -value if is_negative else value
            except ValueError:
                return None
        
        return None


# =============================================================================
# Convenience function for quick validation
# =============================================================================

def validate_table(grid: List[List[Any]], 
                   labels: List[str],
                   normalized_grid: Optional[List[List[Dict]]] = None,
                   tolerance: float = 0.01) -> List[Dict]:
    """
    Quick validation of a table grid.
    
    Args:
        grid: 2D grid of cell values
        labels: Row labels
        normalized_grid: Optional normalized grid
        tolerance: Validation tolerance
        
    Returns:
        List of validation results
    """
    validator = TableValidator(tolerance=tolerance)
    return validator.validate_grid(grid, labels, normalized_grid)
