"""
Rule-based financial validation utilities.
"""
from typing import Dict


def check_equity_balance(assets: float, liabilities: float, equity: float, tolerance: float = 0.01) -> Dict:
    """
    Check Assets ≈ Liabilities + Equity.
    tolerance: relative tolerance (e.g., 0.01 = 1%).
    Returns dict with passed flag and difference.
    """
    diff = assets - (liabilities + equity)
    denom = max(abs(assets), 1.0)
    passed = abs(diff) <= tolerance * denom
    return {
        'passed': passed,
        'diff': diff,
        'tolerance': tolerance,
        'message': 'OK' if passed else 'Assets != Liabilities + Equity'
    }


def check_profit_equation(revenue: float, expenses: float, net_profit: float, tolerance: float = 0.01) -> Dict:
    """
    Check Revenue - Expenses ≈ Net Profit.
    """
    diff = (revenue - expenses) - net_profit
    denom = max(abs(revenue), 1.0)
    passed = abs(diff) <= tolerance * denom
    return {
        'passed': passed,
        'diff': diff,
        'tolerance': tolerance,
        'message': 'OK' if passed else 'Revenue - Expenses != Net Profit'
    }
