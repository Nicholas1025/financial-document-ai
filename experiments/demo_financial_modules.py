"""
Demonstrate numeric normalization, semantic mapping, and rule-based validation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.numeric import normalize_numeric
from modules.semantic import map_alias
from modules.validation import check_equity_balance, check_profit_equation


def demo_numeric():
    samples = [
        "(1,234) RM'000",
        "USD -5,600,000",
        "1.2m",
        "3.4 billion",
        "750k",
        "-123",
    ]
    print("Numeric Normalisation:")
    for s in samples:
        result = normalize_numeric(s, default_currency='USD')
        print(f"  {s:20s} -> value={result['value']}, currency={result['currency']}, unit={result['unit']}, is_negative={result['is_negative']}")
    print()


def demo_semantic():
    samples = ["Turnover", "Sales", "PBT", "PAT", "Total Assets", "Operating Income"]
    print("Semantic Mapping:")
    for s in samples:
        mapped = map_alias(s)
        print(f"  {s:20s} -> {mapped}")
    print()


def demo_validation():
    print("Rule-Based Validation:")
    # Case 1: pass
    result1 = check_equity_balance(assets=1000, liabilities=400, equity=600)
    print(f"  Assets=1000, Liab=400, Equity=600 -> passed={result1['passed']} diff={result1['diff']:.2f}")
    # Case 2: fail
    result2 = check_equity_balance(assets=1000, liabilities=450, equity=600)
    print(f"  Assets=1000, Liab=450, Equity=600 -> passed={result2['passed']} diff={result2['diff']:.2f}")
    # Profit equation
    result3 = check_profit_equation(revenue=1200, expenses=700, net_profit=500)
    print(f"  Revenue=1200, Expenses=700, NetProfit=500 -> passed={result3['passed']} diff={result3['diff']:.2f}")
    print()


def main():
    demo_numeric()
    demo_semantic()
    demo_validation()


if __name__ == "__main__":
    main()
