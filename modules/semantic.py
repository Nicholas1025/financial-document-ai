"""
Semantic mapping for financial terms using alias dictionary.
"""
import re
from typing import Dict, Optional

DEFAULT_ALIAS_MAP: Dict[str, str] = {
    'turnover': 'revenue',
    'sales': 'revenue',
    'revenue': 'revenue',
    'total revenue': 'revenue',
    'pbt': 'profit_before_tax',
    'profit before tax': 'profit_before_tax',
    'ebt': 'profit_before_tax',
    'ebit': 'ebit',
    'ebitda': 'ebitda',
    'pat': 'net_income',
    'net profit': 'net_income',
    'net income': 'net_income',
    'profit after tax': 'net_income',
    'operating profit': 'operating_profit',
    'operating income': 'operating_profit',
    'total assets': 'assets_total',
    'assets': 'assets_total',
    'total liabilities': 'liabilities_total',
    'liabilities': 'liabilities_total',
    'equity': 'equity_total',
    'shareholders equity': 'equity_total',
}

COMMON_OCR_FIXES = {
    'otal': 'Total',
    'perating': 'Operating',
    'ncome': 'Income',
    'evenue': 'Revenue',
    'xpenses': 'Expenses',
    'rofit': 'Profit',
    'ssets': 'Assets',
    'iabilities': 'Liabilities',
    'quity': 'Equity',
    'ash': 'Cash',
    'ctivities': 'Activities',
    'nvesting': 'Investing',
    'inancing': 'Financing',
    'lows': 'Flows',
    'et': 'Net',
    'ross': 'Gross',
    'ost': 'Cost',
    'oods': 'Goods',
    'old': 'Sold',
    'hird': 'Third',
    'irst': 'First',
    'econd': 'Second',
    'ourth': 'Fourth',
}


def correct_ocr_text(text: str) -> str:
    """
    Correct common OCR errors where the first letter is missing.
    """
    if not text:
        return text
        
    words = text.split()
    corrected = []
    for w in words:
        # Strip punctuation for checking
        clean = re.sub(r'[^a-zA-Z]', '', w)
        if not clean:
            corrected.append(w)
            continue
            
        # Check if we have a fix
        replacement = None
        for k, v in COMMON_OCR_FIXES.items():
            if k.lower() == clean.lower():
                replacement = v
                break
        
        if replacement:
            # Preserve punctuation if possible (simple replacement)
            corrected.append(replacement)
        else:
            corrected.append(w)
            
    return ' '.join(corrected)


def _normalize_key(text: str) -> str:
    t = text.lower()
    t = re.sub(r'[^a-z0-9]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def map_alias(label: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    """
    Map a raw label to a canonical financial field.
    If not found, returns the normalized key itself.
    """
    if label is None:
        return ''
    key = _normalize_key(label)
    mapping = alias_map or DEFAULT_ALIAS_MAP
    return mapping.get(key, key)
