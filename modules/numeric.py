"""
Rule-based numeric normalisation for financial tables.
"""
import re
from typing import Optional, Dict

# Unit scaling keywords
_UNIT_SCALES = {
    'billion': 1_000_000_000,
    'bn': 1_000_000_000,
    'b': 1_000_000_000,
    'million': 1_000_000,
    'm': 1_000_000,
    'k': 1_000,
    'thousand': 1_000,
    "'000": 1_000,
}

# Currency patterns (ASCII only tokens)
_CURRENCY_PATTERNS = {
    'USD': [r'\busd\b', r'\$'],
    'RM': [r'\brm\b', r'myr'],
    'EUR': [r'\beur\b', r'euro'],
    'GBP': [r'\bgbp\b', r'gbp', r'\bpound\b'],
    'SGD': [r'\bsgd\b'],
    'AUD': [r'\baud\b'],
    'CAD': [r'\bcad\b'],
    'HKD': [r'\bhkd\b'],
    'JPY': [r'\bjpy\b', r'yen'],
    'CNY': [r'\bcny\b', r'rmb'],
}


def _detect_currency(text: str) -> Optional[str]:
    s = text.lower()
    for code, patterns in _CURRENCY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, s):
                return code
    return None


def _detect_unit(text: str) -> (int, Optional[str]):
    s = text.lower()
    # Explicit word units
    for key, scale in _UNIT_SCALES.items():
        if re.search(rf"\b{re.escape(key)}\b", s):
            return scale, key
    # Suffix units attached to numbers (e.g., 1.2m, 750k, 3b)
    m = re.search(r"\d[\d,\.]*\s*(bn|b|m|k)", s)
    if m:
        key = m.group(1)
        scale = _UNIT_SCALES.get(key, 1)
        return scale, key
    return 1, None


def _extract_number(text: str) -> Optional[float]:
    # Remove thousand separators and quotes
    cleaned = text.replace(',', '').replace("'", '')
    # Find first number pattern
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def normalize_numeric(value_str: str, default_currency: Optional[str] = None) -> Dict:
    """
    Normalize a numeric string into structured fields.
    Returns dict: raw, value, currency, unit, scale, is_negative, normalized
    """
    raw = value_str
    if value_str is None:
        return {
            'raw': None,
            'value': None,
            'currency': default_currency,
            'unit': None,
            'scale': 1,
            'is_negative': None,
            'normalized': None,
        }

    s = value_str.strip()
    is_negative = False

    # Parentheses indicate negative
    if '(' in s and ')' in s:
        is_negative = True
        s = s.replace('(', '').replace(')', '')

    # Leading sign
    if s.startswith('-'):
        is_negative = True
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]

    # Trailing minus
    if s.endswith('-'):
        is_negative = True
        s = s[:-1]

    currency = _detect_currency(s) or default_currency
    scale, unit = _detect_unit(s)

    number = _extract_number(s)
    if number is None:
        return {
            'raw': raw,
            'value': None,
            'currency': currency,
            'unit': unit,
            'scale': scale,
            'is_negative': is_negative,
            'normalized': None,
        }

    value = number * scale
    if is_negative:
        value = -value

    normalized_str = f"{value}"
    if currency:
        normalized_str = f"{currency} {normalized_str}"

    return {
        'raw': raw,
        'value': value,
        'currency': currency,
        'unit': unit,
        'scale': scale,
        'is_negative': is_negative,
        'normalized': normalized_str,
    }
