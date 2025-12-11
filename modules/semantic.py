"""
Semantic mapping for financial terms using alias dictionary.

Enhanced features:
- OCR error correction for common truncated words
- Financial term dictionary for Malaysian/Singapore banks
- Fuzzy matching for similar terms
"""
import re
from typing import Dict, Optional, List

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

# =============================================================================
# Enhanced OCR Correction Dictionary
# =============================================================================

# Common OCR errors where first letter is missing or corrupted
COMMON_OCR_FIXES = {
    # First letter missing
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
    'ntangible': 'Intangible',
    'angible': 'Tangible',
    'eferred': 'Deferred',
    'tatutory': 'Statutory',
    'eposits': 'Deposits',
    'nvestment': 'Investment',
    'roperty': 'Property',
    'quipment': 'Equipment',
    'oodwill': 'Goodwill',
    'erivative': 'Derivative',
    'oans': 'Loans',
    'dvances': 'Advances',
    'inancing': 'Financing',
    # Common typos
    'rassets': 'assets',
    'rasset': 'asset',
    'ther': 'Other',
    'otal assets': 'Total assets',
    'otal liabilities': 'Total liabilities',
    'otal equity': 'Total equity',
}

# Financial terms dictionary for validation/normalization
FINANCIAL_TERMS = {
    # Balance Sheet - Assets
    'cash and short-term funds',
    'cash and cash equivalents',
    'reverse repurchase agreements',
    'deposits and placements',
    'financial investments',
    'debt instruments',
    'equity instruments',
    'derivative financial instruments',
    'loans and advances',
    'loans, advances and financing',
    'other assets',
    'tax recoverable',
    'deferred tax assets',
    'statutory deposits',
    'investment in associates',
    'investment in joint ventures',
    'property, plant and equipment',
    'right-of-use assets',
    'investment properties',
    'goodwill',
    'intangible assets',
    'non-current assets held for sale',
    'total assets',
    
    # Balance Sheet - Liabilities
    'deposits from customers',
    'deposits and placements of banks',
    'obligations on securities',
    'derivative financial liabilities',
    'bills and acceptances payable',
    'other liabilities',
    'provisions',
    'current tax liabilities',
    'deferred tax liabilities',
    'borrowings',
    'subordinated obligations',
    'total liabilities',
    
    # Balance Sheet - Equity
    'share capital',
    'reserves',
    'retained earnings',
    'equity attributable to owners',
    'non-controlling interests',
    'total equity',
    
    # Income Statement
    'interest income',
    'interest expense',
    'net interest income',
    'fee and commission income',
    'fee and commission expense',
    'net fee and commission income',
    'other operating income',
    'total operating income',
    'personnel expenses',
    'other operating expenses',
    'total operating expenses',
    'operating profit',
    'allowance for impairment',
    'profit before tax',
    'tax expense',
    'profit for the year',
    'net profit',
    'earnings per share',
}


def correct_ocr_text(text: str) -> str:
    """
    Correct common OCR errors where the first letter is missing or corrupted.
    
    Handles:
    - First letter truncation (otal -> Total)
    - Common typos (rassets -> assets)
    - Word boundary issues
    
    Only corrects words that EXACTLY match known OCR errors.
    """
    if not text:
        return text
    
    # First, try direct replacement for the whole text (lowercase match)
    text_lower = text.lower().strip()
    for pattern, fix in COMMON_OCR_FIXES.items():
        if text_lower == pattern.lower():
            # Preserve case style
            if text[0].isupper():
                return fix.capitalize() if fix[0].islower() else fix
            return fix.lower()
    
    # Process word by word - only fix words that EXACTLY match
    words = text.split()
    corrected = []
    
    for w in words:
        # Strip punctuation for checking
        clean = re.sub(r'[^a-zA-Z]', '', w).lower()
        if not clean:
            corrected.append(w)
            continue
        
        # Only fix if exact match in dictionary
        found = False
        for pattern, fix in COMMON_OCR_FIXES.items():
            if clean == pattern.lower():
                # Preserve original capitalization style
                if w[0].isupper():
                    corrected.append(fix.capitalize())
                else:
                    corrected.append(fix.lower())
                found = True
                break
        
        if not found:
            corrected.append(w)
    
    return ' '.join(corrected)


def correct_financial_label(label: str) -> str:
    """
    Correct a financial row label using domain knowledge.
    
    Combines OCR correction with fuzzy matching to known financial terms.
    """
    if not label:
        return label
    
    # First apply basic OCR correction
    corrected = correct_ocr_text(label)
    
    # Check if the corrected label is close to a known financial term
    corrected_lower = corrected.lower().strip()
    
    # Direct match
    if corrected_lower in FINANCIAL_TERMS:
        return corrected
    
    # Try to find similar term
    best_match = None
    best_score = 0
    
    for term in FINANCIAL_TERMS:
        # Simple similarity: count matching words
        corrected_words = set(corrected_lower.split())
        term_words = set(term.split())
        
        if not corrected_words or not term_words:
            continue
        
        # Jaccard similarity
        intersection = len(corrected_words & term_words)
        union = len(corrected_words | term_words)
        score = intersection / union if union > 0 else 0
        
        if score > best_score and score > 0.5:  # Threshold for match
            best_score = score
            best_match = term
    
    # If we found a good match, use proper capitalization
    if best_match and best_score > 0.7:
        return best_match.title()
    
    return corrected


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
