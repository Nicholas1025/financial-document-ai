"""
Rule-based numeric normalisation for financial tables.

Key features:
1. Currency detection (USD, RM, EUR, etc.)
2. Unit scaling ('000, million, billion)
3. Column metadata propagation (currency/unit from header)
4. Cell type detection (header, year, note, numeric)
"""
import re
from typing import Optional, Dict, List, Any, Tuple

# Unit scaling keywords
_UNIT_SCALES = {
    'billion': 1_000_000_000,
    'bn': 1_000_000_000,
    'b': 1_000_000_000,
    'million': 1_000_000,
    'mn': 1_000_000,
    'm': 1_000_000,
    'k': 1_000,
    'thousand': 1_000,
    "'000": 1_000,
    "000": 1_000,  
}

# Currency patterns (ASCII only tokens)
_CURRENCY_PATTERNS = {
    # Put more specific patterns before generic '$' to avoid misclassifying 'S$' as USD.
    'SGD': [r'\bsgd\b', r's\$'],
    # '$' is ambiguous; treat as USD only when it is not preceded by a letter (e.g., avoid 'S$').
    'USD': [r'\busd\b', r'us\s*dollar', r'(?<![A-Za-z])\$'],
    'RM': [r'\brm\b', r'\bmyr\b', r'ringgit'],
    # Euro symbol support
    'EUR': [r'\beur\b', r'euro', r'€'],
    'GBP': [r'\bgbp\b', r'£', r'\bpound\b', r'sterling'],
    'AUD': [r'\baud\b', r'a\$'],
    'CAD': [r'\bcad\b', r'c\$'],
    'HKD': [r'\bhkd\b', r'hk\$'],
    'JPY': [r'\bjpy\b', r'¥', r'yen'],
    'CNY': [r'\bcny\b', r'rmb', r'yuan'],
    'IDR': [r'\bidr\b', r'rupiah'],
    'THB': [r'\bthb\b', r'baht'],
    'PHP': [r'\bphp\b', r'peso'],
    'VND': [r'\bvnd\b', r'dong'],
}

# Patterns for header/non-numeric cells
_YEAR_PATTERN = re.compile(r'^(19|20)\d{2}$')  # 1900-2099
_NOTE_PATTERN = re.compile(r'^(note|notes?)$', re.IGNORECASE)
_UNIT_HEADER_PATTERN = re.compile(r"(rm|usd|sgd|eur|gbp|'000|million|billion)", re.IGNORECASE)


def _detect_currency(text: str) -> Optional[str]:
    """Detect currency code from text."""
    s = text.lower()
    for code, patterns in _CURRENCY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, s):
                return code
    return None


def _detect_unit(text: str) -> Tuple[int, Optional[str]]:
    """Detect unit scale from text. Returns (scale, unit_name)."""
    s = text.lower()
    
    # Check for '000 format first (common in financial statements)
    if "'000" in s or "'000" in text:
        return 1000, "'000"
    
    # Explicit word units
    for key, scale in _UNIT_SCALES.items():
        if re.search(rf"\b{re.escape(key)}\b", s):
            return scale, key
    
    # Suffix units attached to numbers (e.g., 1.2m, 750k, 3b)
    m = re.search(r"\d[\d,\.]*\s*(bn|b|mn|m|k)", s)
    if m:
        key = m.group(1)
        scale = _UNIT_SCALES.get(key, 1)
        return scale, key
    return 1, None


def _extract_number(text: str) -> Optional[float]:
    """Extract numeric value from text."""
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


def detect_cell_type(text: str) -> str:
    """
    Detect the type of a cell value.
    
    Returns:
        'year': Year value (2020, 2021, etc.)
        'note': Note reference column
        'unit_header': Unit header (RM'000, USD, etc.)
        'header': Other header text
        'numeric': Numeric value
        'text': Plain text
        'empty': Empty cell
    """
    if not text or not text.strip():
        return 'empty'
    
    s = text.strip()
    
    # Check for year (4-digit year between 1900-2099)
    if _YEAR_PATTERN.match(s):
        return 'year'
    
    # Check for note column header
    if _NOTE_PATTERN.match(s):
        return 'note'
    
    # Check for unit header (RM'000, USD, etc.)
    if _UNIT_HEADER_PATTERN.search(s) and not any(c.isdigit() for c in s.replace("'000", "").replace("000", "")):
        return 'unit_header'
    
    # Check if it's purely numeric or has numeric content
    has_digit = any(c.isdigit() for c in s)
    # Remove common numeric characters to check what's left
    non_numeric = re.sub(r'[\d,.\-+()%$£€¥\s]', '', s)
    
    if has_digit:
        # If mostly numeric characters, it's a numeric value
        if len(non_numeric) <= 2:  # Allow 1-2 non-numeric chars (like currency symbol)
            return 'numeric'
        # Small integers in first/second column are likely note references
        cleaned = s.replace(',', '').replace(' ', '')
        if cleaned.isdigit() and int(cleaned) < 100:
            return 'note'
    
    # Check if it looks like a header (short, possibly with unit info)
    if len(s) < 20 and not has_digit:
        return 'header'
    
    return 'text'


def parse_unit_header(text: str) -> Dict[str, Any]:
    """
    Parse a unit header cell (e.g., "RM'000", "USD Million").
    
    Returns:
        Dict with currency and scale information
    """
    result = {
        'currency': None,
        'scale': 1,
        'unit': None,
        'raw': text
    }
    
    if not text:
        return result
    
    # Detect currency
    result['currency'] = _detect_currency(text)
    
    # Detect scale
    result['scale'], result['unit'] = _detect_unit(text)
    
    return result


def normalize_numeric(value_str: str, 
                      default_currency: Optional[str] = None,
                      default_scale: int = 1,
                      cell_type: Optional[str] = None) -> Dict:
    """
    Normalize a numeric string into structured fields.
    
    Args:
        value_str: Raw cell value
        default_currency: Currency to use if not detected in cell
        default_scale: Scale to apply (from column metadata, e.g., 1000 for '000)
        cell_type: Pre-detected cell type (skip non-numeric types)
    
    Returns:
        dict: raw, value, currency, unit, scale, is_negative, normalized, cell_type
    """
    raw = value_str
    
    # Detect cell type if not provided
    if cell_type is None:
        cell_type = detect_cell_type(value_str) if value_str else 'empty'
    
    # Return early for non-numeric types
    if cell_type in ('empty', 'year', 'note', 'unit_header', 'header', 'text'):
        return {
            'raw': raw,
            'value': None,
            'currency': default_currency,
            'unit': None,
            'scale': default_scale,
            'is_negative': False,
            'normalized': None,
            'cell_type': cell_type,
        }
    
    if value_str is None:
        return {
            'raw': None,
            'value': None,
            'currency': default_currency,
            'unit': None,
            'scale': default_scale,
            'is_negative': None,
            'normalized': None,
            'cell_type': 'empty',
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

    # Detect currency from cell, fall back to default
    cell_currency = _detect_currency(s)
    currency = cell_currency or default_currency
    
    # Detect unit from cell
    cell_scale, unit = _detect_unit(s)
    
    # Use cell scale if detected, otherwise use column default
    scale = cell_scale if cell_scale > 1 else default_scale

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
            'cell_type': cell_type,
        }

    # Apply scale to get actual value
    value = number * scale
    if is_negative:
        value = -value

    normalized_str = f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
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
        'cell_type': cell_type,
    }


class ColumnMetadata:
    """Stores metadata for a table column (currency, unit, type)."""
    
    def __init__(self):
        self.currency: Optional[str] = None
        self.scale: int = 1
        self.unit: Optional[str] = None
        self.column_type: str = 'unknown'  # 'label', 'note', 'year', 'numeric'
        self.year: Optional[int] = None
    
    def __repr__(self):
        return f"ColumnMetadata(currency={self.currency}, scale={self.scale}, type={self.column_type}, year={self.year})"


def extract_column_metadata(grid: List[List[str]], header_rows: int = 3) -> List[ColumnMetadata]:
    """
    Extract column metadata from header rows.
    
    Analyzes the first few rows to determine:
    - Currency (RM, USD, etc.)
    - Scale ('000, million, etc.)
    - Column type (label, note, year, numeric)
    
    Args:
        grid: 2D grid of cell values
        header_rows: Number of rows to analyze for headers
        
    Returns:
        List of ColumnMetadata objects, one per column
    """
    if not grid or not grid[0]:
        return []
    
    num_cols = len(grid[0])
    metadata = [ColumnMetadata() for _ in range(num_cols)]
    
    # Analyze header rows
    for row_idx in range(min(header_rows, len(grid))):
        row = grid[row_idx]
        for col_idx, cell in enumerate(row):
            if col_idx >= num_cols:
                break
            
            cell_type = detect_cell_type(cell)
            meta = metadata[col_idx]
            
            if cell_type == 'year':
                meta.column_type = 'year'
                try:
                    meta.year = int(cell.strip())
                except:
                    pass
            elif cell_type == 'unit_header':
                parsed = parse_unit_header(cell)
                if parsed['currency']:
                    meta.currency = parsed['currency']
                if parsed['scale'] > 1:
                    meta.scale = parsed['scale']
                    meta.unit = parsed['unit']
            elif cell_type == 'note':
                meta.column_type = 'note'
    
    # First column is typically label
    if metadata:
        metadata[0].column_type = 'label'
    
    # Propagate currency/scale to columns without explicit headers
    # (assume same as neighboring numeric columns)
    default_currency = None
    default_scale = 1
    for meta in metadata:
        if meta.currency:
            default_currency = meta.currency
        if meta.scale > 1:
            default_scale = meta.scale
    
    for meta in metadata:
        if meta.column_type not in ('label', 'note') and meta.currency is None:
            meta.currency = default_currency
        if meta.column_type not in ('label', 'note') and meta.scale == 1:
            meta.scale = default_scale
    
    return metadata


def normalize_grid_with_metadata(grid: List[List[str]], 
                                  metadata: Optional[List[ColumnMetadata]] = None,
                                  header_rows: int = 3) -> Tuple[List[List[Dict]], List[ColumnMetadata]]:
    """
    Normalize an entire grid using column metadata.
    
    Args:
        grid: 2D grid of cell values
        metadata: Pre-computed column metadata (will be computed if None)
        header_rows: Number of header rows
        
    Returns:
        Tuple of (normalized_grid, column_metadata)
    """
    if not grid:
        return [], []
    
    # Extract or use provided metadata
    if metadata is None:
        metadata = extract_column_metadata(grid, header_rows)
    
    norm_grid = []
    
    for row_idx, row in enumerate(grid):
        norm_row = []
        for col_idx, cell in enumerate(row):
            # Get column metadata
            if col_idx < len(metadata):
                meta = metadata[col_idx]
                default_currency = meta.currency
                default_scale = meta.scale
            else:
                default_currency = None
                default_scale = 1
            
            # Detect cell type
            cell_type = detect_cell_type(cell)
            
            # Skip header rows for numeric processing
            if row_idx < header_rows:
                cell_type = detect_cell_type(cell)
                if cell_type == 'numeric':
                    # In header rows, numbers are likely years or references
                    cell_type = 'header'
            
            # Normalize the cell
            norm_cell = normalize_numeric(
                cell,
                default_currency=default_currency,
                default_scale=default_scale,
                cell_type=cell_type
            )
            norm_row.append(norm_cell)
        
        norm_grid.append(norm_row)
    
    return norm_grid, metadata
