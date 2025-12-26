"""
Common Table JSON Schema

Defines the intermediate format for comparing table structures across different systems.
This schema is designed to be:
1. System-agnostic (works with docling, our pipeline, or any other system)
2. Rich enough to capture structure (rows, cols, spans) and content
3. Simple enough for easy conversion and comparison

Schema Version: 1.0.0
"""
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


# =============================================================================
# Common Table JSON Schema
# =============================================================================

@dataclass
class CellSpan:
    """Cell spanning information."""
    row_span: int = 1
    col_span: int = 1


@dataclass
class TableCell:
    """
    A single cell in the table.
    
    Attributes:
        row: Row index (0-based)
        col: Column index (0-based)
        text: Cell text content
        is_header: Whether this cell is a header cell
        span: Row/column spanning info
        bbox: Optional bounding box [x1, y1, x2, y2]
        confidence: OCR/detection confidence (0-1)
    """
    row: int
    col: int
    text: str = ""
    is_header: bool = False
    span: CellSpan = field(default_factory=CellSpan)
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if d['bbox'] is None:
            del d['bbox']
        if d['confidence'] is None:
            del d['confidence']
        return d


@dataclass
class TableStructure:
    """
    Structure information for a table.
    
    Attributes:
        num_rows: Number of rows
        num_cols: Number of columns
        has_header: Whether table has header row(s)
        num_header_rows: Number of header rows
        has_spanning_cells: Whether table has merged cells
        row_bboxes: Optional list of row bounding boxes
        col_bboxes: Optional list of column bounding boxes
    """
    num_rows: int
    num_cols: int
    has_header: bool = True
    num_header_rows: int = 1
    has_spanning_cells: bool = False
    row_bboxes: Optional[List[List[float]]] = None
    col_bboxes: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if d['row_bboxes'] is None:
            del d['row_bboxes']
        if d['col_bboxes'] is None:
            del d['col_bboxes']
        return d


@dataclass
class CommonTable:
    """
    Common Table JSON format.
    
    This is the primary interchange format for comparing table structures.
    
    Attributes:
        schema_version: Schema version string
        source_system: System that generated this (e.g., 'docling', 'old_pipeline')
        source_version: Version of the source system
        table_id: Unique identifier for this table
        cells: List of all cells
        structure: Table structure metadata
        html: Optional HTML representation
        grid: Optional 2D text grid (for quick access)
        metadata: Additional metadata
    """
    table_id: str
    source_system: str
    cells: List[TableCell]
    structure: TableStructure
    schema_version: str = "1.0.0"
    source_version: str = ""
    html: Optional[str] = None
    grid: Optional[List[List[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'schema_version': self.schema_version,
            'source_system': self.source_system,
            'source_version': self.source_version,
            'table_id': self.table_id,
            'created_at': self.created_at,
            'structure': self.structure.to_dict(),
            'cells': [c.to_dict() for c in self.cells],
            'html': self.html,
            'grid': self.grid,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'CommonTable':
        """Load from dictionary."""
        cells = [
            TableCell(
                row=c['row'],
                col=c['col'],
                text=c.get('text', ''),
                is_header=c.get('is_header', False),
                span=CellSpan(
                    row_span=c.get('span', {}).get('row_span', 1),
                    col_span=c.get('span', {}).get('col_span', 1)
                ),
                bbox=c.get('bbox'),
                confidence=c.get('confidence')
            )
            for c in d.get('cells', [])
        ]
        
        struct_d = d.get('structure', {})
        structure = TableStructure(
            num_rows=struct_d.get('num_rows', 0),
            num_cols=struct_d.get('num_cols', 0),
            has_header=struct_d.get('has_header', True),
            num_header_rows=struct_d.get('num_header_rows', 1),
            has_spanning_cells=struct_d.get('has_spanning_cells', False),
            row_bboxes=struct_d.get('row_bboxes'),
            col_bboxes=struct_d.get('col_bboxes')
        )
        
        return cls(
            schema_version=d.get('schema_version', '1.0.0'),
            source_system=d.get('source_system', 'unknown'),
            source_version=d.get('source_version', ''),
            table_id=d.get('table_id', ''),
            cells=cells,
            structure=structure,
            html=d.get('html'),
            grid=d.get('grid'),
            metadata=d.get('metadata', {}),
            created_at=d.get('created_at', datetime.now().isoformat())
        )
    
    @classmethod
    def load(cls, path: str) -> 'CommonTable':
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    def to_html(self) -> str:
        """Generate HTML table representation."""
        if self.html:
            return self.html
        
        # Build HTML from cells
        html_parts = ['<table>']
        
        # Group cells by row
        rows_dict: Dict[int, List[TableCell]] = {}
        for cell in self.cells:
            if cell.row not in rows_dict:
                rows_dict[cell.row] = []
            rows_dict[cell.row].append(cell)
        
        # Sort rows and cells within rows
        for row_idx in sorted(rows_dict.keys()):
            row_cells = sorted(rows_dict[row_idx], key=lambda c: c.col)
            
            # Check if header row
            is_header_row = row_idx < self.structure.num_header_rows
            
            html_parts.append('<tr>')
            for cell in row_cells:
                tag = 'th' if (is_header_row or cell.is_header) else 'td'
                
                attrs = []
                if cell.span.row_span > 1:
                    attrs.append(f'rowspan="{cell.span.row_span}"')
                if cell.span.col_span > 1:
                    attrs.append(f'colspan="{cell.span.col_span}"')
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                html_parts.append(f'<{tag}{attr_str}>{cell.text}</{tag}>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        return ''.join(html_parts)
    
    def to_grid(self) -> List[List[str]]:
        """Generate 2D text grid representation."""
        if self.grid:
            return self.grid
        
        # Initialize empty grid
        grid = [
            ['' for _ in range(self.structure.num_cols)]
            for _ in range(self.structure.num_rows)
        ]
        
        # Fill in cells (handling spans)
        for cell in self.cells:
            for r in range(cell.row, min(cell.row + cell.span.row_span, self.structure.num_rows)):
                for c in range(cell.col, min(cell.col + cell.span.col_span, self.structure.num_cols)):
                    if r == cell.row and c == cell.col:
                        grid[r][c] = cell.text
                    # Spanned cells get empty string (already initialized)
        
        return grid


# =============================================================================
# Convenience Functions
# =============================================================================

def create_common_table(
    table_id: str,
    source_system: str,
    grid: List[List[str]],
    num_header_rows: int = 1,
    source_version: str = "",
    metadata: Optional[Dict] = None
) -> CommonTable:
    """
    Create a CommonTable from a simple 2D text grid.
    
    Args:
        table_id: Unique identifier
        source_system: System name
        grid: 2D list of cell texts
        num_header_rows: Number of header rows
        source_version: Version string
        metadata: Additional metadata
        
    Returns:
        CommonTable instance
    """
    if not grid:
        raise ValueError("Grid cannot be empty")
    
    num_rows = len(grid)
    num_cols = max(len(row) for row in grid) if grid else 0
    
    cells = []
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            cells.append(TableCell(
                row=r,
                col=c,
                text=str(text) if text is not None else '',
                is_header=(r < num_header_rows)
            ))
    
    structure = TableStructure(
        num_rows=num_rows,
        num_cols=num_cols,
        has_header=num_header_rows > 0,
        num_header_rows=num_header_rows
    )
    
    return CommonTable(
        table_id=table_id,
        source_system=source_system,
        source_version=source_version,
        cells=cells,
        structure=structure,
        grid=grid,
        metadata=metadata or {}
    )
