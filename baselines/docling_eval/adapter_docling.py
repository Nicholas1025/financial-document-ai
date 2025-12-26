"""
Docling Output Adapter

Converts Docling's output format to Common Table JSON.

Docling outputs tables as DoclingDocument with TableItem objects.
This adapter extracts table data and converts to our common format.
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .common_schema import (
    CommonTable, TableCell, TableStructure, CellSpan,
    create_common_table
)

logger = logging.getLogger(__name__)


# =============================================================================
# Docling Version Info
# =============================================================================

def get_docling_version() -> str:
    """Get docling version string."""
    try:
        import docling
        return getattr(docling, '__version__', 'unknown')
    except ImportError:
        return 'not_installed'


def get_docling_eval_version() -> str:
    """Get docling-eval version string."""
    try:
        import docling_eval
        return getattr(docling_eval, '__version__', 'unknown')
    except ImportError:
        return 'not_installed'


# =============================================================================
# Docling → Common Table Adapter
# =============================================================================

class DoclingAdapter:
    """
    Adapter to convert Docling outputs to Common Table JSON.
    """
    
    SOURCE_SYSTEM = "docling"
    
    def __init__(self):
        self.docling_version = get_docling_version()
        self.docling_eval_version = get_docling_eval_version()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if self.docling_version == 'not_installed':
            logger.warning("docling is not installed. Some features may not work.")
        if self.docling_eval_version == 'not_installed':
            logger.warning("docling-eval is not installed. Some features may not work.")
    
    def from_docling_document(
        self,
        doc: Any,  # DoclingDocument
        table_index: int = 0,
        table_id: Optional[str] = None
    ) -> CommonTable:
        """
        Convert a DoclingDocument's table to CommonTable.
        
        Args:
            doc: DoclingDocument object
            table_index: Index of the table to extract (if multiple)
            table_id: Optional custom table ID
            
        Returns:
            CommonTable instance
        """
        try:
            from docling_core.types.doc.document import DoclingDocument, TableItem
        except ImportError:
            raise ImportError("docling-core is required for this adapter")
        
        # Get tables from document
        tables = [item for item in doc.main_text if hasattr(item, 'data') and 
                  hasattr(item.data, 'table_cells')]
        
        if not tables:
            raise ValueError("No tables found in DoclingDocument")
        
        if table_index >= len(tables):
            raise ValueError(f"Table index {table_index} out of range (found {len(tables)} tables)")
        
        table_item = tables[table_index]
        return self._convert_table_item(table_item, table_id or f"docling_table_{table_index}")
    
    def _convert_table_item(self, table_item: Any, table_id: str) -> CommonTable:
        """Convert a TableItem to CommonTable."""
        # Extract table data
        table_data = table_item.data if hasattr(table_item, 'data') else table_item
        
        cells = []
        max_row = 0
        max_col = 0
        has_spanning = False
        
        # Process each cell
        for cell_data in getattr(table_data, 'table_cells', []):
            row = getattr(cell_data, 'row', 0)
            col = getattr(cell_data, 'col', 0)
            text = getattr(cell_data, 'text', '')
            
            row_span = getattr(cell_data, 'row_span', 1)
            col_span = getattr(cell_data, 'col_span', 1)
            
            if row_span > 1 or col_span > 1:
                has_spanning = True
            
            is_header = getattr(cell_data, 'is_header', False)
            
            # Get bbox if available
            bbox = None
            if hasattr(cell_data, 'bbox'):
                bbox_obj = cell_data.bbox
                if bbox_obj:
                    bbox = [bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b]
            
            cells.append(TableCell(
                row=row,
                col=col,
                text=str(text) if text else '',
                is_header=is_header,
                span=CellSpan(row_span=row_span, col_span=col_span),
                bbox=bbox
            ))
            
            max_row = max(max_row, row + row_span)
            max_col = max(max_col, col + col_span)
        
        # Determine header rows
        header_rows = set()
        for cell in cells:
            if cell.is_header:
                header_rows.add(cell.row)
        num_header_rows = max(header_rows) + 1 if header_rows else 1
        
        structure = TableStructure(
            num_rows=max_row,
            num_cols=max_col,
            has_header=bool(header_rows),
            num_header_rows=num_header_rows,
            has_spanning_cells=has_spanning
        )
        
        return CommonTable(
            table_id=table_id,
            source_system=self.SOURCE_SYSTEM,
            source_version=f"docling={self.docling_version}",
            cells=cells,
            structure=structure,
            metadata={
                'docling_version': self.docling_version,
                'docling_eval_version': self.docling_eval_version
            }
        )
    
    def from_html(self, html: str, table_id: str = "docling_html") -> CommonTable:
        """
        Convert HTML table (Docling's output format) to CommonTable.
        
        This is useful when working with docling-eval benchmark outputs
        which often provide HTML tables.
        
        Args:
            html: HTML table string
            table_id: Table identifier
            
        Returns:
            CommonTable instance
        """
        from lxml import html as lxml_html
        
        tree = lxml_html.fromstring(html)
        
        cells = []
        max_row = 0
        max_col = 0
        has_spanning = False
        header_rows = set()
        
        # Track cell positions (for handling spans)
        occupied = set()
        
        rows = tree.xpath('//tr')
        for row_idx, row in enumerate(rows):
            col_idx = 0
            
            # Skip occupied cells from previous row spans
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            
            for cell in row.xpath('td|th'):
                # Skip occupied columns
                while (row_idx, col_idx) in occupied:
                    col_idx += 1
                
                text = cell.text_content().strip()
                is_header = cell.tag == 'th'
                
                row_span = int(cell.get('rowspan', 1))
                col_span = int(cell.get('colspan', 1))
                
                if row_span > 1 or col_span > 1:
                    has_spanning = True
                
                if is_header:
                    header_rows.add(row_idx)
                
                cells.append(TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=text,
                    is_header=is_header,
                    span=CellSpan(row_span=row_span, col_span=col_span)
                ))
                
                # Mark cells as occupied
                for r in range(row_idx, row_idx + row_span):
                    for c in range(col_idx, col_idx + col_span):
                        occupied.add((r, c))
                
                max_row = max(max_row, row_idx + row_span)
                max_col = max(max_col, col_idx + col_span)
                col_idx += col_span
        
        num_header_rows = max(header_rows) + 1 if header_rows else 0
        
        structure = TableStructure(
            num_rows=max_row,
            num_cols=max_col,
            has_header=bool(header_rows),
            num_header_rows=num_header_rows,
            has_spanning_cells=has_spanning
        )
        
        return CommonTable(
            table_id=table_id,
            source_system=self.SOURCE_SYSTEM,
            source_version=f"docling={self.docling_version}",
            cells=cells,
            structure=structure,
            html=html,
            metadata={
                'docling_version': self.docling_version,
                'source_format': 'html'
            }
        )
    
    def from_otsl(
        self,
        otsl_tokens: List[str],
        cell_texts: List[str],
        table_id: str = "docling_otsl"
    ) -> CommonTable:
        """
        Convert OTSL (Optical Table Structure Language) format to CommonTable.
        
        OTSL is the format used in docling's benchmark datasets (FinTabNet_OTSL, etc.)
        
        Args:
            otsl_tokens: List of OTSL tokens describing structure
            cell_texts: List of cell text contents
            table_id: Table identifier
            
        Returns:
            CommonTable instance
        """
        # OTSL tokens: 'fcel', 'ecel', 'lcel', 'ucel', 'xcel', 'nl', etc.
        # fcel = first cell, ecel = empty cell, lcel = left span, ucel = up span, xcel = cross span
        # nl = new line (row break)
        
        cells = []
        row_idx = 0
        col_idx = 0
        text_idx = 0
        max_col = 0
        has_spanning = False
        
        # Track spans
        span_info = {}  # (row, col) -> (row_span, col_span, text)
        
        for token in otsl_tokens:
            if token == 'nl':
                max_col = max(max_col, col_idx)
                row_idx += 1
                col_idx = 0
            elif token == 'fcel':
                # First cell - new content
                text = cell_texts[text_idx] if text_idx < len(cell_texts) else ''
                text_idx += 1
                span_info[(row_idx, col_idx)] = {'text': text, 'row_span': 1, 'col_span': 1}
                col_idx += 1
            elif token == 'ecel':
                # Empty cell
                span_info[(row_idx, col_idx)] = {'text': '', 'row_span': 1, 'col_span': 1}
                col_idx += 1
            elif token == 'lcel':
                # Left span - extends previous cell horizontally
                has_spanning = True
                prev_col = col_idx - 1
                if (row_idx, prev_col) in span_info:
                    span_info[(row_idx, prev_col)]['col_span'] += 1
                col_idx += 1
            elif token == 'ucel':
                # Up span - extends cell from above vertically
                has_spanning = True
                prev_row = row_idx - 1
                if (prev_row, col_idx) in span_info:
                    span_info[(prev_row, col_idx)]['row_span'] += 1
                col_idx += 1
            elif token == 'xcel':
                # Cross span - both horizontal and vertical
                has_spanning = True
                col_idx += 1
        
        max_col = max(max_col, col_idx)
        num_rows = row_idx + 1 if col_idx > 0 else row_idx
        
        # Convert span_info to cells
        for (r, c), info in span_info.items():
            cells.append(TableCell(
                row=r,
                col=c,
                text=info['text'],
                is_header=(r == 0),  # Assume first row is header
                span=CellSpan(row_span=info['row_span'], col_span=info['col_span'])
            ))
        
        structure = TableStructure(
            num_rows=num_rows,
            num_cols=max_col,
            has_header=True,
            num_header_rows=1,
            has_spanning_cells=has_spanning
        )
        
        return CommonTable(
            table_id=table_id,
            source_system=self.SOURCE_SYSTEM,
            source_version=f"docling={self.docling_version}",
            cells=cells,
            structure=structure,
            metadata={
                'docling_version': self.docling_version,
                'source_format': 'otsl',
                'num_otsl_tokens': len(otsl_tokens),
                'num_cell_texts': len(cell_texts)
            }
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def docling_html_to_common(html: str, table_id: str = "table") -> CommonTable:
    """Convert docling HTML output to CommonTable."""
    adapter = DoclingAdapter()
    return adapter.from_html(html, table_id)


def docling_otsl_to_common(
    otsl_tokens: List[str],
    cell_texts: List[str],
    table_id: str = "table"
) -> CommonTable:
    """Convert docling OTSL output to CommonTable."""
    adapter = DoclingAdapter()
    return adapter.from_otsl(otsl_tokens, cell_texts, table_id)
