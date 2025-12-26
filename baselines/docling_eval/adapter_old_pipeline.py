"""
Old Pipeline Output Adapter

Converts our existing pipeline's output format to Common Table JSON.

Our pipeline outputs:
- grid: List[List[str]] - 2D text grid
- normalized_grid: List[List[Optional[float]]] - Numeric values
- structure: Dict with rows, columns, cells
- headers: List[str]
- labels: List[str]
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .common_schema import (
    CommonTable, TableCell, TableStructure, CellSpan,
    create_common_table
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Version Info
# =============================================================================

def get_pipeline_version() -> str:
    """Get our pipeline version string."""
    try:
        # Try to get git commit
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return f"git:{result.stdout.strip()}"
    except Exception:
        pass
    return "1.0.0"


# =============================================================================
# Old Pipeline → Common Table Adapter
# =============================================================================

class OldPipelineAdapter:
    """
    Adapter to convert our existing pipeline outputs to Common Table JSON.
    """
    
    SOURCE_SYSTEM = "old_pipeline"
    
    def __init__(self):
        self.pipeline_version = get_pipeline_version()
    
    def from_pipeline_output(
        self,
        pipeline_output: Dict[str, Any],
        table_id: Optional[str] = None
    ) -> CommonTable:
        """
        Convert pipeline output dict to CommonTable.
        
        Args:
            pipeline_output: Dictionary from FinancialTablePipeline.process_image()
            table_id: Optional custom table ID
            
        Returns:
            CommonTable instance
        """
        grid = pipeline_output.get('grid', [])
        headers = pipeline_output.get('headers', [])
        labels = pipeline_output.get('labels', [])
        structure = pipeline_output.get('structure', {})
        run_meta = pipeline_output.get('run_meta', {})
        
        if not grid:
            raise ValueError("Pipeline output has no grid data")
        
        # Determine table ID
        if not table_id:
            table_id = pipeline_output.get('file', 'unknown')
            if table_id.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                table_id = Path(table_id).stem
        
        return self._convert_grid_to_common(
            grid=grid,
            table_id=table_id,
            headers=headers,
            structure=structure,
            run_meta=run_meta
        )
    
    def _convert_grid_to_common(
        self,
        grid: List[List[str]],
        table_id: str,
        headers: List[str] = None,
        structure: Dict = None,
        run_meta: Dict = None
    ) -> CommonTable:
        """Convert a 2D grid to CommonTable."""
        if not grid:
            raise ValueError("Grid cannot be empty")
        
        num_rows = len(grid)
        num_cols = max(len(row) for row in grid) if grid else 0
        
        # Determine header rows
        num_header_rows = 1  # Default
        if headers:
            # If we have explicit headers, try to match them
            for i, row in enumerate(grid[:3]):  # Check first 3 rows
                if row and any(h in str(row) for h in headers):
                    num_header_rows = i + 1
        
        # Extract row/column bboxes from structure if available
        row_bboxes = None
        col_bboxes = None
        if structure:
            rows = structure.get('rows', [])
            cols = structure.get('columns', [])
            if rows:
                row_bboxes = [r.get('bbox', r) if isinstance(r, dict) else r for r in rows]
            if cols:
                col_bboxes = [c.get('bbox', c) if isinstance(c, dict) else c for c in cols]
        
        # Create cells
        cells = []
        for r, row in enumerate(grid):
            for c in range(num_cols):
                text = row[c] if c < len(row) else ''
                cells.append(TableCell(
                    row=r,
                    col=c,
                    text=str(text) if text is not None else '',
                    is_header=(r < num_header_rows)
                ))
        
        # Build structure
        table_structure = TableStructure(
            num_rows=num_rows,
            num_cols=num_cols,
            has_header=num_header_rows > 0,
            num_header_rows=num_header_rows,
            has_spanning_cells=False,  # Our old pipeline doesn't detect spans
            row_bboxes=row_bboxes,
            col_bboxes=col_bboxes
        )
        
        # Build metadata
        metadata = {
            'pipeline_version': self.pipeline_version,
            'source_format': 'pipeline_output'
        }
        if run_meta:
            metadata['run_meta'] = run_meta
        
        return CommonTable(
            table_id=table_id,
            source_system=self.SOURCE_SYSTEM,
            source_version=self.pipeline_version,
            cells=cells,
            structure=table_structure,
            grid=grid,
            metadata=metadata
        )
    
    def from_json_file(self, path: str, table_id: Optional[str] = None) -> CommonTable:
        """
        Load pipeline output from JSON file and convert to CommonTable.
        
        Args:
            path: Path to JSON file
            table_id: Optional custom table ID
            
        Returns:
            CommonTable instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if 'grid' in data:
            # Direct pipeline output
            return self.from_pipeline_output(data, table_id)
        elif 'results' in data:
            # Batch results format
            results = data['results']
            if isinstance(results, list) and results:
                return self.from_pipeline_output(results[0], table_id)
            elif isinstance(results, dict):
                first_key = next(iter(results))
                return self.from_pipeline_output(results[first_key], table_id)
        
        raise ValueError(f"Unrecognized JSON format in {path}")
    
    def from_grid(
        self,
        grid: List[List[str]],
        table_id: str = "grid",
        num_header_rows: int = 1
    ) -> CommonTable:
        """
        Create CommonTable directly from a 2D grid.
        
        Args:
            grid: 2D list of cell texts
            table_id: Table identifier
            num_header_rows: Number of header rows
            
        Returns:
            CommonTable instance
        """
        return create_common_table(
            table_id=table_id,
            source_system=self.SOURCE_SYSTEM,
            grid=grid,
            num_header_rows=num_header_rows,
            source_version=self.pipeline_version,
            metadata={'source_format': 'direct_grid'}
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def pipeline_output_to_common(
    pipeline_output: Dict[str, Any],
    table_id: Optional[str] = None
) -> CommonTable:
    """Convert pipeline output to CommonTable."""
    adapter = OldPipelineAdapter()
    return adapter.from_pipeline_output(pipeline_output, table_id)


def pipeline_json_to_common(path: str, table_id: Optional[str] = None) -> CommonTable:
    """Load pipeline JSON and convert to CommonTable."""
    adapter = OldPipelineAdapter()
    return adapter.from_json_file(path, table_id)


def grid_to_common(
    grid: List[List[str]],
    table_id: str = "table",
    num_header_rows: int = 1
) -> CommonTable:
    """Convert simple grid to CommonTable."""
    adapter = OldPipelineAdapter()
    return adapter.from_grid(grid, table_id, num_header_rows)
