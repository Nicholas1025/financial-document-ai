"""pdf_pipeline.py

Minimal PDF → Table Pipeline runner.

Goal: provide a thesis-friendly, reproducible path that starts from an annual-report PDF,
renders one page, optionally crops a table region, then runs the existing
FinancialTablePipeline and exports a JSON result.

Design constraints (intentional):
- No UI / no interactive selection.
- Can run on a single hard-selected page.
- Defaults are stable on Windows: PyTorch on GPU, PaddleOCR on CPU via subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator


def _parse_crop(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse crop string as "x1,y1,x2,y2" (pixel coordinates)."""
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--crop must be 'x1,y1,x2,y2' in pixels")
    x1, y1, x2, y2 = (int(float(p)) for p in parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--crop must have x2>x1 and y2>y1")
    return x1, y1, x2, y2


def _render_pdf_page(pdf_path: str, page_1based: int, zoom: float) -> Image.Image:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF is required for pdf_pipeline.py. Install with: pip install PyMuPDF"
        ) from exc

    doc = fitz.open(pdf_path)
    try:
        if page_1based < 1 or page_1based > len(doc):
            raise ValueError(f"page out of range: {page_1based} (PDF has {len(doc)} pages)")
        page = doc.load_page(page_1based - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return img
    finally:
        doc.close()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FinancialTablePipeline on one PDF page")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--page", type=int, default=1, help="Page number (1-based)")
    parser.add_argument("--zoom", type=float, default=2.0, help="Render zoom (2.0 is a good default)")
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Optional crop box in pixels: x1,y1,x2,y2 (applied after rendering)",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("outputs", "results"),
        help="Output directory",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save rendered page and cropped table image alongside JSON",
    )
    args = parser.parse_args()

    pdf_path = str(Path(args.pdf).expanduser())
    crop = _parse_crop(args.crop)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    _ensure_dir(out_dir)
    base = f"pdf_pipeline_{ts}_p{args.page}"
    out_json = os.path.join(out_dir, f"{base}.json")

    # 1) Render PDF page
    page_img = _render_pdf_page(pdf_path=pdf_path, page_1based=args.page, zoom=float(args.zoom))
    rendered_path = os.path.join(out_dir, f"{base}_page.png")
    table_path = os.path.join(out_dir, f"{base}_table.png")

    if args.save_images:
        page_img.save(rendered_path)

    # 2) Optional crop
    table_img = page_img
    if crop:
        table_img = page_img.crop(crop)

    # Save the table image to a real file path.
    # (This is important because OCR isolation runs via subprocess and expects a file path.)
    table_img.save(table_path)

    # 3) Run existing pipeline on the (cropped) table image
    pipeline = FinancialTablePipeline(config_path=args.config, use_v1_1=True)
    pipeline_out = pipeline.process_image(table_path)

    # 4) Run full validator (pipeline itself only does equity_checks)
    validator = TableValidator(tolerance=0.02)
    grid = pipeline_out.get("grid") or []
    labels = pipeline_out.get("labels") or []
    normalized = pipeline_out.get("normalized_grid") or []
    validations = validator.validate_grid(grid, labels, normalized)
    validations_passed = sum(1 for v in validations if v.get("passed"))

    payload: Dict[str, Any] = {
        "pdf": pdf_path,
        "page": int(args.page),
        "zoom": float(args.zoom),
        "crop": list(crop) if crop else None,
        "rendered_page_image": rendered_path if args.save_images else None,
        "table_image": table_path,
        "pipeline": pipeline_out,
        "validation": {
            "total": len(validations),
            "passed": validations_passed,
            "pass_rate": (validations_passed / len(validations)) if validations else 0.0,
            "checks": validations,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"Saved: {out_json}")

    # Keep workspace tidy if user didn't request page image.
    if not args.save_images:
        try:
            os.remove(rendered_path)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
