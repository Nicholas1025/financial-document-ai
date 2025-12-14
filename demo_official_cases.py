"""demo_official_cases.py

Runs a small, reproducible set of official demo cases (bank statement table crops)
and exports:
- original image copy
- pipeline JSON output (grid, normalized_grid, metadata)
- validation checks (TableValidator)

This is meant to be a stable demo harness for thesis screenshots and results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from modules.pipeline import FinancialTablePipeline
from modules.validation import TableValidator


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_case(
    pipeline: FinancialTablePipeline,
    validator: TableValidator,
    image_path: str,
    out_dir: str,
    case_name: str,
) -> str:
    _ensure_dir(out_dir)
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Missing demo image: {src}")

    # Copy original image for a self-contained demo folder
    copied_image = os.path.join(out_dir, f"{case_name}{src.suffix.lower()}")
    shutil.copyfile(str(src), copied_image)

    pipeline_out = pipeline.process_image(copied_image)
    grid = pipeline_out.get("grid") or []
    labels = pipeline_out.get("labels") or []
    normalized = pipeline_out.get("normalized_grid") or []
    validations = validator.validate_grid(grid, labels, normalized)
    validations_passed = sum(1 for v in validations if v.get("passed"))

    payload: Dict[str, Any] = {
        "case": case_name,
        "image": copied_image,
        "pipeline": pipeline_out,
        "validation": {
            "total": len(validations),
            "passed": validations_passed,
            "pass_rate": (validations_passed / len(validations)) if validations else 0.0,
            "checks": validations,
        },
    }

    out_json = os.path.join(out_dir, f"{case_name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return out_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official demo cases (OCBC/CIMB)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--out_root",
        default=os.path.join("outputs", "demo_cases"),
        help="Root folder for demo outputs",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional comma-separated case names to run (e.g. cimb_balance_sheet)",
    )
    parser.add_argument(
        "--banks",
        type=str,
        default=None,
        help="Optional comma-separated bank names to run (e.g. CIMB,OCBC)",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.out_root, f"demo_{ts}")
    _ensure_dir(out_root)

    pipeline = FinancialTablePipeline(config_path=args.config, use_v1_1=True)
    validator = TableValidator(tolerance=0.02)

    # Minimal “official” set (can extend later)
    cases: List[Dict[str, str]] = [
        {
            "bank": "CIMB",
            "case": "cimb_balance_sheet",
            "image": os.path.join("data", "samples", "CIMB BANK-SAMPLE1.png"),
        },
        {
            "bank": "OCBC",
            "case": "ocbc_balance_sheet",
            "image": os.path.join("data", "samples", "ocbc_127_1.png"),
        },
    ]

    only_cases = None
    if args.only:
        only_cases = {c.strip() for c in args.only.split(",") if c.strip()}
    only_banks = None
    if args.banks:
        only_banks = {b.strip() for b in args.banks.split(",") if b.strip()}

    outputs: List[str] = []
    for c in cases:
        if only_cases is not None and c["case"] not in only_cases:
            continue
        if only_banks is not None and c["bank"] not in only_banks:
            continue
        bank_dir = os.path.join(out_root, c["bank"])
        _ensure_dir(bank_dir)
        out_json = _run_case(
            pipeline=pipeline,
            validator=validator,
            image_path=c["image"],
            out_dir=bank_dir,
            case_name=c["case"],
        )
        outputs.append(out_json)
        print(f"Saved: {out_json}")

    index_path = os.path.join(out_root, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"outputs": outputs}, f, indent=2, ensure_ascii=False)
    print(f"Index: {index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
