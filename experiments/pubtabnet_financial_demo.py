"""
Run a small PubTabNet demo (10 samples) to showcase numeric normalization and semantic mapping
on OCR-aligned cells.
"""
import sys
import os
import json
from datetime import datetime
from collections import Counter
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import load_config
from modules.data_loaders import PubTabNetLoader
from modules.structure import TableStructureRecognizer
from modules.ocr import TableOCR
from modules.numeric import normalize_numeric
from modules.semantic import map_alias


def collect_headers(grid: List[List[str]]) -> List[str]:
    headers = []
    if not grid:
        return headers
    # take first row
    headers.extend(grid[0])
    # take first column (excluding the top-left already captured)
    for r in range(1, len(grid)):
        if grid[r]:
            headers.append(grid[r][0])
    return [h for h in headers if h and h.strip()]


def main(num_samples: int = 10):
    config = load_config('configs/config.yaml')
    loader = PubTabNetLoader(config)
    annotations = loader.load_annotations(split='val', num_samples=num_samples)

    model = TableStructureRecognizer(config)
    ocr = TableOCR(lang='en')

    stats = {
        'num_samples': 0,
        'total_cells': 0,
        'numeric_cells': 0,
        'header_mapped': Counter(),
        'numeric_examples': [],
        'mapping_examples': [],
    }

    for ann in annotations:
        image, annotation = loader.load_sample(ann, split='val')
        gt_html = loader.get_html_structure(annotation)

        structure = model.recognize(image)
        ocr_results = ocr.extract_text(image)
        grid = ocr.align_text_to_grid(
            ocr_results,
            structure.get('rows', []),
            structure.get('columns', []),
            method='hybrid'
        )

        if not grid:
            continue

        stats['num_samples'] += 1

        # headers mapping
        headers = collect_headers(grid)
        for h in headers:
            mapped = map_alias(h)
            stats['header_mapped'][mapped] += 1
            if len(stats['mapping_examples']) < 5:
                stats['mapping_examples'].append({'raw': h, 'mapped': mapped})

        # numeric normalization across cells
        for row in grid:
            for cell in row:
                stats['total_cells'] += 1
                if cell and any(ch.isdigit() for ch in cell):
                    norm = normalize_numeric(cell)
                    if norm['value'] is not None:
                        stats['numeric_cells'] += 1
                        if len(stats['numeric_examples']) < 5:
                            stats['numeric_examples'].append({
                                'raw': cell,
                                'value': norm['value'],
                                'currency': norm['currency'],
                                'unit': norm['unit'],
                                'normalized': norm['normalized'],
                            })

    # Summary
    print("PubTabNet 10-sample financial demo")
    print("=" * 50)
    print(f"Samples processed: {stats['num_samples']}")
    print(f"Total cells: {stats['total_cells']}")
    print(f"Numeric cells (normalized): {stats['numeric_cells']}")
    print("\nTop mapped headers:")
    for k, v in stats['header_mapped'].most_common(10):
        print(f"  {k}: {v}")

    print("\nNumeric examples:")
    for ex in stats['numeric_examples']:
        print(f"  {ex['raw']} -> {ex['normalized']}")

    print("\nMapping examples:")
    for ex in stats['mapping_examples']:
        print(f"  {ex['raw']} -> {ex['mapped']}")

    # Save JSON summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join('outputs', 'results', f'pubtabnet_financial_demo_{ts}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to: {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    main(num_samples=args.num_samples)
