import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _run_ocr(image_path: str, lang: str, use_gpu: bool) -> List[Dict[str, Any]]:
    # Keep worker isolated: do NOT import torch or any project modules.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import logging

    logging.getLogger("ppocr").setLevel(logging.ERROR)

    import paddle

    if use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    from paddleocr import PaddleOCR

    ocr_engine = PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=bool(use_gpu), show_log=False)

    extracted: List[Dict[str, Any]] = []

    # PaddleOCR v3: predict(path)
    if hasattr(ocr_engine, "predict"):
        results = ocr_engine.predict(image_path)
        for result in results or []:
            if not isinstance(result, dict):
                continue
            texts = result.get("rec_texts")
            polys = result.get("rec_polys")
            scores = result.get("rec_scores")
            if not (texts and polys and scores):
                continue
            for text, poly, score in zip(texts, polys, scores):
                x_coords = [p[0] for p in poly]
                y_coords = [p[1] for p in poly]
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                extracted.append({"text": text, "bbox": bbox, "confidence": float(score)})
        return extracted

    # PaddleOCR v2: ocr(path, cls=True)
    results = ocr_engine.ocr(image_path, cls=True)
    if not results:
        return []

    lines = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
    for item in lines:
        try:
            poly = item[0]
            text, score = item[1]
        except Exception:
            continue
        x_coords = [p[0] for p in poly]
        y_coords = [p[1] for p in poly]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        extracted.append({"text": text, "bbox": bbox, "confidence": float(score)})

    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--use_gpu", choices=["0", "1"], default="1")
    args = parser.parse_args()

    image_path = args.image
    out_path = args.out
    lang = args.lang
    use_gpu = args.use_gpu == "1"

    try:
        data = _run_ocr(image_path=image_path, lang=lang, use_gpu=use_gpu)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return 0
    except Exception as exc:
        print(f"ocr_worker failed: {exc!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
