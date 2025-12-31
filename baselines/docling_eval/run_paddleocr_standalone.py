# -*- coding: utf-8 -*-
"""
Standalone PaddleOCR Evaluation Script
Run this in a separate environment with compatible numpy version
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import json
from pathlib import Path
from datetime import datetime

# Dataset path
FINTABNET_PATH = Path('D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure')


def calculate_cer(gt_text: str, pred_text: str) -> float:
    """Calculate Character Error Rate using Levenshtein distance"""
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    
    m, n = len(gt_text), len(pred_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_text[i-1] == pred_text[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--output-dir', type=str, default='./outputs/thesis_figures/step3_ocr')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading PaddleOCR (CPU mode)...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
    
    # Get files
    images_dir = FINTABNET_PATH / 'images'
    words_dir = FINTABNET_PATH / 'words'
    word_files = sorted(list(words_dir.glob('*.json')))[:args.num_samples]
    
    print(f"Processing {len(word_files)} samples...")
    
    results = []
    for i, word_file in enumerate(word_files):
        base_name = word_file.stem.replace('_words', '')
        img_path = images_dir / f'{base_name}.jpg'
        
        if not img_path.exists():
            continue
        
        # Load GT
        with open(word_file, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        gt_words = [w['text'] for w in words_data]
        gt_text = ' '.join(gt_words)
        
        # Run OCR
        try:
            ocr_result = ocr.ocr(str(img_path), cls=True)
            pred_words = []
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        pred_words.append(text)
            pred_text = ' '.join(pred_words)
        except Exception as e:
            pred_words = []
            pred_text = ""
        
        cer = calculate_cer(gt_text.lower(), pred_text.lower())
        
        results.append({
            'sample_id': base_name,
            'img_path': str(img_path),
            'word_file': str(word_file),
            'gt_words': gt_words,
            'pred_words': pred_words,
            'cer': cer,
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(word_files)}")
    
    # Save results
    cache_path = output_dir / 'paddleocr_results_cache.json'
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved: {cache_path}")
    print(f"Total samples: {len(results)}")
    
    import numpy as np
    cer_values = [r['cer'] for r in results]
    print(f"Mean CER: {np.mean(cer_values):.4f}")
    print(f"Min CER: {np.min(cer_values):.4f}")


if __name__ == '__main__':
    main()
