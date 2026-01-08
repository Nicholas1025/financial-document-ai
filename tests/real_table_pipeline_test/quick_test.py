"""
Simplified Pipeline Test - Use uploaded image or sample
"""
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if user image exists, otherwise use sample
TEST_DIR = Path(__file__).parent
IMAGE_PATH = TEST_DIR / 'balance_sheet.png'
GT_PATH = TEST_DIR / 'ground_truth.json'
OUTPUT_DIR = TEST_DIR / 'output'

if not IMAGE_PATH.exists():
    print(f"请将财务报表图片保存到: {IMAGE_PATH}")
    print("或者运行以下命令使用示例图片:")
    print(f"  python {__file__} --use-sample")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-sample', action='store_true')
    args = parser.parse_args()
    
    if args.use_sample:
        # Use FinTabNet sample
        IMAGE_PATH = Path('D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure/images/AAL_2002_page_41_table_1.jpg')
        print(f"Using sample image: {IMAGE_PATH}")
    else:
        sys.exit(1)

from run_pipeline_test import PipelineAnalyzer

# Run test
analyzer = PipelineAnalyzer(str(IMAGE_PATH), str(GT_PATH), str(OUTPUT_DIR))
analyzer.run_full_pipeline()

print(f"\n打开报告: {OUTPUT_DIR / 'SUMMARY.md'}")
