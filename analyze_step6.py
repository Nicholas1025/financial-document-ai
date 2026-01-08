"""Analyze Step 6 QA Validation Results"""
import json
import pandas as pd

# Load results
with open('outputs/thesis_figures/step6_qa_validation/step6_summary.json', 'r') as f:
    summary = json.load(f)

# Overall metrics
print('='*60)
print('STEP 6: QA VALIDATION RESULTS')
print('='*60)
print(f"Dataset: {summary['dataset']}")
print(f"Tables: {summary['num_tables']}")
print(f"Total Questions: {summary['num_questions']}")
print()
print("Overall Accuracy: {:.2f}% ({}/{})".format(
    summary['metrics']['accuracy'],
    summary['metrics']['correct'],
    summary['metrics']['total']
))

# Error analysis
per_sample = summary['per_sample']
df = pd.DataFrame(per_sample)

print()
print('='*60)
print('ERROR ANALYSIS')
print('='*60)

# Row/Col finding stats
row_found = df['row_found'].sum()
col_found = df['col_found'].sum()
both_found = ((df['row_found']) & (df['col_found'])).sum()

print(f"Row Found: {row_found}/{len(df)} ({row_found/len(df)*100:.1f}%)")
print(f"Col Found: {col_found}/{len(df)} ({col_found/len(df)*100:.1f}%)")
print(f"Both Found: {both_found}/{len(df)} ({both_found/len(df)*100:.1f}%)")

# Incorrect cases breakdown
incorrect = df[~df['correct']]
print(f"\nIncorrect Cases: {len(incorrect)}")

# Why incorrect?
no_row = incorrect[~incorrect['row_found']]
no_col = incorrect[~incorrect['col_found']]
row_col_ok_but_wrong = incorrect[(incorrect['row_found']) & (incorrect['col_found'])]

print(f"  - Row not found: {len(no_row)}")
print(f"  - Col not found: {len(no_col)}")
print(f"  - Row+Col found but wrong value: {len(row_col_ok_but_wrong)}")

# Sample some errors
print()
print('Sample Errors (first 5):')
for i, row in incorrect.head(5).iterrows():
    print(f"  Q: '{row['row_key']}' x '{row['col_key']}'")
    print(f"     GT: '{row['gt_answer']}' vs Pred: '{row['pred_answer']}'")
    print()
