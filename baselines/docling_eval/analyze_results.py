"""Quick analysis of results"""
import json

# Load Docling results
with open(r'C:\Users\User\Documents\financial-document-ai\baselines\docling_eval\outputs\docling_results.json') as f:
    docling = json.load(f)

print("=== Docling Results ===")
for r in docling['results']:
    status = "OK" if r['pred_rows'] > 0 else "FAIL"
    print(f"{r['sample_id'][:30]:30} | Pred: {r['pred_rows']:2}x{r['pred_cols']:<2} | GT: {r['gt_rows']:2}x{r['gt_cols']:<2} | TEDS: {r['teds_struct']:.3f} | {status}")

print(f"\nTotal: {docling['samples_with_results']}/{docling['num_samples']} samples with results")
print(f"Mean TEDS: {docling['mean_teds_struct']:.4f}")

# Load Old Pipeline results  
try:
    with open(r'C:\Users\User\Documents\financial-document-ai\baselines\docling_eval\outputs\old_pipeline_results.json') as f:
        old = json.load(f)
    
    print("\n=== Old Pipeline Results ===")
    for r in old['results']:
        status = "OK" if r['pred_rows'] > 0 else "FAIL"
        print(f"{r['sample_id'][:30]:30} | Pred: {r['pred_rows']:2}x{r['pred_cols']:<2} | GT: {r['gt_rows']:2}x{r['gt_cols']:<2} | TEDS: {r['teds_struct']:.3f} | {status}")
    
    print(f"\nTotal: {old['samples_with_results']}/{old['num_samples']} samples with results")
    print(f"Mean TEDS: {old['mean_teds_struct']:.4f}")
except:
    print("\nOld pipeline results not found")
