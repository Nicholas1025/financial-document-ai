"""Test pipeline with LLM validation enabled."""
from modules.pipeline import FinancialTablePipeline
import json

# Test with LLM validation enabled
print('Initializing pipeline with LLM validation...')
pipeline = FinancialTablePipeline(enable_llm_validation=True)

# Process sample image
print('\nProcessing CIMB Bank sample...')
result = pipeline.process_image('data/samples/CIMB_BANK-SAMPLE1.png')

# Show LLM validation results
print('\n' + '='*60)
print('LLM VALIDATION RESULTS (Step 7)')
print('='*60)
llm = result.get('llm_validation', {})
print(f"Model: {llm.get('model', 'N/A')}")
print(f"Cells Verified: {llm.get('cells_verified', 0)}")
print(f"Cells Matched: {llm.get('cells_matched', 0)}")
print(f"Accuracy: {llm.get('accuracy', 0):.1%}")
print()
print('Token Usage:')
tokens = llm.get('token_usage', {})
print(f"  Prompt:  {tokens.get('total_prompt_tokens', 0):,}")
print(f"  Output:  {tokens.get('total_output_tokens', 0):,}")
print(f"  Total:   {tokens.get('total_tokens', 0):,}")
print()
print('Verification Details:')
for d in llm.get('details', []):
    status = '✓' if d.get('match') else '✗'
    row_short = d['row'][:30] if d['row'] else 'N/A'
    print(f"  {status} {row_short:<30} | Pipeline: {str(d['pipeline_value']):>15} | LLM: {str(d['llm_value']):>15}")

if llm.get('discrepancies'):
    print('\nDiscrepancies Found:')
    for disc in llm['discrepancies']:
        print(f"  - {disc['row']}: Pipeline={disc['pipeline_value']}, LLM={disc['llm_value']}")
        
print('='*60)

# Save full result
with open('outputs/validation/pipeline_llm_test.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print('\nFull results saved to outputs/validation/pipeline_llm_test.json')
