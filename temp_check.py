import json

# 检查 GT bbox 尺寸
with open('D:/datasets/FinTabNet_c/FinTabNet.c-Structure/FinTabNet.c-Structure/words/AAL_2002_page_41_table_1_words.json', 'r') as f:
    gt_words = json.load(f)

print('=== GT Word BBoxes (Word-Level) ===')
for w in gt_words[:5]:
    bbox = w['bbox']
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    print(f'  "{w["text"]}" -> bbox={[round(b,1) for b in bbox]}, size={width:.1f}x{height:.1f}')

# 检查 OCR cache 的 bbox
with open('outputs/thesis_figures/step3_ocr_cell/easyocr_cell_results.json', 'r', encoding='utf-8') as f:
    easyocr_results = json.load(f)

print()
print('=== Cell Results Sample ===')
sample = easyocr_results[0]
print(f'Sample: {sample["sample_id"]}')
print(f'Total cells: {sample["total_cells"]}')
print(f'Matched cells: {sample["matched_cells"]}')
print(f'Match rate: {sample["matched_cells"]/sample["total_cells"]*100:.1f}%')
