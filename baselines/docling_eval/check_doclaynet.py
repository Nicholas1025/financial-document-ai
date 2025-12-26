"""检查 DocLayNet 数据集结构"""
import json
from pathlib import Path

data_dir = Path("D:/datasets/DocLayNet")
coco_file = data_dir / "COCO" / "val.json"

print(f"Loading: {coco_file}")
with open(coco_file) as f:
    data = json.load(f)

print("\nCategories:")
for c in data['categories']:
    print(f"  {c['id']}: {c['name']}")

print(f"\nImages: {len(data['images'])}")
print(f"Annotations: {len(data['annotations'])}")

# 统计各类别数量
cat_counts = {}
for ann in data['annotations']:
    cat_id = ann['category_id']
    cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

print("\nAnnotation counts by category:")
id_to_name = {c['id']: c['name'] for c in data['categories']}
for cat_id, count in sorted(cat_counts.items()):
    print(f"  {id_to_name[cat_id]}: {count}")
