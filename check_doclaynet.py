"""Check DocLayNet dataset statistics"""
import json
from collections import Counter

# Load DocLayNet validation annotations
with open('D:/datasets/DocLayNet/COCO/val.json', 'r') as f:
    data = json.load(f)

print('=== DocLayNet Val Set Statistics ===')
print(f'Total images: {len(data["images"])}')
print(f'Total annotations: {len(data["annotations"])}')

# Count by category
category_counts = Counter([ann['category_id'] for ann in data['annotations']])

# Category mapping
categories = {cat['id']: cat['name'] for cat in data['categories']}

print(f'\n=== Annotations by Category ===')
for cat_id, count in sorted(category_counts.items()):
    cat_name = categories.get(cat_id, 'Unknown')
    marker = ' <-- TABLE' if cat_id == 9 else ''
    print(f'  {cat_id}: {cat_name}: {count}{marker}')

# Count images with tables
table_annotations = [ann for ann in data['annotations'] if ann['category_id'] == 9]
images_with_tables = set([ann['image_id'] for ann in table_annotations])

print(f'\n=== Table Statistics ===')
print(f'Total table annotations: {len(table_annotations)}')
print(f'Images with at least 1 table: {len(images_with_tables)}')
print(f'Images without tables: {len(data["images"]) - len(images_with_tables)}')
print(f'Percentage with tables: {100*len(images_with_tables)/len(data["images"]):.1f}%')

# Show first 3 table annotations as example
print(f'\n=== Example Table Annotations ===')
for i, ann in enumerate(table_annotations[:3]):
    img_id = ann['image_id']
    img_info = next(img for img in data['images'] if img['id'] == img_id)
    print(f'\n  Table {i+1}:')
    print(f'    Image: {img_info["file_name"]}')
    print(f'    Bbox: {ann["bbox"]}')
    print(f'    Area: {ann["area"]}')
