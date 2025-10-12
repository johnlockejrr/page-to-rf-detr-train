#!/usr/bin/env python3
"""
Fix COCO Class Indexing for RF-DETR

This script fixes potential class indexing issues that cause CUDA errors.
"""

import json
import sys
from pathlib import Path

def fix_coco_classes(coco_file: Path):
    """Fix class indexing in COCO file for RF-DETR compatibility."""
    print(f"🔧 Fixing classes in {coco_file}")
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Check current categories
    print(f"📋 Current categories: {coco_data['categories']}")
    
    # Check if we need to add dummy class (category_id: 0)
    has_dummy_class = any(cat['id'] == 0 for cat in coco_data['categories'])
    
    if not has_dummy_class:
        print("➕ Adding dummy class (category_id: 0) for RF-DETR compatibility")
        dummy_class = {
            "id": 0,
            "name": "background",
            "supercategory": "none"
        }
        coco_data['categories'].insert(0, dummy_class)
    
    # Ensure all other categories have IDs >= 1
    for i, category in enumerate(coco_data['categories']):
        if i == 0 and category['id'] == 0:
            continue  # Skip dummy class
        if category['id'] != i:
            print(f"⚠️  Fixing category ID: {category['id']} -> {i}")
            category['id'] = i
    
    # Update all annotations to use correct category_id
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == 0:
            print(f"⚠️  Annotation has category_id 0, updating to 1")
            annotation['category_id'] = 1
    
    # Save fixed file
    backup_file = coco_file.with_suffix('.coco.json.backup')
    coco_file.rename(backup_file)
    print(f"💾 Backed up original to: {backup_file}")
    
    with open(coco_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"✅ Fixed classes saved to: {coco_file}")

def main():
    dataset_dir = Path("./datasets/sam_44_mss_coco/")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Fix all splits
    for split in ['train', 'valid', 'test']:
        coco_file = dataset_dir / split / "_annotations.coco.json"
        if coco_file.exists():
            fix_coco_classes(coco_file)
        else:
            print(f"⚠️  Skipping {split} - no COCO file found")
    
    print("\n✅ RF-DETR class indexing fixed!")
    print("🔧 This fixes the CUDA indexing error by adding a dummy background class (id: 0)")
    print("📋 Based on GitHub issues #330 and #349 - known RF-DETR bug with 1-indexed datasets")
    print("\nNow try training again with:")
    print("python train_rtx3060.py --model_size nano --dataset_dir ./datasets/sam_44_mss_coco/ --output_dir ./runs/fixed/")

if __name__ == "__main__":
    main()
