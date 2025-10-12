#!/usr/bin/env python3
"""
COCO Dataset Validation and Fix Script

This script validates and fixes common issues in COCO datasets that can cause
CUDA indexing errors during RF-DETR training.
"""

import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def validate_and_fix_coco_file(coco_file: Path, split_name: str) -> bool:
    """Validate and fix a COCO annotation file."""
    print(f"\n🔍 Validating {split_name} split...")
    
    try:
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {coco_file}: {e}")
        return False
    
    # Get image directory
    image_dir = coco_file.parent
    
    # Track issues
    issues_found = []
    fixes_applied = []
    
    # Validate images
    print(f"📸 Validating {len(coco_data['images'])} images...")
    valid_image_ids = set()
    
    for img in tqdm(coco_data['images'], desc="Checking images"):
        img_path = image_dir / img['file_name']
        
        if not img_path.exists():
            issues_found.append(f"Missing image: {img['file_name']}")
            continue
        
        # Load image to verify dimensions
        try:
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                issues_found.append(f"Cannot load image: {img['file_name']}")
                continue
            
            actual_h, actual_w = cv_img.shape[:2]
            
            # Check if dimensions match
            if img['height'] != actual_h or img['width'] != actual_w:
                print(f"⚠️  Dimension mismatch in {img['file_name']}: "
                      f"COCO says {img['width']}x{img['height']}, "
                      f"actual is {actual_w}x{actual_h}")
                
                # Fix dimensions
                img['width'] = actual_w
                img['height'] = actual_h
                fixes_applied.append(f"Fixed dimensions for {img['file_name']}")
            
            valid_image_ids.add(img['id'])
            
        except Exception as e:
            issues_found.append(f"Error processing {img['file_name']}: {e}")
            continue
    
    # Validate annotations
    print(f"📝 Validating {len(coco_data['annotations'])} annotations...")
    valid_annotations = []
    
    for ann in tqdm(coco_data['annotations'], desc="Checking annotations"):
        # Check if image exists
        if ann['image_id'] not in valid_image_ids:
            issues_found.append(f"Annotation {ann['id']} references non-existent image {ann['image_id']}")
            continue
        
        # Get image info
        img_info = next((img for img in coco_data['images'] if img['id'] == ann['image_id']), None)
        if not img_info:
            continue
        
        img_w, img_h = img_info['width'], img_info['height']
        
        # Validate segmentation
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                if len(seg) < 6:  # At least 3 points (x,y pairs)
                    issues_found.append(f"Invalid segmentation in annotation {ann['id']}: too few points")
                    continue
                
                # Check if coordinates are within image bounds
                coords = np.array(seg).reshape(-1, 2)
                if len(coords) < 3:
                    issues_found.append(f"Invalid segmentation in annotation {ann['id']}: not enough points")
                    continue
                
                # Check bounds
                if np.any(coords < 0) or np.any(coords[:, 0] > img_w) or np.any(coords[:, 1] > img_h):
                    print(f"⚠️  Out-of-bounds segmentation in annotation {ann['id']}")
                    
                    # Clamp coordinates to image bounds
                    coords[:, 0] = np.clip(coords[:, 0], 0, img_w)
                    coords[:, 1] = np.clip(coords[:, 1], 0, img_h)
                    
                    # Update segmentation
                    ann['segmentation'] = [coords.flatten().tolist()]
                    fixes_applied.append(f"Fixed out-of-bounds segmentation in annotation {ann['id']}")
        
        # Validate bbox
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            
            # Check if bbox is valid
            if w <= 0 or h <= 0:
                issues_found.append(f"Invalid bbox in annotation {ann['id']}: {ann['bbox']}")
                continue
            
            # Check if bbox is within image bounds
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                print(f"⚠️  Out-of-bounds bbox in annotation {ann['id']}")
                
                # Clamp bbox to image bounds
                x = max(0, min(x, img_w))
                y = max(0, min(y, img_h))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                ann['bbox'] = [x, y, w, h]
                fixes_applied.append(f"Fixed out-of-bounds bbox in annotation {ann['id']}")
        
        # Validate area
        if 'area' in ann and ann['area'] <= 0:
            # Recalculate area from bbox
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                ann['area'] = w * h
                fixes_applied.append(f"Fixed area for annotation {ann['id']}")
        
        valid_annotations.append(ann)
    
    # Update annotations
    coco_data['annotations'] = valid_annotations
    
    # Report issues
    if issues_found:
        print(f"\n❌ Issues found in {split_name}:")
        for issue in issues_found[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues_found) > 10:
            print(f"   ... and {len(issues_found) - 10} more issues")
    
    if fixes_applied:
        print(f"\n🔧 Fixes applied in {split_name}:")
        for fix in fixes_applied[:10]:  # Show first 10 fixes
            print(f"   - {fix}")
        if len(fixes_applied) > 10:
            print(f"   ... and {len(fixes_applied) - 10} more fixes")
        
        # Save fixed file
        backup_file = coco_file.with_suffix('.coco.json.backup')
        coco_file.rename(backup_file)
        print(f"💾 Backed up original to: {backup_file}")
        
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"✅ Saved fixed annotations to: {coco_file}")
    
    return len(issues_found) == 0

def validate_coco_dataset(dataset_dir: str):
    """Validate entire COCO dataset."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory {dataset_dir} does not exist")
        return False
    
    print(f"🔍 Validating COCO dataset: {dataset_dir}")
    print("=" * 60)
    
    # Check required splits
    splits = ['train', 'valid', 'test']
    all_valid = True
    
    for split in splits:
        split_dir = dataset_path / split
        coco_file = split_dir / '_annotations.coco.json'
        
        if not split_dir.exists():
            print(f"❌ Missing {split} directory")
            all_valid = False
            continue
        
        if not coco_file.exists():
            print(f"❌ Missing _annotations.coco.json in {split} directory")
            all_valid = False
            continue
        
        is_valid = validate_and_fix_coco_file(coco_file, split)
        if not is_valid:
            all_valid = False
    
    if all_valid:
        print(f"\n✅ Dataset validation completed successfully!")
        print(f"🎯 Your dataset is ready for RF-DETR training")
    else:
        print(f"\n⚠️  Dataset validation completed with issues")
        print(f"🔧 Some issues were automatically fixed")
        print(f"💡 Try training again with the fixed dataset")
    
    return all_valid

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and fix COCO dataset for RF-DETR training")
    parser.add_argument("--dataset_dir", required=True, help="Path to COCO dataset directory")
    
    args = parser.parse_args()
    
    validate_coco_dataset(args.dataset_dir)

if __name__ == "__main__":
    main()
