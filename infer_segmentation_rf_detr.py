#!/usr/bin/env python3
"""
RF-DETR Segmentation Inference Script

This script runs inference with a trained RF-DETR segmentation model to get pixel-level masks.
"""

from rfdetr import RFDETRSegPreview
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load your trained segmentation model
    print("🔄 Loading trained segmentation model...")
    model = RFDETRSegPreview(pretrain_weights='./checkpoint_best_total_seg.pth')
    print("✅ Segmentation model loaded!")
    
    # Load image
    image_path = './882241214-0187.jpg'
    print(f"🖼️  Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded: {image_rgb.shape}")
    
    # Run segmentation inference
    print("🔍 Running segmentation inference...")
    detections = model.predict(image_rgb)
    print(f"✅ Found {len(detections)} text line segments")
    
    # Debug: Print detection structure for segmentation
    print("\n🔍 Segmentation detection structure analysis:")
    for i, detection in enumerate(detections[:3]):  # Show first 3
        print(f"Detection {i+1}:")
        print(f"  Type: {type(detection)}")
        print(f"  Length: {len(detection) if hasattr(detection, '__len__') else 'N/A'}")
        print(f"  Content: {detection}")
        
        # Check for mask information
        if hasattr(detection, '__len__') and len(detection) > 0:
            for j, element in enumerate(detection):
                print(f"  Element {j}: {element} (type: {type(element)})")
                if hasattr(element, 'shape'):
                    print(f"    Shape: {element.shape}")
    
    # Create visualization
    print("\n🎨 Creating segmentation visualization...")
    
    # Create output directory
    output_dir = Path('./segmentation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create mask overlay
    mask_overlay = np.zeros_like(image_rgb)
    
    # Process each detection
    for i, detection in enumerate(detections):
        if len(detection) >= 3:
            bbox = detection[0]  # Bounding box
            confidence = detection[2]  # Confidence
            class_id = detection[3]  # Class ID
            
            # For segmentation, we should have masks
            # Check if there's a mask in the detection
            mask = None
            if len(detection) > 4 and detection[4] is not None:
                mask = detection[4]  # Mask might be at index 4
            elif len(detection) > 1 and detection[1] is not None:
                mask = detection[1]  # Or at index 1
            
            if mask is not None:
                print(f"Text {i+1}: Found mask with shape {mask.shape}")
                # Apply mask to overlay
                mask_overlay[mask > 0] = [255, 0, 0]  # Red mask
            else:
                print(f"Text {i+1}: No mask found, using bounding box")
                # Fallback to bounding box
                x1, y1, x2, y2 = map(int, bbox)
                mask_overlay[y1:y2, x1:x2] = [255, 0, 0]  # Red rectangle
    
    # Save results
    # Save original image
    cv2.imwrite(str(output_dir / 'original.jpg'), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Save mask overlay
    mask_vis = cv2.addWeighted(image_rgb, 0.7, mask_overlay, 0.3, 0)
    cv2.imwrite(str(output_dir / 'segmentation_masks.jpg'), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
    
    # Save pure mask
    cv2.imwrite(str(output_dir / 'masks_only.jpg'), cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
    
    print(f"\n📊 Segmentation Results:")
    print(f"   Total segments: {len(detections)}")
    print(f"   Image size: {image_rgb.shape}")
    print(f"   Results saved to: {output_dir}")
    print(f"   - original.jpg: Original image")
    print(f"   - segmentation_masks.jpg: Image with mask overlay")
    print(f"   - masks_only.jpg: Pure masks")

if __name__ == "__main__":
    main()
