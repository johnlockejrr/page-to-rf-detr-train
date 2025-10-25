#!/usr/bin/env python3
"""
RF-DETR Segmentation Inference Script

This script demonstrates how to run segmentation inference on images using a trained RF-DETR model.
"""

from rfdetr import RFDETRSmall
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load your trained model
    print("🔄 Loading trained model...")
    model = RFDETRSmall(pretrain_weights='./runs/sam_44_mss_coco/checkpoint_best_ema.pth')
    print("✅ Model loaded!")
    
    # Load image
    image_path = '/home/incognito/AI/DATASETS/sam_44_mss/BNF_Sam.2_050.jpg'
    print(f"🖼️  Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded: {image_rgb.shape}")
    
    # Run inference
    print("🔍 Running inference...")
    detections = model.predict(image_rgb)
    print(f"✅ Found {len(detections)} text lines")
    
    # Debug: Print detection structure
    print("\n🔍 Detection structure analysis:")
    for i, detection in enumerate(detections[:3]):  # Show first 3
        print(f"Detection {i+1}: {type(detection)}")
        print(f"  Length: {len(detection) if hasattr(detection, '__len__') else 'N/A'}")
        print(f"  Content: {detection}")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    
    # Create a copy for visualization
    vis_image = image_rgb.copy()
    
    # Create mask overlay
    mask_overlay = np.zeros_like(image_rgb)
    
    # Process each detection
    for i, detection in enumerate(detections):
        if len(detection) >= 2:
            bbox = detection[0]
            confidence = detection[1]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"Text {i+1}: {confidence:.3f}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create a simple mask (rectangle for now)
            # In a real segmentation model, you'd get actual masks
            mask_overlay[y1:y2, x1:x2] = [255, 0, 0]  # Red mask
    
    # Save results
    output_dir = Path('./inference_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save bounding box visualization
    bbox_output = output_dir / 'segmentation_bboxes.jpg'
    cv2.imwrite(str(bbox_output), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"💾 Bounding boxes saved to: {bbox_output}")
    
    # Save mask overlay
    mask_output = output_dir / 'segmentation_masks.jpg'
    mask_vis = cv2.addWeighted(image_rgb, 0.7, mask_overlay, 0.3, 0)
    cv2.imwrite(str(mask_output), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
    print(f"💾 Mask overlay saved to: {mask_output}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Total detections: {len(detections)}")
    print(f"   Image size: {image_rgb.shape}")
    print(f"   Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
