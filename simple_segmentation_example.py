#!/usr/bin/env python3
"""
Simple RF-DETR Segmentation Inference Example

A minimal example of how to run segmentation inference on a single image.
"""

from rfdetr import RFDETRSegPreview
import cv2
import numpy as np

def main():
    # Load your trained segmentation model
    print("🔄 Loading trained segmentation model...")
    model = RFDETRSegPreview(pretrain_weights='./runs/sam_44_mss_coco_seg/checkpoint_best_ema.pth')
    print("✅ Segmentation model loaded!")
    
    # Load image
    image_path = './BNF_Sam.2_050.jpg'  # Your image path
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
    
    # Print results - RF-DETR segmentation returns tuples with masks
    for i, detection in enumerate(detections):
        # RF-DETR segmentation returns (bbox, mask, confidence, class_id, None, {})
        if len(detection) >= 4:
            bbox = detection[0]  # Element 0: bbox [x1, y1, x2, y2]
            mask = detection[1]  # Element 1: mask
            confidence = detection[2]  # Element 2: confidence score
            class_id = detection[3]  # Element 3: class ID
            has_mask = mask is not None
            print(f"Text {i+1}: confidence={confidence:.3f}, class_id={class_id}, "
                  f"bbox=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}), "
                  f"has_mask={has_mask}")
        else:
            print(f"Text {i+1}: detection format: {detection}")
    
    # Visualize results
    print("🎨 Drawing segmentation results...")
    
    # Create mask overlay
    mask_overlay = np.zeros_like(image)
    
    for i, detection in enumerate(detections):
        if len(detection) >= 4:
            bbox = detection[0]  # Element 0: bbox
            mask = detection[1]  # Element 1: mask
            confidence = detection[2]  # Element 2: confidence
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{confidence:.3f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Apply mask if available
            if mask is not None and hasattr(mask, 'shape'):
                print(f"  Text {i+1}: Applying mask with shape {mask.shape}")
                mask_overlay[mask > 0] = [255, 0, 0]  # Red mask
            else:
                print(f"  Text {i+1}: No mask, using bounding box")
                mask_overlay[y1:y2, x1:x2] = [255, 0, 0]  # Red rectangle
    
    # Save results
    output_dir = './segmentation_results'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save bounding box visualization
    bbox_output = f'{output_dir}/segmentation_bboxes.jpg'
    cv2.imwrite(bbox_output, image)
    print(f"💾 Bounding boxes saved to: {bbox_output}")
    
    # Save mask overlay
    mask_output = f'{output_dir}/segmentation_masks.jpg'
    mask_vis = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
    cv2.imwrite(mask_output, mask_vis)
    print(f"💾 Mask overlay saved to: {mask_output}")
    
    # Save pure masks
    pure_mask_output = f'{output_dir}/pure_masks.jpg'
    cv2.imwrite(pure_mask_output, mask_overlay)
    print(f"💾 Pure masks saved to: {pure_mask_output}")

if __name__ == "__main__":
    main()
