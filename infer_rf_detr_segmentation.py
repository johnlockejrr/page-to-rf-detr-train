#!/usr/bin/env python3
"""
RF-DETR Segmentation Inference Script

This script demonstrates how to run segmentation inference on images using a trained RF-DETR segmentation model.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def load_model(checkpoint_path: str):
    """Load the trained RF-DETR segmentation model."""
    print(f"🔄 Loading RF-DETR Segmentation model...")
    
    try:
        from rfdetr import RFDETRSegPreview
        model = RFDETRSegPreview(pretrain_weights=checkpoint_path)
        print(f"✅ Segmentation model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path: str, target_size: tuple = None):
    """Load and preprocess image for inference."""
    print(f"🖼️  Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if target size specified
    if target_size:
        image_rgb = cv2.resize(image_rgb, target_size)
        print(f"📏 Resized image to: {target_size}")
    
    print(f"✅ Image loaded: {image_rgb.shape}")
    return image_rgb, image

def run_inference(model, image_rgb, confidence_threshold: float = 0.5):
    """Run segmentation inference on the image."""
    print(f"🔍 Running segmentation inference with confidence threshold: {confidence_threshold}")
    
    try:
        # Run prediction
        detections = model.predict(image_rgb)
        
        # Filter by confidence - RF-DETR segmentation returns tuples (bbox, mask, confidence, class_id, None, {})
        filtered_detections = []
        for detection in detections:
            if len(detection) >= 3:
                confidence = detection[2]  # Element 2 is confidence
                if confidence >= confidence_threshold:
                    filtered_detections.append(detection)
        
        print(f"✅ Found {len(filtered_detections)} text line segments (confidence >= {confidence_threshold})")
        return filtered_detections
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return []

def visualize_results(image, detections, output_path: str = None):
    """Visualize segmentation results on the image."""
    print(f"🎨 Visualizing {len(detections)} segmentation detections...")
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Create mask overlay
    mask_overlay = np.zeros_like(image)
    
    # Draw bounding boxes and masks
    for i, detection in enumerate(detections):
        if len(detection) >= 4:
            # RF-DETR segmentation returns (bbox, mask, confidence, class_id, None, {})
            bbox = detection[0]  # Element 0: bbox
            mask = detection[1]  # Element 1: mask
            confidence = detection[2]  # Element 2: confidence
            class_id = detection[3]  # Element 3: class ID
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Text {i+1}: {confidence:.3f}"
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Apply mask if available
            if mask is not None and hasattr(mask, 'shape'):
                print(f"  Text {i+1}: Mask shape {mask.shape}")
                # Convert mask to overlay
                if len(mask.shape) == 2:  # 2D mask
                    mask_3d = np.stack([mask, mask, mask], axis=2)
                    mask_overlay[mask > 0] = [255, 0, 0]  # Red mask
                else:
                    mask_overlay[mask > 0] = [255, 0, 0]  # Red mask
            else:
                print(f"  Text {i+1}: No mask available, using bounding box")
                # Fallback to bounding box
                mask_overlay[y1:y2, x1:x2] = [255, 0, 0]  # Red rectangle
    
    # Save or display result
    if output_path:
        # Save bounding box visualization
        bbox_output = output_path.replace('.jpg', '_bboxes.jpg')
        cv2.imwrite(bbox_output, vis_image)
        print(f"💾 Bounding boxes saved to: {bbox_output}")
        
        # Save mask overlay
        mask_output = output_path.replace('.jpg', '_masks.jpg')
        mask_vis = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
        cv2.imwrite(mask_output, mask_vis)
        print(f"💾 Mask overlay saved to: {mask_output}")
        
        # Save pure masks
        pure_mask_output = output_path.replace('.jpg', '_pure_masks.jpg')
        cv2.imwrite(pure_mask_output, mask_overlay)
        print(f"💾 Pure masks saved to: {pure_mask_output}")
    else:
        # Display image (if running in GUI environment)
        cv2.imshow('RF-DETR Segmentation', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis_image, mask_overlay

def save_results_json(detections, output_path: str):
    """Save segmentation results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(detections),
        "detections": []
    }
    
    for i, detection in enumerate(detections):
        if len(detection) >= 4:
            bbox = detection[0]  # Element 0: bbox
            mask = detection[1]  # Element 1: mask
            confidence = detection[2]  # Element 2: confidence
            class_id = detection[3]  # Element 3: class ID
            
            detection_data = {
                "id": i + 1,
                "confidence": float(confidence),
                "class_id": int(class_id),
                "bbox": {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3])
                },
                "class": "text_line"
            }
            
            # Add mask information if available
            if mask is not None and hasattr(mask, 'shape'):
                detection_data["mask_shape"] = list(mask.shape)
                detection_data["has_mask"] = True
            else:
                detection_data["has_mask"] = False
            
            results["detections"].append(detection_data)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Segmentation Text Line Detection Inference")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to trained segmentation model checkpoint (.pth file)")
    parser.add_argument("--image", required=True,
                       help="Path to input image")
    parser.add_argument("--output_dir", default="./segmentation_results",
                       help="Output directory for results")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for detections (default: 0.5)")
    parser.add_argument("--target_width", type=int,
                       help="Target image width (optional)")
    parser.add_argument("--target_height", type=int,
                       help="Target image height (optional)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint file {args.checkpoint} does not exist")
        return
    
    if not Path(args.image).exists():
        print(f"❌ Error: Image file {args.image} does not exist")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 RF-DETR Segmentation Text Line Detection Inference")
    print("=" * 50)
    
    # Load model
    model = load_model(args.checkpoint)
    if model is None:
        return
    
    # Load and preprocess image
    target_size = None
    if args.target_width and args.target_height:
        target_size = (args.target_width, args.target_height)
    
    try:
        image_rgb, image_bgr = preprocess_image(args.image, target_size)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Run inference
    detections = run_inference(model, image_rgb, args.confidence)
    
    if not detections:
        print("⚠️  No text line segments detected!")
        return
    
    # Print results
    print(f"\n📊 Segmentation Results:")
    print(f"   Total detections: {len(detections)}")
    print(f"   Confidence threshold: {args.confidence}")
    
    for i, detection in enumerate(detections):
        if len(detection) >= 4:
            bbox = detection[0]
            confidence = detection[2]
            class_id = detection[3]
            has_mask = detection[1] is not None
            print(f"   Text {i+1}: confidence={confidence:.3f}, class_id={class_id}, "
                  f"bbox=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}), "
                  f"has_mask={has_mask}")
    
    # Generate output filenames
    image_name = Path(args.image).stem
    vis_output = output_dir / f"{image_name}_segmentation.jpg"
    json_output = output_dir / f"{image_name}_segmentation_results.json"
    
    # Visualize results
    vis_image, mask_overlay = visualize_results(image_bgr, detections, str(vis_output))
    
    # Save results to JSON
    save_results_json(detections, str(json_output))
    
    print(f"\n✅ Segmentation inference completed!")
    print(f"   Visualization: {vis_output}")
    print(f"   Results JSON: {json_output}")

if __name__ == "__main__":
    main()
