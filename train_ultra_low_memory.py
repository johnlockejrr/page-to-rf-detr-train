#!/usr/bin/env python3
"""
Ultra Low Memory RF-DETR Training Script

This script uses the most aggressive memory optimizations possible for RTX 3060 12GB.
"""

import argparse
import os
import sys
from pathlib import Path

def train_ultra_low_memory(model_size: str, dataset_dir: str, output_dir: str, epochs: int = 20):
    """Train RF-DETR with ultra low memory settings."""
    
    print(f"🚀 Ultra Low Memory Training - RF-DETR {model_size.upper()}")
    print("=" * 70)
    
    # Ultra low memory parameters
    ultra_low_params = {
        "nano": {
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 2e-4,
            "description": "Nano - Ultra low memory"
        }
    }
    
    if model_size.lower() not in ultra_low_params:
        print(f"❌ Only 'nano' model supported for ultra low memory mode")
        return False
    
    params = ultra_low_params[model_size.lower()]
    print(f"📋 {params['description']}")
    print(f"⚙️  Ultra Low Memory Settings:")
    print(f"   - Batch size: {params['batch_size']}")
    print(f"   - Gradient accumulation: {params['grad_accum_steps']}")
    print(f"   - Learning rate: {params['lr']}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Effective batch size: {params['batch_size'] * params['grad_accum_steps']}")
    
    # Get model class
    try:
        from rfdetr import RFDETRNano
        ModelClass = RFDETRNano
        print(f"✅ Loaded RF-DETR NANO model class")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Initialize model
    try:
        model = ModelClass()
        print(f"✅ Initialized RF-DETR NANO model")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False
    
    # Ultra low memory training arguments
    train_args = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": params["batch_size"],
        "grad_accum_steps": params["grad_accum_steps"],
        "lr": params["lr"],
        "output_dir": output_dir,
        "gradient_checkpointing": True,  # Save memory
        "use_ema": False,  # Disable EMA to save memory
        "early_stopping": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.001,
        "resolution": 448,  # Even smaller resolution (448x448)
        "num_workers": 1,  # Minimal workers
        "aux_loss": False,  # Disable auxiliary losses to save memory
        "multi_scale": False,  # Disable multi-scale to save memory
        "expanded_scales": False,  # Disable expanded scales
    }
    
    print(f"\n💾 Ultra Memory Optimizations:")
    print(f"   - Gradient checkpointing: Enabled")
    print(f"   - EMA: Disabled")
    print(f"   - Resolution: 448x448 (ultra small)")
    print(f"   - Workers: 1")
    print(f"   - Auxiliary losses: Disabled")
    print(f"   - Multi-scale: Disabled")
    print(f"   - Expanded scales: Disabled")
    
    print(f"\n🎯 Starting training...")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 70)
    
    try:
        model.train(**train_args)
        print("\n✅ Training completed successfully!")
        print(f"📁 Checkpoints saved to: {output_dir}")
        print(f"🏆 Best model: {output_dir}/checkpoint_best_total.pth")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 If this still fails, try:")
        print("1. Restart your system to clear GPU memory")
        print("2. Use a different dataset")
        print("3. Try training on CPU (very slow but should work)")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ultra Low Memory RF-DETR Training")
    parser.add_argument("--model_size", default="nano", 
                       help="Model size (only nano supported)")
    parser.add_argument("--dataset_dir", required=True, 
                       help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of training epochs (default: 20)")
    
    args = parser.parse_args()
    
    # Validate dataset directory
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory {args.dataset_dir} does not exist")
        sys.exit(1)
    
    # Check for required COCO structure
    required_dirs = ["train", "valid", "test"]
    for split in required_dirs:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"❌ Error: Missing {split} directory in dataset")
            sys.exit(1)
        
        coco_file = split_dir / "_annotations.coco.json"
        if not coco_file.exists():
            print(f"❌ Error: Missing _annotations.coco.json in {split} directory")
            sys.exit(1)
    
    print("✅ Dataset structure validated")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Start training
    success = train_ultra_low_memory(
        model_size=args.model_size,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n💥 Training failed. This might be a deeper RF-DETR compatibility issue.")
        print("\nAlternative solutions:")
        print("1. Try a different RF-DETR version")
        print("2. Use a different model (YOLOv8, DETR, etc.)")
        print("3. Train on a cloud GPU with more memory")
        sys.exit(1)

if __name__ == "__main__":
    main()
