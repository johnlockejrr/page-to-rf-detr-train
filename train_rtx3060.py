#!/usr/bin/env python3
"""
RTX 3060 12GB Optimized RF-DETR Training Script

This script is specifically optimized for RTX 3060 12GB GPU with memory-efficient settings.
"""

import argparse
import os
import sys
from pathlib import Path

def train_rtx3060_optimized(model_size: str, dataset_dir: str, output_dir: str, epochs: int = 20,
                           early_stopping: bool = True, early_stopping_patience: int = 5,
                           early_stopping_min_delta: float = 0.001):
    """Train RF-DETR with RTX 3060 12GB optimized settings."""
    
    print(f"🚀 RTX 3060 12GB Optimized Training - RF-DETR {model_size.upper()}")
    print("=" * 70)
    
    # RTX 3060 12GB optimized parameters
    rtx3060_params = {
        "nano": {
            "batch_size": 4,
            "grad_accum_steps": 4,
            "lr": 2e-4,
            "description": "Nano - Fastest, most memory efficient"
        },
        "small": {
            "batch_size": 2,
            "grad_accum_steps": 8,
            "lr": 1.5e-4,
            "description": "Small - Balanced performance"
        },
        "base": {
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 1e-4,
            "description": "Base - Good performance with memory optimization"
        }
    }
    
    if model_size.lower() not in rtx3060_params:
        print(f"❌ Model size '{model_size}' not supported for RTX 3060 12GB")
        print("✅ Supported sizes: nano, small, base")
        return False
    
    params = rtx3060_params[model_size.lower()]
    print(f"📋 {params['description']}")
    print(f"⚙️  RTX 3060 Optimized Settings:")
    print(f"   - Batch size: {params['batch_size']}")
    print(f"   - Gradient accumulation: {params['grad_accum_steps']}")
    print(f"   - Learning rate: {params['lr']}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Effective batch size: {params['batch_size'] * params['grad_accum_steps']}")
    
    if early_stopping:
        print(f"   - Early stopping: Enabled (patience: {early_stopping_patience}, min_delta: {early_stopping_min_delta})")
    else:
        print(f"   - Early stopping: Disabled")
    
    # Get model class
    try:
        if model_size.lower() == "nano":
            from rfdetr import RFDETRNano
            ModelClass = RFDETRNano
        elif model_size.lower() == "small":
            from rfdetr import RFDETRSmall
            ModelClass = RFDETRSmall
        elif model_size.lower() == "base":
            from rfdetr import RFDETRBase
            ModelClass = RFDETRBase
        
        print(f"✅ Loaded RF-DETR {model_size.upper()} model class")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Initialize model
    try:
        model = ModelClass()
        print(f"✅ Initialized RF-DETR {model_size.upper()} model")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False
    
    # RTX 3060 optimized training arguments
    train_args = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": params["batch_size"],
        "grad_accum_steps": params["grad_accum_steps"],
        "lr": params["lr"],
        "output_dir": output_dir,
        "gradient_checkpointing": True,  # Save memory
        "use_ema": False,  # Disable EMA to save memory
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "resolution": 560,  # Smaller resolution to save memory
        "num_workers": 2,  # Reduce workers to save memory
    }
    
    print(f"\n💾 Memory Optimizations:")
    print(f"   - Gradient checkpointing: Enabled")
    print(f"   - EMA: Disabled")
    print(f"   - Resolution: 560x560")
    print(f"   - Workers: 2")
    
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
        print("\n🔧 Troubleshooting tips:")
        print("1. Try with 'nano' model size first")
        print("2. Check if dataset is valid with: python validate_coco_dataset.py")
        print("3. Monitor GPU memory with: nvidia-smi")
        return False

def main():
    parser = argparse.ArgumentParser(description="RTX 3060 12GB Optimized RF-DETR Training")
    parser.add_argument("--model_size", required=True, 
                       choices=["nano", "small", "base"],
                       help="RF-DETR model size (nano/small/base only for RTX 3060)")
    parser.add_argument("--dataset_dir", required=True, 
                       help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of training epochs (default: 20)")
    parser.add_argument("--no_early_stopping", action="store_true", 
                       help="Disable early stopping (default: enabled)")
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                       help="Number of epochs to wait before stopping (default: 5)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, 
                       help="Minimum change in mAP to qualify as improvement (default: 0.001)")
    
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
    success = train_rtx3060_optimized(
        model_size=args.model_size,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Evaluate your model on the test set")
        print("2. Use the best checkpoint for inference")
        print("3. Consider exporting to ONNX for deployment")
    else:
        print("\n💥 Training failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

