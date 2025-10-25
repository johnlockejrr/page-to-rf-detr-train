#!/usr/bin/env python3
"""
RF-DETR Segmentation Minimal Memory Training Script

This script uses the most aggressive memory optimizations possible for segmentation training.
Use this only if ultra_low_memory.py still fails.
"""

import argparse
import sys
from pathlib import Path

def train_segmentation_minimal_memory(dataset_dir: str, output_dir: str, epochs: int = 20,
                                     batch_size: int = 1, grad_accum_steps: int = 32, lr: float = 5e-5,
                                     early_stopping: bool = True, early_stopping_patience: int = 5,
                                     early_stopping_min_delta: float = 0.001, resume: str = None):
    """Train RF-DETR Segmentation model with minimal memory settings."""
    
    print(f"🚀 Training RF-DETR Segmentation MINIMAL MEMORY")
    print("=" * 70)
    
    print(f"⚙️  Minimal memory training parameters:")
    print(f"   - Model: RFDETRSegPreview")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Gradient accumulation steps: {grad_accum_steps}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Total effective batch size: {batch_size * grad_accum_steps}")
    
    if early_stopping:
        print(f"   - Early stopping: Enabled")
        print(f"     * Patience: {early_stopping_patience} epochs")
        print(f"     * Min delta: {early_stopping_min_delta}")
    else:
        print(f"   - Early stopping: Disabled")
    
    if resume:
        print(f"   - Resume training: {resume}")
        if not Path(resume).exists():
            print(f"❌ Error: Resume checkpoint {resume} does not exist")
            return False
    
    # Initialize model
    try:
        from rfdetr import RFDETRSegPreview
        model = RFDETRSegPreview()
        print(f"✅ Initialized RF-DETR Segmentation model")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False
    
    # Minimal memory training arguments - most aggressive settings
    train_args = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "output_dir": output_dir,
        "gradient_checkpointing": True,  # Essential for memory saving
        "use_ema": False,  # Disable EMA to save memory
        "resolution": 448,  # Minimal rectangular resolution (448x336, divisible by 56)
        "num_workers": 1,  # Minimal workers
        "aux_loss": False,  # Disable auxiliary losses
        "multi_scale": False,  # Disable multi-scale training
        "expanded_scales": False,  # Disable expanded scales
        "warmup_epochs": 0.0,  # No warmup
        "lr_scheduler": "step",  # Simple scheduler
        "lr_min_factor": 0.0,  # No learning rate decay
        "weight_decay": 0.0,  # No weight decay
        "dropout": 0.0,  # No dropout
        "drop_path": 0.0,  # No drop path
        "clip_max_norm": 0.0,  # No gradient clipping
    }
    
    # Add early stopping parameters
    if early_stopping:
        train_args["early_stopping"] = True
        train_args["early_stopping_patience"] = early_stopping_patience
        train_args["early_stopping_min_delta"] = early_stopping_min_delta
        print(f"⏹️  Early stopping enabled (patience: {early_stopping_patience}, min_delta: {early_stopping_min_delta})")
    
    # Add resume parameter
    if resume:
        train_args["resume"] = resume
        print(f"🔄 Resume training from: {resume}")
    
    # Print memory optimizations
    print(f"\n💾 Minimal memory optimizations:")
    print(f"   - Gradient checkpointing: Enabled")
    print(f"   - EMA: Disabled")
    print(f"   - Resolution: 448x336 (minimal rectangular, divisible by 56)")
    print(f"   - Workers: 1")
    print(f"   - Auxiliary losses: Disabled")
    print(f"   - Multi-scale: Disabled")
    print(f"   - Expanded scales: Disabled")
    print(f"   - Warmup: Disabled")
    print(f"   - Weight decay: Disabled")
    print(f"   - Dropout: Disabled")
    print(f"   - Drop path: Disabled")
    print(f"   - Gradient clipping: Disabled")
    print(f"   - Batch size: {batch_size} (minimal)")
    print(f"   - Gradient accumulation: {grad_accum_steps} (very high)")
    
    # Start training
    print(f"\n🎯 Starting minimal memory segmentation training...")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 70)
    
    try:
        model.train(**train_args)
        print("\n✅ Minimal memory segmentation training completed successfully!")
        print(f"📁 Checkpoints saved to: {output_dir}")
        print(f"🏆 Best model: {output_dir}/checkpoint_best_total.pth")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔧 If still getting OOM errors, try:")
        print("1. Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("2. Close all other applications")
        print("3. Restart your system to clear GPU memory")
        print("4. Consider using a smaller dataset")
        print("5. Use CPU training (very slow but works)")
        return False

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Segmentation Minimal Memory Training")
    parser.add_argument("--dataset_dir", required=True,
                       help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--grad_accum_steps", type=int, default=32,
                       help="Gradient accumulation steps (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--no_early_stopping", action="store_true",
                       help="Disable early stopping (default: enabled)")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Number of epochs to wait before stopping (default: 5)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001,
                       help="Minimum change in mAP to qualify as improvement (default: 0.001)")
    parser.add_argument("--resume", type=str,
                       help="Path to checkpoint.pth file to resume training from")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory {args.dataset_dir} does not exist")
        sys.exit(1)
    
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
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success = train_segmentation_minimal_memory(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        resume=args.resume
    )
    
    if success:
        print("\n🎉 Minimal memory segmentation training completed successfully!")
        print("\nNext steps:")
        print("1. Evaluate your segmentation model on the test set")
        print("2. Use the best checkpoint for segmentation inference")
        print("3. Consider exporting to ONNX for deployment")
    else:
        print("\n💥 Training failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
