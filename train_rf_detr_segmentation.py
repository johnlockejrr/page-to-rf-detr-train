#!/usr/bin/env python3
"""
RF-DETR Segmentation Training Script

This script demonstrates how to fine-tune the RF-DETR segmentation model
for text line segmentation on book pages.

Usage:
    python train_rf_detr_segmentation.py --dataset_dir ./coco_dataset --output_dir ./output
    python train_rf_detr_segmentation.py --dataset_dir ./coco_dataset --output_dir ./output --epochs 50
"""

import argparse
import sys
from pathlib import Path
import os

# Make CUDA allocator more flexible to reduce fragmentation-related OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_model_class():
    """Get the RF-DETR segmentation model class."""
    from rfdetr import RFDETRSegPreview
    return RFDETRSegPreview

def get_recommended_params():
    """Get recommended training parameters for RF-DETR segmentation."""
    # RF-DETR segmentation only has one model type, so we provide default parameters
    return {
        "batch_size": 2,
        "grad_accum_steps": 6,
        "lr": 8e-5,
        "description": "RF-DETR Segmentation model - optimized for text line segmentation"
    }

def train_model(dataset_dir: str, output_dir: str, epochs: int = 20, 
                batch_size: int = None, grad_accum_steps: int = None, lr: float = None,
                use_tensorboard: bool = False, use_wandb: bool = False,
                project_name: str = "rf-detr-segmentation-text-lines", run_name: str = None,
                early_stopping: bool = True, early_stopping_patience: int = 5,
                early_stopping_min_delta: float = 0.001, resume: str = None):
    """Train RF-DETR segmentation model with specified parameters."""
    
    print(f"🚀 Training RF-DETR Segmentation for text line segmentation")
    print("=" * 60)
    
    # Get model class
    try:
        ModelClass = get_model_class()
        print(f"✅ Loaded RF-DETR Segmentation model class")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Get recommended parameters
    recommended = get_recommended_params()
    print(f"📋 Model description: {recommended['description']}")
    
    # Use provided parameters or defaults
    final_batch_size = batch_size if batch_size is not None else recommended["batch_size"]
    final_grad_accum = grad_accum_steps if grad_accum_steps is not None else recommended["grad_accum_steps"]
    final_lr = lr if lr is not None else recommended["lr"]
    
    print(f"⚙️  Training parameters:")
    print(f"   - Batch size: {final_batch_size}")
    print(f"   - Gradient accumulation steps: {final_grad_accum}")
    print(f"   - Learning rate: {final_lr}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Total effective batch size: {final_batch_size * final_grad_accum}")
    
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
        model = ModelClass()
        print(f"✅ Initialized RF-DETR Segmentation model")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False
    
    # Prepare training arguments
    train_args = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": final_batch_size,
        "grad_accum_steps": final_grad_accum,
        "lr": final_lr,
        "output_dir": output_dir,
        "gradient_checkpointing": True,  # Enable for segmentation (memory intensive)
        "use_ema": False,  # Disable EMA to save memory
        "resolution": 480,  # Rectangular resolution for text lines (896x640, divisible by 56)
        "num_workers": 4,  # Reduce workers for segmentation
    }
    
    # Add optional logging
    if use_tensorboard:
        train_args["tensorboard"] = True
        print("📊 TensorBoard logging enabled")
    
    if use_wandb:
        train_args["wandb"] = True
        train_args["project"] = project_name
        if run_name:
            train_args["run"] = run_name
        print(f"📊 Weights & Biases logging enabled (project: {project_name})")
    
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
    
    # Add segmentation-specific optimizations
    print(f"💾 Segmentation optimizations:")
    print(f"   - Gradient checkpointing: Enabled")
    print(f"   - EMA: Disabled")
    print(f"   - Resolution: 896x640 (rectangular, divisible by 56)")
    print(f"   - Workers: 2")
    
    # Start training
    print(f"\n🎯 Starting segmentation training...")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 60)
    
    try:
        model.train(**train_args)
        print("\n✅ Segmentation training completed successfully!")
        print(f"📁 Checkpoints saved to: {output_dir}")
        print(f"🏆 Best model: {output_dir}/checkpoint_best_total.pth")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR segmentation model for text line segmentation")
    parser.add_argument("--dataset_dir", required=True, 
                       help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, 
                       help="Batch size (if not provided, uses recommended value)")
    parser.add_argument("--grad_accum_steps", type=int, 
                       help="Gradient accumulation steps (if not provided, uses recommended value)")
    parser.add_argument("--lr", type=float, 
                       help="Learning rate (if not provided, uses recommended value)")
    parser.add_argument("--tensorboard", action="store_true", 
                       help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true", 
                       help="Enable Weights & Biases logging")
    parser.add_argument("--project_name", default="rf-detr-segmentation-text-lines", 
                       help="W&B project name (default: rf-detr-segmentation-text-lines)")
    parser.add_argument("--run_name", 
                       help="W&B run name (if not provided, auto-generated)")
    parser.add_argument("--no_early_stopping", action="store_true", 
                       help="Disable early stopping (default: enabled)")
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                       help="Number of epochs to wait before stopping (default: 5)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, 
                       help="Minimum change in mAP to qualify as improvement (default: 0.001)")
    parser.add_argument("--resume", type=str, 
                       help="Path to checkpoint.pth file to resume training from")
    
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
    success = train_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        project_name=args.project_name,
        run_name=args.run_name,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        resume=args.resume
    )
    
    if success:
        print("\n🎉 Segmentation training completed successfully!")
        print("\nNext steps:")
        print("1. Evaluate your model on the test set")
        print("2. Use the best checkpoint for segmentation inference")
        print("3. Consider exporting to ONNX for deployment")
    else:
        print("\n💥 Training failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
