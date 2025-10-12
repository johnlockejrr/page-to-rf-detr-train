#!/usr/bin/env python3
"""
RF-DETR Training Script with Model Size Selection

This script demonstrates how to fine-tune different RF-DETR model sizes
for text line segmentation on book pages.

Usage:
    python train_rf_detr.py --model_size nano --dataset_dir ./coco_dataset --output_dir ./output
    python train_rf_detr.py --model_size large --dataset_dir ./coco_dataset --output_dir ./output --epochs 50
"""

import argparse
import sys
from pathlib import Path

def get_model_class(model_size: str):
    """Get the appropriate RF-DETR model class based on size."""
    model_size = model_size.lower()
    
    if model_size == "nano":
        from rfdetr import RFDETRNano
        return RFDETRNano
    elif model_size == "small":
        from rfdetr import RFDETRSmall
        return RFDETRSmall
    elif model_size == "base":
        from rfdetr import RFDETRBase
        return RFDETRBase
    elif model_size == "medium":
        from rfdetr import RFDETRMedium
        return RFDETRMedium
    elif model_size == "large":
        from rfdetr import RFDETRLarge
        return RFDETRLarge
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: nano, small, base, medium, large")

def get_recommended_params(model_size: str):
    """Get recommended training parameters based on model size."""
    params = {
        "nano": {
            "batch_size": 8,
            "grad_accum_steps": 2,
            "lr": 2e-4,
            "description": "Fastest, smallest, good for edge devices"
        },
        "small": {
            "batch_size": 6,
            "grad_accum_steps": 3,
            "lr": 1.5e-4,
            "description": "Balanced performance and speed"
        },
        "base": {
            "batch_size": 4,
            "grad_accum_steps": 4,
            "lr": 1e-4,
            "description": "Default, good general performance"
        },
        "medium": {
            "batch_size": 3,
            "grad_accum_steps": 5,
            "lr": 8e-5,
            "description": "Higher accuracy"
        },
        "large": {
            "batch_size": 2,
            "grad_accum_steps": 8,
            "lr": 5e-5,
            "description": "Highest accuracy, requires more resources"
        }
    }
    return params[model_size.lower()]

def train_model(model_size: str, dataset_dir: str, output_dir: str, epochs: int = 20, 
                batch_size: int = None, grad_accum_steps: int = None, lr: float = None,
                use_tensorboard: bool = False, use_wandb: bool = False,
                project_name: str = "rf-detr-text-lines", run_name: str = None,
                early_stopping: bool = True, early_stopping_patience: int = 5,
                early_stopping_min_delta: float = 0.001):
    """Train RF-DETR model with specified parameters."""
    
    print(f"🚀 Training RF-DETR {model_size.upper()} for text line segmentation")
    print("=" * 60)
    
    # Get model class
    try:
        ModelClass = get_model_class(model_size)
        print(f"✅ Loaded RF-DETR {model_size.upper()} model class")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    
    # Get recommended parameters
    recommended = get_recommended_params(model_size)
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
    
    # Initialize model
    try:
        model = ModelClass()
        print(f"✅ Initialized RF-DETR {model_size.upper()} model")
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
        "output_dir": output_dir
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
    
    # Start training
    print(f"\n🎯 Starting training...")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 60)
    
    try:
        model.train(**train_args)
        print("\n✅ Training completed successfully!")
        print(f"📁 Checkpoints saved to: {output_dir}")
        print(f"🏆 Best model: {output_dir}/checkpoint_best_total.pth")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR model for text line segmentation")
    parser.add_argument("--model_size", required=True, 
                       choices=["nano", "small", "base", "medium", "large"],
                       help="RF-DETR model size to use")
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
    parser.add_argument("--project_name", default="rf-detr-text-lines", 
                       help="W&B project name (default: rf-detr-text-lines)")
    parser.add_argument("--run_name", 
                       help="W&B run name (if not provided, auto-generated)")
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
    success = train_model(
        model_size=args.model_size,
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
