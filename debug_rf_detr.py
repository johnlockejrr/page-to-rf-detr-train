#!/usr/bin/env python3
"""
RF-DETR Debug Script

This script helps debug CUDA indexing issues by running with minimal settings
and detailed error reporting.
"""

import os
import sys
import torch
from pathlib import Path

def debug_rf_detr():
    """Debug RF-DETR with minimal settings."""
    
    print("🔍 RF-DETR Debug Mode")
    print("=" * 50)
    
    # Set environment variables for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    print("✅ Debug environment variables set")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared")
    
    try:
        # Import RF-DETR
        from rfdetr import RFDETRNano
        print("✅ RF-DETR imported successfully")
        
        # Initialize model
        model = RFDETRNano()
        print("✅ Model initialized successfully")
        
        # Test with minimal dataset
        print("\n🧪 Testing with minimal settings...")
        
        # Create a minimal test dataset
        test_args = {
            "dataset_dir": "./datasets/sam_44_mss_coco/",
            "epochs": 1,
            "batch_size": 1,
            "grad_accum_steps": 1,
            "lr": 1e-4,
            "output_dir": "./debug_output/",
            "gradient_checkpointing": True,
            "use_ema": False,
            "early_stopping": False,
            "resolution": 224,  # Very small resolution
            "num_workers": 0,  # No workers
            "aux_loss": False,
            "multi_scale": False,
            "expanded_scales": False,
        }
        
        print("⚙️  Test parameters:")
        for key, value in test_args.items():
            print(f"   - {key}: {value}")
        
        print("\n🚀 Starting minimal test...")
        model.train(**test_args)
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # Print detailed error info
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
        
        return False

def main():
    print("This script will test RF-DETR with minimal settings to identify the issue.")
    print("Make sure you're in the rf-detr directory and have the dataset ready.")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    success = debug_rf_detr()
    
    if success:
        print("\n🎉 Debug test passed! The issue might be with batch size or memory.")
        print("Try the ultra low memory script:")
        print("python train_ultra_low_memory.py --dataset_dir ./datasets/sam_44_mss_coco/ --output_dir ./debug_output/")
    else:
        print("\n💥 Debug test failed. This indicates a deeper issue with RF-DETR or your setup.")
        print("\nPossible solutions:")
        print("1. Update RF-DETR: pip install --upgrade rfdetr")
        print("2. Check PyTorch version: pip install torch==2.0.1")
        print("3. Try a different RF-DETR version")
        print("4. Use a different model entirely")

if __name__ == "__main__":
    main()
