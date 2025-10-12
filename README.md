# PAGE-XML to RF-DETR Training Pipeline

A complete pipeline for converting PAGE-XML ground truth data to COCO format and training RF-DETR models for text line segmentation. This repository includes scripts for data conversion, model training, and GPU optimization.

## 🚀 Features

- **PAGE-XML to COCO Conversion**: Handles multiple PAGE-XML namespace schemas
- **RF-DETR Compatible**: Automatically fixes class indexing issues
- **Multiple Model Sizes**: Support for Nano, Small, Base, Medium, and Large models
- **GPU Optimization**: Specialized scripts for RTX 3060 12GB and other GPUs
- **Early Stopping**: Prevents overfitting with configurable parameters
- **Image Resizing**: Optional resizing with coordinate scaling
- **Dataset Splitting**: Optimal train/validation/test splits using scikit-learn

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🏗️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/johnlockejrr/page-to-rf-detr-train.git
cd page-to-rf-detr-train
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python example_usage.py
```

## 📊 Quick Start

### 1. Convert PAGE-XML to COCO Format

```bash
# Basic conversion with default 896x640 resizing
python convert_pagexml_to_coco.py \
    --input_dir /path/to/pagexml/files \
    --output_dir /path/to/coco_dataset

# Custom image size (must be divisible by 56)
python convert_pagexml_to_coco.py \
    --input_dir /path/to/pagexml/files \
    --output_dir /path/to/coco_dataset \
    --target_width 1120 \
    --target_height 800

# Use original image dimensions
python convert_pagexml_to_coco.py \
    --input_dir /path/to/pagexml/files \
    --output_dir /path/to/coco_dataset \
    --no_resize
```

### 2. Train RF-DETR Model

```bash
# Train with default settings
python train_rf_detr.py \
    --model_size base \
    --dataset_dir /path/to/coco_dataset \
    --output_dir ./rf_detr_output \
    --epochs 20

# Train with early stopping
python train_rf_detr.py \
    --model_size small \
    --dataset_dir /path/to/coco_dataset \
    --output_dir ./rf_detr_output \
    --epochs 50 \
    --early_stopping_patience 7 \
    --early_stopping_min_delta 0.002
```

### 3. RTX 3060 12GB Optimization

```bash
# Use memory-optimized training for RTX 3060
python train_rtx3060.py \
    --model_size nano \
    --dataset_dir /path/to/coco_dataset \
    --output_dir ./rf_detr_output \
    --epochs 20
```

## 📁 Project Structure

```
page-to-rf-detr-train/
├── convert_pagexml_to_coco.py    # Main conversion script
├── train_rf_detr.py              # General training script
├── train_rtx3060.py              # RTX 3060 optimized training
├── fix_coco_classes.py           # RF-DETR class indexing fix
├── validate_coco_dataset.py      # Dataset validation
├── debug_rf_detr.py              # Debugging utilities
├── example_usage.py              # Usage examples
├── requirements.txt              # Python dependencies
├── MODEL_SIZES.md                # Model size documentation
└── README.md                     # This file
```

## 🔧 Detailed Usage

### PAGE-XML Conversion

The conversion script handles multiple PAGE-XML namespace schemas and creates RF-DETR-compatible COCO datasets:

```bash
python convert_pagexml_to_coco.py \
    --input_dir /path/to/pagexml \
    --output_dir /path/to/output \
    --test_size 0.2 \
    --val_size 0.1 \
    --target_width 896 \
    --target_height 640
```

**Arguments:**
- `--input_dir`: Directory containing PAGE-XML files and images
- `--output_dir`: Output directory for COCO dataset
- `--test_size`: Proportion for test split (default: 0.2)
- `--val_size`: Proportion for validation split (default: 0.1)
- `--target_width`: Target image width (default: 896)
- `--target_height`: Target image height (default: 640)
- `--no_resize`: Use original image dimensions

### Model Training

#### General Training

```bash
python train_rf_detr.py \
    --model_size {nano|small|base|medium|large} \
    --dataset_dir /path/to/coco_dataset \
    --output_dir /path/to/output \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-4
```

#### RTX 3060 12GB Optimization

```bash
python train_rtx3060.py \
    --model_size {nano|small|base} \
    --dataset_dir /path/to/coco_dataset \
    --output_dir /path/to/output \
    --epochs 20
```

**Memory Optimizations:**
- Reduced batch sizes
- Increased gradient accumulation
- Gradient checkpointing enabled
- EMA disabled
- Smaller resolution (560x560)
- Fewer workers

## 🎯 Model Sizes

| Size | Batch Size | Memory | Speed | Accuracy | Best For |
|------|------------|--------|-------|----------|----------|
| **Nano** | 8 | Low | Fastest | Good | Edge devices, RTX 3060 |
| **Small** | 6 | Medium | Fast | Better | Balanced performance |
| **Base** | 4 | Medium | Good | Good | General use |
| **Medium** | 3 | High | Slower | Better | High accuracy needs |
| **Large** | 2 | Highest | Slowest | Best | Maximum accuracy |

## 📊 Expected Performance

With proper training, you can expect:

- **AP@0.50**: 85-95% (excellent precision)
- **AP@0.75**: 70-85% (high precision)
- **AR@0.50:0.95**: 75-85% (good recall)
- **Class Error**: 0.00% (perfect classification)

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Error: device-side assert triggered**
   - **Solution**: The conversion script now automatically fixes this by adding a dummy background class (id: 0)

2. **GPU Out of Memory**
   - **Solution**: Use `train_rtx3060.py` for RTX 3060 or reduce batch size

3. **Poor Performance**
   - **Solution**: Ensure images are resized to dimensions divisible by 56
   - Check that PAGE-XML files contain valid text line annotations

### Debug Mode

```bash
# Run with minimal settings for debugging
python debug_rf_detr.py
```

## 📈 Training Monitoring

### TensorBoard
```bash
tensorboard --logdir /path/to/output
```

### Early Stopping
- **Default patience**: 5 epochs
- **Min delta**: 0.001
- **Customizable**: Use `--early_stopping_patience` and `--early_stopping_min_delta`

## 🎨 Customization

### Custom Model Sizes
Modify `train_rf_detr.py` to add your own model configurations:

```python
def get_recommended_params(model_size: str):
    params = {
        "custom": {
            "batch_size": 4,
            "grad_accum_steps": 4,
            "lr": 1e-4,
            "description": "Custom configuration"
        }
    }
```

### Custom Image Sizes
Ensure your target dimensions are divisible by 56 for RF-DETR compatibility:

```python
# Good examples
896x640, 1120x800, 1344x960, 1568x1120

# Bad examples (will cause issues)
900x640, 1125x800, 1350x960
```

## 📚 Technical Details

### PAGE-XML Support
- Multiple namespace schemas (2009-2019)
- Automatic namespace detection
- Polygon coordinate extraction
- Coordinate scaling during resizing

### COCO Format
- RF-DETR compatible class structure
- Automatic dummy background class (id: 0)
- Proper supercategory hierarchy
- Segmentation polygon format

### RF-DETR Integration
- Class indexing fix (background class id: 0)
- Proper category structure
- Compatible with all RF-DETR model sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) for the excellent detection model
- [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML) for the document annotation format
- [Roboflow](https://roboflow.com/) for the RF-DETR implementation

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Run the debug script
3. Open an issue on GitHub
4. Include your error logs and system specifications

## 🎉 Success Stories

This pipeline has been successfully used for:
- Historical document text line detection
- Book page text line segmentation
- Manuscript analysis
- OCR preprocessing

---

**Happy Training! 🚀**
