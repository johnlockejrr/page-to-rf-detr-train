# RF-DETR Model Sizes Guide

This guide helps you choose the right RF-DETR model size for your text line segmentation task.

## Available Model Sizes

### 🚀 RF-DETR Nano
- **Best for**: Edge devices, mobile deployment, real-time inference
- **Characteristics**: Fastest inference, smallest model size, lowest memory usage
- **Recommended batch size**: 8
- **Gradient accumulation**: 2
- **Learning rate**: 2e-4
- **Use case**: When speed and efficiency are more important than maximum accuracy

### ⚡ RF-DETR Small
- **Best for**: Balanced performance and speed
- **Characteristics**: Good accuracy with reasonable resource usage
- **Recommended batch size**: 6
- **Gradient accumulation**: 3
- **Learning rate**: 1.5e-4
- **Use case**: General-purpose applications with moderate resource constraints

### 🎯 RF-DETR Base (Default)
- **Best for**: General text line segmentation tasks
- **Characteristics**: Good balance of accuracy and efficiency
- **Recommended batch size**: 4
- **Gradient accumulation**: 4
- **Learning rate**: 1e-4
- **Use case**: Most common choice for document analysis tasks

### 🔬 RF-DETR Medium
- **Best for**: Higher accuracy requirements
- **Characteristics**: Better accuracy than Base, requires more resources
- **Recommended batch size**: 3
- **Gradient accumulation**: 5
- **Learning rate**: 8e-5
- **Use case**: When accuracy is more important than speed

### 🏆 RF-DETR Large
- **Best for**: Maximum accuracy, research applications
- **Characteristics**: Highest accuracy, requires significant resources
- **Recommended batch size**: 2
- **Gradient accumulation**: 8
- **Learning rate**: 5e-5
- **Use case**: When you have powerful hardware and need the best possible results

## Quick Selection Guide

| Scenario | Recommended Model | Reason |
|----------|------------------|---------|
| Mobile/Edge deployment | Nano | Smallest size, fastest inference |
| General document processing | Base | Good balance of accuracy and speed |
| High-accuracy requirements | Medium/Large | Better accuracy for complex documents |
| Research/Experimentation | Large | Maximum accuracy for benchmarking |
| Resource-constrained | Small/Nano | Lower memory and compute requirements |

## Training Examples

### Quick Start (Nano)
```bash
python train_rf_detr.py \
    --model_size nano \
    --dataset_dir ./coco_dataset \
    --output_dir ./output_nano \
    --epochs 15
```

### Balanced Approach (Base)
```bash
python train_rf_detr.py \
    --model_size base \
    --dataset_dir ./coco_dataset \
    --output_dir ./output_base \
    --epochs 20
```

### High Accuracy (Large)
```bash
python train_rf_detr.py \
    --model_size large \
    --dataset_dir ./coco_dataset \
    --output_dir ./output_large \
    --epochs 30 \
    --tensorboard
```

### With Custom Early Stopping
```bash
python train_rf_detr.py \
    --model_size base \
    --dataset_dir ./coco_dataset \
    --output_dir ./output_base \
    --epochs 50 \
    --early_stopping_patience 7 \
    --early_stopping_min_delta 0.002
```

### Without Early Stopping
```bash
python train_rf_detr.py \
    --model_size base \
    --dataset_dir ./coco_dataset \
    --output_dir ./output_base \
    --epochs 20 \
    --no_early_stopping
```

## Hardware Requirements

| Model Size | Minimum GPU Memory | Recommended GPU Memory | Training Time* |
|------------|-------------------|----------------------|----------------|
| Nano | 4GB | 6GB | ~2 hours |
| Small | 6GB | 8GB | ~3 hours |
| Base | 8GB | 12GB | ~4 hours |
| Medium | 12GB | 16GB | ~6 hours |
| Large | 16GB | 24GB+ | ~8 hours |

*Approximate training time for 20 epochs on a typical book page dataset

## Early Stopping Configuration

Early stopping is **enabled by default** and helps prevent overfitting by monitoring validation mAP improvements.

### Default Early Stopping Parameters
- **Patience**: 5 epochs (wait 5 epochs without improvement before stopping)
- **Min Delta**: 0.001 (minimum mAP improvement to qualify as progress)

### When to Adjust Early Stopping

| Scenario | Recommended Settings | Reason |
|----------|---------------------|---------|
| Small dataset | Patience: 3-5, Min Delta: 0.002 | Prevent overfitting quickly |
| Large dataset | Patience: 7-10, Min Delta: 0.001 | Allow more time for convergence |
| Noisy validation | Patience: 5-7, Min Delta: 0.002 | Reduce sensitivity to noise |
| Stable training | Patience: 3-5, Min Delta: 0.0005 | Stop as soon as improvement stops |

### Early Stopping Examples

```bash
# Conservative (stops quickly)
--early_stopping_patience 3 --early_stopping_min_delta 0.002

# Default (balanced)
--early_stopping_patience 5 --early_stopping_min_delta 0.001

# Patient (allows more time)
--early_stopping_patience 10 --early_stopping_min_delta 0.0005

# Disable early stopping
--no_early_stopping
```

## Tips for Model Selection

1. **Start with Base**: If you're unsure, start with RF-DETR Base as it provides a good balance
2. **Consider your hardware**: Make sure you have enough GPU memory for your chosen model
3. **Test different sizes**: Try Nano and Large to see the accuracy/speed tradeoff
4. **Monitor training**: Use TensorBoard or W&B to track training progress
5. **Validate on test set**: Always evaluate on your test set to compare models objectively
6. **Use early stopping**: Prevents overfitting and saves training time

## Model Comparison

| Metric | Nano | Small | Base | Medium | Large |
|--------|------|-------|------|--------|-------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Accuracy | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory Usage | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Model Size | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
