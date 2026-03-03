# Caltech-101 Image Classification Project

## SHBT 261 Mini-Project 1

This project implements and compares multiple image classification approaches on the Caltech-101 dataset, including both classical machine learning and deep learning methods.

## Dataset

The Caltech-101 dataset contains approximately 9,000 images across 101 object categories. The data is located in `data/caltech-101/`.

## Project Structure

```
project1/
├── src/                           # Source code modules
│   ├── data_preparation.py        # Data loading and preprocessing
│   ├── classical_models.py        # Classical ML models (SVM, RF, KNN)
│   ├── deep_models.py             # Deep learning models (ResNet, EfficientNet, ViT)
│   └── evaluation.py              # Evaluation metrics and visualization
├── train_classical.py             # Train classical ML models
├── train_deep.py                  # Train deep learning models
├── run_ablation.py                # Run ablation studies
├── data/                          # Dataset directory
│   └── caltech-101/
├── models/                        # Saved models
├── figures/                       # Generated plots and results
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Pillow
- scikit-image
- OpenCV

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train Classical ML Models

Train all classical models (SVM, Random Forest, KNN) with HOG features:

```bash
python train_classical.py --model all --feature hog --use-pca
```

Train specific model:

```bash
python train_classical.py --model svm --feature hog --image-size 128 --use-pca
```

Options:
- `--model`: Model type (`svm`, `rf`, `knn`, `all`)
- `--feature`: Feature type (`hog`, `raw`)
- `--image-size`: Image size (default: 128)
- `--use-pca`: Use PCA for dimensionality reduction
- `--n-components`: Number of PCA components (default: 100)

### 2. Train Deep Learning Models

Train ResNet-50:

```bash
python train_deep.py --model resnet --version resnet50 --epochs 20 --batch-size 32
```

Train EfficientNet-B0:

```bash
python train_deep.py --model efficientnet --version b0 --epochs 20 --lr 0.001
```

Train Vision Transformer:

```bash
python train_deep.py --model vit --version b_16 --epochs 20 --optimizer adam
```

Train all models:

```bash
python train_deep.py --model all --epochs 20
```

Options:
- `--model`: Model type (`resnet`, `efficientnet`, `vit`, `all`)
- `--version`: Model version (e.g., `resnet18`, `b0`, `b_16`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer (`adam`, `sgd`)
- `--no-augment`: Disable data augmentation
- `--no-pretrained`: Don't use pretrained weights

### 3. Run Ablation Studies

Run all ablation studies:

```bash
python run_ablation.py --study all
```

Run specific ablation study:

```bash
python run_ablation.py --study image_size
python run_ablation.py --study augmentation
python run_ablation.py --study features
python run_ablation.py --study optimizer
```

Available studies:
- `image_size`: Compare 64×64 vs 128×128 images
- `augmentation`: Compare with vs without data augmentation
- `features`: Compare HOG vs raw pixel features
- `optimizer`: Compare Adam vs SGD optimizers

## Data Split

The dataset is split as follows (stratified):
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Per-class accuracy**: Accuracy for each individual class
- **Confusion matrix**: Visualize misclassifications
- **Precision, Recall, F1-Score**: Macro and weighted averages
- **Top-k accuracy**: Top-3 and Top-5 accuracy (for deep learning models)

## Model Implementations

### Classical Machine Learning
1. **SVM** (Support Vector Machine with RBF kernel)
   - Features: HOG (Histogram of Oriented Gradients)
   - Dimensionality reduction: PCA (optional)
   
2. **Random Forest**
   - Features: HOG or raw pixels
   - 100 estimators, max depth 20

3. **K-Nearest Neighbors (KNN)**
   - Features: HOG
   - k=5 neighbors

### Deep Learning
1. **ResNet** (Residual Networks)
   - Versions: ResNet-18, ResNet-34, ResNet-50
   - Pretrained on ImageNet
   
2. **EfficientNet**
   - Versions: EfficientNet-B0, B1, B2
   - Pretrained on ImageNet
   
3. **Vision Transformer (ViT)**
   - Versions: ViT-B/16, ViT-B/32
   - Pretrained on ImageNet

## Outputs

After training, you'll find:

### Models Directory (`models/`)
- Trained model weights (.pkl for classical, .pth for deep learning)

### Figures Directory (`figures/`)
- Confusion matrices (raw and normalized)
- Per-class accuracy plots
- Training history plots (loss and accuracy curves)
- Model comparison plots
- Ablation study results
- Results JSON files with metrics

## Example Workflow

Complete workflow to reproduce all results:

```bash
# 1. Train classical models
python train_classical.py --model all --feature hog --use-pca --image-size 128

# 2. Train deep learning models (shorter epochs for testing)
python train_deep.py --model resnet --version resnet18 --epochs 10
python train_deep.py --model efficientnet --version b0 --epochs 10

# 3. Run ablation studies
python run_ablation.py --study all

# All results will be in figures/ and models/ directories
```

## Tips for Training

### For Faster Iteration (Development)
- Use smaller image size: `--image-size 64`
- Use fewer epochs: `--epochs 5`
- Use smaller models: ResNet-18, EfficientNet-B0
- Reduce batch size if memory limited: `--batch-size 16`

### For Best Performance (Final Results)
- Use larger image size: 224×224 (default for deep learning)
- Train longer: `--epochs 30` or more
- Use larger models: ResNet-50, EfficientNet-B2
- Enable data augmentation (default)
- Use pretrained weights (default)

### For Limited Resources
- Use classical models first (faster, less memory)
- Enable PCA: `--use-pca --n-components 100`
- Train on CPU if no GPU available (will be slower)
- Process smaller batches: `--batch-size 16`

## Hardware Requirements

### Classical ML Models
- **CPU**: Any modern CPU
- **RAM**: 8GB minimum, 16GB recommended
- **Time**: ~10-30 minutes per model

### Deep Learning Models
- **GPU**: Recommended (CUDA-capable)
- **RAM**: 16GB minimum
- **VRAM**: 4GB minimum for batch size 32
- **Time**: 
  - With GPU: ~30-60 minutes per model (20 epochs)
  - With CPU: ~4-8 hours per model (20 epochs)

## Troubleshooting

### Out of Memory (GPU)
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use smaller model versions
- Reduce image size

### Out of Memory (RAM)
- Train one model at a time instead of `--model all`
- Enable PCA for classical models: `--use-pca`
- Reduce number of workers: `--num-workers 2`

### Slow Training
- Enable GPU if available
- Increase number of workers: `--num-workers 8`
- Use smaller models for experiments
- Reduce epochs during development

## Report Template

Your report should include:

1. **Introduction**
   - Brief overview of the task
   - Dataset description

2. **Methods**
   - Classical ML approaches (features + classifiers)
   - Deep learning approaches (architectures)
   - Training details (hyperparameters, augmentation, etc.)

3. **Results**
   - Performance metrics tables
   - Confusion matrices
   - Per-class accuracy plots
   - Training curves (for deep learning)
   - Model comparisons

4. **Ablation Studies**
   - Image size comparison
   - Data augmentation impact
   - Feature extractor comparison
   - Optimizer comparison

5. **Discussion**
   - Best performing models and why
   - Classical ML vs Deep Learning comparison
   - Class imbalance effects
   - Failure cases analysis

6. **Conclusions**
   - Key findings
   - Lessons learned
   - Future improvements

## References

- Caltech-101 Dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/

## Author

SHBT 261 Student
Mini-Project 1: Image Classification with Caltech-101
