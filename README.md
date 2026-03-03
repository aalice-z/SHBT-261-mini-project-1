# Caltech-101 Image Classification Project

This project implements and compares multiple image classification approaches on the Caltech-101 dataset, including both classical machine learning and deep learning methods.

## Dataset

The Caltech-101 dataset contains approximately 9,000 images across 101 object categories. Download the dataset from [Caltech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and place it in `data/caltech-101/`.

## Project Structure

```
SHBT261-mini-project-1/
├── src/                           # Source code modules
│   ├── data_preparation.py        # Data loading and preprocessing
│   ├── classical_models.py        # Classical ML models (SVM, RF, KNN)
│   ├── deep_models.py             # Deep learning models (ResNet, EfficientNet)
│   └── evaluation.py              # Evaluation metrics and visualization
├── train_classical.py             # Train classical ML models
├── train_deep.py                  # Train deep learning models
├── run_ablation.py                # Run ablation studies
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

Train ResNet-18:

```bash
python train_deep.py --model resnet --epochs 20 --batch-size 32
```

Train EfficientNet-B0:

```bash
python train_deep.py --model efficientnet --epochs 20 --lr 0.001
```

Options:
- `--model`: Model type (`resnet`, `efficientnet`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer (`adam`, `sgd`)
- `--no-augment`: Disable data augmentation

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
1. **ResNet-18** (Residual Network)
   - 18 layers
   - Pretrained on ImageNet
   
2. **EfficientNet-B0**
   - Compound scaling
   - Pretrained on ImageNet

## Outputs

After training, you'll find:

### Models Directory (`models/`)
- Trained model weights will be saved here (.pkl for classical, .pth for deep learning)
- Note: Model files are not included in the repository due to size constraints

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

# 2. Train deep learning models
python train_deep.py --model resnet --epochs 20
python train_deep.py --model efficientnet --epochs 20

# 3. Run ablation studies
python run_ablation.py --study all

# All results will be saved in figures/ directory
```

## For Training

### For Faster Iteration (Development)
- Use smaller image size: `--image-size 64`
- Use fewer epochs: `--epochs 5`
- Reduce batch size if memory limited: `--batch-size 16`

### For Best Performance (Final Results)
- Use larger image size: 224×224 (default for deep learning)
- Train longer: `--epochs 30` or more
- Enable data augmentation (default)

### For Limited Resources
- Use classical models first (faster, less memory)
- Enable PCA: `--use-pca --n-components 100`
- Train on CPU if no GPU available (will be slower)
- Process smaller batches: `--batch-size 16`

## Troubleshooting

### Out of Memory (GPU)
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Reduce image size

### Out of Memory (RAM)
- Train one model at a time instead of `--model all`
- Enable PCA for classical models: `--use-pca`

### Slow Training
- Enable GPU if available
- Reduce epochs during development

## Results

This repository includes experimental results in the `figures/` directory:
- Performance metrics for all models
- Confusion matrices and per-class accuracy plots
- Training curves for deep learning models
- Ablation study comparisons

## Author

Yue (Alice) Zhang
Harvard T.H. Chan School of Public Health
