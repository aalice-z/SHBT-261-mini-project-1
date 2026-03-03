"""
Training Script for Deep Learning Models
Train and evaluate ResNet, EfficientNet, and ViT on Caltech-101
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_preparation import CaltechDataLoader
from src.deep_models import ResNetModel, EfficientNetModel, ViTModel
from src.evaluation import ModelEvaluator


def plot_training_history(history, model_name, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models on Caltech-101')
    parser.add_argument('--model', type=str, default='resnet',
                       choices=['resnet', 'efficientnet', 'vit', 'all'],
                       help='Model to train (default: resnet)')
    parser.add_argument('--version', type=str, default=None,
                       help='Model version (e.g., resnet18, b0, b_16)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer (default: adam)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use pretrained weights')
    parser.add_argument('--data-dir', type=str, default='data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Deep Learning Models on Caltech-101")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    if args.version:
        print(f"  Version: {args.version}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Data augmentation: {not args.no_augment}")
    print(f"  Pretrained: {not args.no_pretrained}")
    print()
    
    # Load data
    print("Loading Caltech-101 dataset...")
    loader = CaltechDataLoader(data_dir=args.data_dir, image_size=(224, 224))
    images, labels, class_names = loader.load_data()
    splits = loader.split_data(images, labels)
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names, save_dir='figures')
    
    # Define models to train
    models_to_train = []
    
    if args.model == 'resnet' or args.model == 'all':
        version = args.version if args.version else 'resnet50'
        models_to_train.append((
            'resnet',
            ResNetModel(num_classes, version=version, pretrained=not args.no_pretrained)
        ))
    
    if args.model == 'efficientnet' or args.model == 'all':
        version = args.version if args.version else 'b0'
        models_to_train.append((
            'efficientnet',
            EfficientNetModel(num_classes, version=version, pretrained=not args.no_pretrained)
        ))
    
    if args.model == 'vit' or args.model == 'all':
        version = args.version if args.version else 'b_16'
        models_to_train.append((
            'vit',
            ViTModel(num_classes, version=version, pretrained=not args.no_pretrained)
        ))
    
    # Train and evaluate models
    all_results = {}
    
    for model_type, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model.model_name}")
        print(f"{'='*60}")
        
        # Create data loaders
        dataloaders = model.create_dataloaders(
            splits, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=not args.no_augment
        )
        
        # Train model
        model_filename = f"{model.model_name.lower().replace('-', '_').replace(' ', '_')}.pth"
        model_path = Path(args.save_dir) / model_filename
        
        history = model.train(
            dataloaders['train'],
            dataloaders['val'],
            epochs=args.epochs,
            lr=args.lr,
            optimizer_name=args.optimizer,
            save_path=model_path
        )
        
        # Plot training history
        history_plot_path = f"figures/deep_{model.model_name.replace(' ', '_')}_history.png"
        plot_training_history(history, model.model_name, history_plot_path)
        
        # Load best model for evaluation
        model.load(model_path)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_test, y_test_pred, y_test_proba = model.predict(dataloaders['test'])
        
        results = evaluator.evaluate_model(
            y_test, y_test_pred, y_test_proba,
            model_name=model.model_name,
            save_prefix='deep_'
        )
        
        all_results[model.model_name] = results
    
    # Compare models
    if len(all_results) > 1:
        from src.evaluation import compare_models
        compare_models(all_results, save_path='figures/deep_models_comparison.png')
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Models saved to: {args.save_dir}/")
    print(f"Figures saved to: figures/")
    print(f"Results saved to: figures/")


if __name__ == "__main__":
    main()
