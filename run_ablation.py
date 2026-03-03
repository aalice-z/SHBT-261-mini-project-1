"""
Ablation Studies Script
Run systematic experiments to understand the impact of different hyperparameters
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_preparation import CaltechDataLoader
from src.classical_models import RandomForestModel, train_classical_model
from src.deep_models import ResNetModel
from src.evaluation import ModelEvaluator


def plot_ablation_results(results, study_name, save_path):
    """
    Plot ablation study results
    
    Args:
        results: Dictionary of results for each configuration
        study_name: Name of the ablation study
        save_path: Path to save the plot
    """
    configs = list(results.keys())
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results[config].get(metric, 0) for config in configs]
        
        bars = axes[idx].bar(range(len(configs)), values, color='steelblue', alpha=0.7)
        
        # Color the best bar
        best_idx = np.argmax(values)
        bars[best_idx].set_color('green')
        
        axes[idx].set_xticks(range(len(configs)))
        axes[idx].set_xticklabels(configs, rotation=45, ha='right')
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(f'{study_name} - {metric_name}')
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Annotate values
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Ablation Study: {study_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Ablation results saved to {save_path}")
    plt.close()


def ablation_image_size(loader, class_names, model_type='classical'):
    """
    Ablation study: Compare different image sizes
    
    Args:
        loader: Data loader
        class_names: List of class names
        model_type: 'classical' or 'deep'
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Image Size")
    print("="*60)
    
    image_sizes = [64, 128]
    results = {}
    
    for size in image_sizes:
        print(f"\n--- Testing image size: {size}x{size} ---")
        
        # Reload data with new size
        loader.image_size = (size, size)
        images, labels, _ = loader.load_data()
        splits = loader.split_data(images, labels)
        
        evaluator = ModelEvaluator(class_names, save_dir='figures')
        
        if model_type == 'classical':
            # Train Random Forest with HOG features
            model = RandomForestModel(feature_type='hog', n_estimators=100)
            trained_model, predictions = train_classical_model(
                model, loader, splits,
                image_size=(size, size),
                use_pca=True
            )
            
            y_test, y_test_pred, y_test_proba = predictions['test']
            
        else:  # deep learning
            # Train ResNet
            model = ResNetModel(len(class_names), version='resnet18', pretrained=True)
            dataloaders = model.create_dataloaders(
                splits, batch_size=32, num_workers=4, augment=True
            )
            
            model_path = f'models/ablation_resnet_size{size}.pth'
            model.train(dataloaders['train'], dataloaders['val'],
                       epochs=10, lr=0.001, optimizer_name='adam',
                       save_path=model_path)
            
            y_test, y_test_pred, y_test_proba = model.predict(dataloaders['test'])
        
        # Evaluate
        config_name = f'{size}x{size}'
        model_results = evaluator.evaluate_model(
            y_test, y_test_pred, y_test_proba,
            model_name=f'Size_{size}',
            save_prefix=f'ablation_size{size}_'
        )
        
        results[config_name] = model_results
    
    # Plot comparison
    plot_ablation_results(
        results, 'Image Size Comparison',
        'figures/ablation_image_size.png'
    )
    
    # Save results
    with open('figures/ablation_image_size_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def ablation_data_augmentation(loader, class_names):
    """
    Ablation study: With vs without data augmentation (deep learning)
    
    Args:
        loader: Data loader
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Data Augmentation")
    print("="*60)
    
    images, labels, _ = loader.load_data()
    splits = loader.split_data(images, labels)
    
    augmentation_configs = [
        ('With Augmentation', True),
        ('Without Augmentation', False)
    ]
    
    results = {}
    evaluator = ModelEvaluator(class_names, save_dir='figures')
    
    for config_name, use_augment in augmentation_configs:
        print(f"\n--- Testing: {config_name} ---")
        
        # Train ResNet
        model = ResNetModel(len(class_names), version='resnet18', pretrained=True)
        dataloaders = model.create_dataloaders(
            splits, batch_size=32, num_workers=4, augment=use_augment
        )
        
        model_path = f'models/ablation_resnet_augment_{use_augment}.pth'
        model.train(dataloaders['train'], dataloaders['val'],
                   epochs=10, lr=0.001, optimizer_name='adam',
                   save_path=model_path)
        
        # Evaluate
        y_test, y_test_pred, y_test_proba = model.predict(dataloaders['test'])
        
        model_results = evaluator.evaluate_model(
            y_test, y_test_pred, y_test_proba,
            model_name=config_name.replace(' ', '_'),
            save_prefix=f'ablation_augment_{use_augment}_'
        )
        
        results[config_name] = model_results
    
    # Plot comparison
    plot_ablation_results(
        results, 'Data Augmentation Impact',
        'figures/ablation_augmentation.png'
    )
    
    # Save results
    with open('figures/ablation_augmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def ablation_feature_extractor(loader, class_names):
    """
    Ablation study: HOG vs Raw features (classical ML)
    
    Args:
        loader: Data loader
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Feature Extractor")
    print("="*60)
    
    loader.image_size = (128, 128)
    images, labels, _ = loader.load_data()
    splits = loader.split_data(images, labels)
    
    feature_types = ['hog', 'raw']
    results = {}
    evaluator = ModelEvaluator(class_names, save_dir='figures')
    
    for feature_type in feature_types:
        print(f"\n--- Testing feature type: {feature_type.upper()} ---")
        
        # Train Random Forest
        model = RandomForestModel(feature_type=feature_type, n_estimators=100)
        trained_model, predictions = train_classical_model(
            model, loader, splits,
            image_size=(128, 128),
            use_pca=True
        )
        
        y_test, y_test_pred, y_test_proba = predictions['test']
        
        # Evaluate
        config_name = feature_type.upper()
        model_results = evaluator.evaluate_model(
            y_test, y_test_pred, y_test_proba,
            model_name=f'Feature_{feature_type}',
            save_prefix=f'ablation_feature_{feature_type}_'
        )
        
        results[config_name] = model_results
    
    # Plot comparison
    plot_ablation_results(
        results, 'Feature Extractor Comparison',
        'figures/ablation_features.png'
    )
    
    # Save results
    with open('figures/ablation_features_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def ablation_optimizer(loader, class_names):
    """
    Ablation study: SGD vs Adam optimizer (deep learning)
    
    Args:
        loader: Data loader
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Optimizer")
    print("="*60)
    
    images, labels, _ = loader.load_data()
    splits = loader.split_data(images, labels)
    
    optimizers = ['adam', 'sgd']
    results = {}
    evaluator = ModelEvaluator(class_names, save_dir='figures')
    
    for optimizer_name in optimizers:
        print(f"\n--- Testing optimizer: {optimizer_name.upper()} ---")
        
        # Train ResNet
        model = ResNetModel(len(class_names), version='resnet18', pretrained=True)
        dataloaders = model.create_dataloaders(
            splits, batch_size=32, num_workers=4, augment=True
        )
        
        model_path = f'models/ablation_resnet_{optimizer_name}.pth'
        model.train(dataloaders['train'], dataloaders['val'],
                   epochs=10, lr=0.001, optimizer_name=optimizer_name,
                   save_path=model_path)
        
        # Evaluate
        y_test, y_test_pred, y_test_proba = model.predict(dataloaders['test'])
        
        config_name = optimizer_name.upper()
        model_results = evaluator.evaluate_model(
            y_test, y_test_pred, y_test_proba,
            model_name=f'Optimizer_{optimizer_name}',
            save_prefix=f'ablation_optimizer_{optimizer_name}_'
        )
        
        results[config_name] = model_results
    
    # Plot comparison
    plot_ablation_results(
        results, 'Optimizer Comparison',
        'figures/ablation_optimizer.png'
    )
    
    # Save results
    with open('figures/ablation_optimizer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies on Caltech-101')
    parser.add_argument('--study', type=str, default='all',
                       choices=['image_size', 'augmentation', 'features', 'optimizer', 'all'],
                       help='Ablation study to run (default: all)')
    parser.add_argument('--data-dir', type=str, default='data/caltech-101',
                       help='Path to Caltech-101 dataset')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Running Ablation Studies on Caltech-101")
    print("="*60)
    
    # Load data
    print("Loading Caltech-101 dataset...")
    loader = CaltechDataLoader(data_dir=args.data_dir, image_size=(128, 128))
    images, labels, class_names = loader.load_data()
    
    all_results = {}
    
    # Run selected ablation studies
    if args.study in ['image_size', 'all']:
        all_results['image_size'] = ablation_image_size(loader, class_names, model_type='classical')
    
    if args.study in ['augmentation', 'all']:
        all_results['augmentation'] = ablation_data_augmentation(loader, class_names)
    
    if args.study in ['features', 'all']:
        all_results['features'] = ablation_feature_extractor(loader, class_names)
    
    if args.study in ['optimizer', 'all']:
        all_results['optimizer'] = ablation_optimizer(loader, class_names)
    
    print("\n" + "="*60)
    print("Ablation Studies Completed!")
    print("="*60)
    print(f"Results saved to: figures/")
    
    # Print summary
    print("\nSummary of Best Configurations:")
    for study_name, study_results in all_results.items():
        best_config = max(study_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"  {study_name}: {best_config[0]} (Accuracy: {best_config[1]['accuracy']:.4f})")


if __name__ == "__main__":
    main()
