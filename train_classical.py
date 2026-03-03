"""
Training Script for Classical Machine Learning Models
Train and evaluate SVM, Random Forest, and KNN on Caltech-101
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_preparation import CaltechDataLoader
from src.classical_models import SVMModel, RandomForestModel, KNNModel, train_classical_model
from src.evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train classical ML models on Caltech-101')
    parser.add_argument('--model', type=str, default='all',
                       choices=['svm', 'rf', 'knn', 'all'],
                       help='Model to train (default: all)')
    parser.add_argument('--feature', type=str, default='hog',
                       choices=['hog', 'raw'],
                       help='Feature type (default: hog)')
    parser.add_argument('--image-size', type=int, default=128,
                       help='Image size (default: 128)')
    parser.add_argument('--use-pca', action='store_true',
                       help='Use PCA for dimensionality reduction')
    parser.add_argument('--n-components', type=int, default=100,
                       help='Number of PCA components (default: 100)')
    parser.add_argument('--data-dir', type=str, default='data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Classical ML Models on Caltech-101")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Feature type: {args.feature}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Use PCA: {args.use_pca}")
    if args.use_pca:
        print(f"  PCA components: {args.n_components}")
    print()
    
    # Load data
    print("Loading Caltech-101 dataset...")
    loader = CaltechDataLoader(data_dir=args.data_dir, 
                              image_size=(args.image_size, args.image_size))
    images, labels, class_names = loader.load_data()
    splits = loader.split_data(images, labels)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names, save_dir='figures')
    
    # Define models to train
    models_to_train = []
    
    if args.model == 'svm' or args.model == 'all':
        models_to_train.append(
            SVMModel(feature_type=args.feature, kernel='rbf', C=1.0)
        )
    
    if args.model == 'rf' or args.model == 'all':
        models_to_train.append(
            RandomForestModel(feature_type=args.feature, n_estimators=100, max_depth=20)
        )
    
    if args.model == 'knn' or args.model == 'all':
        models_to_train.append(
            KNNModel(feature_type=args.feature, n_neighbors=5)
        )
    
    # Train and evaluate models
    all_results = {}
    
    for model in models_to_train:
        # Train model
        trained_model, predictions = train_classical_model(
            model, loader, splits, 
            image_size=(args.image_size, args.image_size),
            use_pca=args.use_pca
        )
        
        # Save model
        model_filename = f"{model.model_name.lower().replace(' ', '_')}_{args.feature}"
        if args.use_pca:
            model_filename += f"_pca{args.n_components}"
        model_filename += ".pkl"
        model_path = Path(args.save_dir) / model_filename
        trained_model.save(model_path)
        
        # Evaluate on test set
        y_test, y_test_pred, y_test_proba = predictions['test']
        
        model_display_name = f"{model.model_name} ({args.feature.upper()})"
        results = evaluator.evaluate_model(
            y_test, y_test_pred, y_test_proba,
            model_name=model_display_name,
            save_prefix='classical_'
        )
        
        all_results[model_display_name] = results
    
    # Compare models
    if len(all_results) > 1:
        from src.evaluation import compare_models
        compare_models(all_results, save_path='figures/classical_models_comparison.png')
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Models saved to: {args.save_dir}/")
    print(f"Figures saved to: figures/")
    print(f"Results saved to: figures/")


if __name__ == "__main__":
    main()
