"""
Evaluation Module
Provides comprehensive metrics and visualizations for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
import json
from pathlib import Path


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, class_names, save_dir='figures'):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names
            save_dir: Directory to save figures
        """
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, 
                      model_name='Model', save_prefix=''):
        """
        Comprehensive model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for top-k accuracy)
            model_name: Name of the model
            save_prefix: Prefix for saved files
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        results = {}
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        results['accuracy'] = accuracy
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Precision, Recall, F1-Score (macro and weighted)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        results.update({
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        })
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall:    {recall_macro:.4f}")
        print(f"  F1-Score:  {f1_macro:.4f}")
        
        print(f"\nWeighted Averages:")
        print(f"  Precision: {precision_weighted:.4f}")
        print(f"  Recall:    {recall_weighted:.4f}")
        print(f"  F1-Score:  {f1_weighted:.4f}")
        
        # Top-k accuracy (if probabilities provided)
        if y_pred_proba is not None:
            for k in [3, 5]:
                if y_pred_proba.shape[1] >= k:
                    top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=k)
                    results[f'top_{k}_accuracy'] = top_k_acc
                    print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
        
        # Per-class accuracy
        per_class_acc = self.compute_per_class_accuracy(y_true, y_pred)
        results['per_class_accuracy'] = per_class_acc
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Visualizations
        self.plot_confusion_matrix(cm, model_name, save_prefix)
        self.plot_per_class_accuracy(per_class_acc, model_name, save_prefix)
        
        # Classification report
        print(f"\nClassification Report:")
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      zero_division=0)
        print(report)
        
        # Save results
        self.save_results(results, model_name, save_prefix)
        
        return results
    
    def compute_per_class_accuracy(self, y_true, y_pred):
        """
        Compute accuracy for each class
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary mapping class names to accuracies
        """
        per_class_acc = {}
        
        for idx, class_name in enumerate(self.class_names):
            # Get samples for this class
            mask = y_true == idx
            if mask.sum() > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                per_class_acc[class_name] = class_acc
            else:
                per_class_acc[class_name] = 0.0
        
        return per_class_acc
    
    def plot_confusion_matrix(self, cm, model_name, save_prefix=''):
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            save_prefix: Prefix for saved file
        """
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].tick_params(axis='x', rotation=90)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[1], vmin=0, vmax=1,
                   cbar_kws={'label': 'Accuracy'})
        axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / f'{save_prefix}confusion_matrix_{model_name.replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_per_class_accuracy(self, per_class_acc, model_name, save_prefix=''):
        """
        Plot per-class accuracy
        
        Args:
            per_class_acc: Dictionary of per-class accuracies
            model_name: Name of the model
            save_prefix: Prefix for saved file
        """
        # Sort by accuracy
        sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1])
        classes = [item[0] for item in sorted_items]
        accuracies = [item[1] for item in sorted_items]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(classes) * 0.3)))
        bars = ax.barh(classes, accuracies, color='steelblue')
        
        # Color bars by accuracy
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc < 0.5:
                bar.set_color('coral')
            elif acc < 0.7:
                bar.set_color('gold')
            else:
                bar.set_color('lightgreen')
        
        ax.set_xlabel('Accuracy')
        ax.set_title(f'{model_name} - Per-Class Accuracy')
        ax.set_xlim(0, 1)
        ax.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(accuracies):.3f}')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / f'{save_prefix}per_class_accuracy_{model_name.replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to {save_path}")
        plt.close()
    
    def save_results(self, results, model_name, save_prefix=''):
        """
        Save results to JSON file
        
        Args:
            results: Dictionary of results
            model_name: Name of the model
            save_prefix: Prefix for saved file
        """
        # Convert numpy types to Python types
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value)
            elif isinstance(value, dict):
                results_serializable[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) 
                                            else v for k, v in value.items()}
            else:
                results_serializable[key] = value
        
        save_path = self.save_dir / f'{save_prefix}results_{model_name.replace(" ", "_")}.json'
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Results saved to {save_path}")


def compare_models(results_dict, save_path='figures/model_comparison.png'):
    """
    Compare multiple models
    
    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save comparison plot
    """
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    model_names = list(results_dict.keys())
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            data[metric].append(results_dict[model_name].get(metric, 0))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics)/2 + 0.5)
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Model comparison saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 100
    n_classes = 10
    
    class_names = [f"Class_{i}" for i in range(n_classes)]
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    evaluator = ModelEvaluator(class_names)
    results = evaluator.evaluate_model(y_true, y_pred, model_name='Test Model')
