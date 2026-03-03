"""
Classical Machine Learning Models
Implements SVM, Random Forest, and other classical ML methods with feature extraction
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


class ClassicalModel:
    """Base class for classical ML models"""
    
    def __init__(self, model_name, feature_type='hog'):
        """
        Initialize classical model
        
        Args:
            model_name: Name of the model
            feature_type: Type of features ('hog', 'raw', 'pca')
        """
        self.model_name = model_name
        self.feature_type = feature_type
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_features(self, images, hog_params=None):
        """
        Extract features from images
        
        Args:
            images: Numpy array of images (N, H, W, C)
            hog_params: Parameters for HOG feature extraction
            
        Returns:
            Feature array
        """
        from src.data_preparation import extract_hog_features
        
        if self.feature_type == 'hog':
            if hog_params is None:
                hog_params = {
                    'orientations': 9,
                    'pixels_per_cell': (8, 8),
                    'cells_per_block': (2, 2)
                }
            features = extract_hog_features(images, **hog_params)
            
        elif self.feature_type == 'raw':
            # Flatten images
            features = images.reshape(len(images), -1)
            
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return features
    
    def fit(self, X_train, y_train, use_pca=False, n_components=100):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
        """
        print(f"\nTraining {self.model_name}...")
        print(f"Feature shape: {X_train.shape}")
        
        start_time = time.time()
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA if requested
        if use_pca:
            print(f"Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components, random_state=42)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Training accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        print(f"Training accuracy: {train_acc:.4f}")
        
        return train_time
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature array
            
        Returns:
            Class probabilities
        """
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        elif hasattr(self.model, 'decision_function'):
            # For SVM, use decision function
            decision = self.model.decision_function(X_scaled)
            # Softmax to get probabilities
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)
        else:
            return None
    
    def save(self, save_path):
        """Save model to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_type': self.feature_type,
            'model_name': self.model_name
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {save_path}")
    
    def load(self, load_path):
        """Load model from disk"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.feature_type = model_data['feature_type']
        self.model_name = model_data['model_name']
        print(f"Model loaded from {load_path}")


class SVMModel(ClassicalModel):
    """Support Vector Machine classifier"""
    
    def __init__(self, feature_type='hog', kernel='rbf', C=1.0):
        super().__init__('SVM', feature_type)
        self.model = SVC(kernel=kernel, C=C, probability=True, 
                        random_state=42, verbose=False)


class RandomForestModel(ClassicalModel):
    """Random Forest classifier"""
    
    def __init__(self, feature_type='hog', n_estimators=100, max_depth=None):
        super().__init__('Random Forest', feature_type)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )


class KNNModel(ClassicalModel):
    """K-Nearest Neighbors classifier"""
    
    def __init__(self, feature_type='hog', n_neighbors=5):
        super().__init__('KNN', feature_type)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1
        )


def train_classical_model(model, data_loader, splits, 
                         image_size=(128, 128), use_pca=False):
    """
    Train a classical ML model
    
    Args:
        model: ClassicalModel instance
        data_loader: CaltechDataLoader instance
        splits: Data splits dictionary
        image_size: Image size for loading
        use_pca: Whether to use PCA
        
    Returns:
        Trained model and predictions
    """
    print(f"\n{'='*60}")
    print(f"Training {model.model_name} with {model.feature_type.upper()} features")
    print(f"{'='*60}")
    
    # Load training data
    print("Loading training images...")
    X_train_paths, y_train = splits['train']
    X_train_images = data_loader.load_batch(X_train_paths, target_size=image_size, 
                                           normalize=False)
    
    # Extract features
    print("Extracting features...")
    X_train_features = model.extract_features(X_train_images)
    
    # Train model
    model.fit(X_train_features, y_train, use_pca=use_pca)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    X_val_paths, y_val = splits['val']
    X_val_images = data_loader.load_batch(X_val_paths, target_size=image_size,
                                          normalize=False)
    X_val_features = model.extract_features(X_val_images)
    
    y_val_pred = model.predict(X_val_features)
    y_val_proba = model.predict_proba(X_val_features)
    
    val_acc = (y_val_pred == y_val).mean()
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test_paths, y_test = splits['test']
    X_test_images = data_loader.load_batch(X_test_paths, target_size=image_size,
                                           normalize=False)
    X_test_features = model.extract_features(X_test_images)
    
    y_test_pred = model.predict(X_test_features)
    y_test_proba = model.predict_proba(X_test_features)
    
    test_acc = (y_test_pred == y_test).mean()
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, {
        'val': (y_val, y_val_pred, y_val_proba),
        'test': (y_test, y_test_pred, y_test_proba)
    }


if __name__ == "__main__":
    # Example usage
    from src.data_preparation import CaltechDataLoader
    
    # Load data
    loader = CaltechDataLoader(image_size=(128, 128))
    images, labels, class_names = loader.load_data()
    splits = loader.split_data(images, labels)
    
    # Train SVM with HOG features
    svm_model = SVMModel(feature_type='hog', kernel='rbf', C=1.0)
    trained_model, predictions = train_classical_model(
        svm_model, loader, splits, use_pca=True
    )
    
    # Save model
    trained_model.save('models/svm_hog.pkl')
