"""
Data Preparation Module for Caltech-101 Dataset
Handles loading, splitting, and preprocessing of images
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from collections import defaultdict


class CaltechDataLoader:
    """Load and prepare Caltech-101 dataset"""
    
    def __init__(self, data_dir='data/caltech-101', image_size=(128, 128)):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to Caltech-101 dataset
            image_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.classes = []
        self.class_to_idx = {}
        
    def load_data(self, min_samples_per_class=40):
        """
        Load all images and labels from the dataset
        
        Args:
            min_samples_per_class: Minimum number of samples required per class
            
        Returns:
            images: List of image paths
            labels: List of corresponding class indices
            class_names: List of class names
        """
        print(f"Loading data from {self.data_dir}")
        
        # Get all category directories (excluding BACKGROUND_Google)
        all_classes = sorted([d.name for d in self.data_dir.iterdir() 
                             if d.is_dir() and d.name != 'BACKGROUND_Google'])
        
        # Collect images per class
        class_images = defaultdict(list)
        
        for class_name in all_classes:
            class_dir = self.data_dir / class_name
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
                image_files.extend(class_dir.glob(ext))
            
            if len(image_files) >= min_samples_per_class:
                class_images[class_name] = image_files
        
        # Create class mapping
        self.classes = sorted(class_images.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        images = []
        labels = []
        
        for class_name, image_files in class_images.items():
            class_idx = self.class_to_idx[class_name]
            for img_path in image_files:
                images.append(str(img_path))
                labels.append(class_idx)
        
        print(f"Loaded {len(images)} images from {len(self.classes)} classes")
        print(f"Classes: {self.classes[:5]}... (showing first 5)")
        
        return np.array(images), np.array(labels), self.classes
    
    def split_data(self, images, labels, train_ratio=0.7, val_ratio=0.15, 
                   test_ratio=0.15, random_state=42):
        """
        Split data into train, validation, and test sets (stratified)
        
        Args:
            images: Array of image paths
            labels: Array of labels
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, 
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_state
        )
        
        # Second split: val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_ratio,
            stratify=y_temp,
            random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def load_image(self, image_path, target_size=None):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image
            target_size: Target size (height, width), uses self.image_size if None
            
        Returns:
            Preprocessed image as numpy array
        """
        if target_size is None:
            target_size = self.image_size
            
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def load_batch(self, image_paths, target_size=None, normalize=True):
        """
        Load a batch of images
        
        Args:
            image_paths: List of image paths
            target_size: Target size (height, width)
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Numpy array of images (N, H, W, C)
        """
        images = []
        for path in image_paths:
            img = self.load_image(path, target_size)
            if img is not None:
                images.append(img)
        
        images = np.array(images)
        
        if normalize:
            images = images.astype(np.float32) / 255.0
        
        return images


def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2)):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images
    
    Args:
        images: Numpy array of images (N, H, W, C)
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell
        cells_per_block: Number of cells in each block
        
    Returns:
        HOG features array (N, feature_dim)
    """
    from skimage.feature import hog
    
    features = []
    for img in images:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Extract HOG features
        hog_feat = hog(gray, 
                      orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      visualize=False,
                      transform_sqrt=True,
                      block_norm='L2-Hys')
        features.append(hog_feat)
    
    return np.array(features)


def augment_image(image):
    """
    Apply data augmentation to an image
    
    Args:
        image: Input image (H, W, C)
        
    Returns:
        Augmented image
    """
    import random
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation (-15 to 15 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness adjustment
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    return image


if __name__ == "__main__":
    # Example usage
    loader = CaltechDataLoader()
    images, labels, class_names = loader.load_data()
    splits = loader.split_data(images, labels)
    
    print(f"\nTotal classes: {len(class_names)}")
    print(f"Sample class names: {class_names[:10]}")
