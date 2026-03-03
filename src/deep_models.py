"""
Deep Learning Models
Implements ResNet, EfficientNet, and Vision Transformer (ViT) for image classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import time
from PIL import Image


class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(self, image_paths, labels, transform=None, loader_func=None):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            transform: Torchvision transforms
            loader_func: Function to load images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.loader_func = loader_func
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        if self.loader_func is not None:
            image = self.loader_func(img_path)
        else:
            image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DeepLearningModel:
    """Base class for deep learning models"""
    
    def __init__(self, model_name, num_classes, image_size=224):
        """
        Initialize deep learning model
        
        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            image_size: Input image size
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def get_transforms(self, augment=True):
        """
        Get data transforms
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Transform pipeline
        """
        if augment:
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def create_dataloaders(self, splits, batch_size=32, num_workers=4, augment=True):
        """
        Create data loaders
        
        Args:
            splits: Data splits dictionary
            batch_size: Batch size
            num_workers: Number of worker processes
            augment: Whether to augment training data
            
        Returns:
            Dictionary of data loaders
        """
        train_transform = self.get_transforms(augment=augment)
        val_transform = self.get_transforms(augment=False)
        
        # Create datasets
        train_dataset = ImageDataset(
            splits['train'][0], splits['train'][1],
            transform=train_transform
        )
        val_dataset = ImageDataset(
            splits['val'][0], splits['val'][1],
            transform=val_transform
        )
        test_dataset = ImageDataset(
            splits['test'][0], splits['test'][1],
            transform=val_transform
        )
        
        # Pin memory only for CUDA devices (not supported on MPS/CPU)
        use_pin_memory = torch.cuda.is_available()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.001, 
             optimizer_name='adam', save_path=None):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            optimizer_name: Optimizer name ('adam' or 'sgd')
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        print(f"\nTraining {self.model_name}...")
        print(f"Epochs: {epochs}, Learning rate: {lr}, Optimizer: {optimizer_name}")
        
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, 
                                 momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    self.save(save_path)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self, data_loader, criterion=None):
        """
        Evaluate the model
        
        Args:
            data_loader: Data loader
            criterion: Loss function (optional)
            
        Returns:
            Loss and accuracy (if criterion provided), else just accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total
        
        if criterion:
            avg_loss = total_loss / total
            return avg_loss, accuracy
        else:
            return accuracy
    
    def predict(self, data_loader):
        """
        Make predictions
        
        Args:
            data_loader: Data loader
            
        Returns:
            Predictions and probabilities
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        return all_labels, all_preds, all_probs
    
    def save(self, save_path):
        """Save model to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'image_size': self.image_size
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load(self, load_path):
        """Load model from disk"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")


class ResNetModel(DeepLearningModel):
    """ResNet model"""
    
    def __init__(self, num_classes, version='resnet50', pretrained=True):
        super().__init__(f'ResNet-{version[6:]}', num_classes, image_size=224)
        
        # Load pretrained ResNet
        weights = 'DEFAULT' if pretrained else None
        if version == 'resnet18':
            self.model = models.resnet18(weights=weights)
        elif version == 'resnet34':
            self.model = models.resnet34(weights=weights)
        elif version == 'resnet50':
            self.model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown ResNet version: {version}")
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)


class EfficientNetModel(DeepLearningModel):
    """EfficientNet model"""
    
    def __init__(self, num_classes, version='b0', pretrained=True):
        super().__init__(f'EfficientNet-{version.upper()}', num_classes, image_size=224)
        
        # Load pretrained EfficientNet
        weights = 'DEFAULT' if pretrained else None
        if version == 'b0':
            self.model = models.efficientnet_b0(weights=weights)
        elif version == 'b1':
            self.model = models.efficientnet_b1(weights=weights)
        elif version == 'b2':
            self.model = models.efficientnet_b2(weights=weights)
        else:
            raise ValueError(f"Unknown EfficientNet version: {version}")
        
        # Replace final layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)


class ViTModel(DeepLearningModel):
    """Vision Transformer model"""
    
    def __init__(self, num_classes, version='b_16', pretrained=True):
        super().__init__(f'ViT-{version.upper()}', num_classes, image_size=224)
        
        # Load pretrained ViT
        weights = 'DEFAULT' if pretrained else None
        if version == 'b_16':
            self.model = models.vit_b_16(weights=weights)
        elif version == 'b_32':
            self.model = models.vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unknown ViT version: {version}")
        
        # Replace final layer
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)


if __name__ == "__main__":
    # Example usage
    print("Testing deep learning models...")
    
    num_classes = 101
    
    # Test ResNet
    resnet = ResNetModel(num_classes, version='resnet18', pretrained=True)
    print(f"Created {resnet.model_name}")
    
    # Test EfficientNet
    efficientnet = EfficientNetModel(num_classes, version='b0', pretrained=True)
    print(f"Created {efficientnet.model_name}")
    
    # Test ViT
    vit = ViTModel(num_classes, version='b_16', pretrained=True)
    print(f"Created {vit.model_name}")
