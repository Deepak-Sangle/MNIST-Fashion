"""
Training script for Fashion-MNIST model with cross-validation and hyperparameter tuning.

This training script uses imports relative to the base directory (assignment/).
To run this training script with uv, ensure you're in the root directory (assignment/)
and execute: uv run -m submission.fashion_training
"""
import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from submission import engine
from submission.fashion_model import Net

# Global variable to expose the most recent best validation accuracy from train_fashion_model
# I need to introduce this global variable because as per rules I cannot change the return value of train_fashion_model() function
last_train_best_val_acc: float = 0.0

def get_device(USE_GPU=True):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    elif USE_GPU and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def train_on_each_fold(train_dataset, val_dataset, device, batch_size, learning_rate, weight_decay, n_epochs):
    """
    Helper to train a model on a single train/validation split for a k-fold cross-validation.
    
    Returns:
        best_state_dict, best_val_accuracy
    """
    # Create data loaders
    num_workers = 4 if device.type != "cpu" else 0
    pin_memory = True if device.type != 'cpu' else False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Initialize model, loss function, and optimizer
    model = Net()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    
    # Use Adam optimizer with weight decay for regularization
    # New model uses SGD (vs Adam) 
    # SGD doesn't work that well, Adam again :(
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )

    # Training loop
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training phase
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_accuracy = engine.eval(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{n_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save best model based on validation accuracy
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  -> New best validation accuracy: {best_val_accuracy:.4f}")
    
    return best_model_state, best_val_accuracy


def train_fashion_model(fashion_mnist,
                        n_epochs,
                        batch_size: int = 64,
                        learning_rate: float = 0.1,
                        USE_GPU: bool = True,
                        weight_decay: float = 0.0,
                        k_folds: int = 1):
    """
    Train the Fashion-MNIST model using K-fold cross-validation.
    
    (You can change the default values or add additional keyword arguments if needed.)
    
    Args:
        fashion_mnist: Fashion-MNIST dataset
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        USE_GPU: Whether to use GPU if available
        weight_decay: L2 regularization weight decay
        k_folds: Number of folds for cross-validation (default to 1 which means no k-fold cv)
    
    Returns:
        state_dict: Model's state dictionary (weights)
    """

    global last_train_best_val_acc

    # Optionally use GPU if available
    device = get_device(USE_GPU)

    # K-fold cross-validation (OR single train/val split if k_folds <= 1)
    num_samples = len(fashion_mnist)
    k_folds_int = int(k_folds)

    # If k_folds is 1 or less, use direct train/validation split (no K-fold)
    # We do this during final training
    if k_folds_int <= 1:
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        train_data, val_data = torch.utils.data.random_split(
            fashion_mnist,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        best_model_state, best_val_acc = train_on_each_fold(
            train_data,
            val_data,
            device,
            batch_size,
            learning_rate,
            weight_decay,
            n_epochs,
        )
        last_train_best_val_acc = float(best_val_acc)

        return best_model_state

    # Standard K-fold cross-validation (k_folds >= 2)
    k_folds = max(2, min(k_folds_int, num_samples))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_overall_accuracy = 0.0
    best_overall_state = None

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(fashion_mnist), start=1):
        print(
            f"\nStarting fold {fold_idx}/{k_folds} "
            f"(train={len(train_indices)}, val={len(val_indices)})"
        )

        train_subset = torch.utils.data.Subset(fashion_mnist, train_indices.tolist())
        val_subset = torch.utils.data.Subset(fashion_mnist, val_indices.tolist())

        fold_state, fold_best_acc = train_on_each_fold(train_subset, val_subset, device, batch_size, learning_rate, weight_decay, n_epochs)
        print(f"Fold {fold_idx}/{k_folds} best validation accuracy: {fold_best_acc:.4f}")
        
        if fold_best_acc >= best_overall_accuracy:
            best_overall_accuracy = fold_best_acc
            best_overall_state = fold_state
    
    last_train_best_val_acc = float(best_overall_accuracy)

    # Return the best model's state_dict (weights) across folds
    return best_overall_state

def get_transforms(mode='train'):
    """
    Defines data augmentations and preprocessing transforms.
    
    Args:
        mode: 'train' for training (with augmentation) or 'eval' for evaluation (deterministic)
    
    Returns:
        Compose: torchvision transforms composition
    """
    if mode == 'train':
        # Training transforms with data augmentation
        tfs = torchvision.transforms.Compose([
            # Convert to tensor
            torchvision.transforms.ToTensor(),
            # Random horizontal flip
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # Small random rotation for robustness
            torchvision.transforms.RandomRotation(degrees=5),
            # Normalize to have zero mean and unit variance
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif mode == 'eval':
        # Evaluation transforms: deterministic, no augmentation
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # Same normalization as training for consistency
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # Ensure all transforms are in eval mode (deterministic)
        for tf in tfs.transforms:
            if hasattr(tf, 'train'):
                tf.eval()  # set to eval mode if applicable # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.ipynb! 
    # To use them for training or inference, we need to transform them to tensors. 
    # We set this transform here, as well as any other data preprocessing or augmentation you 
    # wish to apply.
    fashion_mnist.transform = get_transforms(mode='train')
    return fashion_mnist


def main():
    """
    Main training function with hyperparameter tuning and cross-validation.
    
    This function:
    1. Loads the Fashion-MNIST dataset
    2. Performs hyperparameter search using validation set
    3. Saves the best model weights from hyperparameter search
    """
    print("-" * 60)
    print("Fashion-MNIST Model Training")
    print("-" * 60)
    
    # Load training data
    fashion_mnist = load_training_data()
    print(f"Loaded {len(fashion_mnist)} training samples")
    
    # Hyperparameter search space
    hyperparams = [
        {'batch_size': 64, 'learning_rate': 0.001, 'weight_decay': 1e-4, 'n_epochs': 20},
        # {'batch_size': 128, 'learning_rate': 0.1, 'weight_decay': 1e-4, 'n_epochs': 20},
        # {'batch_size': 512, 'learning_rate': 0.005, 'weight_decay': 1e-4, 'n_epochs': 30},
    ]
    
    print("\n" + "-" * 60)
    print("Hyperparameter Search")
    print("-" * 60)
    
    best_accuracy = 0.0
    best_hyperparams = None
    # Number of folds for cross-validation during hyperparameter search/final training
    k_folds = 5
    
    # Hyperparameter search using k-fold cross-validation
    for i, params in enumerate(hyperparams):
        print(f"\n--- Hyperparameter Set {i+1}/{len(hyperparams)} ---")
        print(f"Batch size: {params['batch_size']}, "
              f"Learning rate: {params['learning_rate']}, "
              f"Weight decay: {params['weight_decay']}, "
              f"Epochs: {params['n_epochs']}")
        
        # Train with these hyperparameters (K-fold inside train_fashion_model)
        _ = train_fashion_model(
            fashion_mnist,
            n_epochs=params['n_epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            USE_GPU=True,
            k_folds=k_folds,
        )

        # Use the best validation accuracy recorded during K-fold training
        val_accuracy = last_train_best_val_acc
        print(f"Validation Accuracy for this hyperparameter set: {val_accuracy:.4f}")
        
        if val_accuracy >= best_accuracy:
            best_accuracy = float(val_accuracy)
            best_hyperparams = params
            print(f"  -> New best hyperparameters!")

    print("\n" + "-" * 60)
    print("Best Hyperparameters Summary")
    print("-" * 60)
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    # Train final model with best hyperparameters for more epochs
    print("\nTraining final model with best hyperparameters for 50 epochs...")
    final_epochs = 50
    final_weights = train_fashion_model(
        # Use the full dataset for final training
        fashion_mnist,
        n_epochs=final_epochs,
        batch_size=best_hyperparams['batch_size'],
        learning_rate=best_hyperparams['learning_rate'],
        weight_decay=best_hyperparams['weight_decay'],
        USE_GPU=True,
        # No need to use k-fold finally, train with larger epoch and complete dataset
        k_folds=1,
    )
    
    # Save model weights
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(final_weights, f=model_save_path)
    print(f"\nModel weights saved to {model_save_path}")
    
    # Print model parameter count
    model = Net()
    model.load_state_dict(final_weights)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Parameter limit: 100,000")
    print(f"Within limit: {'Yes' if num_params <= 100000 else 'No'}")
    
    print("\n" + "-" * 60)
    print("Training Complete!")
    print("-" * 60)

if __name__ == "__main__":
    main()
