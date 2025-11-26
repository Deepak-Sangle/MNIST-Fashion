"""
Training and evaluation engine for Fashion-MNIST model.
This module provides helper functions for training and evaluating the model.
"""

import numpy as np
import torch

def train(model, data_loader, criterion, optimizer, device, max_grad_norm=1.0):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run training on (cpu/cuda)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    
    Returns:
        float: Average training loss for the epoch
    """
    model.to(device)
    model.train()

    epoch_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    return avg_loss


def eval(model, data_loader, criterion, device):
    model.to(device)
    model.eval()

    targets = []
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            targets.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            total_loss += loss.item()

    accuracy = (np.array(predictions) == np.array(targets)).mean()
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy

