import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

class Net(nn.Module):
    """
    CNN architecture for Fashion-MNIST classification.
    
    Design choices:
    - Uses convolutional layers for spatial feature extraction (more appropriate for images)
    - Batch normalization for stable training and faster convergence
    - Dropout for regularization to prevent overfitting
    - Compact design to stay under 100,000 parameter limit
    - Global average pooling reduces parameters compared to fully connected layers
    """
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        # First convolutional block: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Second convolutional block: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Third convolutional block: 7x7 -> 4x4 (with padding)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7 -> 4x4 (with padding)
        
        # Global average pooling: 4x4 -> 1x1, then flatten
        # This reduces parameters significantly compared to fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer
        self.fc = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch_size, 1, 28, 28)
        
        # First block: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)  # (batch_size, 32, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # (batch_size, 32, 14, 14)
        
        # Second block: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)  # (batch_size, 64, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (batch_size, 64, 7, 7)
        
        # Third block: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)  # (batch_size, 128, 7, 7)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # (batch_size, 128, 4, 4) -> (batch_size, 128, 3, 3) after pooling
        
        # Global average pooling: (batch_size, 128, 3, 3) -> (batch_size, 128, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (batch_size, 128, 1, 1) -> (batch_size, 128)
        x = x.view(x.size(0), -1)
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc(x)  # (batch_size, 10)
        
        return x
