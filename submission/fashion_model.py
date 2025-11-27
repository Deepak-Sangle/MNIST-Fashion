import torch
import torch.nn as nn


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

        # Convolutional blocks (Conv -> BN -> ReLU -> MaxPool)
        # 28x28 -> 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 14x14 -> 7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 7x7 -> 4x4 (with padding, effectively 3x3 after pooling)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

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

        # Convolutional feature extractor
        x = self.block1(x)  # (batch_size, 32, 14, 14)
        x = self.block2(x)  # (batch_size, 64, 7, 7)
        x = self.block3(x)  # (batch_size, 128, 3, 3)

        # Global average pooling: (batch_size, 128, 3, 3) -> (batch_size, 128, 1, 1)
        x = self.avgpool(x)
        # Flatten: (batch_size, 128, 1, 1) -> (batch_size, 128)
        x = x.view(x.size(0), -1)
        # Dropout for regularization
        x = self.dropout(x)
        # Final classification layer
        x = self.fc(x)  # (batch_size, 10)

        return x
