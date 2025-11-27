import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetCNN(nn.Module):
    """
    CNN architecture for Fashion-MNIST classification.
    This model has 94,410 trainable parameters
    This is the best performing model from my experiments.

    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

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
        self.dropout = nn.Dropout(0.15)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.fc(x)  # (batch_size, num_classes)

        return x


class CnnDropout3(nn.Module):
    """
    CNN architecture (LeNet-style with dropout).
    This model has 44,426 trainable parameters

    This is a balanced model between performance and complexity.
    """

    def __init__(
        self,
        num_classes: int = 10,
        conv_dropout: float = 0.18,
        fc_dropout: float = 0.25,
    ):
        super().__init__()

        # 28x28x1 -> 24x24x6
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding=0,
        )
        # Shared max-pooling for both conv blocks: 2x2, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 12x12x6 -> 8x8x16 (after pool from conv1)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            padding=0,
        )

        # Dropout after each pooling operation
        self.dropout_conv = nn.Dropout(p=conv_dropout)

        # Fully connected layers
        # After second conv + pool we have 4x4x16 = 256 features
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Dropout for fully connected layers
        self.dropout_fc = nn.Dropout(p=fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.conv1(x))     # (N, 6, 24, 24)
        x = self.pool(x)              # (N, 6, 12, 12)
        x = self.dropout_conv(x)

        # Conv block 2
        x = F.relu(self.conv2(x))     # (N, 16, 8, 8)
        x = self.pool(x)              # (N, 16, 4, 4)
        x = self.dropout_conv(x)

        # Flatten
        x = torch.flatten(x, 1)       # (N, 256)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))       # (N, 120)
        x = self.dropout_fc(x)

        x = F.relu(self.fc2(x))       # (N, 84)
        x = self.dropout_fc(x)

        # Final logits (softmax handled by CrossEntropyLoss / eval)
        x = self.fc3(x)               # (N, num_classes)

        return x


class CnnBasic(nn.Module):
    """
    Cnn-simple: compact CNN with two convolutions followed by a fully
    connected layer

    This model has 110,968 trainable parameters.

    This is a basic model that is not very good.
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.25):
        super().__init__()

        # Convolutional part
        # 28x28x1 -> 24x24x6
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding=0,
        )
        # 24x24x6 -> 20x20x12
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=5,
            padding=0,
        )

        # Max pooling: 20x20x12 -> 10x10x12
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout (p=0.25) used after pooling and first dense layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected part
        # After pooling: feature map size is 10x10x12 = 1200
        self.fc1 = nn.Linear(12 * 10 * 10, 90)
        self.fc2 = nn.Linear(90, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = F.relu(self.conv1(x))  # (N, 6, 24, 24)
        x = F.relu(self.conv2(x))  # (N, 12, 20, 20)

        # Max pooling + dropout
        x = self.pool(x)           # (N, 12, 10, 10)
        x = self.dropout(x)

        # Flatten
        x = torch.flatten(x, 1)    # (N, 1200)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))    # (N, 90)
        x = self.dropout(x)
        x = self.fc2(x)            # (N, num_classes)

        # Apply softmax at last
        x = F.softmax(x, dim=1)

        return x
