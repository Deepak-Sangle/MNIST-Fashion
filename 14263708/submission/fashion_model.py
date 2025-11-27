import torch
import torch.nn as nn

from submission.models import ConvNetCNN, CnnDropout3, CnnBasic

class Net(nn.Module):
    """
    Wrapper model that allows selecting between different underlying architectures.

    The actual architectures live in `submission.models`:
    - `ConvNetCNN`: CNN with three conv blocks + global average pooling + output
    - `CnnDropout3`: CNN with two conv blocks + two FC layers + output
    - `CnnBasic`: CNN with two conv blocks + output
    """

    def __init__(self, num_classes: int = 10, model_name: str = "cnn"):
        super().__init__()

        if model_name == "cnn":
            self.model = ConvNetCNN(num_classes=num_classes)
        elif model_name == "dropout3":
            self.model = CnnDropout3(num_classes=num_classes)
        elif model_name == "basic":
            self.model = CnnBasic(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model_type '{model_name}'. Use 'cnn', 'dropout3', or 'basic'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
