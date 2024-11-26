import torch
from .components import *

import torch

class LinearExample(torch.nn.Module):
    def __init__(self, h: int = 224, w: int = 224, num_classes: int = 4):
        super().__init__()
        """
        A simple linear model for classification.

        Args:
            h: height of the input images
            w: width of the input images
            num_classes: number of output class probabilities
        """
        # Calculate the input size based on the height and width
        input_size = h * w  # Since input is (b, 1, h, w), we flatten it to (b, h*w)
        
        # Define a linear layer
        self.linear = torch.nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (b, 1, h, w)

        Returns:
            tensor of shape (b, num_classes) representing class probabilities
        """
        # Flatten the input tensor from (b, 1, h, w) to (b, h*w)
        x = x.view(x.size(0), -1)  # Flatten to (b, h*w)
        return self.linear(x)


class ConvolutionalNN(torch.nn.Module):

    def __init__(self, num_classes: int = 4):
        super().__init__()
        """
        A convolutional network for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        # define layers
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        pass

class VisionTransformer(torch.nn.Module):

    def __init__(self, num_classes: int = 4):
        super().__init__()
        """
        A vision transformer network for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        # define layers
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        pass