import torch
from torchvision.models import vit_b_16
from .components import *


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

class ConvolutionalNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(ConvolutionalNN, self).__init__()
        """
        A convolutional network for image classification.

        Args:
            num_classes: number of output class probabilities
        """

        self.initial_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resblock1 = ResidualBlock(64, 128, 2)
        self.resblock2 = ResidualBlock(128, 256, 2)
        self.resblock3 = ResidualBlock(256, 256, 1)
        self.resblock4 = ResidualBlock(256, 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """

        # # Normalize the input tensor
        # mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(x.device)  # Mean for grayscale
        # std = torch.tensor([0.5]).view(1, 1, 1, 1).to(x.device)    # Std for grayscale
        # x = (x - mean) / std  # Normalize the input

        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        """
        A vision transformer network for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        # Load a pretrained Vision Transformer model
        self.model = vit_b_16()  # Load pretrained weights
        
        # Modify the classifier head to match the number of classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        # Since the pretrained model expects 3 channels, we need to repeat the grayscale channel
        x = x.repeat(1, 3, 1, 1)  # Convert (b, 1, h, w) to (b, 3, h, w)
        
        # Forward pass through the model
        return self.model(x)