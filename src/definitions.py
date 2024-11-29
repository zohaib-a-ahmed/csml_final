import torch
from torchvision.models import densenet121, resnet50, DenseNet121_Weights, ResNet50_Weights
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

        # Increased complexity with more residual blocks and dropout
        self.resblock1 = ResidualBlock(64, 128, 2)
        self.resblock2 = ResidualBlock(128, 256, 2)
        self.resblock3 = ResidualBlock(256, 256, 1)
        self.resblock4 = ResidualBlock(256, 512, 2)
        self.resblock5 = ResidualBlock(512, 512, 1)  # Additional residual block
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer for regularization

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """

        # Normalize the input tensor
        mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(x.device)  # Mean for grayscale
        std = torch.tensor([0.5]).view(1, 1, 1, 1).to(x.device)    # Std for grayscale
        x = (x - mean) / std  # Normalize the input

        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)  # Forward through the additional residual block

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout before the final layer
        x = self.fc(x)

        return x
    
class DenseNetModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(DenseNetModel, self).__init__()
        """
        A pretrained DenseNet model for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        self.model = densenet121(weights = DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)
    
class ResNetModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(ResNetModel, self).__init__()
        """
        A pretrained ResNet model for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)

class DenseNetRawModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(DenseNetRawModel, self).__init__()
        """
        A pretrained DenseNet model for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        self.model = densenet121()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)

class ResNetRawModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(ResNetModel, self).__init__()
        """
        A pretrained ResNet model for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        self.model = resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)