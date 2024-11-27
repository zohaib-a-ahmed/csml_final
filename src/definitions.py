import torch
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

class VisionTransformer(torch.nn.Module):

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768, num_heads: int = 12, num_layers: int = 4, num_classes: int = 4):
        super().__init__()
        """
        A vision transformer network for image classification.

        Args:
            num_classes: number of output class probabilities
        """
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 1, h, w) image

        Returns:
            tensor (b, num_classes) classifications
        """
        # Embed patches
        x = self.patch_embedding(x)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Prepend the classification token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Use the output corresponding to the CLS token for classification
        cls_output = x[:, 0]  # Shape: (batch_size, embed_dim)
        x = self.classifier(cls_output)  # Shape: (batch_size, num_classes)

        return x