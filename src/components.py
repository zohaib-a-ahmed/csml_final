import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (for matching dimensions)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = F.relu(out)
        return out
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Adjusting input channels to 1 for grayscale images
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, 1, height, width)
        x = self.proj(x)  # shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # shape: (batch_size, num_patches, embed_dim)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # First linear layer
            nn.ReLU(),                             # Activation
            nn.Linear(4 * embed_dim, embed_dim)   # Second linear layer
        )
        self.norm1 = nn.LayerNorm(embed_dim)      # Layer normalization after attention
        self.norm2 = nn.LayerNorm(embed_dim)      # Layer normalization after feedforward
        self.dropout1 = nn.Dropout(0.1)           # Dropout after attention
        self.dropout2 = nn.Dropout(0.1)           # Dropout after feedforward

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.self_att(x, x, x)  # x is (batch_size, num_patches, embed_dim)
        x = self.norm1(x + self.dropout1(attn_output))  # Add & normalize

        # Feedforward network with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))  # Add & normalize

        return x