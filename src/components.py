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
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (b, embed_dim, h', w')
        x = x.flatten(2).transpose(1, 2)  # (b, embed_dim, h' * w') -> (b, h' * w', embed_dim)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, 
                 num_heads=12, num_layers=12, num_classes=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        
        # Create transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Create patches and add class token
        x = self.patch_embedding(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # (b, 1 + num_patches, embed_dim)
        x = x + self.pos_embedding  # Add positional embeddings
        
        # Pass through transformer layers
        x = self.transformer_encoder(x)
        
        # Classify using the class token output
        x = self.mlp_head(x[:, 0])  # Use the output of the class token
        return x