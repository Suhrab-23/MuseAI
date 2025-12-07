"""
Decoder network for reconstructing stylized images.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with optional normalization"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        ]
        
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """
    Decoder network to reconstruct images from AdaIN features.
    Mirrors VGG encoder structure (inverted).
    """
    
    def __init__(self, channels=[512, 256, 128, 64]):
        """
        Args:
            channels: List of channel dimensions for each upsampling block
        """
        super(Decoder, self).__init__()
        
        self.channels = channels
        
        # Build decoder blocks
        decoder_layers = []
        
        # First block: 512 -> 256
        decoder_layers.extend([
            ConvBlock(512, 256, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1)
        ])
        
        # Second block: 256 -> 128
        decoder_layers.extend([
            ConvBlock(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 128, 3, 1, 1)
        ])
        
        # Third block: 128 -> 64
        decoder_layers.extend([
            ConvBlock(128, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(64, 64, 3, 1, 1)
        ])
        
        # Final block: 64 -> 3 (RGB)
        decoder_layers.extend([
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        Reconstruct image from features.
        
        Args:
            x: Feature tensor (B, 512, H/8, W/8)
        
        Returns:
            Reconstructed image (B, 3, H, W) with values in [0, 1]
        """
        x = self.decoder(x)
        
        # Sigmoid to ensure [0, 1] range
        x = torch.sigmoid(x)
        
        return x


def test_decoder():
    """Test decoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = Decoder().to(device)
    
    # Test input (features from relu4_1, which is 1/8 resolution)
    x = torch.randn(2, 512, 64, 64).to(device)
    
    output = decoder(x)
    
    print("Decoder Test:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters: {total_params:,}")


if __name__ == '__main__':
    test_decoder()
