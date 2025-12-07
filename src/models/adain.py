"""
Adaptive Instance Normalization (AdaIN) layer.
"""

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    Transfers style from style features to content features.
    """
    
    def __init__(self, eps=1e-5):
        """
        Args:
            eps: Small value for numerical stability
        """
        super(AdaIN, self).__init__()
        self.eps = eps
    
    def calc_mean_std(self, features):
        """
        Calculate channel-wise mean and std.
        
        Args:
            features: (B, C, H, W)
        
        Returns:
            mean: (B, C, 1, 1)
            std: (B, C, 1, 1)
        """
        batch_size, channels = features.size()[:2]
        
        # Reshape to (B, C, H*W)
        features_reshaped = features.view(batch_size, channels, -1)
        
        # Calculate mean and std per channel
        mean = features_reshaped.mean(dim=2, keepdim=True).unsqueeze(3)
        std = features_reshaped.std(dim=2, keepdim=True).unsqueeze(3) + self.eps
        
        return mean, std
    
    def forward(self, content_features, style_features):
        """
        Apply AdaIN.
        
        Args:
            content_features: Content image features (B, C, H, W)
            style_features: Style image features (B, C, H, W)
        
        Returns:
            Stylized features (B, C, H, W)
        """
        # Calculate statistics
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        
        # Normalize content features
        normalized = (content_features - content_mean) / content_std
        
        # Apply style statistics
        stylized = normalized * style_std + style_mean
        
        return stylized


def test_adain():
    """Test AdaIN layer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adain = AdaIN().to(device)
    
    # Test inputs
    content = torch.randn(2, 512, 32, 32).to(device)
    style = torch.randn(2, 512, 32, 32).to(device)
    
    output = adain(content, style)
    
    print("AdaIN Test:")
    print(f"  Content: {content.shape}")
    print(f"  Style: {style.shape}")
    print(f"  Output: {output.shape}")
    
    # Verify that output has style statistics
    def get_stats(x):
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        return mean, std
    
    style_mean, style_std = get_stats(style)
    output_mean, output_std = get_stats(output)
    
    print(f"\n  Style mean: {style_mean[0, :5]}")
    print(f"  Output mean: {output_mean[0, :5]}")
    print(f"  Match: {torch.allclose(style_mean, output_mean, atol=1e-5)}")


if __name__ == '__main__':
    test_adain()
