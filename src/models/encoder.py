"""
VGG19 Encoder for feature extraction.
"""

import torch
import torch.nn as nn
from torchvision import models


class VGGEncoder(nn.Module):
    """
    VGG19 encoder for extracting features at multiple layers.
    Pre-trained on ImageNet and frozen during training.
    """
    
    def __init__(self, layer_names=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']):
        """
        Args:
            layer_names: List of layer names to extract features from
        """
        super(VGGEncoder, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Map layer names to indices
        self.layer_mapping = {
            'relu1_1': 1,
            'relu1_2': 3,
            'relu2_1': 6,
            'relu2_2': 8,
            'relu3_1': 11,
            'relu3_2': 13,
            'relu3_3': 15,
            'relu3_4': 17,
            'relu4_1': 20,
            'relu4_2': 22,
            'relu4_3': 24,
            'relu4_4': 26,
            'relu5_1': 29,
            'relu5_2': 31,
            'relu5_3': 33,
            'relu5_4': 35
        }
        
        self.layer_names = layer_names
        self.layers = nn.ModuleList()
        
        # Build sequential blocks up to each target layer
        prev_idx = 0
        for layer_name in layer_names:
            if layer_name not in self.layer_mapping:
                raise ValueError(f"Unknown layer name: {layer_name}")
            
            layer_idx = self.layer_mapping[layer_name]
            self.layers.append(vgg[prev_idx:layer_idx + 1])
            prev_idx = layer_idx + 1
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input with ImageNet stats"""
        return (x - self.mean) / self.std
    
    def forward(self, x):
        """
        Extract features from multiple layers.
        
        Args:
            x: Input tensor (B, 3, H, W) with values in [0, 1]
        
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Normalize input
        x = self.normalize(x)
        
        features = {}
        for layer_name, layer_block in zip(self.layer_names, self.layers):
            x = layer_block(x)
            features[layer_name] = x
        
        return features
    
    def get_content_features(self, x, content_layer='relu4_1'):
        """Extract features from content layer only"""
        features = self.forward(x)
        return features[content_layer]
    
    def get_style_features(self, x, style_layers=None):
        """Extract features from style layers"""
        if style_layers is None:
            style_layers = self.layer_names
        
        features = self.forward(x)
        return {layer: features[layer] for layer in style_layers if layer in features}


def test_encoder():
    """Test VGG encoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = VGGEncoder().to(device)
    encoder.eval()
    
    # Test input
    x = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        features = encoder(x)
    
    print("VGG Encoder Test:")
    for layer_name, feat in features.items():
        print(f"  {layer_name}: {feat.shape}")


if __name__ == '__main__':
    test_encoder()
