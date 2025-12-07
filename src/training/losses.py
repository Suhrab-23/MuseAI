"""
Loss functions for style transfer training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    Content loss using VGG features.
    Measures L2 distance in feature space.
    """
    
    def __init__(self, layer='relu4_1'):
        super(ContentLoss, self).__init__()
        self.layer = layer
    
    def forward(self, content_features, stylized_features):
        """
        Compute content loss.
        
        Args:
            content_features: Features from content image
            stylized_features: Features from stylized image
        
        Returns:
            Content loss (scalar)
        """
        content_feat = content_features[self.layer]
        stylized_feat = stylized_features[self.layer]
        
        loss = F.mse_loss(stylized_feat, content_feat)
        
        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices.
    Measures style similarity across multiple layers.
    """
    
    def __init__(self, layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']):
        super(StyleLoss, self).__init__()
        self.layers = layers
    
    def gram_matrix(self, features):
        """
        Compute Gram matrix.
        
        Args:
            features: Feature tensor (B, C, H, W)
        
        Returns:
            Gram matrix (B, C, C)
        """
        B, C, H, W = features.size()
        
        # Reshape to (B, C, H*W)
        features_reshaped = features.view(B, C, H * W)
        
        # Compute Gram matrix: (B, C, H*W) @ (B, H*W, C) = (B, C, C)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        
        # Normalize by number of elements
        gram = gram / (C * H * W)
        
        return gram
    
    def forward(self, style_features, stylized_features):
        """
        Compute style loss across multiple layers.
        
        Args:
            style_features: Features from style image
            stylized_features: Features from stylized image
        
        Returns:
            Style loss (scalar)
        """
        loss = 0.0
        
        for layer in self.layers:
            style_feat = style_features[layer]
            stylized_feat = stylized_features[layer]
            
            # Compute Gram matrices
            style_gram = self.gram_matrix(style_feat)
            stylized_gram = self.gram_matrix(stylized_feat)
            
            # L2 distance between Gram matrices
            layer_loss = F.mse_loss(stylized_gram, style_gram)
            loss += layer_loss
        
        # Average across layers
        loss = loss / len(self.layers)
        
        return loss


class IdentityLoss(nn.Module):
    """
    Identity preservation loss using FaceNet embeddings.
    Ensures the stylized face maintains the subject's identity.
    """
    
    def __init__(self):
        super(IdentityLoss, self).__init__()
    
    def forward(self, content_embedding, stylized_embedding):
        """
        Compute identity loss.
        Uses cosine distance between embeddings.
        
        Args:
            content_embedding: Embedding from content image (B, 512)
            stylized_embedding: Embedding from stylized image (B, 512)
        
        Returns:
            Identity loss (scalar)
        """
        # Cosine similarity (already normalized embeddings)
        similarity = (content_embedding * stylized_embedding).sum(dim=1).mean()
        
        # Convert to loss (1 - similarity)
        # similarity = 1 means identical, loss = 0
        # similarity = 0 means different, loss = 1
        loss = 1.0 - similarity
        
        return loss


class TotalVariationLoss(nn.Module):
    """
    Total variation loss for image smoothness.
    Reduces noise and artifacts.
    """
    
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, image):
        """
        Compute TV loss.
        
        Args:
            image: Image tensor (B, C, H, W)
        
        Returns:
            TV loss (scalar)
        """
        # Difference between adjacent pixels
        diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        
        # Sum of absolute differences
        tv_loss = diff_h.mean() + diff_w.mean()
        
        return tv_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for style transfer training.
    """
    
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        
        # Loss components
        self.content_loss = ContentLoss(layer=config['model']['content_layer'])
        self.style_loss = StyleLoss(layers=config['model']['style_layers'])
        self.identity_loss = IdentityLoss()
        self.tv_loss = TotalVariationLoss()
        
        # Loss weights
        self.weights = config['training']['loss_weights']
    
    def forward(self, features, stylized_img):
        """
        Compute total loss.
        
        Args:
            features: Dictionary with features from model
            stylized_img: Stylized image for TV loss
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Content loss
        content_loss = self.content_loss(
            features['content_features'],
            features['stylized_features']
        )
        
        # Style loss
        style_loss = self.style_loss(
            features['style_features'],
            features['stylized_features']
        )
        
        # Identity loss
        identity_loss = self.identity_loss(
            features['content_embedding'],
            features['stylized_embedding']
        )
        
        # TV loss
        tv_loss = self.tv_loss(stylized_img)
        
        # Total weighted loss
        total_loss = (
            self.weights['content'] * content_loss +
            self.weights['style'] * style_loss +
            self.weights['identity'] * identity_loss +
            self.weights['tv'] * tv_loss
        )
        
        return {
            'total': total_loss,
            'content': content_loss.item(),
            'style': style_loss.item(),
            'identity': identity_loss.item(),
            'tv': tv_loss.item()
        }


def test_losses():
    """Test loss functions"""
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy features
    B, C, H, W = 2, 512, 64, 64
    
    features = {
        'content_features': {
            'relu4_1': torch.randn(B, C, H, W).to(device)
        },
        'style_features': {
            'relu1_1': torch.randn(B, 64, 256, 256).to(device),
            'relu2_1': torch.randn(B, 128, 128, 128).to(device),
            'relu3_1': torch.randn(B, 256, 64, 64).to(device),
            'relu4_1': torch.randn(B, C, H, W).to(device)
        },
        'stylized_features': {
            'relu1_1': torch.randn(B, 64, 256, 256).to(device),
            'relu2_1': torch.randn(B, 128, 128, 128).to(device),
            'relu3_1': torch.randn(B, 256, 64, 64).to(device),
            'relu4_1': torch.randn(B, C, H, W).to(device)
        },
        'content_embedding': torch.randn(B, 512).to(device),
        'stylized_embedding': torch.randn(B, 512).to(device)
    }
    
    stylized_img = torch.rand(B, 3, 512, 512).to(device)
    
    # Test combined loss
    criterion = CombinedLoss(config).to(device)
    losses = criterion(features, stylized_img)
    
    print("Loss Test:")
    for name, value in losses.items():
        if name == 'total':
            print(f"  {name}: {value.item():.4f}")
        else:
            print(f"  {name}: {value:.4f}")


if __name__ == '__main__':
    test_losses()
