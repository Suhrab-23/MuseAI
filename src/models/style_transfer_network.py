"""
Main Style Transfer Network combining encoder, AdaIN, and decoder.
"""

import torch
import torch.nn as nn
from .encoder import VGGEncoder
from .adain import AdaIN
from .decoder import Decoder
from .facenet_identity import FaceNetIdentity


class StyleTransferNetwork(nn.Module):
    """
    Complete style transfer network with artist conditioning.
    """
    
    def __init__(self, num_artists=2, artist_embed_dim=128):
        """
        Args:
            num_artists: Number of artists to support
            artist_embed_dim: Dimension of artist embeddings
        """
        super(StyleTransferNetwork, self).__init__()
        
        # Components
        self.encoder = VGGEncoder(layer_names=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        self.adain = AdaIN()
        self.decoder = Decoder()
        self.facenet = FaceNetIdentity()
        
        # Artist embeddings for conditioning
        self.artist_embeddings = nn.Embedding(num_artists, artist_embed_dim)
        
        # Artist-specific conditional normalization
        # Applied to AdaIN output before decoding
        self.artist_transform = nn.Sequential(
            nn.Linear(artist_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
        # Initialize embeddings
        nn.init.normal_(self.artist_embeddings.weight, mean=0, std=0.02)
    
    def forward(self, content_img, style_img, artist_id):
        """
        Forward pass through the network.
        
        Args:
            content_img: Content image (B, 3, H, W) in [0, 1]
            style_img: Style image (B, 3, H, W) in [0, 1]
            artist_id: Artist ID tensor (B,) with values in [0, num_artists-1]
        
        Returns:
            stylized_img: Stylized image (B, 3, H, W) in [0, 1]
        """
        # Extract features
        content_features = self.encoder(content_img)
        style_features = self.encoder(style_img)
        
        # Use relu4_1 for AdaIN
        content_feat = content_features['relu4_1']
        style_feat = style_features['relu4_1']
        
        # Apply AdaIN
        stylized_feat = self.adain(content_feat, style_feat)
        
        # Apply artist-specific transformation
        artist_embed = self.artist_embeddings(artist_id)  # (B, embed_dim)
        artist_transform = self.artist_transform(artist_embed)  # (B, 512)
        
        # Apply transformation as channel-wise affine
        artist_transform = artist_transform.unsqueeze(2).unsqueeze(3)  # (B, 512, 1, 1)
        stylized_feat = stylized_feat + artist_transform
        
        # Decode to image
        stylized_img = self.decoder(stylized_feat)
        
        return stylized_img
    
    def get_features_for_loss(self, content_img, style_img, stylized_img):
        """
        Extract features for computing losses.
        
        Args:
            content_img: Original content image
            style_img: Style reference image
            stylized_img: Generated stylized image
        
        Returns:
            Dictionary with features for loss computation
        """
        content_features = self.encoder(content_img)
        style_features = self.encoder(style_img)
        stylized_features = self.encoder(stylized_img)
        
        # Face embeddings for identity loss
        content_embedding = self.facenet(content_img)
        stylized_embedding = self.facenet(stylized_img)
        
        return {
            'content_features': content_features,
            'style_features': style_features,
            'stylized_features': stylized_features,
            'content_embedding': content_embedding,
            'stylized_embedding': stylized_embedding
        }
    
    def stylize(self, content_img, style_img, artist_id, alpha=1.0):
        """
        Stylize with controllable strength.
        
        Args:
            content_img: Content image (B, 3, H, W)
            style_img: Style image (B, 3, H, W)
            artist_id: Artist ID (B,)
            alpha: Style strength in [0, 1] (1 = full style, 0 = no style)
        
        Returns:
            Stylized image
        """
        stylized = self.forward(content_img, style_img, artist_id)
        
        if alpha < 1.0:
            # Blend with original content
            stylized = alpha * stylized + (1 - alpha) * content_img
        
        return stylized


def test_style_transfer_network():
    """Test complete network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleTransferNetwork(num_artists=2).to(device)
    
    # Test inputs
    content = torch.rand(2, 3, 512, 512).to(device)
    style = torch.rand(2, 3, 512, 512).to(device)
    artist_id = torch.tensor([0, 1]).to(device)
    
    # Forward pass
    stylized = model(content, style, artist_id)
    
    print("StyleTransferNetwork Test:")
    print(f"  Content: {content.shape}")
    print(f"  Style: {style.shape}")
    print(f"  Artist IDs: {artist_id}")
    print(f"  Stylized: {stylized.shape}")
    print(f"  Output range: [{stylized.min():.3f}, {stylized.max():.3f}]")
    
    # Test feature extraction for loss
    features = model.get_features_for_loss(content, style, stylized)
    print(f"\nFeatures for loss computation:")
    print(f"  Content features: {list(features['content_features'].keys())}")
    print(f"  Content embedding: {features['content_embedding'].shape}")
    print(f"  Stylized embedding: {features['stylized_embedding'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")


if __name__ == '__main__':
    test_style_transfer_network()
