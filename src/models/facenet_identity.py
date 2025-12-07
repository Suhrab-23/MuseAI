"""
FaceNet model for identity preservation.
"""

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNetIdentity(nn.Module):
    """
    FaceNet model for extracting face embeddings.
    Used to preserve identity in stylized faces.
    """
    
    def __init__(self, pretrained='vggface2'):
        """
        Args:
            pretrained: Which pretrained weights to use ('vggface2' or 'casia-webface')
        """
        super(FaceNetIdentity, self).__init__()
        
        # Load pretrained FaceNet
        self.facenet = InceptionResnetV1(pretrained=pretrained)
        self.facenet.eval()
        
        # Freeze all parameters
        for param in self.facenet.parameters():
            param.requires_grad = False
    
    def preprocess(self, x):
        """
        Preprocess images for FaceNet.
        FaceNet expects images in [-1, 1] range.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1] range
        
        Returns:
            Preprocessed tensor in [-1, 1] range
        """
        # Convert [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        
        # Resize to 160x160 (FaceNet input size)
        if x.size(2) != 160 or x.size(3) != 160:
            x = nn.functional.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        return x
    
    def forward(self, x):
        """
        Extract face embeddings.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1] range
        
        Returns:
            Face embeddings (B, 512)
        """
        x = self.preprocess(x)
        
        with torch.no_grad():
            embeddings = self.facenet(x)
        
        # Normalize embeddings to unit length
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (B, 512)
            embedding2: Second embedding (B, 512)
        
        Returns:
            Similarity scores (B,) in [-1, 1]
        """
        # Cosine similarity
        similarity = (embedding1 * embedding2).sum(dim=1)
        return similarity


def test_facenet():
    """Test FaceNet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facenet = FaceNetIdentity().to(device)
    
    # Test input
    x = torch.rand(2, 3, 512, 512).to(device)
    
    embeddings = facenet(x)
    
    print("FaceNet Test:")
    print(f"  Input: {x.shape}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Embedding norm: {embeddings.norm(dim=1)}")
    
    # Test similarity
    sim = facenet.compute_similarity(embeddings[0:1], embeddings[1:2])
    print(f"  Similarity between two images: {sim.item():.4f}")


if __name__ == '__main__':
    test_facenet()
