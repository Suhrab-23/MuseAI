"""
Evaluation metrics for style transfer.
"""

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np


class Evaluator:
    """
    Evaluator for style transfer quality metrics.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # LPIPS metric (perceptual similarity)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
    
    def compute_ssim(self, img1, img2):
        """
        Compute SSIM between two images.
        
        Args:
            img1: First image tensor (B, 3, H, W) in [0, 1]
            img2: Second image tensor (B, 3, H, W) in [0, 1]
        
        Returns:
            Average SSIM score
        """
        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()
        
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            # Compute SSIM per channel and average
            score = ssim(
                img1_np[i].transpose(1, 2, 0),
                img2_np[i].transpose(1, 2, 0),
                channel_axis=2,
                data_range=1.0
            )
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
    
    def compute_lpips(self, img1, img2):
        """
        Compute LPIPS perceptual distance.
        
        Args:
            img1: First image tensor (B, 3, H, W) in [0, 1]
            img2: Second image tensor (B, 3, H, W) in [0, 1]
        
        Returns:
            Average LPIPS distance
        """
        with torch.no_grad():
            # Convert to [-1, 1] range for LPIPS
            img1_scaled = img1 * 2.0 - 1.0
            img2_scaled = img2 * 2.0 - 1.0
            
            distance = self.lpips_model(img1_scaled, img2_scaled)
        
        return distance.mean().item()
    
    def compute_gram_distance(self, features1, features2, layers=None):
        """
        Compute Gram matrix distance between style features.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            layers: List of layer names to compute distance over
        
        Returns:
            Average Gram matrix distance
        """
        if layers is None:
            layers = list(features1.keys())
        
        distances = []
        
        for layer in layers:
            feat1 = features1[layer]
            feat2 = features2[layer]
            
            # Compute Gram matrices
            gram1 = self._gram_matrix(feat1)
            gram2 = self._gram_matrix(feat2)
            
            # L2 distance
            distance = F.mse_loss(gram1, gram2)
            distances.append(distance.item())
        
        return np.mean(distances)
    
    def _gram_matrix(self, features):
        """Compute Gram matrix"""
        B, C, H, W = features.size()
        features_reshaped = features.view(B, C, H * W)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        gram = gram / (C * H * W)
        return gram
    
    def compute_identity_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between face embeddings.
        
        Args:
            embedding1: First face embedding (B, 512)
            embedding2: Second face embedding (B, 512)
        
        Returns:
            Average cosine similarity
        """
        similarity = (embedding1 * embedding2).sum(dim=1)
        return similarity.mean().item()
    
    def evaluate_batch(self, content_img, style_img, stylized_img, model):
        """
        Compute all metrics for a batch.
        
        Args:
            content_img: Content image tensor
            style_img: Style image tensor
            stylized_img: Stylized image tensor
            model: Style transfer model
        
        Returns:
            Dictionary with all metrics
        """
        with torch.no_grad():
            # Get features
            features = model.get_features_for_loss(content_img, style_img, stylized_img)
            
            # Compute metrics
            metrics = {
                'ssim': self.compute_ssim(content_img, stylized_img),
                'lpips': self.compute_lpips(content_img, stylized_img),
                'gram_distance': self.compute_gram_distance(
                    features['style_features'],
                    features['stylized_features']
                ),
                'identity_similarity': self.compute_identity_similarity(
                    features['content_embedding'],
                    features['stylized_embedding']
                )
            }
        
        return metrics


def test_evaluator():
    """Test evaluator"""
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).parent.parent))
    from models import StyleTransferNetwork
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create evaluator and model
    evaluator = Evaluator(device=device)
    model = StyleTransferNetwork(num_artists=2).to(device)
    model.eval()
    
    # Test images
    content = torch.rand(2, 3, 512, 512).to(device)
    style = torch.rand(2, 3, 512, 512).to(device)
    artist_id = torch.tensor([0, 1]).to(device)
    
    with torch.no_grad():
        stylized = model(content, style, artist_id)
    
    # Evaluate
    metrics = evaluator.evaluate_batch(content, style, stylized, model)
    
    print("Evaluator Test:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == '__main__':
    test_evaluator()
