"""
Inference script for loading trained model and stylizing images.
"""

import sys
from pathlib import Path
import yaml
import torch
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent.parent))
from models import StyleTransferNetwork


class StyleTransferInference:
    """
    Inference wrapper for style transfer model.
    """
    
    def __init__(self, checkpoint_path, config_path=None, device=None):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        
        # Create model
        self.model = StyleTransferNetwork(
            num_artists=len(self.config['artists']),
            artist_embed_dim=self.config['model']['artist_embed_dim']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {checkpoint_path}")
        print(f"   Trained for {checkpoint['epoch']} epochs")
        
        # Artist mapping
        self.artist_to_id = {name: idx for idx, name in enumerate(self.config['artists'])}
        
        # Transform for input images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        return img_tensor
    
    def stylize_from_paths(self, content_path, style_path, artist_name, alpha=1.0):
        """
        Stylize using image paths.
        
        Args:
            content_path: Path to content image (face)
            style_path: Path to style image (painting)
            artist_name: Name of artist ('picasso' or 'rembrandt')
            alpha: Style strength (0 to 1)
        
        Returns:
            PIL Image of stylized result
        """
        # Load images
        content_img = self.load_image(content_path).to(self.device)
        style_img = self.load_image(style_path).to(self.device)
        
        # Get artist ID
        if artist_name not in self.artist_to_id:
            raise ValueError(f"Unknown artist: {artist_name}. Choose from {list(self.artist_to_id.keys())}")
        
        artist_id = torch.tensor([self.artist_to_id[artist_name]]).to(self.device)
        
        # Stylize
        with torch.no_grad():
            stylized = self.model.stylize(content_img, style_img, artist_id, alpha=alpha)
        
        # Convert to PIL Image
        stylized = stylized.squeeze(0).cpu()
        stylized = stylized.clamp(0, 1)
        stylized = transforms.ToPILImage()(stylized)
        
        return stylized
    
    def stylize_from_tensors(self, content_tensor, style_tensor, artist_name, alpha=1.0):
        """
        Stylize using pre-loaded tensors.
        
        Args:
            content_tensor: Content image tensor (1, 3, H, W)
            style_tensor: Style image tensor (1, 3, H, W)
            artist_name: Name of artist
            alpha: Style strength
        
        Returns:
            PIL Image of stylized result
        """
        artist_id = torch.tensor([self.artist_to_id[artist_name]]).to(self.device)
        
        content_tensor = content_tensor.to(self.device)
        style_tensor = style_tensor.to(self.device)
        
        with torch.no_grad():
            stylized = self.model.stylize(content_tensor, style_tensor, artist_id, alpha=alpha)
        
        # Convert to PIL Image
        stylized = stylized.squeeze(0).cpu()
        stylized = stylized.clamp(0, 1)
        stylized = transforms.ToPILImage()(stylized)
        
        return stylized
    
    def stylize_with_random_style(self, content_path, artist_name, alpha=1.0):
        """
        Stylize using a random painting from the specified artist.
        
        Args:
            content_path: Path to content image
            artist_name: Name of artist
            alpha: Style strength
        
        Returns:
            PIL Image of stylized result
        """
        import random
        
        # Get random style image from artist
        style_dir = Path(self.config['paths']['style_processed']) / artist_name / 'test'
        style_images = list(style_dir.glob('*.jpg')) + list(style_dir.glob('*.png'))
        
        if not style_images:
            raise ValueError(f"No style images found for {artist_name}")
        
        style_path = random.choice(style_images)
        
        return self.stylize_from_paths(content_path, style_path, artist_name, alpha)


def demo_inference():
    """Demo script for testing inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MuseAI Style Transfer Inference')
    parser.add_argument('--content', required=True, help='Path to content image (face)')
    parser.add_argument('--style', help='Path to style image (painting). If not provided, uses random.')
    parser.add_argument('--artist', required=True, choices=['picasso', 'rembrandt'], help='Artist name')
    parser.add_argument('--output', default='output.jpg', help='Output path')
    parser.add_argument('--checkpoint', default='checkpoints/final_model.pth', help='Checkpoint path')
    parser.add_argument('--alpha', type=float, default=1.0, help='Style strength (0-1)')
    
    args = parser.parse_args()
    
    # Create inference model
    inference = StyleTransferInference(checkpoint_path=args.checkpoint)
    
    # Stylize
    if args.style:
        print(f"Stylizing {args.content} with {args.style} ({args.artist})...")
        stylized = inference.stylize_from_paths(args.content, args.style, args.artist, args.alpha)
    else:
        print(f"Stylizing {args.content} with random {args.artist} painting...")
        stylized = inference.stylize_with_random_style(args.content, args.artist, args.alpha)
    
    # Save
    stylized.save(args.output)
    print(f"✅ Stylized image saved to {args.output}")


if __name__ == '__main__':
    demo_inference()