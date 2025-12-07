"""
PyTorch Dataset classes for style transfer training.
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class StyleTransferDataset(Dataset):
    """
    Dataset for style transfer training.
    Pairs content images (faces) with style images (paintings).
    """
    
    def __init__(self, content_dir, style_dirs, artist_names, transform=None, mode='train'):
        """
        Args:
            content_dir: Path to content images directory
            style_dirs: Dictionary mapping artist names to style image directories
            artist_names: List of artist names
            transform: Optional transform to apply
            mode: 'train', 'val', or 'test'
        """
        self.content_dir = Path(content_dir) / mode
        self.style_dirs = {name: Path(style_dirs[name]) / mode for name in artist_names}
        self.artist_names = artist_names
        self.mode = mode
        
        # Get all content images
        self.content_images = self._get_image_files(self.content_dir)
        
        # Get style images per artist
        self.style_images = {}
        for artist in artist_names:
            self.style_images[artist] = self._get_image_files(self.style_dirs[artist])
        
        # Create artist ID mapping
        self.artist_to_id = {name: idx for idx, name in enumerate(artist_names)}
        
        if not self.content_images:
            raise ValueError(f"No content images found in {self.content_dir}")
        
        for artist in artist_names:
            if not self.style_images[artist]:
                raise ValueError(f"No style images found for {artist} in {self.style_dirs[artist]}")
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        print(f"Loaded {mode} dataset:")
        print(f"  Content images: {len(self.content_images)}")
        for artist in artist_names:
            print(f"  {artist.capitalize()} style images: {len(self.style_images[artist])}")
    
    def _get_image_files(self, directory):
        """Get all image files from directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        
        if not directory.exists():
            return files
        
        for ext in image_extensions:
            files.extend(list(directory.glob(f'*{ext}')))
            files.extend(list(directory.glob(f'*{ext.upper()}')))
        
        return sorted(files)
    
    def __len__(self):
        return len(self.content_images)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            content_img: Content image tensor (3, H, W)
            style_img: Style image tensor (3, H, W)
            artist_id: Artist ID (scalar)
        """
        # Load content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')
        content_img = self.transform(content_img)
        
        # Randomly select an artist
        artist = random.choice(self.artist_names)
        artist_id = self.artist_to_id[artist]
        
        # Randomly select a style image from this artist
        style_path = random.choice(self.style_images[artist])
        style_img = Image.open(style_path).convert('RGB')
        style_img = self.transform(style_img)
        
        return content_img, style_img, artist_id


def get_dataloaders(config, batch_size=None, num_workers=4):
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        batch_size: Batch size (uses config if None)
        num_workers: Number of worker processes
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # Paths
    content_dir = config['paths']['content_processed']
    style_base = config['paths']['style_processed']
    
    style_dirs = {}
    for artist in config['artists']:
        style_dirs[artist] = str(Path(style_base) / artist)
    
    # Create datasets
    datasets = {}
    for mode in ['train', 'val', 'test']:
        datasets[mode] = StyleTransferDataset(
            content_dir=content_dir,
            style_dirs=style_dirs,
            artist_names=config['artists'],
            mode=mode
        )
    
    # Create dataloaders
    dataloaders = {}
    for mode in ['train', 'val', 'test']:
        shuffle = (mode == 'train')
        dataloaders[mode] = torch.utils.data.DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        )
    
    return dataloaders


def test_dataset():
    """Test dataset loading"""
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataloaders = get_dataloaders(config, batch_size=4, num_workers=0)
    
    print("\nTesting dataloader:")
    for mode, dataloader in dataloaders.items():
        print(f"\n{mode.upper()} set:")
        content, style, artist_id = next(iter(dataloader))
        print(f"  Content batch: {content.shape}")
        print(f"  Style batch: {style.shape}")
        print(f"  Artist IDs: {artist_id}")
        print(f"  Content range: [{content.min():.3f}, {content.max():.3f}]")
        print(f"  Style range: [{style.min():.3f}, {style.max():.3f}]")


if __name__ == '__main__':
    test_dataset()
