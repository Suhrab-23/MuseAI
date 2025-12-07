"""
Preprocessing script for style images (paintings).
Resizes and crops paintings to 512x512 and splits into train/val/test.
"""

import os
import yaml
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def preprocess_style_image(image_path, output_size=512, crop_mode='center'):
    """
    Preprocess a single style image (painting).
    
    Args:
        image_path: Path to input image
        output_size: Target size (512x512)
        crop_mode: 'center' or 'random' crop
    
    Returns:
        PIL Image object
    """
    img = Image.open(image_path).convert('RGB')
    
    # Resize shortest side to output_size
    width, height = img.size
    if width < height:
        new_width = output_size
        new_height = int(height * (output_size / width))
    else:
        new_height = output_size
        new_width = int(width * (output_size / height))
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Crop to square
    width, height = img.size
    if crop_mode == 'center':
        left = (width - output_size) // 2
        top = (height - output_size) // 2
    else:  # random crop
        left = random.randint(0, width - output_size) if width > output_size else 0
        top = random.randint(0, height - output_size) if height > output_size else 0
    
    right = left + output_size
    bottom = top + output_size
    
    img = img.crop((left, top, right, bottom))
    
    return img


def split_and_process_artist(config, artist_name):
    """
    Process all images for one artist and split into train/val/test.
    
    Args:
        config: Configuration dictionary
        artist_name: Name of artist (e.g., 'picasso', 'rembrandt')
    """
    print(f"\n{'='*60}")
    print(f"Processing {artist_name.upper()} paintings")
    print(f"{'='*60}")
    
    # Paths
    raw_dir = Path(config['paths']['style_raw']) / artist_name
    processed_base = Path(config['paths']['style_processed']) / artist_name
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(raw_dir.glob(f'*{ext}')))
        image_files.extend(list(raw_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.seed(config['preprocessing']['random_seed'])
    random.shuffle(image_files)
    
    train_ratio = config['preprocessing']['train_split']
    val_ratio = config['preprocessing']['val_split']
    
    n_train = int(len(image_files) * train_ratio)
    n_val = int(len(image_files) * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Process and save each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        output_dir = processed_base / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split_name} set...")
        for img_path in tqdm(files, desc=f"{artist_name}/{split_name}"):
            try:
                # Preprocess image
                processed_img = preprocess_style_image(
                    img_path,
                    output_size=config['preprocessing']['image_size'],
                    crop_mode=config['preprocessing']['style_crop_mode']
                )
                
                # Save with original filename
                output_path = output_dir / img_path.name
                processed_img.save(output_path, quality=95)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
    
    print(f"✅ Completed processing {artist_name}")


def create_metadata(config):
    """Create metadata CSV file listing all processed style images"""
    import csv
    
    metadata_path = Path(config['paths']['metadata']) / 'style_catalog.csv'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating metadata catalog")
    print("="*60)
    
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['artist', 'split', 'filename', 'path'])
        
        for artist in config['artists']:
            artist_dir = Path(config['paths']['style_processed']) / artist
            for split in ['train', 'val', 'test']:
                split_dir = artist_dir / split
                if split_dir.exists():
                    for img_path in split_dir.glob('*'):
                        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            rel_path = img_path.relative_to(Path(config['paths']['style_processed']).parent)
                            writer.writerow([artist, split, img_path.name, str(rel_path)])
    
    print(f"✅ Metadata saved to {metadata_path}")


def main():
    """Main preprocessing pipeline for style images"""
    print("\n" + "="*60)
    print("MUSEAI - STYLE IMAGE PREPROCESSING")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Check if raw data exists
    style_raw_path = Path(config['paths']['style_raw'])
    if not style_raw_path.exists():
        print(f"❌ Error: Style raw data directory not found: {style_raw_path}")
        return
    
    # Process each artist
    for artist in config['artists']:
        artist_raw_dir = style_raw_path / artist
        if not artist_raw_dir.exists():
            print(f"⚠️ Warning: {artist} directory not found: {artist_raw_dir}")
            continue
        
        split_and_process_artist(config, artist)
    
    # Create metadata
    create_metadata(config)
    
    print("\n" + "="*60)
    print("✅ ALL STYLE IMAGES PROCESSED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
