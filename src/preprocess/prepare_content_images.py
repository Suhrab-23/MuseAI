"""
Preprocessing script for content images (faces).
Detects faces using MTCNN, crops with expansion, and resizes to 512x512.
"""

import os
import yaml
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
import warnings
warnings.filterwarnings('ignore')


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def preprocess_face_image(image_path, mtcnn, output_size=512, expand_factor=1.4):
    """
    Detect face and crop with expansion.
    
    Args:
        image_path: Path to input image
        mtcnn: MTCNN face detector instance
        output_size: Target size (512x512)
        expand_factor: Factor to expand bbox (1.4 = 40% larger)
    
    Returns:
        PIL Image object or None if face not detected
    """
    img = Image.open(image_path).convert('RGB')
    
    # Detect face
    boxes, probs = mtcnn.detect(img)
    
    if boxes is None or len(boxes) == 0:
        # Fallback: center crop
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((output_size, output_size), Image.LANCZOS)
        return img
    
    # Use the most confident face
    box = boxes[0]
    x1, y1, x2, y2 = box
    
    # Calculate expanded bbox
    width = x2 - x1
    height = y2 - y1
    
    # Expand bbox
    expand_w = width * (expand_factor - 1) / 2
    expand_h = height * (expand_factor - 1) / 2
    
    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(img.width, x2 + expand_w)
    y2 = min(img.height, y2 + expand_h)
    
    # Make square by expanding shorter side
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width > bbox_height:
        diff = bbox_width - bbox_height
        y1 = max(0, y1 - diff / 2)
        y2 = min(img.height, y2 + diff / 2)
    else:
        diff = bbox_height - bbox_width
        x1 = max(0, x1 - diff / 2)
        x2 = min(img.width, x2 + diff / 2)
    
    # Crop and resize
    face_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
    face_img = face_img.resize((output_size, output_size), Image.LANCZOS)
    
    return face_img


def split_and_process_faces(config):
    """
    Process all face images and split into train/val/test.
    
    Args:
        config: Configuration dictionary
    """
    print(f"\n{'='*60}")
    print("PROCESSING FACE IMAGES")
    print(f"{'='*60}")
    
    # Initialize MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    mtcnn = MTCNN(
        image_size=config['preprocessing']['image_size'],
        margin=0,
        device=device,
        post_process=False
    )
    
    # Paths
    raw_dir = Path(config['paths']['content_raw'])
    processed_base = Path(config['paths']['content_processed'])
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(raw_dir.glob(f'*{ext}')))
        image_files.extend(list(raw_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        return
    
    print(f"Found {len(image_files)} face images")
    
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
    
    failed_count = 0
    
    for split_name, files in splits.items():
        output_dir = processed_base / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split_name} set...")
        for img_path in tqdm(files, desc=f"faces/{split_name}"):
            try:
                # Preprocess image
                processed_img = preprocess_face_image(
                    img_path,
                    mtcnn,
                    output_size=config['preprocessing']['image_size'],
                    expand_factor=config['preprocessing']['content_face_expand']
                )
                
                if processed_img is not None:
                    # Save with original filename
                    output_path = output_dir / img_path.name
                    processed_img.save(output_path, quality=95)
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                failed_count += 1
    
    if failed_count > 0:
        print(f"\n⚠️ Failed to process {failed_count} images (fell back to center crop)")
    
    print(f"✅ Completed processing face images")


def create_metadata(config):
    """Create metadata CSV file listing all processed face images"""
    import csv
    
    metadata_path = Path(config['paths']['metadata']) / 'content_catalog.csv'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating metadata catalog")
    print("="*60)
    
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'filename', 'path'])
        
        faces_dir = Path(config['paths']['content_processed'])
        for split in ['train', 'val', 'test']:
            split_dir = faces_dir / split
            if split_dir.exists():
                for img_path in split_dir.glob('*'):
                    if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        rel_path = img_path.relative_to(Path(config['paths']['content_processed']).parent)
                        writer.writerow([split, img_path.name, str(rel_path)])
    
    print(f"✅ Metadata saved to {metadata_path}")


def main():
    """Main preprocessing pipeline for face images"""
    print("\n" + "="*60)
    print("MUSEAI - FACE IMAGE PREPROCESSING")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Check if raw data exists
    content_raw_path = Path(config['paths']['content_raw'])
    if not content_raw_path.exists():
        print(f"❌ Error: Content raw data directory not found: {content_raw_path}")
        return
    
    # Process faces
    split_and_process_faces(config)
    
    # Create metadata
    create_metadata(config)
    
    print("\n" + "="*60)
    print("✅ ALL FACE IMAGES PROCESSED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
