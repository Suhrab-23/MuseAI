"""
Main training script for MuseAI style transfer.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models import StyleTransferNetwork
from training.dataset import get_dataloaders
from training.losses import CombinedLoss


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, losses, config, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config
    }, checkpoint_path)
    
    print(f"✅ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['losses']


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = {
        'total': 0.0,
        'content': 0.0,
        'style': 0.0,
        'identity': 0.0,
        'tv': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (content_img, style_img, artist_id) in enumerate(pbar):
        # Move to device
        content_img = content_img.to(device)
        style_img = style_img.to(device)
        artist_id = artist_id.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        stylized_img = model(content_img, style_img, artist_id)
        
        # Get features for loss computation
        features = model.get_features_for_loss(content_img, style_img, stylized_img)
        
        # Compute loss
        losses = criterion(features, stylized_img)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses.keys():
            if key == 'total':
                epoch_losses[key] += losses[key].item()
            else:
                epoch_losses[key] += losses[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'content': f"{losses['content']:.4f}",
            'style': f"{losses['style']:.4f}",
            'identity': f"{losses['identity']:.4f}"
        })
    
    # Average losses
    num_batches = len(dataloader)
    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    
    val_losses = {
        'total': 0.0,
        'content': 0.0,
        'style': 0.0,
        'identity': 0.0,
        'tv': 0.0
    }
    
    with torch.no_grad():
        for content_img, style_img, artist_id in tqdm(dataloader, desc="Validating"):
            # Move to device
            content_img = content_img.to(device)
            style_img = style_img.to(device)
            artist_id = artist_id.to(device)
            
            # Forward pass
            stylized_img = model(content_img, style_img, artist_id)
            
            # Get features for loss
            features = model.get_features_for_loss(content_img, style_img, stylized_img)
            
            # Compute loss
            losses = criterion(features, stylized_img)
            
            # Accumulate losses
            for key in val_losses.keys():
                if key == 'total':
                    val_losses[key] += losses[key].item()
                else:
                    val_losses[key] += losses[key]
    
    # Average losses
    num_batches = len(dataloader)
    for key in val_losses.keys():
        val_losses[key] /= num_batches
    
    return val_losses


def save_sample_images(model, dataloader, device, epoch, config, num_samples=4):
    """Save sample stylized images"""
    model.eval()
    
    output_dir = Path(config['paths']['outputs']) / 'training_samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get one batch
    content_img, style_img, artist_id = next(iter(dataloader))
    
    # Take only num_samples
    content_img = content_img[:num_samples].to(device)
    style_img = style_img[:num_samples].to(device)
    artist_id = artist_id[:num_samples].to(device)
    
    with torch.no_grad():
        stylized_img = model(content_img, style_img, artist_id)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Content
        axes[i, 0].imshow(content_img[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title('Content')
        axes[i, 0].axis('off')
        
        # Style
        axes[i, 1].imshow(style_img[i].cpu().permute(1, 2, 0))
        artist_name = config['artists'][artist_id[i].item()]
        axes[i, 1].set_title(f'Style ({artist_name})')
        axes[i, 1].axis('off')
        
        # Stylized
        axes[i, 2].imshow(stylized_img[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[i, 2].set_title('Stylized')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'epoch_{epoch:03d}.png', dpi=150)
    plt.close()
    
    print(f"📸 Sample images saved to {output_dir}")


def plot_training_curves(train_losses, val_losses, config):
    """Plot and save training curves"""
    output_dir = Path(config['paths']['outputs']) / 'training_curves'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(train_losses['total']) + 1)
    
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses['total'], label='Train', marker='o')
    plt.plot(epochs, val_losses['total'], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'total_loss.png', dpi=150)
    plt.close()
    
    # Plot individual losses
    loss_names = ['content', 'style', 'identity', 'tv']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, loss_name in enumerate(loss_names):
        axes[idx].plot(epochs, train_losses[loss_name], label='Train', marker='o')
        axes[idx].plot(epochs, val_losses[loss_name], label='Val', marker='s')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(f'{loss_name.capitalize()} Loss')
        axes[idx].set_title(f'{loss_name.capitalize()} Loss')
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_losses.png', dpi=150)
    plt.close()
    
    print(f"📊 Training curves saved to {output_dir}")


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("MUSEAI - TRAINING")
    print("="*70)
    
    # Load config
    config = load_config()
    print(f"\n✅ Configuration loaded")
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create dataloaders
    print("\n📁 Loading datasets...")
    dataloaders = get_dataloaders(
        config,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    print("\n🏗️  Building model...")
    model = StyleTransferNetwork(
        num_artists=len(config['artists']),
        artist_embed_dim=config['model']['artist_embed_dim']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create criterion and optimizer
    criterion = CombinedLoss(config).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    if config['training']['lr_scheduler']['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_scheduler']['step_size'],
            gamma=config['training']['lr_scheduler']['gamma']
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    
    # Training history
    train_history = {key: [] for key in ['total', 'content', 'style', 'identity', 'tv']}
    val_history = {key: [] for key in ['total', 'content', 'style', 'identity', 'tv']}
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    start_time = datetime.now()
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_losses = train_epoch(model, dataloaders['train'], criterion, optimizer, device, epoch)
        
        # Validate
        val_losses = validate(model, dataloaders['val'], criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        for key in train_history.keys():
            train_history[key].append(train_losses[key])
            val_history[key].append(val_losses[key])
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_losses['total']:.4f}")
        print(f"   Val Loss:   {val_losses['total']:.4f}")
        print(f"   LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save sample images
        if epoch % 5 == 0 or epoch == 1:
            save_sample_images(model, dataloaders['val'], device, epoch, config)
        
        # Save checkpoint
        if epoch % config['training']['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_history, 'val': val_history},
                config,
                filename=f'checkpoint_epoch_{epoch:03d}.pth'
            )
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_history, 'val': val_history},
                config,
                filename='best_model.pth'
            )
            print(f"   ⭐ New best model! Val loss: {best_val_loss:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, config['training']['num_epochs'],
        {'train': train_history, 'val': val_history},
        config,
        filename='final_model.pth'
    )
    
    # Plot training curves
    plot_training_curves(train_history, val_history, config)
    
    # Training complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"   Duration: {duration}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final model saved: checkpoints/final_model.pth")
    print("="*70)


if __name__ == '__main__':
    main()
