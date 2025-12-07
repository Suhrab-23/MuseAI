"""
Quick test script to verify all components are working.
"""

import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models import StyleTransferNetwork, VGGEncoder, AdaIN, Decoder, FaceNetIdentity
        print("✅ Models imported successfully")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from training import StyleTransferDataset, CombinedLoss
        print("✅ Training modules imported successfully")
    except Exception as e:
        print(f"❌ Training import failed: {e}")
        return False
    
    try:
        from utils import Evaluator
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True


def test_model():
    """Test model forward pass"""
    print("\nTesting model forward pass...")
    
    try:
        from models import StyleTransferNetwork
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = StyleTransferNetwork(num_artists=2).to(device)
        
        # Test inputs
        content = torch.rand(2, 3, 512, 512).to(device)
        style = torch.rand(2, 3, 512, 512).to(device)
        artist_id = torch.tensor([0, 1]).to(device)
        
        # Forward pass
        with torch.no_grad():
            stylized = model(content, style, artist_id)
        
        print(f"✅ Model forward pass successful")
        print(f"   Input shape: {content.shape}")
        print(f"   Output shape: {stylized.shape}")
        print(f"   Output range: [{stylized.min():.3f}, {stylized.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("MUSEAI SYSTEM TEST")
    print("="*70 + "\n")
    
    success = True
    
    success &= test_imports()
    success &= test_model()
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED - System is ready!")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    print("="*70)


if __name__ == '__main__':
    main()
