"""Verify installation and run basic checks."""

import sys

def main():
    print("🔍 Verifying Free Transformer installation...\n")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPUs: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        from free_transformer import (
            FreeTransformer,ModelConfig
        )
        print("✓ Free Transformer package")
    except ImportError as e:
        print(f"✗ Free Transformer import failed: {e}")
        return False
    
    # Test instantiation
    try:
        config = ModelConfig(vocab_size=1000, hidden_dim=128, num_layers=4, num_heads=4)
        model = FreeTransformer(config)
        print(f"✓ Model instantiation (params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M)")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False
    
    # Test forward pass
    try:
        tokens = torch.randint(0, 1000, (2, 64))
        logits, z_logits = model(tokens, mode='training')
        print("✓ Forward pass (training mode)")
        
        logits, _ = model(tokens, mode='inference')
        print("✓ Forward pass (inference mode)")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    print("\n✅ All checks passed! Installation verified.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
