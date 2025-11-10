"""
Quick Environment Check - Minimal Version
Run this for a quick check of your environment
"""

import sys

def quick_check():
    print("="*50)
    print("QUICK ENVIRONMENT CHECK")
    print("="*50)
    
    # Python version
    print(f"\nPython: {sys.version.split()[0]}")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("PyTorch: NOT INSTALLED")
    
    # DGL
    try:
        import dgl
        print(f"DGL: {dgl.__version__}")
    except:
        print("DGL: NOT INSTALLED")
    
    # Config
    import os
    if os.path.exists("config.yaml"):
        print("Config: ✅ Found")
    else:
        print("Config: ❌ Not found")
    
    print("="*50)

if __name__ == "__main__":
    quick_check()
    input("\nPress Enter to exit...")

