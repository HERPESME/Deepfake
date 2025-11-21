#!/usr/bin/env python3
"""
Quick test script to verify project setup.
Tests all major components without requiring datasets.
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 80)
    print("TESTING IMPORTS")
    print("=" * 80)
    
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    all_passed = True
    for module_name, display_name in tests:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - NOT FOUND")
            all_passed = False
    
    # Test project modules
    print("\nTesting project modules:")
    project_modules = [
        "src.data.preprocessing",
        "src.data.dataloader",
        "src.models.baseline_models",
        "src.models.advanced_models",
        "src.training.trainer",
        "src.evaluation.metrics",
        "src.utils.config",
    ]
    
    for module_name in project_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name} - ERROR: {e}")
            all_passed = False
    
    return all_passed


def test_model_creation():
    """Test if models can be created."""
    print("\n" + "=" * 80)
    print("TESTING MODEL CREATION")
    print("=" * 80)
    
    try:
        import torch
        from src.models.baseline_models import create_model as create_baseline_model
        
        models_to_test = ["xception", "efficientnet_b0", "resnet50"]
        
        for model_name in models_to_test:
            try:
                model = create_baseline_model(model_name, num_classes=2, pretrained=False)
                num_params = sum(p.numel() for p in model.parameters())
                print(f"✅ {model_name}: {num_params:,} parameters")
            except Exception as e:
                print(f"❌ {model_name}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_data_loader():
    """Test if data loader works."""
    print("\n" + "=" * 80)
    print("TESTING DATA LOADER")
    print("=" * 80)
    
    sample_data_path = Path("data/processed/sample")
    
    if not sample_data_path.exists():
        print("⚠️  Sample data not found. Run: python scripts/create_sample_data.py")
        return False
    
    try:
        from src.data.dataloader import DataLoaderFactory
        
        train_loader, val_loader, test_loader = DataLoaderFactory.create_dataloaders(
            str(sample_data_path),
            batch_size=4,
            num_workers=0  # Use 0 to avoid multiprocessing issues in testing
        )
        
        print(f"✅ Train batches: {len(train_loader)}")
        print(f"✅ Val batches: {len(val_loader)}")
        print(f"✅ Test batches: {len(test_loader)}")
        
        # Test loading one batch
        try:
            images, labels = next(iter(train_loader))
            print(f"✅ Batch shape: {images.shape}")
            print(f"✅ Labels shape: {labels.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION")
    print("=" * 80)
    
    try:
        from src.utils.config import ConfigLoader
        
        # Test default config
        config = ConfigLoader.create_default_config()
        print(f"✅ Default config created")
        print(f"   Model: {config.model.name}")
        print(f"   Dataset: {config.data.dataset}")
        print(f"   Epochs: {config.training.epochs}")
        
        # Test loading from file
        config_path = Path("configs/training_config.yaml")
        if config_path.exists():
            config = ConfigLoader.load_config(str(config_path))
            print(f"✅ Config file loaded successfully")
        else:
            print("⚠️  Config file not found (this is okay)")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DEEPFAKE DETECTION PROJECT - SETUP VERIFICATION")
    print("=" * 80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Configuration", test_config()))
    results.append(("Data Loader", test_data_loader()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED! Setup is complete.")
        print("\nNext steps:")
        print("1. Create sample data: python scripts/create_sample_data.py")
        print("2. Run training: python main.py train --model xception --dataset sample --epochs 5")
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create sample data: python scripts/create_sample_data.py")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
