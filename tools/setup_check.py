#!/usr/bin/env python3
"""
Setup verification script for Deepfake Detection Project.
Checks dependencies, environment, and data availability.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(
            f"   ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)"
        )
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    import_name = import_name or package_name
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"   ✅ {package_name} ({version})")
            return True
        except Exception as e:
            print(f"   ⚠️  {package_name} (found but error: {e})")
            return False
    else:
        print(f"   ❌ {package_name} (not installed)")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\n📦 Checking dependencies...")

    packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "tensorflow": "tensorflow",
        "transformers": "transformers",
        "timm": "timm",
        "opencv-python": "cv2",
        "Pillow": "PIL",
        "albumentations": "albumentations",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "omegaconf": "omegaconf",
    }

    results = {}
    for package, import_name in packages.items():
        results[package] = check_package(package, import_name)

    return results


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\n📦 Checking optional dependencies...")

    packages = {
        "mtcnn": "mtcnn",
        "dlib": "dlib",
        "face-recognition": "face_recognition",
        "wandb": "wandb",
        "tensorboard": "tensorboard",
        "reportlab": "reportlab",
        "shap": "shap",
        "grad-cam": "pytorch_grad_cam",
    }

    results = {}
    for package, import_name in packages.items():
        results[package] = check_package(package, import_name)

    return results


def check_gpu():
    """Check GPU availability."""
    print("\n🖥️  Checking GPU availability...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"   ✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️  No GPU found (will use CPU)")
            return False
    except:
        print("   ❌ Cannot check GPU (torch not installed)")
        return False


def check_data_directories():
    """Check data directory structure."""
    print("\n📁 Checking data directories...")

    base_path = Path("data")
    dirs_to_check = {
        "data/raw": "Raw datasets",
        "data/processed": "Processed datasets",
        "data/splits": "Data splits",
    }

    results = {}
    for dir_path, description in dirs_to_check.items():
        path = Path(dir_path)
        if path.exists():
            count = len(list(path.iterdir())) if path.is_dir() else 0
            print(f"   ✅ {dir_path} ({count} items)")
            results[dir_path] = True
        else:
            print(f"   ❌ {dir_path} (missing)")
            results[dir_path] = False

    return results


def check_datasets():
    """Check available datasets."""
    print("\n📊 Checking datasets...")

    datasets = ["faceforensics", "celebd", "dfdc", "wilddeepfake"]
    processed_path = Path("data/processed")

    available = []
    for dataset in datasets:
        dataset_path = processed_path / dataset
        if dataset_path.exists() and any(dataset_path.iterdir()):
            print(f"   ✅ {dataset}")
            available.append(dataset)
        else:
            print(f"   ❌ {dataset} (not found)")

    return available


def check_config_files():
    """Check configuration files."""
    print("\n⚙️  Checking configuration files...")

    configs = {
        "configs/training_config.yaml": "Training config",
    }

    results = {}
    for config_path, description in configs.items():
        path = Path(config_path)
        if path.exists():
            print(f"   ✅ {config_path}")
            results[config_path] = True
        else:
            print(f"   ❌ {config_path} (missing)")
            results[config_path] = False

    return results


def create_missing_directories():
    """Create missing directories."""
    print("\n🔨 Creating missing directories...")

    dirs = [
        "data/raw",
        "data/processed",
        "data/splits",
        "experiments",
        "reports",
        "logs",
    ]

    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created {dir_path}")
        else:
            print(f"   ⏭️  {dir_path} already exists")


def install_missing_packages(missing_packages):
    """Install missing packages."""
    if not missing_packages:
        print("\n✅ All required packages are installed!")
        return

    print(f"\n⚠️  {len(missing_packages)} packages are missing")
    print("   Missing:", ", ".join(missing_packages))

    response = input("\n❓ Would you like to install missing packages? (y/n): ")
    if response.lower() == "y":
        print("\n📥 Installing missing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Installation complete!")
    else:
        print("\n⚠️  Please install manually: pip install " + " ".join(missing_packages))


def generate_report(all_results):
    """Generate final report."""
    print("\n" + "=" * 80)
    print("📋 SETUP VERIFICATION REPORT")
    print("=" * 80)

    # Count results
    total_checks = sum(
        len(v) if isinstance(v, dict) else 1 for v in all_results.values()
    )
    passed_checks = sum(
        sum(v.values()) if isinstance(v, dict) else (1 if v else 0)
        for v in all_results.values()
    )

    print(f"\nOverall Status: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("\n✅ Your environment is ready!")
        print("\nNext steps:")
        print("  1. Download datasets: python scripts/download_datasets.py")
        print(
            "  2. Preprocess data: python src/data/preprocessing.py --dataset faceforensics --data_path data/raw/faceforensics --output_path data/processed/faceforensics"
        )
        print(
            "  3. Train model: python main.py train --model xception --dataset faceforensics"
        )
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Create directories: Run this script again")
        print("  - Download datasets: python scripts/download_datasets.py")


def main():
    """Main verification function."""
    print("=" * 80)
    print("🔍 DEEPFAKE DETECTION PROJECT - SETUP VERIFICATION")
    print("=" * 80)

    all_results = {}

    # Run checks
    all_results["python"] = check_python_version()
    all_results["dependencies"] = check_dependencies()
    all_results["optional"] = check_optional_dependencies()
    all_results["gpu"] = check_gpu()

    # Create missing directories
    create_missing_directories()

    # Check structure
    all_results["directories"] = check_data_directories()
    all_results["datasets"] = check_datasets()
    all_results["configs"] = check_config_files()

    # Find missing packages
    missing_core = [
        pkg for pkg, installed in all_results["dependencies"].items() if not installed
    ]

    # Generate report
    generate_report(all_results)

    # Offer to install
    if missing_core:
        install_missing_packages(missing_core)


if __name__ == "__main__":
    main()
