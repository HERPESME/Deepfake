#!/bin/bash

# GitHub Upload Helper Script
# Run this to prepare and push your Deepfake Detection project to GitHub

set -e

echo "=========================================="
echo "🚀 GITHUB UPLOAD PREPARATION"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    echo "Run: git init"
    exit 1
fi

# Show current git status
echo "📊 Current repository status:"
echo ""
git status --short | head -20
echo ""
echo "Total files to commit: $(git status --short | wc -l)"
echo ""

# Show what's being ignored
echo "🚫 Files being ignored (will NOT be uploaded):"
echo ""
git status --ignored --short | grep '^!!' | head -15
echo ""

# Summary
echo "=========================================="
echo "📋 WHAT WILL BE UPLOADED TO GITHUB:"
echo "=========================================="
echo ""
echo "✅ Source code (src/)"
echo "✅ Configuration files (configs/)"
echo "✅ Documentation (*.md)"
echo "✅ Scripts (scripts/, *.sh, *.py)"
echo "✅ Notebooks (notebooks/)"
echo "✅ Figures and results visualizations"
echo "✅ Requirements and setup files"
echo ""
echo "❌ NOT uploaded (as per .gitignore):"
echo "❌ Virtual environments (deepfake_env/, .venv/)"
echo "❌ Large datasets (data/raw/faceforensics, data/raw/celebd, etc.)"
echo "❌ Processed datasets (data/processed/*)"
echo "❌ Model checkpoints (experiments/**/*.pth)"
echo "❌ Python cache (__pycache__/)"
echo "❌ Environment files (.env)"
echo ""

# Ask for confirmation
read -p "❓ Ready to commit? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
fi

# Commit
echo ""
echo "📝 Creating commit..."
git add -A

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "⚠️  No changes to commit"
    exit 0
fi

# Create commit
read -p "📝 Enter commit message (or press Enter for default): " commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Add deepfake detection project with multiple models, training pipeline, and comprehensive documentation"
fi

git commit -m "$commit_msg"

echo ""
echo "✅ Commit created successfully!"
echo ""

# Show commit info
git log -1 --stat | head -30

echo ""
echo "=========================================="
echo "📤 NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. Create a GitHub repository:"
echo "   - Go to: https://github.com/new"
echo "   - Name: Deepfake-Detection"
echo "   - Description: State-of-the-art deepfake detection using CNNs and Vision Transformers"
echo "   - Public or Private (your choice)"
echo "   - Do NOT initialize with README (we already have one)"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/Deepfake-Detection.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. (Optional) If you want to share model weights:"
echo "   - Install Git LFS: brew install git-lfs && git lfs install"
echo "   - Uncomment model files in .gitignore"
echo "   - Track with LFS: git lfs track '*.pth'"
echo "   - Commit and push: git add .gitattributes && git commit -m 'Add LFS tracking' && git push"
echo ""
echo "=========================================="
echo "✅ READY TO PUSH!"
echo "=========================================="
