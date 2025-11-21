# Deepfake Detection Project
## State-of-the-Art Deepfake Detection for Image and Forensics Security Course

This project implements a comprehensive, state-of-the-art deepfake detection system that combines cutting-edge machine learning techniques with practical forensic analysis tools. It's designed to be both educational and research-oriented, providing a solid foundation for understanding and advancing deepfake detection technology.

## 🎯 Project Overview

This project implements cutting-edge deepfake detection models using the latest techniques including Vision Transformers, CLIP-based contrastive learning, and hybrid CNN-Transformer architectures. It combines comprehensive evaluation and explainability tools to create a robust deepfake detection pipeline.

## 🚀 Key Features

### **Latest Models & Techniques**
- **Vision Transformers (ViT)**: Pure transformer architecture for global context understanding
- **CLIP-based Detection**: Contrastive learning approach leveraging pre-trained CLIP models
- **Hybrid CNN-Transformer**: Combines CNN feature extraction with transformer attention
- **Contrastive Learning**: Self-supervised learning for better feature representations
- **Multi-scale Feature Fusion**: Combines features from multiple scales and resolutions

### **Comprehensive Dataset Support**
- **FaceForensics++**: 1,000 videos with 4 manipulation methods
- **Celeb-DF (v2)**: High-quality celebrity deepfakes (5,639 fake + 590 real)
- **DFDC**: Massive dataset with 100,000+ clips from 3,426 actors
- **WildDeepfake**: Real-world internet-sourced deepfakes (most challenging)

### **Advanced Evaluation**
- **Cross-dataset Testing**: Robust generalization evaluation
- **Comprehensive Metrics**: AUC, accuracy, precision, recall, F1-score
- **Statistical Analysis**: Mean, std, min, max across datasets
- **Generalization Gap Analysis**: Overfitting detection

### **Explainability & Interpretability**
- **Grad-CAM**: Visual explanations of model decisions
- **Grad-CAM++**: Improved localization and accuracy
- **Attention Visualization**: Transformer attention heatmaps
- **SHAP Analysis**: Feature importance analysis
- **Forensic Reports**: Detailed analysis with visualizations

### **Automated Reporting**
- **PDF Reports**: Professional forensic analysis reports
- **HTML Reports**: Interactive web-based reports
- **JSON Summaries**: Machine-readable experiment data
- **Visualization Suite**: Comprehensive charts and graphs

## 📁 Project Structure

```
Deepfake/
├── data/                          # Data management
│   ├── raw/                      # Raw dataset downloads
│   ├── processed/                # Preprocessed face crops
│   └── splits/                   # Train/val/test splits
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── preprocessing.py      # Dataset preprocessing pipeline
│   │   └── dataloader.py        # PyTorch data loaders
│   ├── models/                   # Model architectures
│   │   ├── baseline_models.py    # XceptionNet, EfficientNet, ViT
│   │   └── advanced_models.py    # CLIP, Hybrid, Contrastive
│   ├── training/                 # Training pipeline
│   │   └── trainer.py            # Comprehensive trainer
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py            # Metrics and visualizations
│   ├── explainability/          # Model interpretability
│   │   └── gradcam.py           # Grad-CAM, attention, SHAP
│   ├── reporting/               # Report generation
│   │   └── report_generator.py  # PDF/HTML report generation
│   └── utils/                   # Utility functions
│       └── config.py            # Configuration management
├── configs/                     # Configuration files
│   └── training_config.yaml     # Training configuration
├── scripts/                     # Utility scripts
│   └── download_datasets.py     # Dataset download helper
├── notebooks/                   # Jupyter notebooks
│   └── deepfake_detection_demo.ipynb  # Complete demo
├── reports/                     # Generated reports
├── experiments/                 # Training experiments
└── tests/                       # Unit tests
```

## 🛠️ Technical Implementation

### **Model Architectures**

1. **Baseline Models**
   - **XceptionNet**: Proven CNN baseline from FaceForensics++
   - **EfficientNet**: Efficient scaling with compound coefficients
   - **Vision Transformer**: Pure transformer with patch embeddings
   - **ResNet Variants**: ResNet18/34/50/101/152

2. **Advanced Models**
   - **CLIP-based Detector**: Leverages OpenAI's CLIP for visual understanding
   - **Hybrid CNN-Transformer**: Best of both CNN and transformer worlds
   - **Contrastive Learning**: Self-supervised feature learning
   - **Multi-scale Fusion**: Combines multiple feature scales

### **Training Pipeline**
- **Advanced Optimizers**: Adam, AdamW, SGD with momentum
- **Learning Rate Scheduling**: Cosine, Step, Plateau, Warmup-Cosine
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Mixed Precision**: Optional FP16 training for efficiency
- **Gradient Clipping**: Prevents exploding gradients
- **Label Smoothing**: Regularization technique
- **Focal Loss**: Handles class imbalance

### **Data Processing**
- **Face Detection**: MTCNN, dlib, OpenCV cascade classifiers
- **Face Cropping**: Automatic face extraction with padding
- **Data Augmentation**: Albumentations for robust training
- **Normalization**: ImageNet normalization standards
- **Batch Processing**: Efficient data loading with multiple workers

## 📊 Performance Metrics

### **Standard Metrics**
- **Accuracy**: Overall classification accuracy
- **AUC**: Area Under ROC Curve (primary metric)
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **Average Precision**: Area under precision-recall curve

### **Cross-Dataset Evaluation**
- **Generalization Testing**: Train on one dataset, test on others
- **Performance Drop Analysis**: Quantify overfitting
- **Statistical Aggregation**: Mean, std, min, max across datasets
- **Robustness Assessment**: Model stability across domains

## 🔍 Explainability Features

### **Visual Explanations**
- **Grad-CAM**: Highlights important regions in input images
- **Grad-CAM++**: Improved localization and accuracy
- **Attention Maps**: Visualize transformer attention patterns
- **Feature Visualizations**: Understand learned representations

### **Quantitative Analysis**
- **SHAP Values**: Feature importance scores
- **Confidence Analysis**: Model uncertainty quantification
- **Error Analysis**: Detailed failure case examination
- **Forensic Reports**: Comprehensive analysis documentation

## 📈 Reporting System

### **Automated Report Generation**
- **PDF Reports**: Professional forensic analysis reports
- **HTML Reports**: Interactive web-based reports
- **JSON Summaries**: Machine-readable experiment data
- **Visualization Suite**: Comprehensive charts and graphs

### **Report Contents**
- **Executive Summary**: High-level performance overview
- **Training Curves**: Loss, accuracy, AUC over time
- **Confusion Matrices**: Detailed classification analysis
- **ROC/PR Curves**: Performance visualization
- **Cross-dataset Comparison**: Generalization analysis
- **Model Comparison**: Architecture performance comparison
- **Conclusions**: Key findings and recommendations

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Deepfake
```

2. **Create a virtual environment**:
```bash
python -m venv deepfake_env
source deepfake_env/bin/activate  # On Windows: deepfake_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

### **1. Dataset Setup**
```bash
# Download datasets (manual process)
python scripts/download_datasets.py --dataset all

# Check dataset structure
python scripts/download_datasets.py --check_structure
```

### **2. Data Preprocessing**
```bash
# Preprocess FaceForensics++ dataset
python src/data/preprocessing.py --dataset faceforensics --data_path data/raw/faceforensics --output_path data/processed/faceforensics

# Preprocess Celeb-DF dataset
python src/data/preprocessing.py --dataset celebd --data_path data/raw/celebd --output_path data/processed/celebd
```

### **3. Model Training**
```bash
# Train XceptionNet on FaceForensics++
python main.py train --model xception --dataset faceforensics --data_root data/processed --epochs 50

# Train Vision Transformer
python main.py train --model vit --dataset faceforensics --data_root data/processed --epochs 50
```

### **4. Evaluation**
```bash
# Evaluate trained model
python main.py evaluate --model_path experiments/xception_faceforensics/final_model.pth --dataset celebd --data_root data/processed

# Cross-dataset evaluation
python main.py cross-dataset --model xception --train_dataset faceforensics --test_datasets celebd dfdc
```

**Note**: Reports are automatically generated after training. For manual report generation, see `src/reporting/report_generator.py` - `main()` function.

### **5. Quick Demo**
```bash
# Run the complete demo notebook
jupyter notebook notebooks/deepfake_detection_demo.ipynb
```

## 📊 Datasets

### FaceForensics++
- **Size**: 1,000 original videos manipulated by 4 methods
- **Methods**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **Download**: Requires Google Form request

### Celeb-DF (v2)
- **Size**: 590 real videos + 5,639 deepfake videos
- **Quality**: High-quality modern deepfakes
- **Use**: Cross-dataset testing

### DFDC (DeepFake Detection Challenge)
- **Size**: 100,000+ video clips from 3,426 actors
- **Methods**: Multiple face-swap techniques
- **Use**: Large-scale training

### WildDeepfake
- **Size**: 7,314 face sequences from 707 videos
- **Source**: Internet-sourced real-world manipulations
- **Challenge**: Most realistic test set

## 🎓 Educational Value

### **For Image and Forensics Security Course**
- **Comprehensive Coverage**: End-to-end deepfake detection pipeline
- **State-of-the-Art Techniques**: Latest models and methods
- **Practical Implementation**: Real-world applicable code
- **Research Integration**: Incorporates recent academic advances
- **Industry Standards**: Follows best practices and conventions

### **Learning Objectives**
1. **Understand Deepfake Technology**: How deepfakes are created and detected
2. **Master Computer Vision**: CNN, transformer, and hybrid architectures
3. **Learn Evaluation Methods**: Comprehensive metrics and cross-dataset testing
4. **Explore Explainability**: Model interpretability and forensic analysis
5. **Develop Research Skills**: Experimentation, analysis, and reporting

## 🔬 Research Contributions

### **Novel Approaches**
- **CLIP-based Detection**: Leveraging contrastive learning for deepfake detection
- **Hybrid Architectures**: Combining CNN and transformer strengths
- **Multi-scale Fusion**: Improved feature representation
- **Comprehensive Evaluation**: Cross-dataset generalization analysis

### **Technical Innovations**
- **Automated Pipeline**: End-to-end processing from raw data to reports
- **Explainability Integration**: Built-in interpretability tools
- **Cross-dataset Testing**: Robust evaluation methodology
- **Professional Reporting**: Automated forensic analysis reports

## 📚 References

### **Key Papers**
1. **FaceForensics++**: Rössler et al. "FaceForensics++: Learning to Detect Manipulated Facial Images"
2. **Celeb-DF**: Li et al. "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics"
3. **DFDC**: Dolhansky et al. "The DeepFake Detection Challenge (DFDC) Dataset"
4. **WildDeepfake**: Zi et al. "WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection"
5. **Vision Transformer**: Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
6. **CLIP**: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision"

### **Technical Resources**
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer library
- **Albumentations**: Advanced data augmentation
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **SHAP**: SHapley Additive exPlanations

## 🎯 Future Enhancements

### **Potential Improvements**
1. **Real-time Detection**: Optimize for live video processing
2. **Audio-Visual Fusion**: Combine visual and audio features
3. **Temporal Modeling**: Leverage video sequence information
4. **Adversarial Training**: Improve robustness against attacks
5. **Federated Learning**: Distributed training across multiple sources

### **Research Directions**
1. **Novel Architectures**: Explore new model designs
2. **Self-supervised Learning**: Reduce dependency on labeled data
3. **Domain Adaptation**: Better cross-domain generalization
4. **Interpretability**: Enhanced explainability methods
5. **Deployment**: Production-ready model serving

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{deepfake_detection_2024,
  title={State-of-the-Art Deepfake Detection: A Comprehensive Framework},
  author={Your Name},
  year={2024},
  institution={Your Institution}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions, issues, or contributions:
- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Contributions**: Submit pull requests for improvements
- **Documentation**: See `QUICKSTART.md` for quick setup or `PROJECT_ARCHITECTURE_GUIDE.md` for detailed technical documentation

---

## 📚 **Additional Documentation**

- **`QUICKSTART.md`** - Quick 3-step setup guide and common commands
- **`PROJECT_ARCHITECTURE_GUIDE.md`** - Complete technical architecture, how everything works, and detailed learning guide

---
