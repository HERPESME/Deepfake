# Deepfake Detection Research

A comprehensive deepfake detection research project comprising two papers: a **Comparative Study** of CNN and Transformer architectures, and a follow-up study on **Bridging the Generalization Gap** through ensemble and hybrid strategies.

## Project Structure

```
Deepfake/
├── papers/                          # Research papers & LaTeX assets
│   ├── comparative_study/           # Paper 1: Comparative Study
│   │   ├── SpringConference.tex
│   │   └── figures/                 # Confusion matrices, ROC/PR curves, Grad-CAM
│   │
│   └── bridging_the_gap/            # Paper 2: Bridging the Generalization Gap
│       ├── NewProject.tex
│       └── figures/                 # Adversarial plots, architecture diagrams, Grad-CAM grids
│
├── docs/                            # Documentation
│   ├── comparative_study/           # Docs specific to Paper 1
│   ├── bridging_the_gap/            # Docs specific to Paper 2
│   └── shared/                      # Docs applicable to both projects
│
├── src/                             # Core source code
│   ├── data/                        # Data loading and preprocessing
│   ├── models/                      # Model architectures (if applicable)
│   ├── training/                    # Training loop and utilities
│   ├── evaluation/                  # Metrics and adversarial evaluation
│   ├── explainability/              # Grad-CAM and interpretability tools
│   ├── reporting/                   # Report generation
│   └── utils/                       # Configuration and utility functions
│
├── tools/                           # Utility scripts
│   ├── generate_arch_diagrams.py    # Architecture diagram generator
│   ├── generate_paper_plots.py      # Adversarial plot generator
│   ├── run_ensemble_eval.py         # Ensemble evaluation pipeline
│   ├── run_phase5_eval.py           # Phase 5 evaluation pipeline
│   ├── run_phase6_gradcam.py        # Grad-CAM generation pipeline
│   └── ...                          # Other utility scripts
│
├── scripts/                         # Shell scripts for pipeline automation
├── configs/                         # Training configuration (YAML)
├── notebooks/                       # Jupyter notebooks for experimentation
├── figures/                         # Root-level output figures (Phase 1)
├── data/                            # Datasets (gitignored — download separately)
├── main.py                          # Main pipeline entry point
└── requirements.txt                 # Python dependencies
```

## Papers

### Paper 1: Comparative Study
**"Deepfake Detection on Facial Features Using Deep Learning: A Comparative Study of CNN and Transformer Architectures"**

Evaluates EfficientNet-B0, XceptionNet, ResNet50, and Vision Transformer on Celeb-DF and FaceForensics++ datasets. Establishes baseline performance and identifies the generalization gap.

→ [`papers/comparative_study/SpringConference.tex`](papers/comparative_study/SpringConference.tex)

### Paper 2: Bridging the Generalization Gap
**"Bridging the Generalization Gap in Deepfake Detection: Ensemble and Hybrid Strategies for Cross-Domain Robustness"**

Introduces three novel architectures (Hybrid CNN-Transformer, Multi-Scale Feature Fusion, Frequency-Aware CNN) and comprehensive ensemble/adversarial evaluation. Key findings include ViT's adversarial invariance and the three-tier robustness hierarchy.

→ [`papers/bridging_the_gap/NewProject.tex`](papers/bridging_the_gap/NewProject.tex)

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the main pipeline
python main.py
```

For detailed setup and training instructions, see [`docs/shared/QUICKSTART.md`](docs/shared/QUICKSTART.md) and [`docs/shared/TRAIN_CELEBDF.md`](docs/shared/TRAIN_CELEBDF.md).

## Key Results

| Model | Celeb-DF Acc. | Celeb-DF AUC | FF++ Acc. | Params |
|-------|:---:|:---:|:---:|:---:|
| EfficientNet-B0 | 98.70% | 99.54% | 49.66% | 5.3M |
| XceptionNet | 98.76% | 98.57% | 56.33% | 38.9M |
| Vision Transformer | 90.54% | 50.35% | **66.06%** | 86.6M |
| Frequency-Aware CNN | 99.03% | **99.84%** | 54.64% | 5.5M |
| **Ensemble (Soft-Vote)** | **99.51%** | 99.80% | 55.43% | — |

## License

See [LICENSE](LICENSE) for details.
