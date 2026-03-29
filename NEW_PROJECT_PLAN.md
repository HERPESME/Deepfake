# 🔬 Bridging the Generalization Gap in Deepfake Detection: Ensemble and Hybrid Strategies for Cross-Domain Robustness

## Research Project Plan — Extension of Deepfake Detection Framework

> **Timeline:** 3 Weeks (March 24 – April 13, 2026)
> **Builds on:** Existing Deepfake Detection codebase (same `src/`, `data/`, `experiments/` infrastructure)
> **Predecessor Paper:** *"Deepfake Detection on Facial Features Using Deep Learning: A Comparative Study of CNN and Transformer Architectures"* (SpringConference)

---

## 🎯 Research Motivation

Our first paper revealed a critical finding: **CNNs achieve near-perfect in-distribution accuracy (98.7%) but catastrophically fail on unseen datasets (dropping to ~50%), while Vision Transformers generalize better (only 24.48% drop) despite lower in-distribution performance.** This creates a clear research gap:

> *Can we combine the precision of CNNs with the robustness of Transformers to build a deepfake detector that is both accurate AND generalizable?*

This new project directly addresses this open problem.

---

## 📖 Proposed Paper Title

**"Bridging the Generalization Gap in Deepfake Detection: Ensemble and Hybrid Strategies for Cross-Domain Robustness"**

### Alternative Titles:
- "Beyond Single-Model Detection: Ensemble and Frequency-Aware Methods for Generalizable Deepfake Forensics"
- "Towards Robust Deepfake Detection: A Multi-Strategy Approach Combining CNNs, Transformers, and Spectral Analysis"

---

## 🧪 Core Research Questions

1. **Ensemble Question:** Can strategic ensemble combinations of CNNs and ViT close the generalization gap while retaining high in-distribution accuracy?
2. **Hybrid Question:** Does the Hybrid CNN-Transformer architecture (already implemented but untested) outperform both standalone CNNs and ViT?
3. **Frequency Domain Question:** Do frequency-domain features (FFT/DCT) capture manipulation artifacts that spatial-domain models miss, improving cross-dataset performance?
4. **Multi-Scale Question:** Does multi-scale feature fusion improve detection of subtle artifacts at different spatial resolutions?
5. **Adversarial Question:** How robust are these approaches against adversarial perturbations designed to fool detectors?

---

## 🌟 Novelty & Expected Contributions

### Contribution 1: Strategic Ensemble Fusion
- **What:** Combine EfficientNet-B0 (highest in-distribution accuracy) + ViT (best cross-dataset resilience) via soft voting, weighted averaging, and learned stacking
- **Why novel:** No existing work has systematically studied ensemble strategies specifically designed to address the CNN-vs-Transformer generalization gap in deepfake detection
- **Expected finding:** Ensemble should achieve >95% in-distribution accuracy (from CNN strength) AND >60% cross-dataset accuracy (from ViT robustness), beating any single model

### Contribution 2: Hybrid CNN-Transformer Evaluation
- **What:** Train and evaluate the `HybridCNNTransformer` model (code exists in `advanced_models.py`, never trained)
- **Why novel:** This architecture extracts CNN features and processes them through Transformer attention — a fundamentally different approach from standalone models
- **Expected finding:** The hybrid should capture both local textures (CNN) and global relationships (Transformer) in a single forward pass

### Contribution 3: Frequency-Domain Augmented Detection
- **What:** Build a new model branch that processes FFT/DCT spectral features alongside spatial features
- **Why novel:** Deepfake artifacts often manifest as specific frequency patterns (GAN fingerprints) that are invisible in spatial domain but clear in frequency domain
- **Expected finding:** Spectral features should be more dataset-agnostic, significantly improving cross-dataset generalization

### Contribution 4: Multi-Scale Feature Fusion Evaluation
- **What:** Train and evaluate the `MultiScaleFeatureFusion` model (code exists in `advanced_models.py`, never trained)
- **Why novel:** Manipulation artifacts exist at different scales — blending boundaries are coarse, texture inconsistencies are fine
- **Expected finding:** Multi-scale features should capture both macro and micro-level artifacts

### Contribution 5: Adversarial Robustness Analysis
- **What:** Test all models against FGSM and PGD adversarial attacks
- **Why novel:** Demonstrates real-world robustness beyond just cross-dataset generalization
- **Expected finding:** Ensemble and hybrid models should be more robust than single models due to attack diversity requirements

---

## 📋 3-Week Execution Plan

---

### WEEK 1: Foundation Models & Ensemble Building (March 24–30)

#### Day 1–2: Hybrid CNN-Transformer Training
- **Task:** Train the existing `HybridCNNTransformer` from `src/models/advanced_models.py` on Celeb-DF
- **Command:**
  ```bash
  python main.py train --model hybrid_cnn_transformer --dataset celebd --data_root data/processed --epochs 30 --batch_size 16 --experiment_name hybrid_celebd
  ```
- **Evaluate:** On both Celeb-DF test set and FaceForensics++ (cross-dataset)
- **Deliverable:** Hybrid model metrics (accuracy, AUC, F1) on both datasets

#### Day 2–3: Multi-Scale Feature Fusion Training
- **Task:** Train the existing `MultiScaleFeatureFusion` from `src/models/advanced_models.py` on Celeb-DF
- **Command:**
  ```bash
  python main.py train --model multiscale --dataset celebd --data_root data/processed --epochs 30 --batch_size 16 --experiment_name multiscale_celebd
  ```
- **Evaluate:** On both Celeb-DF test set and FaceForensics++ (cross-dataset)
- **Deliverable:** Multi-scale model metrics on both datasets

#### Day 4–5: Ensemble Methods Implementation
- **Task:** Create `src/models/ensemble_models.py` with three ensemble strategies:
  1. **Hard Voting:** Majority vote across models
  2. **Soft Voting (Weighted Average):** Weighted probability averaging
  3. **Learned Stacking:** Train a meta-classifier on model outputs

- **Models in Ensemble:**
  - EfficientNet-B0 (from `experiments/effb0_celebd_full/final_model.pth`)
  - ResNet50 (from `experiments/resnet50_celebd_optimized/final_model.pth`)
  - XceptionNet (from best checkpoint in `experiments/xception_celebd/`)
  - ViT (from best checkpoint in `experiments/vit_celebd_optimized/`)

- **Sub-ensembles to test:**
  - CNN-only: EfficientNet + XceptionNet + ResNet50
  - Full: All 4 models
  - Best-of-both: EfficientNet + ViT (precision + robustness)

- **Deliverable:** Ensemble evaluation results on Celeb-DF and FaceForensics++

#### Day 6–7: Results Compilation & Figures
- Generate comparison tables and visualization figures for all Week 1 experiments
- Create training curves, confusion matrices, ROC/PR curves for new models
- **Deliverable:** All figures ready for the paper

---

### WEEK 2: Frequency Domain & Adversarial Testing (March 31–April 6)

#### Day 1–3: Frequency-Domain Feature Extractor
- **Task:** Create `src/models/frequency_models.py` containing:
  1. **FFTFeatureExtractor:** Applies 2D FFT to face images, extracts magnitude spectrum
  2. **FrequencyAwareCNN:** Dual-branch model — spatial CNN branch + frequency CNN branch, late fusion
  3. **DCTFeatureExtractor:** Discrete Cosine Transform features (used in JPEG, may capture compression artifacts)

- **Architecture of FrequencyAwareCNN:**
  ```
  Input Image (224×224)
      ├─→ [Spatial Branch] EfficientNet-B0 → 1280-dim features
      ├─→ [Frequency Branch] FFT → Magnitude → CNN → 512-dim features
      └─→ [Fusion] Concatenate → FC(1792, 512) → FC(512, 2) → Output
  ```

- **Training:**
  ```bash
  python main.py train --model frequency_aware --dataset celebd --data_root data/processed --epochs 30 --batch_size 16 --experiment_name freq_celebd
  ```

- **Evaluate:** On both Celeb-DF and FaceForensics++
- **Key hypothesis:** Frequency features are more domain-invariant, improving cross-dataset performance
- **Deliverable:** Frequency-aware model metrics + spectrum visualizations

#### Day 4–5: Adversarial Robustness Testing
- **Task:** Create `src/evaluation/adversarial.py` with:
  1. **FGSM (Fast Gradient Sign Method):** Single-step white-box attack
  2. **PGD (Projected Gradient Descent):** Multi-step iterative attack
  3. **Test at multiple epsilon values:** ε = {0.01, 0.02, 0.05, 0.1}

- **Models to attack:**
  - All 4 baseline models (EfficientNet, XceptionNet, ResNet50, ViT)
  - Hybrid CNN-Transformer
  - Multi-Scale Fusion
  - Best ensemble
  - Frequency-Aware CNN

- **Metrics:** Accuracy under attack, AUC under attack, robustness score (area under accuracy-vs-epsilon curve)
- **Deliverable:** Adversarial robustness comparison table + accuracy-vs-epsilon plots

#### Day 6–7: Failure Case Analysis & Grad-CAM for New Models
- Run Grad-CAM on all new models (Hybrid, Multi-Scale, Frequency-Aware, Ensemble)
- Identify systematic failure patterns across models
- Analyze which model catches which failure modes
- **Deliverable:** Grad-CAM visualizations for all new models + failure analysis

---

### WEEK 3: Paper Writing, Polish & Presentation (April 7–13)

#### Day 1–2: Comprehensive Results Table
- Compile ALL results into a master comparison table:

| Model | Params | Celeb-DF Acc | Celeb-DF AUC | FF++ Acc | FF++ AUC | Drop | FGSM@0.02 | PGD@0.02 |
|-------|--------|-------------|-------------|---------|---------|------|-----------|----------|
| EfficientNet-B0 | 4.6M | 98.70% | 99.54% | 49.66% | 73.43% | -49.04% | ? | ? |
| XceptionNet | 38.9M | 98.76% | 98.57% | 56.33% | 68.71% | -42.43% | ? | ? |
| ResNet50 | 25M | 98.22% | 99.50% | 56.11% | 72.98% | -42.11% | ? | ? |
| ViT | 86M | 90.54% | 50.35% | 66.06% | 52.75% | -24.48% | ? | ? |
| **Hybrid CNN-T** | ~?M | ? | ? | ? | ? | ? | ? | ? |
| **Multi-Scale** | ~?M | ? | ? | ? | ? | ? | ? | ? |
| **Freq-Aware** | ~?M | ? | ? | ? | ? | ? | ? | ? |
| **Ensemble** | — | ? | ? | ? | ? | ? | ? | ? |

#### Day 3–5: Paper Writing
- **New LaTeX file:** `SpringConference_Package/NewProject.tex` (or appropriate venue format)
- Paper structure:

  1. **Abstract:** Focus on generalization gap problem + our multi-strategy solution
  2. **Introduction:** Motivation from Paper 1 findings, the generalization problem as central issue
  3. **Related Work:** Ensemble methods in deepfake detection, frequency-domain approaches, adversarial robustness studies
  4. **Methodology:**
     - Ensemble strategies (voting, stacking)
     - Hybrid CNN-Transformer architecture
     - Frequency-aware dual-branch model (novel contribution)
     - Multi-scale feature fusion
     - Adversarial attack methodology
  5. **Experiments & Results:**
     - In-distribution comparison (Celeb-DF)
     - Cross-dataset generalization (FaceForensics++)
     - Adversarial robustness analysis
     - Ablation: which components matter most
     - Model interpretability (Grad-CAM comparison)
  6. **Discussion:** Why frequency features help, why ensemble bridges the gap, practical implications
  7. **Conclusion & Future Work**

#### Day 6–7: Polish, Review & Presentation
- Review all figures and tables for consistency
- Proofread paper
- Prepare presentation slides if needed
- Final codebase cleanup and documentation

---

## 📂 New Files to Create

| File | Purpose | Week |
|------|---------|------|
| `src/models/ensemble_models.py` | Ensemble strategies (voting, stacking) | 1 |
| `src/models/frequency_models.py` | FFT/DCT feature extractors + dual-branch model | 2 |
| `src/evaluation/adversarial.py` | FGSM/PGD adversarial attacks | 2 |
| `SpringConference_Package/NewProject.tex` | New research paper | 3 |

## 📂 Existing Files to Modify

| File | Change | Week |
|------|--------|------|
| `main.py` | Add `ensemble`, `frequency_aware` model support + adversarial evaluation mode | 1–2 |
| `src/models/advanced_models.py` | Minor fixes if needed during hybrid/multiscale training | 1 |

## 📂 Existing Code That Can Be Reused As-Is

| File | What's Reused |
|------|--------------|
| `src/models/advanced_models.py` → `HybridCNNTransformer` | Train it, no code changes needed |
| `src/models/advanced_models.py` → `MultiScaleFeatureFusion` | Train it, no code changes needed |
| `src/data/dataloader.py` | All data loading |
| `src/training/trainer.py` | Training loop + cross-dataset evaluation |
| `src/evaluation/metrics.py` | All metrics |
| `src/explainability/gradcam.py` | Grad-CAM for new models |
| `src/reporting/report_generator.py` | Report generation |
| All trained model checkpoints in `experiments/` | Ensemble building |

---

## 🔬 Expected Research Narrative

The paper tells this story:

1. **Problem:** Paper 1 showed CNNs overfit to dataset-specific artifacts while ViT generalizes but underperforms
2. **Hypothesis:** Combining multiple detection strategies — spatial CNNs, global attention (ViT), frequency-domain analysis, and multi-scale features — can produce a detector that is both accurate AND robust
3. **Approach:** We test 4 strategies: ensemble fusion, hybrid architectures, frequency-aware dual-branch models, and multi-scale feature fusion
4. **Evidence:** Cross-dataset evaluation + adversarial robustness testing, with Grad-CAM interpretability analysis
5. **Finding:** [To be determined by experiments — expected: ensemble of EfficientNet+ViT + frequency features achieves best overall balance of accuracy and generalizability]

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Hybrid/MultiScale training takes too long on MPS/CPU | Use smaller batch size (8-16), reduce epochs to 20, early stopping |
| Ensemble doesn't improve cross-dataset | Focus on frequency domain results as the novel contribution |
| Frequency-aware model is complex to implement | Start with simple FFT magnitude concatenation, iterate |
| ViT checkpoints don't have `final_model.pth` | Use best available checkpoint (epoch 15) for ensemble |
| XceptionNet missing `final_model.pth` | Use best available checkpoint (epoch 10) for ensemble |

---

## 📊 Success Criteria

The project is successful if we can demonstrate at least ONE of:
1. An ensemble or hybrid approach that achieves **>95% Celeb-DF accuracy AND >60% FaceForensics++ accuracy** (beating any single model on both)
2. A frequency-aware model that significantly improves cross-dataset AUC compared to purely spatial models
3. A clear adversarial robustness advantage for ensemble/hybrid approaches over single models

---

*Created: March 24, 2026*
*Project: Deepfake Detection — Phase 2 Research*
