#!/bin/bash
# Script to copy visualization images to figures directory for LaTeX

FIGURES_DIR="figures"

# Copy Celeb-DF confusion matrices
cp reports/evaluation/visualizations/efficientnet_b0_celebd/confusion_matrix.png ${FIGURES_DIR}/efficientnet_b0_celebd_confusion_matrix.png
cp reports/evaluation/visualizations/xception_celebd/confusion_matrix.png ${FIGURES_DIR}/xception_celebd_confusion_matrix.png
cp reports/evaluation/visualizations/resnet50_celebd/confusion_matrix.png ${FIGURES_DIR}/resnet50_celebd_confusion_matrix.png
cp reports/evaluation/visualizations/vit_celebd/confusion_matrix.png ${FIGURES_DIR}/vit_celebd_confusion_matrix.png

# Copy Celeb-DF ROC curves
cp reports/evaluation/visualizations/efficientnet_b0_celebd/roc_curve.png ${FIGURES_DIR}/efficientnet_b0_celebd_roc_curve.png
cp reports/evaluation/visualizations/xception_celebd/roc_curve.png ${FIGURES_DIR}/xception_celebd_roc_curve.png
cp reports/evaluation/visualizations/resnet50_celebd/roc_curve.png ${FIGURES_DIR}/resnet50_celebd_roc_curve.png
cp reports/evaluation/visualizations/vit_celebd/roc_curve.png ${FIGURES_DIR}/vit_celebd_roc_curve.png

# Copy FaceForensics++ confusion matrices
cp reports/evaluation/visualizations/efficientnet_b0_faceforensics/confusion_matrix.png ${FIGURES_DIR}/efficientnet_b0_faceforensics_confusion_matrix.png
cp reports/evaluation/visualizations/xception_faceforensics/confusion_matrix.png ${FIGURES_DIR}/xception_faceforensics_confusion_matrix.png
cp reports/evaluation/visualizations/resnet50_faceforensics/confusion_matrix.png ${FIGURES_DIR}/resnet50_faceforensics_confusion_matrix.png
cp reports/evaluation/visualizations/vit_faceforensics/confusion_matrix.png ${FIGURES_DIR}/vit_faceforensics_confusion_matrix.png

# Copy Grad-CAM visualizations
cp reports/explainability/id0_id1_0000_frame000000_frame_000000_efficientb0_gradcam_overlay.png ${FIGURES_DIR}/efficientnet_b0_gradcam_fake.png 2>/dev/null || true
cp reports/explainability/id0_0000_frame000000_frame_000000_efficientb0_gradcam_overlay.png ${FIGURES_DIR}/efficientnet_b0_gradcam_real.png 2>/dev/null || true
cp reports/explainability/id0_id1_0000_frame000000_frame_000000_xception_gradcam_overlay.png ${FIGURES_DIR}/xception_gradcam_fake.png 2>/dev/null || true
cp reports/explainability/id0_0000_frame000000_frame_000000_xception_gradcam_overlay.png ${FIGURES_DIR}/xception_gradcam_real.png 2>/dev/null || true

echo "Images copied to ${FIGURES_DIR}/ directory"
echo "Total images: $(ls ${FIGURES_DIR}/*.png 2>/dev/null | wc -l | xargs)"
