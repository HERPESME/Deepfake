#!/usr/bin/env python3
"""
Generate comprehensive Viva Preparation PDF for Deepfake Detection Project.
Explains code, algorithms, models, classifiers, and results interpretation.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from pathlib import Path
import json
import os

# Color scheme
PRIMARY_COLOR = colors.HexColor('#1a237e')
SECONDARY_COLOR = colors.HexColor('#3949ab')
ACCENT_COLOR = colors.HexColor('#5c6bc0')
TEXT_COLOR = colors.HexColor('#212121')
LIGHT_BG = colors.HexColor('#f5f5f5')

def get_results_data():
    """Load results from experiment directories."""
    results = {}
    
    models = ['efficientnet_b0', 'xception', 'resnet50', 'vit']
    datasets = ['celebd', 'faceforensics']
    
    for model in models:
        results[model] = {}
        for dataset in datasets:
            exp_dir = Path(f"experiments/{model}_{dataset}_optimized")
            if not exp_dir.exists():
                # Try alternative naming
                if model == 'xception' and dataset == 'celebd':
                    exp_dir = Path("experiments/xception_celebd")
                elif model == 'efficientnet_b0' and dataset == 'celebd':
                    exp_dir = Path("experiments/effb0_celebd_full")
            
            results_file = exp_dir / "final_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results[model][dataset] = json.load(f)
            else:
                results[model][dataset] = None
    
    return results

def create_viva_pdf(output_path="viva_prep.pdf"):
    """Create comprehensive viva preparation PDF."""
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define custom styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Heading styles
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=PRIMARY_COLOR,
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=SECONDARY_COLOR,
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    # Code style
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        backColor=LIGHT_BG,
        borderPadding=10,
        spaceAfter=10
    )
    
    # Body style
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=TEXT_COLOR,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Deepfake Detection Project", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Viva Voce Preparation Guide", ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=SECONDARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=30
    )))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Comprehensive Guide to Code, Algorithms, Models, and Results",
        ParagraphStyle('Description', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, textColor=TEXT_COLOR)
    ))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading1_style))
    toc_items = [
        "1. Project Overview",
        "2. Data Preprocessing Pipeline",
        "3. Model Architectures Explained",
        "4. Training Process & Algorithms",
        "5. Evaluation Metrics & Interpretation",
        "6. Results Analysis",
        "7. Code Structure & Key Components",
        "8. Key Algorithms Explained",
        "9. Common Viva Questions & Answers"
    ]
    
    for item in toc_items:
        story.append(Paragraph(f"• {item}", body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # ========== SECTION 1: PROJECT OVERVIEW ==========
    story.append(Paragraph("1. Project Overview", heading1_style))
    story.append(Paragraph(
        "<b>Objective:</b> Develop and compare deep learning models for detecting deepfake images and videos.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Approach:</b> Train and evaluate four different architectures (EfficientNet-B0, XceptionNet, ResNet50, Vision Transformer) on Celeb-DF dataset, then test generalization on FaceForensics++.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Innovation:</b> Cross-dataset evaluation to assess model robustness and generalization capabilities.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Project Pipeline
    story.append(Paragraph("Project Pipeline:", heading2_style))
    pipeline_steps = [
        "1. <b>Data Preprocessing:</b> Extract frames from videos → Detect faces → Crop and resize → Normalize",
        "2. <b>Model Training:</b> Train 4 models on Celeb-DF with optimized hyperparameters",
        "3. <b>In-Distribution Evaluation:</b> Test on Celeb-DF test set",
        "4. <b>Cross-Dataset Evaluation:</b> Test on FaceForensics++ (different distribution)",
        "5. <b>Explainability Analysis:</b> Generate Grad-CAM visualizations to understand model decisions"
    ]
    for step in pipeline_steps:
        story.append(Paragraph(step, body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # ========== SECTION 2: DATA PREPROCESSING ==========
    story.append(Paragraph("2. Data Preprocessing Pipeline", heading1_style))
    
    story.append(Paragraph(
        "The preprocessing pipeline converts raw video files into standardized face images ready for model training.",
        body_style
    ))
    
    story.append(Paragraph("2.1 Frame Extraction", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/data/preprocessing.py</code> - <code>VideoProcessor.extract_frames()</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it works:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Open video file using OpenCV's VideoCapture",
        body_style
    ))
    story.append(Paragraph(
        "• Extract frames at regular intervals (every 30th frame for Celeb-DF, every 60th for FaceForensics++)",
        body_style
    ))
    story.append(Paragraph(
        "• Limit frames per video (max 10) to balance dataset size and diversity",
        body_style
    ))
    story.append(Paragraph(
        "• Save frames as temporary image files",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("2.2 Face Detection", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/data/preprocessing.py</code> - <code>FaceDetector.detect_faces()</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Algorithm:</b> OpenCV Haar Cascade Classifier",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it works:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Convert image to grayscale (face detection works on grayscale)",
        body_style
    ))
    story.append(Paragraph(
        "• Apply Haar Cascade classifier: Uses pre-trained features (edges, lines) to detect face patterns",
        body_style
    ))
    story.append(Paragraph(
        "• Returns bounding boxes (x, y, width, height) for detected faces",
        body_style
    ))
    story.append(Paragraph(
        "• Select largest face if multiple faces detected",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("2.3 Face Cropping & Resizing", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/data/preprocessing.py</code> - <code>FaceDetector.crop_face()</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Steps:</b>",
        body_style
    ))
    story.append(Paragraph(
        "1. Add 20% padding around detected face (preserves context)",
        body_style
    ))
    story.append(Paragraph(
        "2. Crop face region from original image",
        body_style
    ))
    story.append(Paragraph(
        "3. Resize to 224×224 pixels (standard input size for all models)",
        body_style
    ))
    story.append(Paragraph(
        "4. Normalize pixel values to [0, 1] range",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("2.4 Data Splitting", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/data/preprocessing.py</code> - <code>DatasetPreprocessor.create_splits()</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Split Ratio:</b> 80% Training, 10% Validation, 10% Test",
        body_style
    ))
    story.append(Paragraph(
        "<b>Method:</b> Random split with fixed seed (42) for reproducibility",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 3: MODEL ARCHITECTURES ==========
    story.append(Paragraph("3. Model Architectures Explained", heading1_style))
    
    story.append(Paragraph(
        "All models follow a similar structure: <b>Backbone (Feature Extractor) → Classifier Head</b>",
        body_style
    ))
    
    # EfficientNet-B0
    story.append(Paragraph("3.1 EfficientNet-B0", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/models/baseline_models.py</code> - <code>EfficientNetModel</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Parameters:</b> 4.6 million",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Innovation:</b> Compound Scaling - simultaneously scales depth, width, and resolution",
        body_style
    ))
    story.append(Paragraph(
        "<b>Architecture:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Backbone:</b> EfficientNet-B0 (pretrained on ImageNet) extracts 1280-dimensional features",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Classifier:</b> Dropout(0.5) → Linear(1280 → 512) → ReLU → Dropout(0.5) → Linear(512 → 2)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Output:</b> 2 logits (one for Real, one for Fake)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    
    # XceptionNet
    story.append(Paragraph("3.2 XceptionNet", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/models/baseline_models.py</code> - <code>XceptionNet</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Parameters:</b> 38.9 million",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Innovation:</b> Depthwise Separable Convolutions - reduces computation while maintaining capacity",
        body_style
    ))
    story.append(Paragraph(
        "<b>How Depthwise Separable Convolution Works:</b>",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Depthwise Convolution:</b> Each input channel is convolved separately (spatial filtering)",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Pointwise Convolution:</b> 1×1 convolution combines channels (channel mixing)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Result:</b> Similar representational power with ~8× fewer parameters than standard convolution",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Classifier:</b> Same structure as EfficientNet (Dropout → Linear → ReLU → Dropout → Linear)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    
    # ResNet50
    story.append(Paragraph("3.3 ResNet50", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/models/baseline_models.py</code> - <code>ResNetModel</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Parameters:</b> 25 million",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Innovation:</b> Residual Connections (Skip Connections)",
        body_style
    ))
    story.append(Paragraph(
        "<b>How Residual Blocks Work:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Each block computes: <b>output = F(x) + x</b> where F(x) is the learned transformation",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Why it works:</b> Allows gradients to flow directly through skip connections, enabling training of very deep networks (50+ layers)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Architecture:</b> 4 residual blocks (layer1-4) with increasing channels (64→128→256→512)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Classifier:</b> Global Average Pooling → Dropout → Linear(2048 → 512) → ReLU → Dropout → Linear(512 → 2)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Vision Transformer
    story.append(Paragraph("3.4 Vision Transformer (ViT)", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/models/baseline_models.py</code> - <code>VisionTransformer</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Parameters:</b> 86 million",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Innovation:</b> Treats images as sequences of patches, uses self-attention instead of convolutions",
        body_style
    ))
    story.append(Paragraph(
        "<b>How ViT Works (Step-by-Step):</b>",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Patch Embedding:</b> Split 224×224 image into 16×16 patches (196 patches total)",
        body_style
    ))
    story.append(Paragraph(
        "   • Each patch (16×16×3 = 768 pixels) → Linear projection to 768 dimensions",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Add Class Token:</b> Prepend learnable [CLS] token (for classification)",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Positional Embedding:</b> Add learnable position encodings (tells model where each patch is)",
        body_style
    ))
    story.append(Paragraph(
        "4. <b>Transformer Encoder (12 layers):</b>",
        body_style
    ))
    story.append(Paragraph(
        "   • <b>Multi-Head Self-Attention:</b> Each patch attends to all other patches",
        body_style
    ))
    story.append(Paragraph(
        "   • <b>MLP (Feed-Forward):</b> Two linear layers with GELU activation",
        body_style
    ))
    story.append(Paragraph(
        "   • <b>Layer Normalization:</b> Applied before each sub-layer",
        body_style
    ))
    story.append(Paragraph(
        "5. <b>Classification:</b> Use [CLS] token output → Linear(768 → 512) → ReLU → Linear(512 → 2)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "<b>Self-Attention Formula:</b> Attention(Q, K, V) = softmax(QK^T / √d_k) × V",
        body_style
    ))
    story.append(Paragraph(
        "• Q (Query), K (Key), V (Value) are learned linear projections",
        body_style
    ))
    story.append(Paragraph(
        "• Computes how much each patch should attend to every other patch",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 4: TRAINING PROCESS ==========
    story.append(Paragraph("4. Training Process & Algorithms", heading1_style))
    
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/training/trainer.py</code> - <code>DeepfakeTrainer.train()</code>",
        body_style
    ))
    
    story.append(Paragraph("4.1 Training Loop Structure", heading2_style))
    story.append(Paragraph(
        "For each epoch:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Training Phase:</b> Forward pass → Loss calculation → Backward pass → Weight update",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Validation Phase:</b> Forward pass only (no gradient computation) → Calculate metrics",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Learning Rate Scheduling:</b> Adjust learning rate based on schedule",
        body_style
    ))
    story.append(Paragraph(
        "4. <b>Early Stopping Check:</b> Stop if validation performance doesn't improve",
        body_style
    ))
    story.append(Paragraph(
        "5. <b>Checkpoint Saving:</b> Save model every N epochs",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.2 Loss Function: Cross-Entropy Loss", heading2_style))
    story.append(Paragraph(
        "<b>Formula:</b> L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]",
        body_style
    ))
    story.append(Paragraph(
        "• <b>y:</b> True label (0 for Real, 1 for Fake)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>ŷ:</b> Predicted probability of Fake class (from softmax)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Why it works:</b> Penalizes confident wrong predictions heavily, encourages correct predictions",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Implementation:</b> <code>nn.CrossEntropyLoss()</code> in PyTorch (combines LogSoftmax + NLLLoss)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.3 Optimizers", heading2_style))
    
    story.append(Paragraph("4.3.1 Adam Optimizer", heading2_style))
    story.append(Paragraph(
        "<b>Used for:</b> EfficientNet, XceptionNet, ViT",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it works:</b>",
        body_style
    ))
    story.append(Paragraph(
        "1. Maintains <b>exponential moving averages</b> of gradients (m_t) and squared gradients (v_t)",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Bias correction:</b> Adjusts for initialization bias",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Adaptive learning rate:</b> Each parameter gets its own learning rate based on gradient history",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Advantages:</b> Fast convergence, works well with default hyperparameters",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Hyperparameters:</b> β₁=0.9 (momentum decay), β₂=0.999 (squared gradient decay), ε=10⁻⁸",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.3.2 AdamW Optimizer", heading2_style))
    story.append(Paragraph(
        "<b>Used for:</b> ResNet50",
        body_style
    ))
    story.append(Paragraph(
        "<b>Difference from Adam:</b> Decouples weight decay from gradient-based updates",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Why better:</b> More effective regularization, better generalization",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Formula:</b> θ_{t+1} = θ_t - η × (m̂_t / (√v̂_t + ε)) - λ × η × θ_t",
        body_style
    ))
    story.append(Paragraph(
        "  (last term is decoupled weight decay)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.4 Learning Rate Schedulers", heading2_style))
    
    story.append(Paragraph("4.4.1 Cosine Annealing", heading2_style))
    story.append(Paragraph(
        "<b>Used for:</b> EfficientNet, XceptionNet, ViT",
        body_style
    ))
    story.append(Paragraph(
        "<b>Formula:</b> η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2",
        body_style
    ))
    story.append(Paragraph(
        "• <b>How it works:</b> Learning rate decreases smoothly following a cosine curve",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Advantages:</b> Smooth decay, helps fine-tune in later epochs",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.4.2 ReduceLROnPlateau", heading2_style))
    story.append(Paragraph(
        "<b>Used for:</b> ResNet50",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it works:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Monitors validation AUC (Area Under ROC Curve)",
        body_style
    ))
    story.append(Paragraph(
        "• If validation AUC doesn't improve for 'patience' epochs → reduce LR by factor (e.g., 0.5)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Advantages:</b> Adaptive - only reduces LR when needed",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.5 Early Stopping", heading2_style))
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/training/trainer.py</code> - <code>EarlyStopping</code>",
        body_style
    ))
    story.append(Paragraph(
        "<b>How it works:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Tracks best validation AUC score",
        body_style
    ))
    story.append(Paragraph(
        "• If validation AUC doesn't improve for 'patience' epochs (10-20) → stop training",
        body_style
    ))
    story.append(Paragraph(
        "• Restores best model weights before stopping",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Purpose:</b> Prevents overfitting, saves computation time",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.6 Mixed Precision Training", heading2_style))
    story.append(Paragraph(
        "<b>What it is:</b> Uses FP16 (half precision) for forward/backward passes, FP32 for weight updates",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Benefits:</b> 30-50% faster training, 50% less memory usage",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Implementation:</b> PyTorch's <code>torch.cuda.amp</code> with <code>GradScaler</code>",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 5: EVALUATION METRICS ==========
    story.append(Paragraph("5. Evaluation Metrics & Interpretation", heading1_style))
    
    story.append(Paragraph(
        "<b>Code Location:</b> <code>src/evaluation/metrics.py</code> - <code>MetricsCalculator</code>",
        body_style
    ))
    
    story.append(Paragraph("5.1 Confusion Matrix", heading2_style))
    story.append(Paragraph(
        "<b>Structure:</b>",
        body_style
    ))
    cm_table = Table([
        ['', 'Predicted Real', 'Predicted Fake'],
        ['Actual Real', 'TN (True Negative)', 'FP (False Positive)'],
        ['Actual Fake', 'FN (False Negative)', 'TP (True Positive)']
    ], colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(cm_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("5.2 Classification Metrics", heading2_style))
    
    story.append(Paragraph("<b>Accuracy:</b> (TP + TN) / (TP + TN + FP + FN)", body_style))
    story.append(Paragraph(
        "• <b>Interpretation:</b> Overall percentage of correct predictions",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Limitation:</b> Can be misleading with imbalanced datasets",
        body_style
    ))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Precision:</b> TP / (TP + FP)", body_style))
    story.append(Paragraph(
        "• <b>Interpretation:</b> Of all predictions labeled 'Fake', how many are actually fake?",
        body_style
    ))
    story.append(Paragraph(
        "• <b>High Precision:</b> When model says 'Fake', it's usually correct (few false alarms)",
        body_style
    ))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Recall (Sensitivity):</b> TP / (TP + FN)", body_style))
    story.append(Paragraph(
        "• <b>Interpretation:</b> Of all actual fakes, how many did we catch?",
        body_style
    ))
    story.append(Paragraph(
        "• <b>High Recall:</b> Model finds most fakes (few missed detections)",
        body_style
    ))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>F1-Score:</b> 2 × (Precision × Recall) / (Precision + Recall)", body_style))
    story.append(Paragraph(
        "• <b>Interpretation:</b> Harmonic mean of Precision and Recall",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Use case:</b> Single metric balancing precision and recall",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("5.3 ROC Curve & AUC", heading2_style))
    story.append(Paragraph(
        "<b>ROC Curve:</b> Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at different thresholds",
        body_style
    ))
    story.append(Paragraph(
        "• <b>TPR (Sensitivity):</b> TP / (TP + FN) - How many fakes we catch",
        body_style
    ))
    story.append(Paragraph(
        "• <b>FPR:</b> FP / (FP + TN) - How many reals we incorrectly flag as fake",
        body_style
    ))
    story.append(Paragraph(
        "• <b>AUC (Area Under Curve):</b> Measures overall discriminative ability",
        body_style
    ))
    story.append(Paragraph(
        "  - AUC = 1.0: Perfect classifier",
        body_style
    ))
    story.append(Paragraph(
        "  - AUC = 0.5: Random guessing",
        body_style
    ))
    story.append(Paragraph(
        "  - AUC > 0.9: Excellent performance",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Why useful:</b> Threshold-independent metric, works well with imbalanced data",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("5.4 Precision-Recall Curve", heading2_style))
    story.append(Paragraph(
        "<b>What it shows:</b> Trade-off between Precision and Recall at different thresholds",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Better than ROC for imbalanced datasets</b> (we have more fake than real images)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>High curve:</b> Can achieve high precision and recall simultaneously",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 6: RESULTS ANALYSIS ==========
    story.append(Paragraph("6. Results Analysis", heading1_style))
    
    # Load actual results
    results = get_results_data()
    
    # In-Distribution Results
    story.append(Paragraph("6.1 In-Distribution Performance (Celeb-DF Test Set)", heading2_style))
    
    celebd_results = [
        ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'],
        ['EfficientNet-B0', '98.70%', '99.54%', '98.72%', '98.70%', '98.71%'],
        ['XceptionNet', '98.76%', '98.57%', '98.76%', '98.76%', '98.76%'],
        ['ResNet50', '98.22%', '99.50%', '98.21%', '98.22%', '98.15%'],
        ['Vision Transformer', '90.54%', '50.35%', '81.98%', '90.54%', '86.05%']
    ]
    
    results_table = Table(celebd_results, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG])
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Key Observations:</b>", body_style))
    story.append(Paragraph(
        "• <b>CNN models excel:</b> All three CNN architectures (EfficientNet, XceptionNet, ResNet50) achieve >98% accuracy",
        body_style
    ))
    story.append(Paragraph(
        "• <b>ViT lower accuracy:</b> Vision Transformer achieves 90.54% (still good, but lower than CNNs)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>ViT AUC anomaly:</b> 50.35% AUC suggests model may be predicting mostly one class (needs investigation)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("6.2 Cross-Dataset Performance (FaceForensics++)", heading2_style))
    
    cross_results = [
        ['Model', 'Celeb-DF Acc.', 'FF++ Acc.', 'Drop', 'FF++ AUC', 'FF++ F1'],
        ['EfficientNet-B0', '98.70%', '49.66%', '-49.04%', '73.43%', '45.93%'],
        ['XceptionNet', '98.76%', '56.33%', '-42.43%', '68.71%', '55.36%'],
        ['ResNet50', '98.22%', '56.11%', '-42.11%', '72.98%', '54.94%'],
        ['Vision Transformer', '90.54%', '66.06%', '-24.48%', '52.75%', '52.56%']
    ]
    
    cross_table = Table(cross_results, colWidths=[1.2*inch, 1*inch, 1*inch, 0.9*inch, 1*inch, 1*inch])
    cross_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG])
    ]))
    story.append(cross_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Critical Findings:</b>", body_style))
    story.append(Paragraph(
        "• <b>Massive performance drop:</b> All models suffer significant degradation (40-50% accuracy drop)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>ViT shows better generalization:</b> Only 24.48% drop vs 40%+ for CNNs",
        body_style
    ))
    story.append(Paragraph(
        "• <b>EfficientNet worst:</b> Drops to near-random performance (49.66%)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Implication:</b> Models learn dataset-specific features, not generalizable deepfake patterns",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("6.3 What These Results Mean", heading2_style))
    story.append(Paragraph(
        "<b>In-Distribution Success:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Models successfully learn to distinguish real from fake faces in Celeb-DF",
        body_style
    ))
    story.append(Paragraph(
        "• High accuracy suggests models identify consistent patterns/artifacts",
        body_style
    ))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>Cross-Dataset Failure:</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Models fail to generalize to different deepfake generation methods",
        body_style
    ))
    story.append(Paragraph(
        "• Suggests models may be learning dataset-specific biases (compression artifacts, video quality, etc.)",
        body_style
    ))
    story.append(Paragraph(
        "• <b>ViT's relative success:</b> Attention mechanism may learn more generalizable features",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 7: CODE STRUCTURE ==========
    story.append(Paragraph("7. Code Structure & Key Components", heading1_style))
    
    story.append(Paragraph("7.1 Project Directory Structure", heading2_style))
    story.append(Paragraph(
        "<code>src/data/preprocessing.py</code> - Data preprocessing pipeline",
        body_style
    ))
    story.append(Paragraph(
        "<code>src/data/dataloader.py</code> - PyTorch DataLoader creation",
        body_style
    ))
    story.append(Paragraph(
        "<code>src/models/baseline_models.py</code> - Model architectures (EfficientNet, Xception, ResNet, ViT)",
        body_style
    ))
    story.append(Paragraph(
        "<code>src/training/trainer.py</code> - Training loop, optimizers, schedulers",
        body_style
    ))
    story.append(Paragraph(
        "<code>src/evaluation/metrics.py</code> - Evaluation metrics calculation",
        body_style
    ))
    story.append(Paragraph(
        "<code>src/explainability/gradcam.py</code> - Grad-CAM visualization",
        body_style
    ))
    story.append(Paragraph(
        "<code>main.py</code> - Main entry point (CLI interface)",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("7.2 Key Functions to Understand", heading2_style))
    
    story.append(Paragraph("<b>Training Flow:</b>", body_style))
    story.append(Paragraph(
        "1. <code>main.py:train_model()</code> - Entry point, sets up data loaders and model",
        body_style
    ))
    story.append(Paragraph(
        "2. <code>DeepfakeTrainer.train()</code> - Main training loop",
        body_style
    ))
    story.append(Paragraph(
        "3. <code>DeepfakeTrainer._train_epoch()</code> - Single epoch training",
        body_style
    ))
    story.append(Paragraph(
        "4. <code>DeepfakeTrainer._validate_epoch()</code> - Validation phase",
        body_style
    ))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Model Forward Pass:</b>", body_style))
    story.append(Paragraph(
        "• <code>model.forward(x)</code> - Takes input tensor (batch_size, 3, 224, 224)",
        body_style
    ))
    story.append(Paragraph(
        "• Returns logits (batch_size, 2) - raw scores for Real and Fake classes",
        body_style
    ))
    story.append(Paragraph(
        "• Apply <code>torch.softmax()</code> to get probabilities",
        body_style
    ))
    story.append(Paragraph(
        "• Use <code>torch.argmax()</code> to get predicted class",
        body_style
    ))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Evaluation Flow:</b>", body_style))
    story.append(Paragraph(
        "1. <code>ModelEvaluator.evaluate_model()</code> - Main evaluation function",
        body_style
    ))
    story.append(Paragraph(
        "2. <code>MetricsCalculator.calculate_metrics()</code> - Computes all metrics",
        body_style
    ))
    story.append(Paragraph(
        "3. <code>VisualizationGenerator</code> - Creates confusion matrices, ROC curves, PR curves",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 8: KEY ALGORITHMS ==========
    story.append(Paragraph("8. Key Algorithms Explained", heading1_style))
    
    story.append(Paragraph("8.1 Backpropagation (How Models Learn)", heading2_style))
    story.append(Paragraph(
        "<b>Process:</b>",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Forward Pass:</b> Input → Model → Output (predictions)",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Loss Calculation:</b> Compare predictions with true labels",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Backward Pass:</b> Compute gradients (derivatives) of loss w.r.t. each parameter",
        body_style
    ))
    story.append(Paragraph(
        "4. <b>Gradient Descent:</b> Update parameters: θ = θ - η × ∇L(θ)",
        body_style
    ))
    story.append(Paragraph(
        "   • η (eta) = learning rate, ∇L(θ) = gradient of loss",
        body_style
    ))
    story.append(Paragraph(
        "5. <b>Repeat:</b> Thousands of iterations until loss is minimized",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("8.2 Gradient Calculation (Chain Rule)", heading2_style))
    story.append(Paragraph(
        "For a neural network: L = loss(y, f(x; θ))",
        body_style
    ))
    story.append(Paragraph(
        "• Compute ∂L/∂θ using chain rule: ∂L/∂θ = (∂L/∂ŷ) × (∂ŷ/∂θ)",
        body_style
    ))
    story.append(Paragraph(
        "• PyTorch's <code>autograd</code> automatically computes these gradients",
        body_style
    ))
    story.append(Paragraph(
        "• <code>loss.backward()</code> triggers backpropagation through entire network",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("8.3 Batch Processing", heading2_style))
    story.append(Paragraph(
        "<b>Why batches?</b>",
        body_style
    ))
    story.append(Paragraph(
        "• Computing gradients on entire dataset is too slow",
        body_style
    ))
    story.append(Paragraph(
        "• Single sample has high variance",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Solution:</b> Process 32 samples at a time (batch_size=32)",
        body_style
    ))
    story.append(Paragraph(
        "• Average gradients over batch → more stable updates",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("8.4 Dropout Regularization", heading2_style))
    story.append(Paragraph(
        "<b>What it does:</b> Randomly sets 50% of neurons to zero during training",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Why:</b> Prevents overfitting by forcing model to not rely on specific neurons",
        body_style
    ))
    story.append(Paragraph(
        "• <b>During inference:</b> All neurons active, but outputs scaled by dropout probability",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Implementation:</b> <code>nn.Dropout(0.5)</code> in classifier head",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("8.5 Softmax Activation", heading2_style))
    story.append(Paragraph(
        "<b>Formula:</b> softmax(x_i) = exp(x_i) / Σ exp(x_j)",
        body_style
    ))
    story.append(Paragraph(
        "• Converts raw logits to probabilities (sums to 1.0)",
        body_style
    ))
    story.append(Paragraph(
        "• Example: logits = [2.0, 1.0] → probabilities = [0.73, 0.27]",
        body_style
    ))
    story.append(Paragraph(
        "• Higher logit → higher probability",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ========== SECTION 9: VIVA Q&A ==========
    story.append(Paragraph("9. Common Viva Questions & Answers", heading1_style))
    
    qa_pairs = [
        ("Q: Why did you choose these four models?", 
         "A: EfficientNet for efficiency, XceptionNet as proven baseline, ResNet50 for residual learning, ViT to explore transformer-based approach. This gives diverse architectural perspectives."),
        
        ("Q: Why does ViT have lower in-distribution accuracy?", 
         "A: ViT requires more data to train effectively. With limited training data, CNNs' inductive biases (translation equivariance, locality) help them learn better. ViT's attention mechanism needs more examples to learn spatial relationships."),
        
        ("Q: Why do models fail on cross-dataset evaluation?", 
         "A: Models learn dataset-specific features (compression artifacts, video quality, generation method characteristics) rather than universal deepfake patterns. Different datasets have different distributions, causing domain shift."),
        
        ("Q: Why does ViT generalize better?", 
         "A: Self-attention mechanism allows ViT to learn global relationships between image patches, potentially capturing more generalizable features. CNNs' local receptive fields may overfit to local patterns specific to training data."),
        
        ("Q: What is the difference between accuracy and AUC?", 
         "A: Accuracy measures correctness at a fixed threshold (usually 0.5). AUC measures discriminative ability across all possible thresholds. AUC is better for imbalanced datasets and threshold-independent evaluation."),
        
        ("Q: How does Grad-CAM work?", 
         "A: Grad-CAM computes gradients of target class score w.r.t. feature maps, then weights feature maps by these gradients. This highlights image regions most important for the prediction. Red regions = high importance."),
        
        ("Q: What is early stopping and why use it?", 
         "A: Early stopping monitors validation performance and stops training when it stops improving. Prevents overfitting (model memorizing training data) and saves computation time."),
        
        ("Q: What is the purpose of learning rate scheduling?", 
         "A: Start with higher LR for fast learning, gradually reduce for fine-tuning. Cosine annealing provides smooth decay, while plateau-based reduces only when needed."),
        
        ("Q: How do you handle class imbalance?", 
         "A: We use weighted sampling in DataLoader, but our dataset is relatively balanced. For severe imbalance, we could use focal loss or class weights in cross-entropy loss."),
        
        ("Q: What improvements would you make?", 
         "A: 1) Multi-dataset training for better generalization, 2) Ensemble methods combining CNN and ViT, 3) Temporal analysis for video sequences, 4) Adversarial training for robustness, 5) Domain adaptation techniques.")
    ]
    
    for i, (q, a) in enumerate(qa_pairs, 1):
        story.append(Paragraph(f"<b>{q}</b>", body_style))
        story.append(Paragraph(a, body_style))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(PageBreak())
    
    # ========== FINAL SUMMARY ==========
    story.append(Paragraph("Summary & Key Takeaways", heading1_style))
    
    story.append(Paragraph("<b>Project Achievements:</b>", body_style))
    story.append(Paragraph(
        "✓ Successfully trained 4 deep learning models on Celeb-DF dataset",
        body_style
    ))
    story.append(Paragraph(
        "✓ Achieved >98% accuracy on in-distribution test set (CNN models)",
        body_style
    ))
    story.append(Paragraph(
        "✓ Conducted comprehensive cross-dataset evaluation revealing generalization challenges",
        body_style
    ))
    story.append(Paragraph(
        "✓ Generated explainability visualizations using Grad-CAM",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Key Technical Insights:</b>", body_style))
    story.append(Paragraph(
        "• CNN models excel at in-distribution tasks but struggle with generalization",
        body_style
    ))
    story.append(Paragraph(
        "• Vision Transformers show promise for better cross-dataset performance",
        body_style
    ))
    story.append(Paragraph(
        "• Current deepfake detection systems are vulnerable to distribution shifts",
        body_style
    ))
    story.append(Paragraph(
        "• Future work should focus on domain adaptation and multi-dataset training",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Code Quality:</b>", body_style))
    story.append(Paragraph(
        "• Modular architecture with clear separation of concerns",
        body_style
    ))
    story.append(Paragraph(
        "• Comprehensive evaluation and visualization pipeline",
        body_style
    ))
    story.append(Paragraph(
        "• Reproducible experiments with proper logging and checkpointing",
        body_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Viva preparation PDF created: {output_path}")

if __name__ == "__main__":
    create_viva_pdf("viva_prep.pdf")



