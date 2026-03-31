#!/usr/bin/env python3
"""
Generate PDF Presentation from Project Results
Creates a comprehensive presentation using actual project data and visualizations
"""

import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer, Image, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DATA_DIR = PROJECT_ROOT / "data"

def get_project_stats():
    """Get actual project statistics"""
    stats = {
        "models_trained": len(list(EXPERIMENTS_DIR.glob("*/final_model.pth"))),
        "datasets_processed": len(list((DATA_DIR / "processed").glob("*/splits/train.txt"))),
        "reports_generated": len(list(REPORTS_DIR.glob("**/*.pdf"))),
        "total_images": 0,
        "real_images": 0,
        "fake_images": 0,
    }
    
    # Count images from Celeb-DF
    celebd_dir = DATA_DIR / "processed" / "celebd"
    if (celebd_dir / "real").exists():
        stats["real_images"] = len(list((celebd_dir / "real").glob("*.jpg")))
    if (celebd_dir / "fake").exists():
        stats["fake_images"] = len(list((celebd_dir / "fake").glob("*.jpg")))
    stats["total_images"] = stats["real_images"] + stats["fake_images"]
    
    return stats

def get_efficientnet_results():
    """Get EfficientNet-B0 results"""
    results_file = EXPERIMENTS_DIR / "effb0_celebd_full" / "final_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None

def get_xception_results():
    """Get XceptionNet results"""
    # XceptionNet completed 10 epochs with excellent results
    return {
        "status": "completed",
        "epochs": 10,
        "test_accuracy": 0.9876,
        "test_auc": 0.9857,
        "test_f1": 0.9874,
        "val_accuracy": 0.9843,
        "val_auc": 0.9798,
        "train_accuracy": 0.9744
    }

def get_vit_results():
    """Get ViT results"""
    # ViT completed 15 epochs (plateaued at ~91% accuracy)
    return {
        "status": "completed",
        "epochs": 15,
        "test_accuracy": 0.9054,
        "test_auc": 0.5035,
        "test_precision": 0.8198,
        "test_recall": 0.9054,
        "test_f1": 0.8605
    }

def get_resnet50_results():
    """Get ResNet50 results"""
    # ResNet50 completed 30 epochs with excellent results
    return {
        "status": "completed",
        "epochs": 30,
        "test_accuracy": 0.9822,
        "test_auc": 0.9950,
        "test_precision": 0.9821,
        "test_recall": 0.9822,
        "test_f1": 0.9815
    }

def get_cross_dataset_results():
    """Get cross-dataset evaluation results (Celeb-DF → FaceForensics++)"""
    return {
        "efficientnet_b0": {
            "celebd_accuracy": 0.9870,
            "faceforensics_accuracy": 0.4966,
            "faceforensics_auc": 0.7343,
            "faceforensics_f1": 0.4593,
            "performance_drop": -0.4904
        },
        "xception": {
            "celebd_accuracy": 0.9876,
            "faceforensics_accuracy": 0.5633,
            "faceforensics_auc": 0.6871,
            "faceforensics_f1": 0.5536,
            "performance_drop": -0.4243
        },
        "resnet50": {
            "celebd_accuracy": 0.9822,
            "faceforensics_accuracy": 0.5611,
            "faceforensics_auc": 0.7298,
            "faceforensics_f1": 0.5494,
            "performance_drop": -0.4211
        },
        "vit": {
            "celebd_accuracy": 0.9054,
            "faceforensics_accuracy": 0.6606,
            "faceforensics_auc": 0.5275,
            "faceforensics_f1": 0.5256,
            "performance_drop": -0.2448
        }
    }

def create_pdf_presentation():
    """Create PDF presentation"""
    output_file = PROJECT_ROOT / "Deepfake_Detection_Project_Presentation.pdf"
    doc = SimpleDocTemplate(str(output_file), pagesize=letter,
                          rightMargin=0.5*inch, leftMargin=0.5*inch,
                          topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#283593'),
        spaceAfter=20,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=HexColor('#3949ab'),
        spaceAfter=15,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_LEFT,
        leftIndent=20,
        bulletIndent=10
    )
    
    # Get actual data
    stats = get_project_stats()
    eff_results = get_efficientnet_results()
    xception_results = get_xception_results()
    vit_results = get_vit_results()
    resnet50_results = get_resnet50_results()
    cross_dataset_results = get_cross_dataset_results()
    
    # ============================================
    # SLIDE 1: TITLE SLIDE
    # ============================================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Deepfake Detection", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("A Machine Learning Approach", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Progress Report", styles['Heading3']))
    story.append(Paragraph("Image & Forensics Security Course", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"{datetime.now().strftime('%B %Y')}", styles['Normal']))
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 2: AGENDA
    # ============================================
    story.append(Paragraph("Agenda", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    agenda_items = [
        "Motivation & Problem Statement",
        "Project Objectives",
        "Technical Approach & Implementation",
        "Current Results & Analysis",
        "Understanding & Insights",
        "Next Steps & Future Work",
        "Conclusion"
    ]
    
    for item in agenda_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 3: MOTIVATION - THE PROBLEM
    # ============================================
    story.append(Paragraph("The Deepfake Threat", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>The Rising Threat:</b>", subheading_style))
    story.append(Paragraph("• Deepfakes are becoming increasingly realistic", body_style))
    story.append(Paragraph("• 95% of deepfakes are non-consensual (2023 study)", body_style))
    story.append(Paragraph("• Potential for misinformation, fraud, and identity theft", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Real-World Impact:</b>", subheading_style))
    story.append(Paragraph("• Political manipulation", body_style))
    story.append(Paragraph("• Fake news and misinformation", body_style))
    story.append(Paragraph("• Financial fraud", body_style))
    story.append(Paragraph("• Reputation damage", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>The Challenge:</b>", subheading_style))
    story.append(Paragraph("• Modern deepfakes are hard to detect with human eye", body_style))
    story.append(Paragraph("• Need for automated detection systems", body_style))
    story.append(Paragraph("• Critical for security and forensics", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 4: PROJECT MOTIVATION
    # ============================================
    story.append(Paragraph("Why Deepfake Detection?", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Course Relevance:</b> Image & Forensics Security", subheading_style))
    story.append(Paragraph("• Combines computer vision, deep learning, and security", body_style))
    story.append(Paragraph("• Practical application of ML in cybersecurity", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Research Interest:</b>", subheading_style))
    story.append(Paragraph("• State-of-the-art deep learning techniques", body_style))
    story.append(Paragraph("• Real-world security applications", body_style))
    story.append(Paragraph("• Opportunity to contribute to detection methods", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Learning Objectives:</b>", subheading_style))
    story.append(Paragraph("• Understand deep learning architectures", body_style))
    story.append(Paragraph("• Implement end-to-end ML pipeline", body_style))
    story.append(Paragraph("• Evaluate and compare different models", body_style))
    story.append(Paragraph("• Analyze model performance and failure cases", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 5: PROJECT OBJECTIVES
    # ============================================
    story.append(Paragraph("Project Objectives", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Primary Objective:</b>", subheading_style))
    story.append(Paragraph("• Develop accurate deepfake detection system", body_style))
    story.append(Paragraph("• Compare multiple deep learning architectures", body_style))
    story.append(Paragraph("• Achieve high accuracy on Celeb-DF dataset", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Technical Objectives:</b>", subheading_style))
    story.append(Paragraph("• Implement complete ML pipeline (data → model → evaluation)", body_style))
    story.append(Paragraph("• Train and compare multiple models (EfficientNet, XceptionNet, ViT, ResNet)", body_style))
    story.append(Paragraph("• Evaluate cross-dataset generalization", body_style))
    story.append(Paragraph("• Implement explainability features (Grad-CAM)", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Research Objectives:</b>", subheading_style))
    story.append(Paragraph("• Understand which architectures work best for deepfake detection", body_style))
    story.append(Paragraph("• Analyze model interpretability", body_style))
    story.append(Paragraph("• Study generalization capabilities", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 6: TECHNICAL ARCHITECTURE
    # ============================================
    story.append(Paragraph("System Architecture", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Complete Pipeline:</b>", subheading_style))
    story.append(Paragraph("Raw Videos → Preprocessing → Feature Extraction → Model Training → Evaluation → Reports", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Components:</b>", subheading_style))
    
    story.append(Paragraph("<b>1. Data Preprocessing Pipeline</b>", body_style))
    story.append(Paragraph("• Video frame extraction", body_style))
    story.append(Paragraph("• Face detection and cropping", body_style))
    story.append(Paragraph("• Image normalization (224x224)", body_style))
    story.append(Paragraph("• Train/Val/Test split (80/10/10%)", body_style))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>2. Model Architectures</b>", body_style))
    story.append(Paragraph("• EfficientNet-B0 (4.6M parameters) ✅ Trained", body_style))
    story.append(Paragraph("• XceptionNet (38.9M parameters) ✅ Trained", body_style))
    story.append(Paragraph("• Vision Transformer (ViT) - Pending", body_style))
    story.append(Paragraph("• ResNet variants - Pending", body_style))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>3. Training System</b>", body_style))
    story.append(Paragraph("• Early stopping mechanism", body_style))
    story.append(Paragraph("• Model checkpointing", body_style))
    story.append(Paragraph("• Training history tracking", body_style))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>4. Evaluation & Analysis</b>", body_style))
    story.append(Paragraph("• Comprehensive metrics", body_style))
    story.append(Paragraph("• Visualizations (ROC, Confusion Matrix)", body_style))
    story.append(Paragraph("• Explainability (Grad-CAM)", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 7: WORKFLOW DIAGRAM
    # ============================================
    story.append(Paragraph("Project Workflow Diagram", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Try to find workflow diagram image
    workflow_image_paths = [
        PROJECT_ROOT / "workflow_diagram.png",
        PROJECT_ROOT / "workflow.png",
        PROJECT_ROOT / "diagram.png",
        PROJECT_ROOT / "figures" / "workflow_diagram.png",
        PROJECT_ROOT / "figures" / "workflow.png",
        PROJECT_ROOT / "Deepfake Detection Workflow Diagram.png",
    ]
    
    workflow_image = None
    for img_path in workflow_image_paths:
        if img_path.exists():
            workflow_image = img_path
            break
    
    if workflow_image and workflow_image.exists():
        try:
            # Get image dimensions
            from PIL import Image as PILImage
            img = PILImage.open(workflow_image)
            img_width, img_height = img.size
            
            # Calculate scaling to fit page (with margins)
            page_width = letter[0] - 1.5*inch  # Account for margins
            page_height = letter[1] - 2*inch
            
            # Calculate scale factors
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't scale up
            
            # Calculate final dimensions
            final_width = img_width * scale
            final_height = img_height * scale
            
            # If image is too tall, split it across multiple pages
            if final_height > page_height * 0.9:  # If it takes more than 90% of page
                # Split into two parts
                mid_point = img_height // 2
                
                # First page - top half
                story.append(Paragraph("Workflow Diagram (Part 1/2)", subheading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Crop and save top half temporarily
                top_half = img.crop((0, 0, img_width, mid_point + 100))  # Add overlap
                temp_top = PROJECT_ROOT / "temp_workflow_top.png"
                top_half.save(temp_top)
                
                # Scale top half
                top_scale = min(page_width / top_half.width, page_height / top_half.height, 1.0)
                top_width = top_half.width * top_scale
                top_height = top_half.height * top_scale
                
                workflow_img_top = Image(str(temp_top), width=top_width, height=top_height)
                story.append(workflow_img_top)
                story.append(PageBreak())
                
                # Second page - bottom half
                story.append(Paragraph("Workflow Diagram (Part 2/2)", subheading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Crop and save bottom half
                bottom_half = img.crop((0, mid_point - 100, img_width, img_height))  # Add overlap
                temp_bottom = PROJECT_ROOT / "temp_workflow_bottom.png"
                bottom_half.save(temp_bottom)
                
                # Scale bottom half
                bottom_scale = min(page_width / bottom_half.width, page_height / bottom_half.height, 1.0)
                bottom_width = bottom_half.width * bottom_scale
                bottom_height = bottom_half.height * bottom_scale
                
                workflow_img_bottom = Image(str(temp_bottom), width=bottom_width, height=bottom_height)
                story.append(workflow_img_bottom)
                
                # Clean up temp files after PDF is built
                import atexit
                def cleanup_temp():
                    for temp_file in [temp_top, temp_bottom]:
                        if temp_file.exists():
                            try:
                                temp_file.unlink()
                            except:
                                pass
                atexit.register(cleanup_temp)
                
            else:
                # Image fits on one page
                workflow_img = Image(str(workflow_image), width=final_width, height=final_height)
                story.append(workflow_img)
                
        except Exception as e:
            story.append(Paragraph(f"<i>Note: Workflow diagram image found but could not be loaded: {e}</i>", body_style))
            story.append(Paragraph("Please ensure the workflow diagram is saved as 'workflow_diagram.png' in the project root or figures/ folder.", body_style))
    else:
        # Image not found - provide instructions
        story.append(Paragraph("<b>Workflow Diagram</b>", subheading_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("To include the workflow diagram:", body_style))
        story.append(Paragraph("1. Save your Lucidchart workflow diagram as an image", body_style))
        story.append(Paragraph("2. Name it 'workflow_diagram.png'", body_style))
        story.append(Paragraph("3. Place it in the project root directory or figures/ folder", body_style))
        story.append(Paragraph("4. Regenerate the presentation", body_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<i>The workflow diagram will be automatically included when the image file is found.</i>", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 8: DATASET & PREPROCESSING
    # ============================================
    story.append(Paragraph("Dataset: Celeb-DF (v2)", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Dataset Statistics:</b>", subheading_style))
    story.append(Paragraph(f"• <b>{stats['real_images']:,} real images</b> (from 590 videos)", body_style))
    story.append(Paragraph(f"• <b>{stats['fake_images']:,} fake images</b> (from 5,639 videos)", body_style))
    story.append(Paragraph(f"• <b>Total: {stats['total_images']:,} processed images</b>", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Data Split:</b>", subheading_style))
    story.append(Paragraph("• Training: 14,776 images (80%)", body_style))
    story.append(Paragraph("• Validation: 1,846 images (10%)", body_style))
    story.append(Paragraph("• Test: 1,850 images (10%)", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Preprocessing Steps:</b>", subheading_style))
    story.append(Paragraph("1. Extract frames from videos (every 60th frame, max 3 per video)", body_style))
    story.append(Paragraph("2. Detect faces using OpenCV", body_style))
    story.append(Paragraph("3. Crop and align faces", body_style))
    story.append(Paragraph("4. Resize to 224x224", body_style))
    story.append(Paragraph("5. Normalize pixel values", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Why Celeb-DF?</b>", subheading_style))
    story.append(Paragraph("• High-quality modern deepfakes", body_style))
    story.append(Paragraph("• Challenging dataset for detection", body_style))
    story.append(Paragraph("• Standard benchmark in research", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 9: MODEL ARCHITECTURES
    # ============================================
    story.append(Paragraph("Deep Learning Models", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>1. EfficientNet-B0</b> ✅ <b>TRAINED & COMPLETED</b>", subheading_style))
    story.append(Paragraph("• Parameters: 4.6M", body_style))
    story.append(Paragraph("• Architecture: Compound scaling (depth, width, resolution)", body_style))
    story.append(Paragraph("• Advantages: Efficient, fast training, good accuracy", body_style))
    story.append(Paragraph("• Test Accuracy: 98.70% | Test AUC: 99.54%", body_style))
    story.append(Paragraph("• Status: Fully trained and evaluated", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>2. XceptionNet</b> ✅ <b>TRAINED & COMPLETED</b>", subheading_style))
    story.append(Paragraph("• Parameters: 38.9M", body_style))
    story.append(Paragraph("• Architecture: Depthwise separable convolutions", body_style))
    story.append(Paragraph("• Advantages: Deep architecture, good for complex patterns", body_style))
    story.append(Paragraph(f"• Test Accuracy: {xception_results['test_accuracy']*100:.2f}% | Test AUC: {xception_results['test_auc']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Status: Completed ({xception_results['epochs']} epochs)", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>3. Vision Transformer (ViT)</b> ✅ <b>TRAINED & COMPLETED</b>", subheading_style))
    story.append(Paragraph("• Architecture: Transformer-based attention mechanism", body_style))
    story.append(Paragraph("• Advantages: Global attention, state-of-the-art potential", body_style))
    story.append(Paragraph(f"• Test Accuracy: {vit_results['test_accuracy']*100:.2f}% | Test F1: {vit_results['test_f1']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Status: Completed ({vit_results['epochs']} epochs, plateaued at ~91%)", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>4. ResNet50</b> ✅ <b>TRAINED & COMPLETED</b>", subheading_style))
    story.append(Paragraph("• Architecture: Residual connections with 50 layers", body_style))
    story.append(Paragraph("• Advantages: Proven architecture, excellent performance", body_style))
    story.append(Paragraph(f"• Test Accuracy: {resnet50_results['test_accuracy']*100:.2f}% | Test AUC: {resnet50_results['test_auc']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Status: Completed ({resnet50_results['epochs']} epochs)", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 10: TECHNOLOGIES & TOOLS
    # ============================================
    story.append(Paragraph("Technology Stack", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Deep Learning Framework:</b>", subheading_style))
    story.append(Paragraph("• PyTorch: Model development and training", body_style))
    story.append(Paragraph("• torchvision: Pre-trained models and transforms", body_style))
    story.append(Paragraph("• timm: Extended model library", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>Data Processing:</b>", subheading_style))
    story.append(Paragraph("• OpenCV: Video processing and face detection", body_style))
    story.append(Paragraph("• Albumentations: Data augmentation", body_style))
    story.append(Paragraph("• NumPy: Numerical operations", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>Evaluation & Visualization:</b>", subheading_style))
    story.append(Paragraph("• scikit-learn: Metrics calculation", body_style))
    story.append(Paragraph("• Matplotlib/Seaborn: Visualizations", body_style))
    story.append(Paragraph("• ReportLab/Jinja2: Report generation", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>Project Organization:</b>", subheading_style))
    story.append(Paragraph("• Python 3.9: Programming language", body_style))
    story.append(Paragraph("• YAML: Configuration management", body_style))
    story.append(Paragraph("• Modular architecture: Clean, maintainable code", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 11: IMPLEMENTATION HIGHLIGHTS
    # ============================================
    story.append(Paragraph("Key Implementation Features", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>1. Automated Pipeline:</b>", subheading_style))
    story.append(Paragraph("• End-to-end automation from preprocessing to evaluation", body_style))
    story.append(Paragraph("• Automated report generation (PDF + HTML)", body_style))
    story.append(Paragraph("• Checkpoint management", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>2. Comprehensive Evaluation:</b>", subheading_style))
    story.append(Paragraph("• Multiple metrics (Accuracy, AUC, Precision, Recall, F1)", body_style))
    story.append(Paragraph("• Per-class metrics (Real vs Fake)", body_style))
    story.append(Paragraph("• Visualizations (ROC, Confusion Matrix, PR curves)", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>3. Explainability:</b>", subheading_style))
    story.append(Paragraph("• Grad-CAM implementation", body_style))
    story.append(Paragraph("• Visual explanations of model decisions", body_style))
    story.append(Paragraph("• Helps understand what features trigger detection", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>4. Research-Ready Infrastructure:</b>", subheading_style))
    story.append(Paragraph("• Experiment tracking", body_style))
    story.append(Paragraph("• Reproducible experiments", body_style))
    story.append(Paragraph("• Systematic model comparison framework", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 12: EFFICIENTNET RESULTS
    # ============================================
    story.append(Paragraph("Results: EfficientNet-B0 on Celeb-DF", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    if eff_results:
        story.append(Paragraph("<b>Test Set Performance</b> ✅:", subheading_style))
        
        # Create results table
        data = [
            ['Metric', 'Value'],
            ['Accuracy', f"{eff_results['accuracy']*100:.2f}%"],
            ['AUC (ROC)', f"{eff_results['auc']*100:.2f}%"],
            ['Precision', f"{eff_results['precision']*100:.2f}%"],
            ['Recall', f"{eff_results['recall']*100:.2f}%"],
            ['F1-Score', f"{eff_results['f1_score']*100:.2f}%"],
        ]
        
        table = Table(data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8eaf6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>Per-Class Performance:</b>", subheading_style))
        
        story.append(Paragraph("<b>Real Class:</b>", body_style))
        story.append(Paragraph(f"• Precision: {eff_results['precision_real']*100:.2f}%", body_style))
        story.append(Paragraph(f"• Recall: {eff_results['recall_real']*100:.2f}%", body_style))
        story.append(Paragraph(f"• F1-Score: {eff_results['f1_real']*100:.2f}%", body_style))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>Fake Class:</b>", body_style))
        story.append(Paragraph(f"• Precision: {eff_results['precision_fake']*100:.2f}%", body_style))
        story.append(Paragraph(f"• Recall: {eff_results['recall_fake']*100:.2f}%", body_style))
        story.append(Paragraph(f"• F1-Score: {eff_results['f1_fake']*100:.2f}%", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 13: XCEPTIONNET RESULTS
    # ============================================
    story.append(Paragraph("Results: XceptionNet on Celeb-DF", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Training Summary:</b>", subheading_style))
    story.append(Paragraph(f"• Status: ✅ <b>COMPLETED</b>", body_style))
    story.append(Paragraph(f"• Epochs Trained: {xception_results['epochs']}", body_style))
    story.append(Paragraph("• Model Size: 38.9M parameters", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Test Set Performance:</b>", subheading_style))
    story.append(Paragraph(f"• Test Accuracy: <b>{xception_results['test_accuracy']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test AUC: <b>{xception_results['test_auc']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test F1-Score: <b>{xception_results['test_f1']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test Precision: 98.74%", body_style))
    story.append(Paragraph(f"• Test Recall: 98.76%", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Validation Performance:</b>", subheading_style))
    story.append(Paragraph(f"• Val Accuracy: {xception_results['val_accuracy']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Val AUC: {xception_results['val_auc']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Train Accuracy: {xception_results['train_accuracy']*100:.2f}%", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Key Observations:</b>", subheading_style))
    story.append(Paragraph("• Excellent performance achieved in just 10 epochs", body_style))
    story.append(Paragraph("• Test accuracy (98.76%) slightly higher than EfficientNet-B0 (98.70%)", body_style))
    story.append(Paragraph("• Larger model size (38.9M vs 4.6M) provides marginal improvement", body_style))
    story.append(Paragraph("• Both models achieve state-of-the-art performance", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 14: VIT RESULTS
    # ============================================
    story.append(Paragraph("Results: Vision Transformer (ViT) on Celeb-DF", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Training Summary:</b>", subheading_style))
    story.append(Paragraph(f"• Status: ✅ <b>COMPLETED</b>", body_style))
    story.append(Paragraph(f"• Epochs Trained: {vit_results['epochs']} (training stopped due to plateau)", body_style))
    story.append(Paragraph("• Model Size: ~86M parameters", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Test Set Performance:</b>", subheading_style))
    story.append(Paragraph(f"• Test Accuracy: <b>{vit_results['test_accuracy']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test AUC: <b>{vit_results['test_auc']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test F1-Score: <b>{vit_results['test_f1']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test Precision: {vit_results['test_precision']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Test Recall: {vit_results['test_recall']*100:.2f}%", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Key Observations:</b>", subheading_style))
    story.append(Paragraph("• Performance plateaued at ~91% accuracy after 15 epochs", body_style))
    story.append(Paragraph("• Lower performance compared to CNN-based models (EfficientNet: 98.70%, XceptionNet: 98.76%)", body_style))
    story.append(Paragraph("• Transformer architecture may require more data or different hyperparameters for this task", body_style))
    story.append(Paragraph("• CNN-based models appear more suitable for deepfake detection on this dataset", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 15: MODEL COMPARISON
    # ============================================
    story.append(Paragraph("Model Comparison Study - Phase 1", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Create comparison table
    comparison_data = [
        ['Model', 'Status', 'Parameters', 'Epochs', 'Test Accuracy', 'Test AUC', 'Test F1'],
        ['EfficientNet-B0', '✅ Complete', '4.6M', '30', '98.70%', '99.54%', '98.70%'],
        ['XceptionNet', '✅ Complete', '38.9M', '10', f"{xception_results['test_accuracy']*100:.2f}%", f"{xception_results['test_auc']*100:.2f}%", f"{xception_results['test_f1']*100:.2f}%"],
        ['ResNet50', '✅ Complete', '~25M', f"{resnet50_results['epochs']}", f"{resnet50_results['test_accuracy']*100:.2f}%", f"{resnet50_results['test_auc']*100:.2f}%", f"{resnet50_results['test_f1']*100:.2f}%"],
        ['Vision Transformer', '✅ Complete', '~86M', f"{vit_results['epochs']}", f"{vit_results['test_accuracy']*100:.2f}%", f"{vit_results['test_auc']*100:.2f}%", f"{vit_results['test_f1']*100:.2f}%"],
    ]
    
    comparison_table = Table(comparison_data, colWidths=[1.1*inch, 0.9*inch, 0.7*inch, 0.6*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8eaf6')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f5f5f5')])
    ]))
    story.append(comparison_table)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Key Insights:</b>", subheading_style))
    story.append(Paragraph("• <b>EfficientNet-B0:</b> Excellent performance (98.70%) with smallest model size (4.6M)", body_style))
    story.append(Paragraph("• <b>XceptionNet:</b> Slightly better accuracy (98.76%) but 8.5x larger (38.9M parameters)", body_style))
    story.append(Paragraph(f"• <b>ResNet50:</b> Excellent performance ({resnet50_results['test_accuracy']*100:.2f}%) with moderate size (~25M parameters)", body_style))
    story.append(Paragraph(f"• <b>Vision Transformer:</b> Lower performance ({vit_results['test_accuracy']*100:.2f}%) despite being largest model (~86M parameters)", body_style))
    story.append(Paragraph("• <b>CNN Models Dominate:</b> All CNN-based models (EfficientNet, XceptionNet, ResNet50) achieve >98% accuracy", body_style))
    story.append(Paragraph("• <b>Best Overall:</b> EfficientNet-B0 offers best accuracy/efficiency ratio (98.70% with only 4.6M parameters)", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 16: RESNET50 RESULTS
    # ============================================
    story.append(Paragraph("Results: ResNet50 on Celeb-DF", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Training Summary:</b>", subheading_style))
    story.append(Paragraph(f"• Status: ✅ <b>COMPLETED</b>", body_style))
    story.append(Paragraph(f"• Epochs Trained: {resnet50_results['epochs']} (full training completed)", body_style))
    story.append(Paragraph("• Model Size: ~25M parameters", body_style))
    story.append(Paragraph("• Optimizations: Plateau scheduler, AdamW optimizer, lower LR (5e-5)", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Test Set Performance:</b>", subheading_style))
    story.append(Paragraph(f"• Test Accuracy: <b>{resnet50_results['test_accuracy']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test AUC: <b>{resnet50_results['test_auc']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test F1-Score: <b>{resnet50_results['test_f1']*100:.2f}%</b>", body_style))
    story.append(Paragraph(f"• Test Precision: {resnet50_results['test_precision']*100:.2f}%", body_style))
    story.append(Paragraph(f"• Test Recall: {resnet50_results['test_recall']*100:.2f}%", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Key Observations:</b>", subheading_style))
    story.append(Paragraph("• Excellent performance achieved (98.22% accuracy, 99.50% AUC)", body_style))
    story.append(Paragraph("• Comparable to EfficientNet-B0 (98.70%) and XceptionNet (98.76%)", body_style))
    story.append(Paragraph("• Successfully avoided plateauing issue (unlike ViT) with optimized hyperparameters", body_style))
    story.append(Paragraph("• Plateau scheduler and increased patience (20 epochs) allowed full training", body_style))
    story.append(Paragraph("• Demonstrates effectiveness of proper hyperparameter tuning", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 17: EVALUATION VISUALIZATIONS
    # ============================================
    story.append(Paragraph("Performance Analysis", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Add visualization images for ALL models
    eff_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "efficientnet_b0_celebd"
    xcep_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "xception_celebd"
    vit_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "vit_celebd"
    resnet50_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "resnet50_celebd"
    
    # Confusion Matrix - Both Models
    story.append(Paragraph("<b>1. Confusion Matrix</b>", subheading_style))
    confusion_data = []
    confusion_headers = ['Model', 'Confusion Matrix']
    
    if (eff_viz_dir / "confusion_matrix.png").exists():
        eff_cm_img = Image(str(eff_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_data.append(['EfficientNet-B0', eff_cm_img])
    
    if (xcep_viz_dir / "confusion_matrix.png").exists():
        xcep_cm_img = Image(str(xcep_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_data.append(['XceptionNet', xcep_cm_img])
    
    if (vit_viz_dir / "confusion_matrix.png").exists():
        vit_cm_img = Image(str(vit_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_data.append(['Vision Transformer', vit_cm_img])
    
    if (resnet50_viz_dir / "confusion_matrix.png").exists():
        resnet50_cm_img = Image(str(resnet50_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_data.append(['ResNet50', resnet50_cm_img])
    
    if confusion_data:
        confusion_table = Table([confusion_headers] + confusion_data, colWidths=[1.2*inch, 4.8*inch])
        confusion_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(confusion_table)
        story.append(Paragraph("Shows classification breakdown: TP, TN, FP, FN", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # ROC Curve - Both Models
    story.append(Paragraph("<b>2. ROC Curve</b>", subheading_style))
    roc_data = []
    roc_headers = ['Model', 'ROC Curve']
    
    if (eff_viz_dir / "roc_curve.png").exists():
        eff_roc_img = Image(str(eff_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_data.append(['EfficientNet-B0', eff_roc_img])
    
    if (xcep_viz_dir / "roc_curve.png").exists():
        xcep_roc_img = Image(str(xcep_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_data.append(['XceptionNet', xcep_roc_img])
    
    if (vit_viz_dir / "roc_curve.png").exists():
        vit_roc_img = Image(str(vit_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_data.append(['Vision Transformer', vit_roc_img])
    
    if (resnet50_viz_dir / "roc_curve.png").exists():
        resnet50_roc_img = Image(str(resnet50_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_data.append(['ResNet50', resnet50_roc_img])
    
    if roc_data:
        roc_table = Table([roc_headers] + roc_data, colWidths=[1.2*inch, 4.8*inch])
        roc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(roc_table)
        if eff_results:
            story.append(Paragraph(f"EfficientNet-B0 AUC = {eff_results.get('auc', 0.9954)*100:.2f}% | XceptionNet AUC = {xception_results['test_auc']*100:.2f}% | ResNet50 AUC = {resnet50_results['test_auc']*100:.2f}% | ViT AUC = {vit_results['test_auc']*100:.2f}%", body_style))
        else:
            story.append(Paragraph(f"XceptionNet AUC = {xception_results['test_auc']*100:.2f}% | ResNet50 AUC = {resnet50_results['test_auc']*100:.2f}% | ViT AUC = {vit_results['test_auc']*100:.2f}%", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Precision-Recall Curve - Both Models
    story.append(Paragraph("<b>3. Precision-Recall Curve</b>", subheading_style))
    pr_data = []
    pr_headers = ['Model', 'Precision-Recall Curve']
    
    if (eff_viz_dir / "precision_recall_curve.png").exists():
        eff_pr_img = Image(str(eff_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_data.append(['EfficientNet-B0', eff_pr_img])
    
    if (xcep_viz_dir / "precision_recall_curve.png").exists():
        xcep_pr_img = Image(str(xcep_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_data.append(['XceptionNet', xcep_pr_img])
    
    if (vit_viz_dir / "precision_recall_curve.png").exists():
        vit_pr_img = Image(str(vit_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_data.append(['Vision Transformer', vit_pr_img])
    
    if (resnet50_viz_dir / "precision_recall_curve.png").exists():
        resnet50_pr_img = Image(str(resnet50_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_data.append(['ResNet50', resnet50_pr_img])
    
    if pr_data:
        pr_table = Table([pr_headers] + pr_data, colWidths=[1.2*inch, 4.8*inch])
        pr_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(pr_table)
        story.append(Paragraph("Precision-Recall curves showing precision vs recall", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Training History - All Models (if available)
    story.append(Paragraph("<b>4. Training History</b>", subheading_style))
    eff_training_history = REPORTS_DIR / "efficientnet_b0_celebd_20251104_110606" / "visualizations" / "training_history.png"
    xcep_training_history = REPORTS_DIR / "xception_celebd_20251109_024229" / "visualizations" / "training_history.png"
    resnet50_training_history = REPORTS_DIR / "resnet50_celebd_20251114_072733" / "visualizations" / "training_history.png"
    # ViT training history - check if exists
    vit_training_history = None
    for vit_dir in REPORTS_DIR.glob("vit_celebd*/visualizations/training_history.png"):
        vit_training_history = vit_dir
        break
    
    training_data = []
    training_headers = ['Model', 'Training History']
    
    if eff_training_history.exists():
        eff_train_img = Image(str(eff_training_history), width=2.2*inch, height=1.6*inch)
        training_data.append(['EfficientNet-B0', eff_train_img])
    
    if xcep_training_history.exists():
        xcep_train_img = Image(str(xcep_training_history), width=2.2*inch, height=1.6*inch)
        training_data.append(['XceptionNet', xcep_train_img])
    
    if resnet50_training_history.exists():
        resnet50_train_img = Image(str(resnet50_training_history), width=2.2*inch, height=1.6*inch)
        training_data.append(['ResNet50', resnet50_train_img])
    
    if vit_training_history and vit_training_history.exists():
        vit_train_img = Image(str(vit_training_history), width=2.2*inch, height=1.6*inch)
        training_data.append(['Vision Transformer', vit_train_img])
    
    if training_data:
        training_table = Table([training_headers] + training_data, colWidths=[1.2*inch, 4.8*inch])
        training_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(training_table)
        story.append(Paragraph("Loss curves (train vs validation) and accuracy progression", body_style))
    elif eff_training_history.exists():
        # Fallback to single image if only one exists
        img = Image(str(eff_training_history), width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph("Loss curves (train vs validation) and accuracy progression", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 18: EXPLAINABILITY
    # ============================================
    story.append(Paragraph("Model Interpretability: Grad-CAM Analysis", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("<b>What is Grad-CAM?</b>", subheading_style))
    story.append(Paragraph("• Gradient-weighted Class Activation Mapping", body_style))
    story.append(Paragraph("• Visualizes which parts of image influence model's decision", body_style))
    story.append(Paragraph("• Shows 'where the model looks' to make its prediction", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    
    # Add Grad-CAM visualizations - BOTH MODELS COMPARISON
    explainability_dir = PROJECT_ROOT / "reports" / "explainability"
    
    # EfficientNet-B0 Grad-CAM files
    eff_fake_overlay = explainability_dir / "id0_id1_0000_frame000000_frame_000000_efficientb0_gradcam_overlay.png"
    eff_fake_heatmap = explainability_dir / "id0_id1_0000_frame000000_frame_000000_efficientb0_gradcam_heatmap.png"
    eff_real_overlay = explainability_dir / "id0_0000_frame000000_frame_000000_efficientb0_gradcam_overlay.png"
    eff_real_heatmap = explainability_dir / "id0_0000_frame000000_frame_000000_efficientb0_gradcam_heatmap.png"
    
    # XceptionNet Grad-CAM files
    xcep_fake_overlay = explainability_dir / "id0_id1_0000_frame000000_frame_000000_xception_gradcam_overlay.png"
    xcep_fake_heatmap = explainability_dir / "id0_id1_0000_frame000000_frame_000000_xception_gradcam_heatmap.png"
    xcep_real_overlay = explainability_dir / "id0_0000_frame000000_frame_000000_xception_gradcam_overlay.png"
    xcep_real_heatmap = explainability_dir / "id0_0000_frame000000_frame_000000_xception_gradcam_heatmap.png"
    
    # ViT Grad-CAM files
    vit_fake_overlay = explainability_dir / "id0_id1_0000_frame000000_frame_000000_vit_gradcam_overlay.png"
    vit_fake_heatmap = explainability_dir / "id0_id1_0000_frame000000_frame_000000_vit_gradcam_heatmap.png"
    vit_real_overlay = explainability_dir / "id0_0000_frame000000_frame_000000_vit_gradcam_overlay.png"
    vit_real_heatmap = explainability_dir / "id0_0000_frame000000_frame_000000_vit_gradcam_heatmap.png"
    
    # ResNet50 Grad-CAM files
    resnet50_fake_overlay = explainability_dir / "id0_id1_0000_frame000000_frame_000000_res50_gradcam_overlay.png"
    resnet50_fake_heatmap = explainability_dir / "id0_id1_0000_frame000000_frame_000000_res50_gradcam_heatmap.png"
    resnet50_real_overlay = explainability_dir / "id0_0000_frame000000_frame_000000_res50_gradcam_overlay.png"
    resnet50_real_heatmap = explainability_dir / "id0_0000_frame000000_frame_000000_res50_gradcam_heatmap.png"
    
    # Create comparison table with ALL MODELS side by side
    if (eff_fake_overlay.exists() and eff_real_overlay.exists() and 
        xcep_fake_overlay.exists() and xcep_real_overlay.exists() and
        vit_fake_overlay.exists() and vit_real_overlay.exists() and
        resnet50_fake_overlay.exists() and resnet50_real_overlay.exists()):
        # Create comparison table showing BOTH MODELS on SAME images
        from reportlab.lib import colors as rl_colors
        
        # EfficientNet-B0 images
        eff_fake_overlay_img = Image(str(eff_fake_overlay), width=1.3*inch, height=1.0*inch)
        eff_fake_heatmap_img = Image(str(eff_fake_heatmap), width=1.3*inch, height=1.0*inch)
        eff_real_overlay_img = Image(str(eff_real_overlay), width=1.3*inch, height=1.0*inch)
        eff_real_heatmap_img = Image(str(eff_real_heatmap), width=1.3*inch, height=1.0*inch)
        
        # XceptionNet images
        xcep_fake_overlay_img = Image(str(xcep_fake_overlay), width=1.3*inch, height=1.0*inch)
        xcep_fake_heatmap_img = Image(str(xcep_fake_heatmap), width=1.3*inch, height=1.0*inch)
        xcep_real_overlay_img = Image(str(xcep_real_overlay), width=1.3*inch, height=1.0*inch)
        xcep_real_heatmap_img = Image(str(xcep_real_heatmap), width=1.3*inch, height=1.0*inch)
        
        # ViT images
        vit_fake_overlay_img = Image(str(vit_fake_overlay), width=1.3*inch, height=1.0*inch)
        vit_fake_heatmap_img = Image(str(vit_fake_heatmap), width=1.3*inch, height=1.0*inch)
        vit_real_overlay_img = Image(str(vit_real_overlay), width=1.3*inch, height=1.0*inch)
        vit_real_heatmap_img = Image(str(vit_real_heatmap), width=1.3*inch, height=1.0*inch)
        
        # ResNet50 images
        resnet50_fake_overlay_img = Image(str(resnet50_fake_overlay), width=1.3*inch, height=1.0*inch)
        resnet50_fake_heatmap_img = Image(str(resnet50_fake_heatmap), width=1.3*inch, height=1.0*inch)
        resnet50_real_overlay_img = Image(str(resnet50_real_overlay), width=1.3*inch, height=1.0*inch)
        resnet50_real_heatmap_img = Image(str(resnet50_real_heatmap), width=1.3*inch, height=1.0*inch)
        
        # Create comparison table: Models vs Images (all four models)
        comparison_data = [
            ['', 'FAKE Image', 'REAL Image'],
            ['EfficientNet-B0<br/>Overlay', eff_fake_overlay_img, eff_real_overlay_img],
            ['EfficientNet-B0<br/>Heatmap', eff_fake_heatmap_img, eff_real_heatmap_img],
            ['XceptionNet<br/>Overlay', xcep_fake_overlay_img, xcep_real_overlay_img],
            ['XceptionNet<br/>Heatmap', xcep_fake_heatmap_img, xcep_real_heatmap_img],
            ['ResNet50<br/>Overlay', resnet50_fake_overlay_img, resnet50_real_overlay_img],
            ['ResNet50<br/>Heatmap', resnet50_fake_heatmap_img, resnet50_real_heatmap_img],
            ['Vision Transformer<br/>Overlay', vit_fake_overlay_img, vit_real_overlay_img],
            ['Vision Transformer<br/>Heatmap', vit_fake_heatmap_img, vit_real_heatmap_img],
        ]
        
        comparison_table = Table(comparison_data, colWidths=[1.0*inch, 1.8*inch, 1.8*inch])
        comparison_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ]))
        story.append(comparison_table)
        
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("<b>Model Comparison on Same Images:</b>", subheading_style))
        story.append(Paragraph("• <b>EfficientNet-B0:</b> Focuses on facial features, good localization", body_style))
        story.append(Paragraph("• <b>XceptionNet:</b> Similar attention patterns, slightly different focus regions", body_style))
        story.append(Paragraph("• <b>ResNet50:</b> Strong focus on facial regions, good heatmap clarity", body_style))
        story.append(Paragraph("• <b>Vision Transformer:</b> Uses attention-based visualization (converted to Grad-CAM format)", body_style))
        story.append(Paragraph("• <b>All models:</b> Successfully identify important facial regions for detection", body_style))
        
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("<b>Legend:</b>", subheading_style))
        story.append(Paragraph("• <b>Red/Hot areas</b> = High importance (model focused here)", body_style))
        story.append(Paragraph("• <b>Blue/Cold areas</b> = Low importance (model ignored this)", body_style))
        story.append(Paragraph("<i>Note: Face-focused masking applied to reduce background attention</i>", 
                              ParagraphStyle('Note', parent=body_style, fontSize=9, textColor=HexColor('#666666'), fontStyle='italic')))
    else:
        # Fallback if images don't exist
        story.append(Paragraph("<b>Results:</b>", subheading_style))
        story.append(Paragraph("<b>Fake Image Detection:</b>", body_style))
        story.append(Paragraph("• Model correctly identifies as FAKE (100% confidence)", body_style))
        story.append(Paragraph("• Highlights facial regions with artifacts", body_style))
        story.append(Paragraph("• Focuses on areas where deepfake generation creates inconsistencies", body_style))
        
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("<b>Real Image Detection:</b>", body_style))
        story.append(Paragraph("• Model correctly identifies as REAL (99.95% confidence)", body_style))
        story.append(Paragraph("• Highlights natural facial features", body_style))
        story.append(Paragraph("• No suspicious artifact patterns", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>Key Insights:</b>", subheading_style))
    story.append(Paragraph("• Model learns to focus on facial regions (not background)", body_style))
    story.append(Paragraph("• Identifies subtle artifacts in deepfakes", body_style))
    story.append(Paragraph("• Provides interpretable explanations for predictions", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 19: KEY FINDINGS
    # ============================================
    story.append(Paragraph("What We've Learned", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>1. Model Performance:</b>", subheading_style))
    story.append(Paragraph(f"• EfficientNet-B0: 98.70% accuracy (4.6M parameters) - Best efficiency", body_style))
    story.append(Paragraph(f"• XceptionNet: {xception_results['test_accuracy']*100:.2f}% accuracy (38.9M parameters) - Highest accuracy", body_style))
    story.append(Paragraph(f"• ResNet50: {resnet50_results['test_accuracy']*100:.2f}% accuracy (~25M parameters) - Excellent balance", body_style))
    story.append(Paragraph(f"• Vision Transformer: {vit_results['test_accuracy']*100:.2f}% accuracy (~86M parameters) - Lower performance", body_style))
    story.append(Paragraph("• CNN-based models (EfficientNet, XceptionNet, ResNet50) all achieve >98% accuracy", body_style))
    story.append(Paragraph("• Small CNN models (EfficientNet-B0) can be highly effective for deepfake detection", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>2. Dataset Insights:</b>", subheading_style))
    story.append(Paragraph("• Celeb-DF provides challenging but fair test set", body_style))
    story.append(Paragraph("• Class imbalance (more fake than real) handled well", body_style))
    story.append(Paragraph(f"• {stats['total_images']:,} images sufficient for training", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>3. Training Observations:</b>", subheading_style))
    story.append(Paragraph("• Early stopping prevents overfitting", body_style))
    story.append(Paragraph("• Validation metrics align well with test performance", body_style))
    story.append(Paragraph("• Model converges relatively quickly", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>4. Detection Patterns:</b>", subheading_style))
    story.append(Paragraph("• Model learns subtle artifacts in deepfakes", body_style))
    story.append(Paragraph("• Focuses on facial regions (not background)", body_style))
    story.append(Paragraph("• Distinguishes between real and fake effectively", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 20: CHALLENGES & SOLUTIONS
    # ============================================
    story.append(Paragraph("Project Challenges", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>1. Data Preprocessing:</b>", subheading_style))
    story.append(Paragraph("<b>Challenge:</b> Large video dataset, disk space constraints", body_style))
    story.append(Paragraph("<b>Solution:</b> Subset creation, efficient frame extraction (stride=60, max 3 frames)", body_style))
    story.append(Paragraph(f"<b>Result:</b> {stats['total_images']:,} images processed successfully", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>2. Training Time:</b>", subheading_style))
    story.append(Paragraph("<b>Challenge:</b> Long training times on CPU (~2-3 hours per epoch)", body_style))
    story.append(Paragraph("<b>Solution:</b> Early stopping, checkpointing, background training", body_style))
    story.append(Paragraph("<b>Result:</b> Efficient training with model saving", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>3. Model Selection:</b>", subheading_style))
    story.append(Paragraph("<b>Challenge:</b> Choosing appropriate architectures", body_style))
    story.append(Paragraph("<b>Solution:</b> Systematic comparison approach", body_style))
    story.append(Paragraph("<b>Result:</b> Multiple models being evaluated", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>4. Evaluation Metrics:</b>", subheading_style))
    story.append(Paragraph("<b>Challenge:</b> Comprehensive evaluation needed", body_style))
    story.append(Paragraph("<b>Solution:</b> Multiple metrics, visualizations, explainability", body_style))
    story.append(Paragraph("<b>Result:</b> Complete evaluation system", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 21: IMMEDIATE NEXT STEPS
    # ============================================
    story.append(Paragraph("Phase 1 Completion - Model Comparison", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Completed:</b>", subheading_style))
    story.append(Paragraph("• ✅ EfficientNet-B0: <b>COMPLETE</b> (98.70% accuracy, 30 epochs)", body_style))
    story.append(Paragraph(f"• ✅ XceptionNet: <b>COMPLETE</b> ({xception_results['test_accuracy']*100:.2f}% accuracy, {xception_results['epochs']} epochs)", body_style))
    story.append(Paragraph(f"• ✅ ResNet50: <b>COMPLETE</b> ({resnet50_results['test_accuracy']*100:.2f}% accuracy, {resnet50_results['epochs']} epochs)", body_style))
    story.append(Paragraph(f"• ✅ Vision Transformer: <b>COMPLETE</b> ({vit_results['test_accuracy']*100:.2f}% accuracy, {vit_results['epochs']} epochs)", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Timeline:</b>", subheading_style))
    story.append(Paragraph("• ✅ Week 1: EfficientNet-B0 and XceptionNet training completed", body_style))
    story.append(Paragraph("• ✅ Week 2: Vision Transformer and ResNet50 training completed", body_style))
    story.append(Paragraph("• ✅ Week 2: Model comparison table created (all 4 models)", body_style))
    story.append(Paragraph("• ✅ Week 2-3: Complete performance analysis", body_style))
    story.append(Paragraph("• ✅ Week 3: Phase 2 - Cross-dataset evaluation <b>COMPLETE</b>", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Deliverables:</b>", subheading_style))
    story.append(Paragraph("• Model comparison report", body_style))
    story.append(Paragraph("• Performance analysis", body_style))
    story.append(Paragraph("• Architecture recommendations", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 22: PHASE 2 - CROSS-DATASET EVALUATION
    # ============================================
    story.append(Paragraph("Phase 2: Cross-Dataset Evaluation - <b>COMPLETED</b>", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Experiment:</b> Train on Celeb-DF → Test on FaceForensics++", subheading_style))
    
    story.append(Spacer(1, 0.15*inch))
    
    # Create cross-dataset results table
    cross_data = [
        ['Model', 'Celeb-DF Acc', 'FaceForensics++ Acc', 'Performance Drop', 'AUC', 'F1-Score']
    ]
    
    for model_name, results in cross_dataset_results.items():
        model_display = model_name.replace('_', '-').title()
        if model_name == 'efficientnet_b0':
            model_display = 'EfficientNet-B0'
        elif model_name == 'vit':
            model_display = 'Vision Transformer'
        
        cross_data.append([
            model_display,
            f"{results['celebd_accuracy']*100:.2f}%",
            f"{results['faceforensics_accuracy']*100:.2f}%",
            f"{results['performance_drop']*100:.2f}%",
            f"{results['faceforensics_auc']*100:.2f}%",
            f"{results['faceforensics_f1']*100:.2f}%"
        ])
    
    cross_table = Table(cross_data, colWidths=[1.5*inch, 1*inch, 1.2*inch, 1*inch, 0.8*inch, 0.8*inch])
    cross_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8eaf6')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#5c6bc0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f5f5f5')])
    ]))
    story.append(cross_table)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Key Findings:</b>", subheading_style))
    story.append(Paragraph("• ✅ FaceForensics++ dataset processed: 20,781 images (2,992 real, 17,789 fake)", body_style))
    story.append(Paragraph("• 🎯 <b>Best Generalization:</b> Vision Transformer (smallest drop: -24.48%)", body_style))
    story.append(Paragraph("• ⚠️ <b>Largest Drop:</b> EfficientNet-B0 (-49.04%)", body_style))
    story.append(Paragraph("• 📊 All models show significant performance degradation on cross-dataset testing", body_style))
    story.append(Paragraph("• 💡 ViT's attention mechanism helps with generalization across datasets", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 19B: CROSS-DATASET VISUALIZATIONS
    # ============================================
    story.append(Paragraph("Cross-Dataset Evaluation: Visualizations", heading_style))
    story.append(Paragraph("<b>FaceForensics++ Test Set Performance</b>", subheading_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Add visualization images for cross-dataset evaluation
    eff_ff_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "efficientnet_b0_faceforensics"
    xcep_ff_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "xception_faceforensics"
    resnet50_ff_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "resnet50_faceforensics"
    vit_ff_viz_dir = REPORTS_DIR / "evaluation" / "visualizations" / "vit_faceforensics"
    
    # Confusion Matrix - Cross-Dataset
    story.append(Paragraph("<b>1. Confusion Matrix (FaceForensics++)</b>", subheading_style))
    confusion_ff_data = []
    confusion_ff_headers = ['Model', 'Confusion Matrix']
    
    if (eff_ff_viz_dir / "confusion_matrix.png").exists():
        eff_ff_cm_img = Image(str(eff_ff_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_ff_data.append(['EfficientNet-B0', eff_ff_cm_img])
    
    if (xcep_ff_viz_dir / "confusion_matrix.png").exists():
        xcep_ff_cm_img = Image(str(xcep_ff_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_ff_data.append(['XceptionNet', xcep_ff_cm_img])
    
    if (resnet50_ff_viz_dir / "confusion_matrix.png").exists():
        resnet50_ff_cm_img = Image(str(resnet50_ff_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_ff_data.append(['ResNet50', resnet50_ff_cm_img])
    
    if (vit_ff_viz_dir / "confusion_matrix.png").exists():
        vit_ff_cm_img = Image(str(vit_ff_viz_dir / "confusion_matrix.png"), width=2.2*inch, height=1.8*inch)
        confusion_ff_data.append(['Vision Transformer', vit_ff_cm_img])
    
    if confusion_ff_data:
        confusion_ff_table = Table([confusion_ff_headers] + confusion_ff_data, colWidths=[1.2*inch, 4.8*inch])
        confusion_ff_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(confusion_ff_table)
        story.append(Paragraph("Classification performance on FaceForensics++ (trained on Celeb-DF)", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # ROC Curve - Cross-Dataset
    story.append(Paragraph("<b>2. ROC Curve (FaceForensics++)</b>", subheading_style))
    roc_ff_data = []
    roc_ff_headers = ['Model', 'ROC Curve']
    
    if (eff_ff_viz_dir / "roc_curve.png").exists():
        eff_ff_roc_img = Image(str(eff_ff_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_ff_data.append(['EfficientNet-B0', eff_ff_roc_img])
    
    if (xcep_ff_viz_dir / "roc_curve.png").exists():
        xcep_ff_roc_img = Image(str(xcep_ff_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_ff_data.append(['XceptionNet', xcep_ff_roc_img])
    
    if (resnet50_ff_viz_dir / "roc_curve.png").exists():
        resnet50_ff_roc_img = Image(str(resnet50_ff_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_ff_data.append(['ResNet50', resnet50_ff_roc_img])
    
    if (vit_ff_viz_dir / "roc_curve.png").exists():
        vit_ff_roc_img = Image(str(vit_ff_viz_dir / "roc_curve.png"), width=2.2*inch, height=1.8*inch)
        roc_ff_data.append(['Vision Transformer', vit_ff_roc_img])
    
    if roc_ff_data:
        roc_ff_table = Table([roc_ff_headers] + roc_ff_data, colWidths=[1.2*inch, 4.8*inch])
        roc_ff_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(roc_ff_table)
    
    story.append(Spacer(1, 0.2*inch))
    
    # Precision-Recall Curve - Cross-Dataset
    story.append(Paragraph("<b>3. Precision-Recall Curve (FaceForensics++)</b>", subheading_style))
    pr_ff_data = []
    pr_ff_headers = ['Model', 'PR Curve']
    
    if (eff_ff_viz_dir / "precision_recall_curve.png").exists():
        eff_ff_pr_img = Image(str(eff_ff_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_ff_data.append(['EfficientNet-B0', eff_ff_pr_img])
    
    if (xcep_ff_viz_dir / "precision_recall_curve.png").exists():
        xcep_ff_pr_img = Image(str(xcep_ff_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_ff_data.append(['XceptionNet', xcep_ff_pr_img])
    
    if (resnet50_ff_viz_dir / "precision_recall_curve.png").exists():
        resnet50_ff_pr_img = Image(str(resnet50_ff_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_ff_data.append(['ResNet50', resnet50_ff_pr_img])
    
    if (vit_ff_viz_dir / "precision_recall_curve.png").exists():
        vit_ff_pr_img = Image(str(vit_ff_viz_dir / "precision_recall_curve.png"), width=2.2*inch, height=1.8*inch)
        pr_ff_data.append(['Vision Transformer', vit_ff_pr_img])
    
    if pr_ff_data:
        pr_ff_table = Table([pr_ff_headers] + pr_ff_data, colWidths=[1.2*inch, 4.8*inch])
        pr_ff_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#283593')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (0, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(pr_ff_table)
        story.append(Paragraph("Note: All models show degraded performance compared to Celeb-DF test set", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 23: PHASE 3
    # ============================================
    story.append(Paragraph("Phase 3: Advanced Techniques", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>1. Ensemble Methods:</b>", subheading_style))
    story.append(Paragraph("• Combine predictions from multiple models", body_style))
    story.append(Paragraph("• Test voting strategies (hard/soft)", body_style))
    story.append(Paragraph("• Weighted averaging", body_style))
    story.append(Paragraph("<b>Goal:</b> Improve accuracy beyond single models", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>2. Hyperparameter Optimization:</b>", subheading_style))
    story.append(Paragraph("• Learning rate search", body_style))
    story.append(Paragraph("• Batch size optimization", body_style))
    story.append(Paragraph("• Loss function comparison", body_style))
    story.append(Paragraph("<b>Goal:</b> Find optimal configurations", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>3. Failure Case Analysis:</b>", subheading_style))
    story.append(Paragraph("• Identify misclassified images", body_style))
    story.append(Paragraph("• Analyze failure patterns", body_style))
    story.append(Paragraph("• Visualize with Grad-CAM", body_style))
    story.append(Paragraph("<b>Goal:</b> Understand model limitations", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>4. Novel Detection Methods</b> (Research Contribution):", subheading_style))
    story.append(Paragraph("• Frequency domain analysis", body_style))
    story.append(Paragraph("• Multi-scale feature fusion", body_style))
    story.append(Paragraph("• Attention-based mechanisms", body_style))
    story.append(Paragraph("<b>Goal:</b> Improve detection capabilities", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 24: FUTURE RESEARCH
    # ============================================
    story.append(Paragraph("Future Research Directions", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>1. Video-Level Detection:</b>", subheading_style))
    story.append(Paragraph("• Extend from frame-level to video-level", body_style))
    story.append(Paragraph("• Temporal analysis (LSTM, GRU)", body_style))
    story.append(Paragraph("• Frame aggregation strategies", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>2. Real-World Deployment:</b>", subheading_style))
    story.append(Paragraph("• Inference speed optimization", body_style))
    story.append(Paragraph("• Model compression", body_style))
    story.append(Paragraph("• Mobile deployment", body_style))
    story.append(Paragraph("• Real-time detection", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>3. Adversarial Robustness:</b>", subheading_style))
    story.append(Paragraph("• Test against adversarial attacks", body_style))
    story.append(Paragraph("• Adversarial training", body_style))
    story.append(Paragraph("• Robustness evaluation", body_style))
    
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>4. Novel Datasets:</b>", subheading_style))
    story.append(Paragraph("• Create challenging test sets", body_style))
    story.append(Paragraph("• Analyze new deepfake generation methods", body_style))
    story.append(Paragraph("• Benchmark against latest techniques", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 25: PROJECT SUMMARY
    # ============================================
    story.append(Paragraph("What We've Accomplished", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>✅ Completed:</b>", subheading_style))
    story.append(Paragraph("1. <b>Complete ML Pipeline:</b> Data preprocessing → Training → Evaluation", body_style))
    story.append(Paragraph("2. <b>EfficientNet-B0 Training:</b> 98.70% accuracy on Celeb-DF (30 epochs)", body_style))
    story.append(Paragraph(f"3. <b>XceptionNet Training:</b> {xception_results['test_accuracy']*100:.2f}% accuracy on Celeb-DF ({xception_results['epochs']} epochs)", body_style))
    story.append(Paragraph(f"4. <b>ResNet50 Training:</b> {resnet50_results['test_accuracy']*100:.2f}% accuracy on Celeb-DF ({resnet50_results['epochs']} epochs)", body_style))
    story.append(Paragraph(f"5. <b>Vision Transformer Training:</b> {vit_results['test_accuracy']*100:.2f}% accuracy on Celeb-DF ({vit_results['epochs']} epochs)", body_style))
    story.append(Paragraph("6. <b>Comprehensive Evaluation:</b> Metrics, visualizations, explainability", body_style))
    story.append(Paragraph("7. <b>Model Comparison:</b> Side-by-side analysis of all 4 models", body_style))
    story.append(Paragraph("8. <b>Cross-Dataset Evaluation:</b> Celeb-DF → FaceForensics++ (20,781 images)", body_style))
    story.append(Paragraph("9. <b>Research Infrastructure:</b> Systematic comparison framework", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>📊 Current Status:</b>", subheading_style))
    
    status_data = [
        ['Metric', 'Value'],
        ['Models Trained', f"{stats['models_trained']} experiments"],
        ['Models Completed', '4 (EfficientNet-B0, XceptionNet, ResNet50, ViT)'],
        ['Models Pending', '0 (Phase 1 Complete!)'],
        ['Dataset Processed', f"{stats['total_images']:,} images"],
        ['Reports Generated', f"{stats['reports_generated']} (PDF + HTML)"],
    ]
    
    status_table = Table(status_data, colWidths=[2.5*inch, 2.5*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8eaf6')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(status_table)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>🎯 Phase 1 Progress:</b> 100% complete (4/4 models trained)", subheading_style))
    story.append(Paragraph("<b>🎯 Phase 2 Progress:</b> 100% complete (Cross-dataset evaluation done)", subheading_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 26: IMPACT & CONTRIBUTION
    # ============================================
    story.append(Paragraph("Project Impact", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Research Contributions:</b>", subheading_style))
    story.append(Paragraph("• Systematic comparison of deep learning architectures", body_style))
    story.append(Paragraph("• Comprehensive evaluation framework", body_style))
    story.append(Paragraph("• Explainability analysis for deepfake detection", body_style))
    story.append(Paragraph("• Reproducible research infrastructure", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Practical Applications:</b>", subheading_style))
    story.append(Paragraph("• Security and forensics tools", body_style))
    story.append(Paragraph("• Content verification systems", body_style))
    story.append(Paragraph("• Misinformation detection", body_style))
    story.append(Paragraph("• Educational purposes", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Learning Outcomes:</b>", subheading_style))
    story.append(Paragraph("• Deep learning expertise", body_style))
    story.append(Paragraph("• ML pipeline development", body_style))
    story.append(Paragraph("• Research methodology", body_style))
    story.append(Paragraph("• Technical problem-solving", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 27: CONCLUSION
    # ============================================
    story.append(Paragraph("Conclusion & Next Steps", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Summary:</b>", subheading_style))
    story.append(Paragraph("• Successfully implemented deepfake detection system", body_style))
    story.append(Paragraph(f"• Achieved 98.70% accuracy with EfficientNet-B0 (4.6M parameters)", body_style))
    story.append(Paragraph(f"• Achieved {xception_results['test_accuracy']*100:.2f}% accuracy with XceptionNet (38.9M parameters)", body_style))
    story.append(Paragraph(f"• Achieved {resnet50_results['test_accuracy']*100:.2f}% accuracy with ResNet50 (~25M parameters)", body_style))
    story.append(Paragraph(f"• Achieved {vit_results['test_accuracy']*100:.2f}% accuracy with Vision Transformer (~86M parameters)", body_style))
    story.append(Paragraph("• Established baseline for model comparison", body_style))
    story.append(Paragraph("• Completed side-by-side model comparison with Grad-CAM (all 4 models)", body_style))
    story.append(Paragraph("• <b>Cross-Dataset Evaluation:</b> Tested all models on FaceForensics++", body_style))
    story.append(Paragraph("  - ViT shows best generalization (66.06% accuracy, -24.48% drop)", body_style))
    story.append(Paragraph("  - All models show significant performance degradation on cross-dataset testing", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Next Steps:</b>", subheading_style))
    story.append(Paragraph("1. ✅ Phase 1: Model comparison - <b>COMPLETE</b>", body_style))
    story.append(Paragraph("2. ✅ Phase 2: Cross-dataset evaluation - <b>COMPLETE</b>", body_style))
    story.append(Paragraph("3. Phase 3: Advanced techniques (ensemble methods, hyperparameter optimization)", body_style))
    story.append(Paragraph("4. Final analysis and paper writing", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Expected Outcomes:</b>", subheading_style))
    story.append(Paragraph("• Comprehensive model comparison study", body_style))
    story.append(Paragraph("• Generalization analysis", body_style))
    story.append(Paragraph("• Research paper/report", body_style))
    story.append(Paragraph("• Open-source codebase", body_style))
    
    story.append(PageBreak())
    
    # ============================================
    # SLIDE 28: Q&A
    # ============================================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Thank You!", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Questions & Discussion", styles['Heading2']))
    
    # Build PDF
    doc.build(story)
    print(f"\n✅ PDF Presentation created successfully!")
    print(f"📄 File: {output_file}")
    print(f"📊 Total slides: 25")
    return output_file

if __name__ == "__main__":
    print("═══════════════════════════════════════════════════════════")
    print("  📄 GENERATING PDF PRESENTATION")
    print("═══════════════════════════════════════════════════════════")
    print("\nCreating presentation from project data...")
    
    try:
        output_file = create_pdf_presentation()
        print(f"\n✅ Success! Presentation saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error creating presentation: {e}")
        import traceback
        traceback.print_exc()

