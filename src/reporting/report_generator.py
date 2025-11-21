"""
Automated report generation for deepfake detection experiments.
Creates comprehensive PDF and HTML reports with visualizations and analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import os

# Report generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. Install with: pip install reportlab")

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: jinja2 not available. Install with: pip install jinja2")


class ReportGenerator:
    """Comprehensive report generator for deepfake detection experiments."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_experiment_report(self, 
                                  experiment_data: Dict[str, Any],
                                  model_name: str,
                                  dataset_name: str,
                                  output_format: str = "both") -> Dict[str, str]:
        """
        Generate comprehensive experiment report.
        
        Args:
            experiment_data: Experiment results and metadata
            model_name: Name of the model
            dataset_name: Name of the dataset
            output_format: Output format ("pdf", "html", or "both")
            
        Returns:
            Dictionary with paths to generated reports
        """
        report_paths = {}
        
        # Create experiment-specific directory
        experiment_dir = self.output_dir / f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        viz_paths = self._generate_visualizations(experiment_data, experiment_dir)
        
        # Generate PDF report
        if output_format in ["pdf", "both"] and REPORTLAB_AVAILABLE:
            pdf_path = self._generate_pdf_report(experiment_data, viz_paths, experiment_dir, model_name, dataset_name)
            report_paths["pdf"] = str(pdf_path)
        
        # Generate HTML report
        if output_format in ["html", "both"] and JINJA2_AVAILABLE:
            html_path = self._generate_html_report(experiment_data, viz_paths, experiment_dir, model_name, dataset_name)
            report_paths["html"] = str(html_path)
        
        # Generate JSON summary
        json_path = self._generate_json_summary(experiment_data, experiment_dir)
        report_paths["json"] = str(json_path)
        
        return report_paths
    
    def _generate_visualizations(self, 
                               experiment_data: Dict[str, Any], 
                               output_dir: Path) -> Dict[str, str]:
        """Generate all visualizations for the report."""
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        viz_paths = {}
        
        # Training history plots
        if "training_history" in experiment_data:
            viz_paths["training_history"] = self._plot_training_history(
                experiment_data["training_history"], viz_dir
            )
        
        # Confusion matrix
        if "confusion_matrix" in experiment_data:
            viz_paths["confusion_matrix"] = self._plot_confusion_matrix(
                experiment_data["confusion_matrix"], viz_dir
            )
        
        # ROC curve
        if "roc_curve" in experiment_data:
            viz_paths["roc_curve"] = self._plot_roc_curve(
                experiment_data["roc_curve"], viz_dir
            )
        
        # Precision-Recall curve
        if "precision_recall_curve" in experiment_data:
            viz_paths["precision_recall_curve"] = self._plot_precision_recall_curve(
                experiment_data["precision_recall_curve"], viz_dir
            )
        
        # Cross-dataset comparison
        if "cross_dataset_results" in experiment_data:
            viz_paths["cross_dataset_comparison"] = self._plot_cross_dataset_comparison(
                experiment_data["cross_dataset_results"], viz_dir
            )
        
        # Model comparison
        if "model_comparison" in experiment_data:
            viz_paths["model_comparison"] = self._plot_model_comparison(
                experiment_data["model_comparison"], viz_dir
            )
        
        return viz_paths
    
    def _plot_training_history(self, history: Dict[str, List[float]], output_dir: Path) -> str:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history.get('train_loss', []), label='Train Loss', color='blue')
        axes[0, 0].plot(history.get('val_loss', []), label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history.get('train_acc', []), label='Train Accuracy', color='blue')
        axes[0, 1].plot(history.get('val_acc', []), label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC plot
        axes[1, 0].plot(history.get('val_auc', []), label='Validation AUC', color='green')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='orange')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        save_path = output_dir / "training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_confusion_matrix(self, cm_data: Dict[str, Any], output_dir: Path) -> str:
        """Plot confusion matrix."""
        cm = cm_data["matrix"]
        labels = cm_data.get("labels", ["Real", "Fake"])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        save_path = output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_roc_curve(self, roc_data: Dict[str, Any], output_dir: Path) -> str:
        """Plot ROC curve."""
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc = roc_data["auc"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        save_path = output_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_precision_recall_curve(self, pr_data: Dict[str, Any], output_dir: Path) -> str:
        """Plot precision-recall curve."""
        precision = pr_data["precision"]
        recall = pr_data["recall"]
        ap = pr_data["average_precision"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {ap:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        save_path = output_dir / "precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_cross_dataset_comparison(self, cross_data: Dict[str, Any], output_dir: Path) -> str:
        """Plot cross-dataset comparison."""
        datasets = list(cross_data.keys())
        metrics = ["auc", "accuracy", "precision", "recall", "f1_score"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            values = [cross_data[dataset].get(metric, 0) for dataset in datasets]
            
            bars = axes[i].bar(datasets, values, color='skyblue', edgecolor='navy', alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        
        save_path = output_dir / "cross_dataset_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_model_comparison(self, model_data: Dict[str, Any], output_dir: Path) -> str:
        """Plot model comparison."""
        models = list(model_data.keys())
        metrics = ["auc", "accuracy", "precision", "recall", "f1_score"]
        
        # Create comparison table
        comparison_data = []
        for model in models:
            row = [model]
            for metric in metrics:
                row.append(model_data[model].get(metric, 0))
            comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data, columns=["Model"] + [m.upper() for m in metrics])
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the plot
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [model_data[model].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _generate_pdf_report(self, 
                           experiment_data: Dict[str, Any],
                           viz_paths: Dict[str, str],
                           output_dir: Path,
                           model_name: str,
                           dataset_name: str) -> Path:
        """Generate PDF report."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF report generation")
        
        pdf_path = output_dir / f"{model_name}_{dataset_name}_report.pdf"
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(f"Deepfake Detection Report", title_style))
        story.append(Paragraph(f"Model: {model_name}", styles['Heading2']))
        story.append(Paragraph(f"Dataset: {dataset_name}", styles['Heading2']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = self._generate_summary_text(experiment_data)
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Metrics Table
        if "metrics" in experiment_data:
            story.append(Paragraph("Performance Metrics", styles['Heading2']))
            metrics_table = self._create_metrics_table(experiment_data["metrics"])
            story.append(metrics_table)
            story.append(Spacer(1, 20))
        
        # Visualizations
        story.append(Paragraph("Visualizations", styles['Heading2']))
        
        for viz_name, viz_path in viz_paths.items():
            if os.path.exists(viz_path):
                story.append(Paragraph(f"{viz_name.replace('_', ' ').title()}", styles['Heading3']))
                img = Image(viz_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
        
        # Cross-dataset Results
        if "cross_dataset_results" in experiment_data:
            story.append(Paragraph("Cross-Dataset Evaluation", styles['Heading2']))
            cross_table = self._create_cross_dataset_table(experiment_data["cross_dataset_results"])
            story.append(cross_table)
            story.append(Spacer(1, 20))
        
        # Conclusions
        story.append(Paragraph("Conclusions", styles['Heading2']))
        conclusions = self._generate_conclusions(experiment_data)
        story.append(Paragraph(conclusions, styles['Normal']))
        
        doc.build(story)
        return pdf_path
    
    def _generate_html_report(self, 
                            experiment_data: Dict[str, Any],
                            viz_paths: Dict[str, str],
                            output_dir: Path,
                            model_name: str,
                            dataset_name: str) -> Path:
        """Generate HTML report."""
        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is required for HTML report generation")
        
        html_path = output_dir / f"{model_name}_{dataset_name}_report.html"
        
        # HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; }
                h3 { color: #7f8c8d; }
                .summary { background-color: #ecf0f1; padding: 20px; border-radius: 5px; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .metrics-table th, .metrics-table td { border: 1px solid #bdc3c7; padding: 10px; text-align: left; }
                .metrics-table th { background-color: #3498db; color: white; }
                .visualization { text-align: center; margin: 30px 0; }
                .visualization img { max-width: 100%; height: auto; }
                .conclusions { background-color: #f8f9fa; padding: 20px; border-left: 4px solid #28a745; }
            </style>
        </head>
        <body>
            <h1>Deepfake Detection Report</h1>
            <h2>Model: {{ model_name }}</h2>
            <h2>Dataset: {{ dataset_name }}</h2>
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            
            <h2>Executive Summary</h2>
            <div class="summary">
                {{ summary }}
            </div>
            
            {% if metrics %}
            <h2>Performance Metrics</h2>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% for metric, value in metrics.items() %}
                <tr>
                    <td>{{ metric.replace('_', ' ').title() }}</td>
                    <td>{{ "%.4f"|format(value) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
            
            <h2>Visualizations</h2>
            {% for viz_name, viz_path in visualizations.items() %}
            <div class="visualization">
                <h3>{{ viz_name.replace('_', ' ').title() }}</h3>
                <img src="{{ viz_path }}" alt="{{ viz_name }}">
            </div>
            {% endfor %}
            
            {% if cross_dataset_results %}
            <h2>Cross-Dataset Evaluation</h2>
            <table class="metrics-table">
                <tr>
                    <th>Dataset</th>
                    <th>AUC</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
                {% for dataset, results in cross_dataset_results.items() %}
                <tr>
                    <td>{{ dataset }}</td>
                    <td>{{ "%.4f"|format(results.get('auc', 0)) }}</td>
                    <td>{{ "%.4f"|format(results.get('accuracy', 0)) }}</td>
                    <td>{{ "%.4f"|format(results.get('precision', 0)) }}</td>
                    <td>{{ "%.4f"|format(results.get('recall', 0)) }}</td>
                    <td>{{ "%.4f"|format(results.get('f1_score', 0)) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
            
            <h2>Conclusions</h2>
            <div class="conclusions">
                {{ conclusions }}
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        
        # Prepare data for template
        template_data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "summary": self._generate_summary_text(experiment_data),
            "metrics": experiment_data.get("metrics", {}),
            "visualizations": viz_paths,
            "cross_dataset_results": experiment_data.get("cross_dataset_results", {}),
            "conclusions": self._generate_conclusions(experiment_data)
        }
        
        html_content = template.render(**template_data)
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_json_summary(self, 
                             experiment_data: Dict[str, Any], 
                             output_dir: Path) -> Path:
        """Generate JSON summary."""
        json_path = output_dir / "experiment_summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "experiment_data": experiment_data
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return json_path
    
    def _create_metrics_table(self, metrics: Dict[str, float]) -> Table:
        """Create metrics table for PDF."""
        data = [["Metric", "Value"]]
        for metric, value in metrics.items():
            data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_cross_dataset_table(self, cross_data: Dict[str, Any]) -> Table:
        """Create cross-dataset table for PDF."""
        datasets = list(cross_data.keys())
        metrics = ["auc", "accuracy", "precision", "recall", "f1_score"]
        
        data = [["Dataset"] + [m.upper() for m in metrics]]
        for dataset in datasets:
            row = [dataset]
            for metric in metrics:
                row.append(f"{cross_data[dataset].get(metric, 0):.4f}")
            data.append(row)
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _generate_summary_text(self, experiment_data: Dict[str, Any]) -> str:
        """Generate executive summary text."""
        metrics = experiment_data.get("metrics", {})
        
        summary_parts = []
        
        if "auc" in metrics:
            summary_parts.append(f"The model achieved an AUC score of {metrics['auc']:.4f}.")
        
        if "accuracy" in metrics:
            summary_parts.append(f"Overall accuracy was {metrics['accuracy']:.4f}.")
        
        if "precision" in metrics and "recall" in metrics:
            summary_parts.append(f"Precision and recall were {metrics['precision']:.4f} and {metrics['recall']:.4f} respectively.")
        
        if "cross_dataset_results" in experiment_data:
            cross_results = experiment_data["cross_dataset_results"]
            avg_auc = np.mean([results.get("auc", 0) for results in cross_results.values()])
            summary_parts.append(f"Cross-dataset evaluation showed an average AUC of {avg_auc:.4f} across {len(cross_results)} datasets.")
        
        return " ".join(summary_parts) if summary_parts else "No summary available."
    
    def _generate_conclusions(self, experiment_data: Dict[str, Any]) -> str:
        """Generate conclusions text."""
        conclusions = []
        
        metrics = experiment_data.get("metrics", {})
        
        if metrics.get("auc", 0) > 0.9:
            conclusions.append("The model demonstrates excellent performance with high AUC scores.")
        elif metrics.get("auc", 0) > 0.8:
            conclusions.append("The model shows good performance with satisfactory AUC scores.")
        else:
            conclusions.append("The model performance could be improved with further training or architecture modifications.")
        
        if "cross_dataset_results" in experiment_data:
            cross_results = experiment_data["cross_dataset_results"]
            generalization_gap = self._calculate_generalization_gap(cross_results)
            
            if generalization_gap < 0.1:
                conclusions.append("The model shows good generalization across different datasets.")
            else:
                conclusions.append("The model shows some overfitting and could benefit from regularization techniques.")
        
        return " ".join(conclusions)
    
    def _calculate_generalization_gap(self, cross_results: Dict[str, Any]) -> float:
        """Calculate generalization gap."""
        if "train" in cross_results:
            train_auc = cross_results["train"].get("auc", 0)
            test_aucs = [results.get("auc", 0) for dataset, results in cross_results.items() if dataset != "train"]
            if test_aucs:
                return train_auc - np.mean(test_aucs)
        return 0.0


def main():
    """Test report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate deepfake detection report")
    parser.add_argument("--experiment_data", required=True, help="Path to experiment data JSON")
    parser.add_argument("--model_name", default="TestModel", help="Model name")
    parser.add_argument("--dataset_name", default="TestDataset", help="Dataset name")
    parser.add_argument("--output_dir", default="reports", help="Output directory")
    parser.add_argument("--format", default="both", choices=["pdf", "html", "both"], help="Output format")
    
    args = parser.parse_args()
    
    # Load experiment data
    with open(args.experiment_data, 'r') as f:
        experiment_data = json.load(f)
    
    # Generate report
    generator = ReportGenerator(args.output_dir)
    report_paths = generator.generate_experiment_report(
        experiment_data, args.model_name, args.dataset_name, args.format
    )
    
    print("Report generation completed!")
    for format_type, path in report_paths.items():
        print(f"{format_type.upper()} report: {path}")


if __name__ == "__main__":
    main()
