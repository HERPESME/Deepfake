"""
Comprehensive evaluation metrics for deepfake detection.
Includes accuracy, precision, recall, F1-score, AUC, and confusion matrix analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path


class MetricsCalculator:
    """Comprehensive metrics calculator for deepfake detection."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = []
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics["precision_real"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics["precision_fake"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["recall_real"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics["recall_fake"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["f1_real"] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics["f1_fake"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # ROC AUC (if probabilities provided)
        if y_prob is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob)
                metrics["average_precision"] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics["auc"] = 0.0
                metrics["average_precision"] = 0.0
        else:
            metrics["auc"] = 0.0
            metrics["average_precision"] = 0.0
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def calculate_cross_dataset_metrics(self, 
                                      results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate cross-dataset evaluation metrics.
        
        Args:
            results: Dictionary of results from different datasets
            
        Returns:
            Aggregated cross-dataset metrics
        """
        cross_dataset_metrics = {}
        
        # Extract metrics for each dataset
        datasets = list(results.keys())
        metrics_names = list(results[datasets[0]].keys())
        
        # Calculate mean and std for each metric across datasets
        for metric_name in metrics_names:
            values = [results[dataset][metric_name] for dataset in datasets]
            cross_dataset_metrics[f"{metric_name}_mean"] = np.mean(values)
            cross_dataset_metrics[f"{metric_name}_std"] = np.std(values)
            cross_dataset_metrics[f"{metric_name}_min"] = np.min(values)
            cross_dataset_metrics[f"{metric_name}_max"] = np.max(values)
        
        # Calculate generalization gap (train vs test performance)
        if "train" in results and len(datasets) > 1:
            train_auc = results["train"].get("auc", 0)
            test_aucs = [results[dataset].get("auc", 0) for dataset in datasets if dataset != "train"]
            cross_dataset_metrics["generalization_gap"] = train_auc - np.mean(test_aucs)
        
        return cross_dataset_metrics


class VisualizationGenerator:
    """Generate visualizations for evaluation results."""
    
    def __init__(self, output_dir: str = "reports/visualizations"):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'],
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, 
                       y_true: np.ndarray, 
                       y_prob: np.ndarray,
                       title: str = "ROC Curve",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_prob: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             title: str = "Training History",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history.get('train_loss', []), label='Train Loss')
        axes[0, 0].plot(history.get('val_loss', []), label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history.get('train_acc', []), label='Train Acc')
        axes[0, 1].plot(history.get('val_acc', []), label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC plot
        axes[1, 0].plot(history.get('val_auc', []), label='Val AUC', color='green')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cross_dataset_comparison(self, 
                                    results: Dict[str, Dict[str, float]],
                                    metric: str = "auc",
                                    title: str = "Cross-Dataset Performance",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cross-dataset performance comparison.
        
        Args:
            results: Cross-dataset results
            metric: Metric to compare
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        datasets = list(results.keys())
        values = [results[dataset].get(metric, 0) for dataset in datasets]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(datasets, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metrics_table(self, 
                           results: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create metrics comparison table.
        
        Args:
            results: Results dictionary
            save_path: Path to save table
            
        Returns:
            Pandas DataFrame
        """
        df = pd.DataFrame(results).T
        
        # Round numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        if save_path:
            df.to_csv(save_path)
        
        return df


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize model evaluator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = VisualizationGenerator(str(self.output_dir / "visualizations"))
    
    def evaluate_model(self, 
                     y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None,
                     model_name: str = "Model",
                     dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Evaluation results
        """
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        # Generate visualizations
        viz_dir = self.output_dir / "visualizations" / f"{model_name}_{dataset_name}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            y_true, y_pred, 
            title=f"Confusion Matrix - {model_name} on {dataset_name}",
            save_path=str(viz_dir / "confusion_matrix.png")
        )
        
        # ROC curve (if probabilities available)
        if y_prob is not None:
            self.visualizer.plot_roc_curve(
                y_true, y_prob,
                title=f"ROC Curve - {model_name} on {dataset_name}",
                save_path=str(viz_dir / "roc_curve.png")
            )
            
            self.visualizer.plot_precision_recall_curve(
                y_true, y_prob,
                title=f"Precision-Recall Curve - {model_name} on {dataset_name}",
                save_path=str(viz_dir / "precision_recall_curve.png")
            )
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=['Real', 'Fake'],
                                     output_dict=True)
        
        results = {
            "metrics": metrics,
            "classification_report": report,
            "visualization_dir": str(viz_dir)
        }
        
        return results
    
    def evaluate_cross_dataset(self, 
                             cross_dataset_results: Dict[str, Dict[str, float]],
                             experiment_name: str = "CrossDataset") -> Dict[str, Any]:
        """
        Evaluate cross-dataset performance.
        
        Args:
            cross_dataset_results: Cross-dataset results
            experiment_name: Name of the experiment
            
        Returns:
            Cross-dataset evaluation results
        """
        # Calculate cross-dataset metrics
        cross_metrics = self.metrics_calculator.calculate_cross_dataset_metrics(cross_dataset_results)
        
        # Generate comparison plots
        viz_dir = self.output_dir / "visualizations" / experiment_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # AUC comparison
        self.visualizer.plot_cross_dataset_comparison(
            cross_dataset_results,
            metric="auc",
            title=f"AUC Comparison - {experiment_name}",
            save_path=str(viz_dir / "auc_comparison.png")
        )
        
        # Accuracy comparison
        self.visualizer.plot_cross_dataset_comparison(
            cross_dataset_results,
            metric="accuracy",
            title=f"Accuracy Comparison - {experiment_name}",
            save_path=str(viz_dir / "accuracy_comparison.png")
        )
        
        # Create metrics table
        metrics_table = self.visualizer.create_metrics_table(
            cross_dataset_results,
            save_path=str(viz_dir / "metrics_table.csv")
        )
        
        results = {
            "cross_dataset_metrics": cross_metrics,
            "individual_results": cross_dataset_results,
            "metrics_table": metrics_table,
            "visualization_dir": str(viz_dir)
        }
        
        return results


def main():
    """Test evaluation metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test evaluation metrics")
    parser.add_argument("--output_dir", default="reports", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_prob = np.random.rand(1000)
    
    # Test metrics calculator
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
    print("Sample metrics:", metrics)
    
    # Test evaluator
    evaluator = ModelEvaluator(args.output_dir)
    results = evaluator.evaluate_model(y_true, y_pred, y_prob, "TestModel", "TestDataset")
    print("Evaluation completed!")
    print(f"Results saved to: {results['visualization_dir']}")


if __name__ == "__main__":
    main()
