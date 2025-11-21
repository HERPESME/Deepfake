"""
Configuration utilities for deepfake detection project.
Handles YAML configuration loading and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "xception"
    num_classes: int = 2
    pretrained: bool = True
    dropout_rate: float = 0.5


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "faceforensics"
    data_root: str = "data/processed"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    use_albumentations: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    save_frequency: int = 5


@dataclass
class CrossDatasetConfig:
    """Cross-dataset evaluation configuration."""
    enabled: bool = True
    train_dataset: str = "faceforensics"
    test_datasets: list = field(default_factory=lambda: ["celebd", "dfdc", "wilddeepfake"])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = False
    wandb_project: str = "deepfake-detection"
    log_frequency: int = 10


@dataclass
class OutputConfig:
    """Output configuration."""
    experiment_name: Optional[str] = None
    output_dir: str = "experiments"
    save_predictions: bool = True
    generate_report: bool = True


@dataclass
class AdvancedConfig:
    """Advanced training configuration."""
    mixed_precision: bool = False
    gradient_clipping: float = 1.0
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cross_dataset: CrossDatasetConfig = field(default_factory=CrossDatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)


class ConfigLoader:
    """Configuration loader and validator."""
    
    @staticmethod
    def load_config(config_path: str) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf for better handling
        config_omega = OmegaConf.create(config_dict)
        
        # Convert to dataclass
        config = ConfigLoader._dict_to_config(config_omega)
        
        # Validate configuration
        ConfigLoader.validate_config(config)
        
        return config
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config dataclass."""
        config = Config()
        
        # Model config
        if 'model' in config_dict:
            model_dict = config_dict['model']
            config.model = ModelConfig(
                name=model_dict.get('name', 'xception'),
                num_classes=model_dict.get('num_classes', 2),
                pretrained=model_dict.get('pretrained', True),
                dropout_rate=model_dict.get('dropout_rate', 0.5)
            )
        
        # Data config
        if 'data' in config_dict:
            data_dict = config_dict['data']
            config.data = DataConfig(
                dataset=data_dict.get('dataset', 'faceforensics'),
                data_root=data_dict.get('data_root', 'data/processed'),
                image_size=data_dict.get('image_size', 224),
                batch_size=data_dict.get('batch_size', 32),
                num_workers=data_dict.get('num_workers', 4),
                use_albumentations=data_dict.get('use_albumentations', True)
            )
        
        # Training config
        if 'training' in config_dict:
            training_dict = config_dict['training']
            config.training = TrainingConfig(
                epochs=training_dict.get('epochs', 100),
                learning_rate=training_dict.get('learning_rate', 1e-4),
                weight_decay=training_dict.get('weight_decay', 1e-4),
                optimizer=training_dict.get('optimizer', 'adam'),
                scheduler=training_dict.get('scheduler', 'cosine'),
                early_stopping_patience=training_dict.get('early_stopping_patience', 10),
                save_frequency=training_dict.get('save_frequency', 10)
            )
        
        # Cross-dataset config
        if 'cross_dataset' in config_dict:
            cross_dict = config_dict['cross_dataset']
            config.cross_dataset = CrossDatasetConfig(
                enabled=cross_dict.get('enabled', True),
                train_dataset=cross_dict.get('train_dataset', 'faceforensics'),
                test_datasets=cross_dict.get('test_datasets', ['celebd', 'dfdc', 'wilddeepfake'])
            )
        
        # Logging config
        if 'logging' in config_dict:
            logging_dict = config_dict['logging']
            config.logging = LoggingConfig(
                use_wandb=logging_dict.get('use_wandb', False),
                wandb_project=logging_dict.get('wandb_project', 'deepfake-detection'),
                log_frequency=logging_dict.get('log_frequency', 10)
            )
        
        # Output config
        if 'output' in config_dict:
            output_dict = config_dict['output']
            config.output = OutputConfig(
                experiment_name=output_dict.get('experiment_name'),
                output_dir=output_dict.get('output_dir', 'experiments'),
                save_predictions=output_dict.get('save_predictions', True),
                generate_report=output_dict.get('generate_report', True)
            )
        
        # Advanced config
        if 'advanced' in config_dict:
            advanced_dict = config_dict['advanced']
            config.advanced = AdvancedConfig(
                mixed_precision=advanced_dict.get('mixed_precision', False),
                gradient_clipping=advanced_dict.get('gradient_clipping', 1.0),
                warmup_epochs=advanced_dict.get('warmup_epochs', 5),
                label_smoothing=advanced_dict.get('label_smoothing', 0.0),
                focal_loss=advanced_dict.get('focal_loss', False),
                focal_alpha=advanced_dict.get('focal_alpha', 0.25),
                focal_gamma=advanced_dict.get('focal_gamma', 2.0)
            )
        
        return config
    
    @staticmethod
    def validate_config(config: Config):
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
        """
        # Model validation
        valid_models = ["xception", "efficientnet_b0", "efficientnet_b4", "vit", 
                       "resnet50", "clip", "hybrid_cnn_transformer", "contrastive", "multiscale"]
        if config.model.name not in valid_models:
            raise ValueError(f"Invalid model name: {config.model.name}. Valid options: {valid_models}")
        
        if config.model.num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        
        if not 0 <= config.model.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        # Data validation
        valid_datasets = ["faceforensics", "celebd", "dfdc", "wilddeepfake"]
        if config.data.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {config.data.dataset}. Valid options: {valid_datasets}")
        
        if config.data.image_size <= 0:
            raise ValueError("Image size must be positive")
        
        if config.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.data.num_workers < 0:
            raise ValueError("Number of workers must be non-negative")
        
        # Training validation
        if config.training.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if config.training.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        
        valid_optimizers = ["adam", "adamw", "sgd"]
        if config.training.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {config.training.optimizer}. Valid options: {valid_optimizers}")
        
        valid_schedulers = ["cosine", "step", "plateau", "warmup_cosine"]
        if config.training.scheduler not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {config.training.scheduler}. Valid options: {valid_schedulers}")
        
        if config.training.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be positive")
        
        # Cross-dataset validation
        if config.cross_dataset.enabled:
            if config.cross_dataset.train_dataset not in valid_datasets:
                raise ValueError(f"Invalid train dataset: {config.cross_dataset.train_dataset}")
            
            for test_dataset in config.cross_dataset.test_datasets:
                if test_dataset not in valid_datasets:
                    raise ValueError(f"Invalid test dataset: {test_dataset}")
        
        # Advanced validation
        if config.advanced.gradient_clipping < 0:
            raise ValueError("Gradient clipping must be non-negative")
        
        if config.advanced.warmup_epochs < 0:
            raise ValueError("Warmup epochs must be non-negative")
        
        if not 0 <= config.advanced.label_smoothing <= 1:
            raise ValueError("Label smoothing must be between 0 and 1")
        
        if not 0 <= config.advanced.focal_alpha <= 1:
            raise ValueError("Focal alpha must be between 0 and 1")
        
        if config.advanced.focal_gamma < 0:
            raise ValueError("Focal gamma must be non-negative")
    
    @staticmethod
    def save_config(config: Config, output_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        config_dict = ConfigLoader._config_to_dict(config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        """Convert Config dataclass to dictionary."""
        return {
            'model': {
                'name': config.model.name,
                'num_classes': config.model.num_classes,
                'pretrained': config.model.pretrained,
                'dropout_rate': config.model.dropout_rate
            },
            'data': {
                'dataset': config.data.dataset,
                'data_root': config.data.data_root,
                'image_size': config.data.image_size,
                'batch_size': config.data.batch_size,
                'num_workers': config.data.num_workers,
                'use_albumentations': config.data.use_albumentations
            },
            'training': {
                'epochs': config.training.epochs,
                'learning_rate': config.training.learning_rate,
                'weight_decay': config.training.weight_decay,
                'optimizer': config.training.optimizer,
                'scheduler': config.training.scheduler,
                'early_stopping_patience': config.training.early_stopping_patience,
                'save_frequency': config.training.save_frequency
            },
            'cross_dataset': {
                'enabled': config.cross_dataset.enabled,
                'train_dataset': config.cross_dataset.train_dataset,
                'test_datasets': config.cross_dataset.test_datasets
            },
            'logging': {
                'use_wandb': config.logging.use_wandb,
                'wandb_project': config.logging.wandb_project,
                'log_frequency': config.logging.log_frequency
            },
            'output': {
                'experiment_name': config.output.experiment_name,
                'output_dir': config.output.output_dir,
                'save_predictions': config.output.save_predictions,
                'generate_report': config.output.generate_report
            },
            'advanced': {
                'mixed_precision': config.advanced.mixed_precision,
                'gradient_clipping': config.advanced.gradient_clipping,
                'warmup_epochs': config.advanced.warmup_epochs,
                'label_smoothing': config.advanced.label_smoothing,
                'focal_loss': config.advanced.focal_loss,
                'focal_alpha': config.advanced.focal_alpha,
                'focal_gamma': config.advanced.focal_gamma
            }
        }
    
    @staticmethod
    def create_default_config() -> Config:
        """
        Create a default configuration object.
        
        Returns:
            Default configuration object
        """
        return Config()


def main():
    """Test configuration loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test configuration loading")
    parser.add_argument("--config", default="configs/training_config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output configuration file")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(args.config)
        print("Configuration loaded successfully!")
        print(f"Model: {config.model.name}")
        print(f"Dataset: {config.data.dataset}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Learning rate: {config.training.learning_rate}")
        
        # Save configuration if output specified
        if args.output:
            ConfigLoader.save_config(config, args.output)
            print(f"Configuration saved to: {args.output}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")


if __name__ == "__main__":
    main()
