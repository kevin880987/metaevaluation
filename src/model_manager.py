"""
Model Manager for Health Indicator Generation
Centralizes training, loading, and configuration of all autoencoder models
"""

import os
import numpy as np
import torch
import joblib
import pandas as pd
import warnings
from dataclasses import dataclass
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

from model.cvae import CVAE
from model.cae_m import CAE_M  
from model.vq_vae import VQVAE


def get_model_configs(config):
    """
    Centralized model configuration factory
    Returns complete model configuration for any model type
    """
    
    model_configs = {}

    # === SKLEARN MODELS ===
    
    # PCA
    model_configs["pca"] = {
        "model_type": "sklearn",
        "model_class": PCA,
        "kwargs": {
            "n_components": None,
        },
        "description": "Principal Component Analysis for dimensionality reduction"
    }

    # ICA
    model_configs["ica"] = {
        "model_type": "sklearn", 
        "model_class": FastICA,
        "kwargs": {
            "n_components": None,
            "max_iter": 10000,
        },
        "description": "Independent Component Analysis for source separation"
    }
    
    # === TORCH MODELS ===

    base_config = {
        'device': config.device,
        'seed': config.seed,
        'batch_size': 64,
        'shuffle': True,
        'stopping_count': 50,
    }

    # CVAE
    def create_architecture(input_channels, output_channels, input_length, output_length):
        """CVAE: encoder -> latent -> decoder"""
        base_channels = max(16, min(64, max(input_channels, output_channels) // 4))
        encoder_channels = [base_channels * (2**i) for i in range(3)]
        latent_dim = max(8, min(256, max(input_channels, output_channels) // 2))
        decoder_channels = encoder_channels[::-1][1:]  # Mirror without final layer
        return (encoder_channels, latent_dim, decoder_channels)
    model_configs["cvae"] = {
        "model_type": "torch",
        "model_class": CVAE,
        "learning_rate": 0.0001,
        "n_epochs": 5000,
        "beta_start": 0.001,
        "beta_max": 0.01,
        "beta_annealing_rate": 0.01,
        "architecture_fn": staticmethod(create_architecture),
        "description": "Conditional Variational Autoencoder for health indicator generation"
    }
    if base_config:
        model_configs["cvae"].update(base_config)

    # CAE-M
    def create_architecture(input_channels, output_channels, input_length, output_length):
        """CAE-M: more complex architecture with additional components"""
        base_channels = max(8, min(16, max(input_channels, output_channels) // 8))
        encoder_channels = [base_channels * (2**i) for i in range(4)]
        decoder_channels = encoder_channels[::-1][1:]  # Mirror without final layer
        latent_dim = max(16, min(512, max(input_channels, output_channels) // 2))
        return (encoder_channels, latent_dim, decoder_channels)
    model_configs["cae_m"] = {
        "model_type": "torch",
        "model_class": CAE_M,
        "learning_rate": 0.0001,
        "n_epochs": 2000,
        "lambda_mmd": 1e-4,
        "lambda_lp": 0.5,
        "lambda_np": 0.5,
        "sequence_length": 10,
        "architecture_fn": staticmethod(create_architecture),
        "description": "Convolutional Autoencoder with MMD regularization"
    }
    if base_config:
        model_configs["cae_m"].update(base_config)

    # VQ-VAE
    def create_architecture(input_channels, output_channels, input_length, output_length):
        """VQ-VAE: use configured values with data-driven adjustments"""
        hidden_dim = max(64, min(256, max(input_channels, output_channels) * 2))
        embedding_dim = 16  # From centralized config
        num_embeddings = 32  # From centralized config
        
        # Adaptive block count based on sequence length to prevent MaxPool1d errors
        min_length = min(input_length, output_length)
        if min_length >= 32:
            n_blocks = 4  # Original setting for longer sequences
        elif min_length >= 16:
            n_blocks = 3  # Reduce by 1 for medium sequences  
        elif min_length >= 8:
            n_blocks = 2  # Reduce by 2 for short sequences
        else:
            n_blocks = 1  # Minimum for very short sequences
            
        return (hidden_dim, embedding_dim, num_embeddings, n_blocks)
    model_configs["vq_vae"] = {
        "model_type": "torch",
        "model_class": VQVAE,
        "learning_rate": 0.0002,
        "n_epochs": 2000,
        "commitment_cost": 2.0,
        "use_ema": True,
        "ema_decay": 0.99,
        "vq_loss_weight": 1.0,
        "architecture_fn": staticmethod(create_architecture),
        "description": "Vector Quantized Variational Autoencoder for discrete representation learning"
    }
    if base_config:
        model_configs["vq_vae"].update(base_config)

    for m, c in model_configs.items():
        c = type('Config', (), {**c,})()
        model_configs[m] = c
    return model_configs


class SklearnModelManager:
    """
    Manages training, loading, and configuration of scikit-learn models
    Specifically designed for PCA, ICA, and other dimensionality reduction models
    """
    
    def __init__(self, config):
        self.config = config

        self.models = {}
        self.feature_names = {}
        # Filter only sklearn models
        all_configs = get_model_configs(config)
        self.model_configs = {k: v for k, v in all_configs.items() if v.model_type == "sklearn"}

        print(f"SklearnModelManager initialized")
        print(f"Available sklearn models: {list(self.model_configs.keys())}")
    
    def _create_model_instance(self, model_name: str):
        """Create scikit-learn model instance"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model config for {model_name} not found.")
        model_config = self.model_configs[model_name]
        return model_config.model_class(**model_config.kwargs)

    def _get_model_path(self, model_name: str) -> str:
        """Get file path for model persistence"""
        return os.path.join(self.config.training_dir, f"{model_name}_sklearn.joblib")
    
    def fit_model(self, model_name: str, X: np.array, feature_names: list = None, 
                  force_refit: bool = False):
        """
        Fit a scikit-learn model or load if already fitted
        
        Args:
            model_name: Name of the model ('pca' or 'ica')
            X: Training data
            feature_names: List of feature names for the data
            force_refit: Force refitting even if model exists
        """
        # Check if model already exists
        if not force_refit:
            try:
                print(f"Loading existing {model_name.upper()} model...")
                model = self.load_model(model_name)
                return model
            except Exception as e:
                print(f"Failed to load existing {model_name.upper()} model: {str(e)}")

        print(f"Fitting {model_name.upper()} model...")
        
        # Create model instance
        model = self._create_model_instance(model_name)
        
        # Prepare data with feature names
        if isinstance(X, pd.DataFrame):
            X_data = X
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_data = pd.DataFrame(X, columns=feature_names)
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_data)
        
        # Set feature names to avoid warnings during transform
        if hasattr(model, 'feature_names_in_'):
            model.feature_names_in_ = np.array(feature_names, dtype=object)
        
        # Store feature names and model
        self.feature_names[model_name] = feature_names
        self.models[model_name] = model
        
        # Save model
        self._save_model(model_name, model)
        
        print(f"✓ {model_name.upper()} fitted successfully")
        print(f"  Components: {getattr(model, 'n_components_', 'N/A')}")
        print(f"  Features: {len(feature_names)}")
        
        return model
    
    def _save_model(self, model_name: str, model):
        """Save model and feature names using joblib"""
        os.makedirs(self.config.training_dir, exist_ok=True)
        
        model_path = self._get_model_path(model_name)
        
        # Prepare model data for joblib serialization
        model_data = {
            'model': model,
            'feature_names': self.feature_names.get(model_name, []),
            'model_type': model_name,
            'model_class': model.__class__.__name__
        }
        
        # Save using joblib (more efficient and preserves exact model state)
        joblib.dump(model_data, model_path, compress=3)
        
        print(f"  {model_name.upper()} model saved to {model_path}")
    
    def load_model(self, model_name: str):
        """Load a pre-fitted model using joblib"""
        model_path = self._get_model_path(model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_name.upper()} model not found at {model_path}")
        
        # Load model data using joblib
        model_data = joblib.load(model_path)
        
        # Extract model and feature names
        model = model_data['model']
        feature_names = model_data.get('feature_names', [])
        
        # Store in memory
        self.models[model_name] = model
        self.feature_names[model_name] = feature_names
        
        return model
    
    def fit_all_models(self, X: np.array, feature_names: list = None, 
                      force_refit: bool = False):
        """
        Fit all available models
        
        Args:
            X: Training data
            feature_names: Feature names for the data
            force_refit: Force refitting of all models
            
        Returns:
            Dictionary of fitted models
        """
        print("=" * 60)
        print("FITTING/LOADING ALL SKLEARN MODELS")
        print("=" * 60)
        
        fitted_models = {}
        available_models = self.list_available_models()
        
        for model_name in available_models:
            try:
                fitted_models[model_name] = self.fit_model(
                    model_name, X, feature_names, force_refit
                )
                print(f"✓ {model_name.upper()} ready")
                print("-" * 30)
            except Exception as e:
                print(f"✗ Failed to fit {model_name.upper()}: {str(e)}")
                print("-" * 30)
                continue
                
        print(f"Completed: {len(fitted_models)}/{len(available_models)} models ready")
        print("=" * 60)
        
        self.models.update(fitted_models)
        return fitted_models
    
    def get_model(self, model_name: str):
        """Get a fitted model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not fitted. Call fit_model() first.")
        return self.models[model_name]
    
    def list_available_models(self) -> list:
        """List all available model types"""
        return list(self.model_configs.keys())
    
    def list_fitted_models(self) -> list:
        """List currently fitted models"""
        return list(self.models.keys())
    
    def get_feature_names(self, model_name: str) -> list:
        """Get stored feature names for a model"""
        return self.feature_names.get(model_name, [])
    
    def clear_models(self):
        """Clear all fitted models from memory"""
        self.models.clear()
        self.feature_names.clear()
        print("All sklearn models cleared from memory")


class TorchModelManager:
    """
    Manages training, loading, and configuration of autoencoder models
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.seed = config.seed

        self.models = {}
        # Filter only torch models
        all_configs = get_model_configs(config)
        self.model_configs = {k: v for k, v in all_configs.items() if v.model_type == "torch"}
        
        print(f"TorchModelManager initialized with device: {self.device}")
        print(f"Available torch models: {list(self.model_configs.keys())}")
        
    def _create_model_instance(self, model_name: str, X_in: np.array, X_out: np.array):
        """Create model instance"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model config for {model_name} not found.")
        
        model_config = self.model_configs[model_name]
        
        # Extract dimensions from 3D data - ALWAYS derive from data shape
        # X_in shape: [n_samples, input_channels, input_sequence_length]  
        # X_out shape: [n_samples, output_channels, output_sequence_length]
        input_channels = X_in.shape[1]
        output_channels = X_out.shape[1]
        input_length = X_in.shape[2]
        output_length = X_out.shape[2]
        
        # Create architecture using centralized architecture function
        architecture = model_config.architecture_fn(input_channels, output_channels, input_length, output_length)
        model_config.architecture = architecture
        
        # Add dimension information to config for models to use
        model_config.input_channels = input_channels
        model_config.output_channels = output_channels
        model_config.input_length = input_length
        model_config.output_length = output_length

        # Create training directory
        training_dir = self.config.training_dir + model_name + os.sep
        os.makedirs(training_dir, exist_ok=True)
        model_config.training_dir = training_dir

        return model_config.model_class(model_config)

    def train_model(self, model_name: str, X_in: np.array, X_out: np.array, 
                   force_retrain: bool = False):
        """
        Train a specific model or load if already trained with proper device management
        
        Args:
            model_name: Name of the model to train
            X_in: Input training data
            X_out: Output training data  
            force_retrain: Force retraining even if model exists
            
        Returns:
            Trained model instance
        """
        # Create model instance
        model = self._create_model_instance(model_name, X_in, X_out)
        
        # Check if model already exists and we don't want to force retrain
        if not force_retrain:
            try:
                # Attempt to load existing model
                print(f"Loading existing {model_name.upper()} model...")
                # Model should automatically be on correct device from loading
                model.load_model()
                return model
            except Exception as e:
                print(f"Failed to load existing {model_name.upper()} model: {str(e)}")

        # Train the model if not loaded
        print(f"Training {model_name.upper()} model...")
        model.train(X_in, X_out)
        # Save model to cpu for portability after training
        model.save_model("cpu")

        return model
    
    def train_all_models(self, X_in: np.array, X_out: np.array, 
                        force_retrain: bool = False):
        """
        Train or load all available models
        
        Args:
            X_in: Input training data
            X_out: Output training data
            force_retrain: Force retraining of all models
            
        Returns:
            Dictionary of trained models
        """
        print("=" * 20)
        print("TRAINING/LOADING ALL MODELS")
        print("=" * 20)
        
        trained_models = {}
        available_models = self.list_available_models()
        
        for model_name in available_models:

            # if "vq_vae" in model_name or "cae_m" in model_name:#"cvae" in model_name: # 
            #     force_retrain = False
            # else:
            #     force_retrain = True

            # try:
            trained_models[model_name] = self.train_model(
                model_name, X_in, X_out, force_retrain
            )
            print(f"✓ {model_name.upper()} ready")
            print("-" * 40)
            # except Exception as e:
            #     print(f"✗ Failed to train/load {model_name.upper()}: {str(e)}")
            #     print("-" * 40)
            #     continue
                
        print(f"Completed: {len(trained_models)}/{len(available_models)} models ready")
        print("=" * 20)
        
        self.models.update(trained_models)
        return trained_models
        
    def get_model(self, model_name: str):
        """Get a loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Call train_model() or load_model() first.")
        return self.models[model_name]
    
    def list_available_models(self) -> list:
        """List all available model types"""
        return list(self.model_configs.keys())
    
    def list_loaded_models(self) -> list:
        """List currently loaded models"""
        return list(self.models.keys())
    
    def clear_models(self):
        """Clear all loaded models from memory"""
        self.models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None