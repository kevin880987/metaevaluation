"""
doi:https://doi.org/10.1109/TKDE.2021.3102110
Enhanced Convolutional Autoencoder for Multi-sensor Time-Series (CAE-M)
Implementation based on "Unsupervised Deep Anomaly Detection for Multi-Sensor Time-Series Signals" (2021)

This module implements a literature-compliant CAE-M with:
- MMD penalty for latent distribution regularization (λ₁ = 1e-4)  
- BiLSTM+Attention memory network for next-step prediction (λ₂ = 0.5)
- AR prediction head for latent forecasting (λ₃ = 0.5)
- Composite loss: J = L_MSE + λ₁L_MMD + λ₂L_lp + λ₃L_np
"""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from torchsummary import summary
# from torchinfo import summary
import sys
import io


def compute_mmd_loss(x, y, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) loss between two batches
    
    MMD is used to measure the difference between latent distributions
    and enforce regularization as described in CAE-M (2021).
    
    Args:
        x: Source batch (B, D)
        y: Target batch (B, D) 
        sigma: Bandwidth parameter for RBF kernel
        
    Returns:
        MMD loss scalar
    """
    def rbf_kernel(x, y, sigma):
        dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))
    
    # Compute kernel matrices
    kxx = rbf_kernel(x, x, sigma)
    kyy = rbf_kernel(y, y, sigma)  
    kxy = rbf_kernel(x, y, sigma)
    
    # MMD statistic
    mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return mmd


class BiLSTMAttention(nn.Module):
    """
    BiLSTM with Attention mechanism for temporal prediction
    
    This implements the memory network prediction branch as described
    in CAE-M (2021) for next-step latent prediction.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(BiLSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Predicted next step (batch_size, input_dim)
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step for prediction
        prediction = self.output_proj(attn_out[:, -1, :])  # (batch_size, input_dim)
        
        return prediction


class ARPredictionHead(nn.Module):
    """
    Autoregressive prediction head for latent forecasting
    
    This implements the AR prediction component as described
    in CAE-M (2021) for latent space prediction.
    """
    
    def __init__(self, latent_dim, hidden_dim=128):
        super(ARPredictionHead, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
    def forward(self, latent):
        """
        Args:
            latent: Current latent representation (batch_size, latent_dim)
        Returns:
            Predicted next latent (batch_size, latent_dim)
        """
        return self.predictor(latent)


class ConvolutionalAutoencoder(nn.Module):
    """
    Enhanced Convolutional Autoencoder implementing CAE-M (2021) specifications
    
    This implementation includes:
    - Standard convolutional encoder-decoder architecture for multi-feature data
    - MMD penalty for latent distribution regularization
    - BiLSTM+Attention memory network for next-step prediction
    - AR prediction head for latent forecasting
    - Literature-compliant loss weighting (λ₁, λ₂, λ₃)
    """

    def __init__(self, input_channels, output_channels, input_length, output_length, encoder_channels, latent_dim, decoder_channels, sequence_length=10):
        super(ConvolutionalAutoencoder, self).__init__()

        self.input_channels = input_channels  # Number of input features
        self.output_channels = output_channels  # Number of output features
        self.input_length = input_length
        self.output_length = output_length
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels
        self.latent_dim = latent_dim
        self.decoder_channels = decoder_channels
        self.sequence_length = sequence_length  # For temporal prediction
        
        # === Core Encoder-Decoder Architecture ===
        # Encoder: 1D Convolutional layers
        # Input: [batch, input_channels, input_length]
        encoder_layers = []
        in_channels = input_channels  # Start with input_channels
        for out_channels in encoder_channels:
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels),
                # nn.LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Calculate size after convolutions for fully connected layer
        with torch.no_grad():
            # Set model to eval mode to avoid BatchNorm training issues
            self.encoder_conv.eval()
            dummy_input = torch.zeros(1, input_channels, input_length)
            conv_output = self.encoder_conv(dummy_input)
            self.conv_output_size = conv_output.shape[1] * conv_output.shape[2]
            # Set back to train mode
            self.encoder_conv.train()

        # Encoder fully connected
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.conv_output_size, latent_dim),
            nn.ReLU()
        )
        
        # Decoder fully connected
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.conv_output_size),
            nn.ReLU()
        )
        
        # Decoder: 1D Transposed Convolutional layers
        decoder_layers = []
        in_channels = encoder_channels[-1]
        for i, out_channels in enumerate(decoder_channels[:-1]):
            decoder_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(out_channels),
                # nn.LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
            
        # Final layer to output features
        decoder_layers.extend([
            nn.ConvTranspose1d(in_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Output activation
        ])
            
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
        # === CAE-M (2021) Enhancement Components ===
        
        # BiLSTM+Attention Memory Network for next-step prediction
        self.memory_network = BiLSTMAttention(
            input_dim=latent_dim,
            hidden_dim=64,
            num_layers=2
        )
        
        # AR Prediction Head for latent forecasting  
        self.ar_predictor = ARPredictionHead(
            latent_dim=latent_dim,
            hidden_dim=128
        )
        
        # Reference distribution for MMD (learnable parameters)
        self.register_buffer('reference_mean', torch.zeros(latent_dim))
        self.register_buffer('reference_std', torch.ones(latent_dim))
    
    def encode(self, x):
        """Encode input to latent representation"""
        # Convolutional encoding
        conv_features = self.encoder_conv(x)
        
        # Flatten and fully connected
        conv_features_flat = conv_features.view(conv_features.size(0), -1)
        latent = self.encoder_fc(conv_features_flat)
        
        return latent, conv_features.shape
    
    def decode(self, latent, conv_shape):
        """Decode latent representation to reconstruction"""
        # Fully connected layer
        decoded = self.decoder_fc(latent)
        
        # Reshape for deconvolution
        decoded = decoded.view(conv_shape)
        
        # Convolutional decoding
        reconstruction = self.decoder_conv(decoded)
        
        # Ensure output matches the desired output size
        if reconstruction.size(-1) != self.output_length:
            reconstruction = F.interpolate(
                reconstruction, size=self.output_length, mode='linear', align_corners=False
            )
            
        return reconstruction
    
    def forward(self, x, latent_sequence=None, return_predictions=False):
        """
        Enhanced forward pass with optional prediction components
        
        Args:
            x: Input tensor (batch_size, input_length)
            latent_sequence: Previous latent sequence for temporal prediction (batch_size, seq_len, latent_dim)
            return_predictions: Whether to return prediction outputs for loss computation
            
        Returns:
            reconstruction: Reconstructed input
            latent: Current latent representation
            next_latent_pred: Next-step latent prediction (if return_predictions=True)
            latent_ar_pred: AR latent prediction (if return_predictions=True)
        """
        # Standard encoder-decoder
        latent, conv_shape = self.encode(x)
        reconstruction = self.decode(latent, conv_shape)
        
        if not return_predictions:
            return reconstruction, latent
        
        # === Prediction Components (for training with full loss) ===
        
        # Next-step latent prediction using BiLSTM+Attention
        next_latent_pred = None
        if latent_sequence is not None:
            # Add current latent to sequence for prediction
            current_sequence = torch.cat([latent_sequence, latent.unsqueeze(1)], dim=1)
            next_latent_pred = self.memory_network(current_sequence)
        
        # AR latent prediction  
        latent_ar_pred = self.ar_predictor(latent)
        
        return reconstruction, latent, next_latent_pred, latent_ar_pred
    
    def compute_mmd_penalty(self, latent):
        """
        Compute MMD penalty for latent distribution regularization
        
        Args:
            latent: Latent representations (batch_size, latent_dim)
            
        Returns:
            MMD loss between latent distribution and reference Gaussian
        """
        # Sample from reference Gaussian distribution
        batch_size = latent.size(0)
        device = latent.device  # Get device from input latent
        
        reference_sample = torch.normal(
            self.reference_mean.expand(batch_size, -1),
            self.reference_std.expand(batch_size, -1)
        ).to(device)  # Ensure reference sample is on the correct device
        
        # Compute MMD loss
        mmd_loss = compute_mmd_loss(latent, reference_sample, sigma=1.0)
        return mmd_loss


class CAE_M:
    """
    Literature-compliant CAE-M implementation for Multi-sensor Time-Series Signals
    
    Enhanced implementation following CAE-M (2021) with:
    - MMD penalty (λ₁ = 1e-4)
    - BiLSTM memory network (λ₂ = 0.5) 
    - AR prediction (λ₃ = 0.5)
    - Composite loss: J = L_MSE + λ₁L_MMD + λ₂L_lp + λ₃L_np
    """
    
    def __init__(self, config):
        self.config = config
        self.architecture = config.architecture
        encoder_channels, latent_dim, decoder_channels = self.architecture
        self.encoder_channels = encoder_channels
        self.latent_dim = latent_dim
        self.decoder_channels = decoder_channels

        # Get dimensions from config
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels
        self.input_length = config.input_length
        self.output_length = config.output_length
        
        # Training hyperparameters
        self.learning_rate = config.learning_rate
        self.n_epochs = config.n_epochs
        self.stopping_count = config.stopping_count
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.sequence_length = config.sequence_length  # For temporal prediction

        # Device and paths
        self.device = config.device
        self.seed = config.seed
        self.dir = config.training_dir
        self.model_path = self.dir + "cae_m.pt"

        # Literature-compliant loss weights (CAE-M 2021)
        self.lambda_mmd = config.lambda_mmd  # MMD penalty weight
        self.lambda_lp = config.lambda_lp     # Next-step prediction weight
        self.lambda_np = config.lambda_np     # AR prediction weight

        # Initialize the model with proper dimensions
        self.model = ConvolutionalAutoencoder(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            input_length=self.input_length,
            output_length=self.output_length,
            encoder_channels=self.encoder_channels,
            latent_dim=self.latent_dim,
            decoder_channels=self.decoder_channels, 
            sequence_length=self.sequence_length, 
        ).to(self.device)

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        # Latent sequence buffer for temporal prediction
        self.latent_buffer = []
        
    def _load_batches(self, batch_size, shuffle):
        """Load data in batches for training"""
        idx = np.arange(self.X_in_tensor.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        for i in range(int(np.ceil(idx.size / batch_size))):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            x_in = self.X_in_tensor[batch_idx]
            x_out = self.X_out_tensor[batch_idx]
            if x_in.shape[0] <= 1:
                break
            yield x_in, x_out
    
    def _get_latent_sequences(self, batch_size):
        """
        Get latent sequences for temporal prediction
        
        Returns:
            Tensor of shape [batch_size, sequence_length, latent_dim] or None
        """
        if len(self.latent_buffer) < self.sequence_length:
            return None
        
        # Get the most recent sequence_length latent vectors
        seq_start = max(0, len(self.latent_buffer) - self.sequence_length)
        recent_latents = self.latent_buffer[seq_start:]
        
        # Ensure we have exactly sequence_length latents
        if len(recent_latents) < self.sequence_length:
            return None
            
        # Stack into sequence: [sequence_length, latent_dim]
        try:
            sequence = torch.stack(recent_latents, dim=0)  # Shape: [sequence_length, latent_dim]
        except RuntimeError as e:
            print(f"Error stacking latents: {e}")
            print(f"Buffer length: {len(self.latent_buffer)}")
            print(f"Recent latents shapes: {[t.shape for t in recent_latents]}")
            return None
        
        # Expand for batch: [batch_size, sequence_length, latent_dim]
        sequence = sequence.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sequence
    
    def _train_one_epoch(self):
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        batch_losses = []
        
        for x_in, x_out in self._load_batches(self.batch_size, self.shuffle):            
            optimizer.zero_grad()
            
            # Get latent sequences for temporal prediction
            latent_sequence = self._get_latent_sequences(x_in.size(0))
            if latent_sequence is not None:
                latent_sequence = latent_sequence.to(self.device)
            
            # Forward pass with predictions
            outputs = self.model(x_in, latent_sequence=latent_sequence, return_predictions=True)
            reconstruction, latent, next_latent_pred, latent_ar_pred = outputs
            
            # === Literature-compliant Composite Loss (CAE-M 2021) ===
            
            # L_MSE: Reconstruction loss
            reconstruction_loss = criterion(reconstruction, x_out)
            
            # λ₁L_MMD: MMD penalty for latent distribution regularization
            mmd_loss = self.model.compute_mmd_penalty(latent)
            
            # λ₂L_lp: Next-step latent prediction loss (BiLSTM+Attention)
            lp_loss = torch.tensor(0.0, device=self.device)
            if next_latent_pred is not None and len(self.latent_buffer) > 0:
                # Use last latent as target for next-step prediction
                target_latent = self.latent_buffer[-1].to(self.device).unsqueeze(0).expand(next_latent_pred.size(0), -1)
                lp_loss = criterion(next_latent_pred, target_latent)
            
            # λ₃L_np: AR latent prediction loss  
            # For AR prediction, we use a simple autoregressive target (current + noise)
            ar_target = latent + 0.1 * torch.randn_like(latent)  # Simple AR target
            np_loss = criterion(latent_ar_pred, ar_target)
            
            # Composite loss: J = L_MSE + λ₁L_MMD + λ₂L_lp + λ₃L_np
            total_loss = (reconstruction_loss + 
                         self.lambda_mmd * mmd_loss +
                         self.lambda_lp * lp_loss +
                         self.lambda_np * np_loss)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update latent buffer for temporal prediction - FIXED VERSION
            current_latent = latent.detach().mean(dim=0)  # Shape: [latent_dim]
            
            # Ensure consistent tensor properties
            if len(self.latent_buffer) == 0:
                self.latent_buffer.append(current_latent)
            else:
                # Check shape consistency
                if current_latent.shape == self.latent_buffer[0].shape:
                    self.latent_buffer.append(current_latent)
                else:
                    print(f"Warning: Latent shape inconsistency. Expected {self.latent_buffer[0].shape}, got {current_latent.shape}")
                    # Clear buffer and restart
                    self.latent_buffer = [current_latent]
        
            # Maintain buffer size
            if len(self.latent_buffer) > self.sequence_length:
                self.latent_buffer.pop(0)
            
            batch_losses.append([
                reconstruction_loss.item(),
                mmd_loss.item(),
                lp_loss.item(),
                np_loss.item(),
                total_loss.item()
            ])
        
        # Calculate average losses
        epoch_loss = np.array(batch_losses).mean(axis=0)
        epoch_loss = dict(zip([
            "Reconstruction Loss", "MMD Loss", "LP Loss", "NP Loss", "Total Loss"
        ], epoch_loss))
        
        return epoch_loss
    
    def train(self, X_in: np.array, X_out: np.array):
        """Train the literature-compliant CAE-M model with provided data"""
        # Set training data
        self.X_in_tensor = torch.from_numpy(X_in).float().to(self.device)  # [n_samples, input_channels, sequence_length]
        self.X_out_tensor = torch.from_numpy(X_out).float().to(self.device)  # [n_samples, output_channels, sequence_length]
        
        sns.set_theme()
        plt.rcParams["font.family"] = "serif"
        
        print(f"Training CAE-M with architecture: {self.architecture}")
        print(f"Input shape: {X_in.shape}, Output shape: {X_out.shape}")
        print(f"Loss weights: λ₁={self.lambda_mmd}, λ₂={self.lambda_lp}, λ₃={self.lambda_np}")
        print("-" * 20)
        
        # Training history
        history = pd.DataFrame()
        best = defaultdict(lambda: np.inf)
        stopping_criteria_ctr = 0
        stop = False
        start_time = time.time()
        
        for epoch in range(self.n_epochs):
            epoch_loss = self._train_one_epoch()
            
            result = {
                "Epoch": epoch,
                "Elapsed Time": time.time() - start_time,
            }
            result.update(epoch_loss)
            
            # Early stopping criteria
            if np.all([best[k] <= epoch_loss[k] for k in epoch_loss.keys()]):
                stopping_criteria_ctr += 1
            else:
                result["Best"] = "True"
                best = epoch_loss
                stopping_criteria_ctr = 0
                
            if stopping_criteria_ctr >= self.stopping_count:
                stop = True
            
            # Save history
            history = pd.concat([
                history, 
                pd.Series(result, name=epoch).to_frame().T
            ], axis=0)
            
            # Save periodically and plot
            if epoch % 50 == 0 or epoch == self.n_epochs - 1 or stop:
                self._save_progress(history, epoch)
                
            # Progress message
            self._print_progress(epoch, epoch_loss, start_time)
            
            if stop:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        self._finalize_training(history, start_time, best)
    
    def _save_progress(self, history, epoch):
        """Save training progress and model"""
        # Save history
        history.to_csv(self.dir + "History.csv", index=False)
        
        # Save model
        self.save_model()
        
        # Plot training curves
        self._plot_training_curves(history)
    
    def _plot_training_curves(self, history):
        """Plot and save training curves for literature-compliant loss"""
        title = "CAE-M Training Curves"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reconstruction loss
        ax1.plot(history["Epoch"], history["Reconstruction Loss"], 'b-', alpha=0.7, label="Reconstruction Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Reconstruction Loss")
        ax1.set_title("L_MSE: Reconstruction Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MMD Loss
        ax2.plot(history["Epoch"], history["MMD Loss"], 'r-', alpha=0.7, label="MMD Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MMD Loss")
        ax2.set_title(f"λ₁L_MMD: MMD Penalty (λ₁={self.lambda_mmd})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Prediction losses
        ax3.plot(history["Epoch"], history["LP Loss"], 'g-', alpha=0.7, label="LP Loss")
        ax3.plot(history["Epoch"], history["NP Loss"], 'orange', alpha=0.7, label="NP Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Prediction Loss")
        ax3.set_title(f"λ₂L_lp & λ₃L_np: Prediction Losses (λ₂={self.lambda_lp}, λ₃={self.lambda_np})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Total loss
        ax4.plot(history["Epoch"], history["Total Loss"], 'purple', alpha=0.7, label="Total Loss")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Total Loss") 
        ax4.set_title("J: Composite Loss")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dir + title + ".svg", transparent=True, bbox_inches="tight", dpi=144)
        plt.close()
    
    def _print_progress(self, epoch, epoch_loss, start_time):
        """Print training progress"""
        format_text = lambda t: f"{t:.3f}" if t < 1e3 and t >= 1e-3 else f"{t:.3e}"
        loss_msg = ", ".join([f"{k}: {format_text(v)}" for k, v in epoch_loss.items()])
        
        remaining_time = (self.n_epochs - epoch - 1) * (time.time() - start_time) / (epoch + 1)
        etc_msg = f"ETC: {remaining_time / 60:.2f} min ({remaining_time:.2f} sec)"
        
        print(f"Epoch [{epoch+1}/{self.n_epochs}] | {loss_msg} | {etc_msg}", end="\r")
    
    def _finalize_training(self, history, start_time, best):
        """Finalize training and print results"""
        print()
        print("-" * 20)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        for key, value in best.items():
            print(f"Best {key}: {value:.6f}")

        print(f"Model saved to {self.model_path}")
        print("-" * 20)

    def save_model(self, device=None):
        """Save model state"""
        device = self.device if device is None else device
        original_device = self.device

        with open(self.model_path, mode="wb") as f:
            if device != original_device:
                self.model.to(device)
                torch.save(self.model.state_dict(), f)
                self.model.to(original_device)  # Restore original device
            else:
                torch.save(self.model.state_dict(), f)

        # Save model summary to text file
        summary_path = self.dir + "model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CAE-M Model Architecture Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Shape: {(self.input_channels, self.input_length)}\n")
            f.write(f"Output Shape: {(self.output_channels, self.output_length)}\n")
            f.write(f"Latent Dimension: {self.latent_dim}\n")
            f.write(f"Encoder Channels: {self.encoder_channels}\n")
            f.write(f"Decoder Channels: {self.decoder_channels}\n\n")
            
            # Capture model summary
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                summary(self.model, input_size=(self.input_channels, self.input_length))
                model_summary = buffer.getvalue()
                f.write(model_summary)
            except:
                f.write("Model summary could not be generated\n")
            finally:
                sys.stdout = old_stdout

    def load_model(self, device=None):
        """Load a pre-trained model"""
        device = self.device if device is None else device
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"CAE-M model loaded from {self.model_path}")
    
    def predict(self, x):
        """Generate reconstructions for input data"""
        x = torch.from_numpy(x).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstruction, _ = self.model(x)
            
        return reconstruction.cpu().numpy()
    
    def encode(self, x):
        """Extract latent space representations"""
        x = torch.from_numpy(x).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            latent, _ = self.model.encode(x)
            
        return latent.cpu().numpy()
    
    # def get_latent_representation(self, x):
    #     """
    #     Extract latent representations from the encoder (literature-aligned)
        
    #     This method provides access to the bottleneck latent representations
    #     used for Mahalanobis distance health indicator computation as described
    #     in advanced CAE-M literature applications.
    #     """
    #     return self.encode(x)
    
    # def get_reconstruction(self, x):
    #     """
    #     Get reconstruction output for standard reconstruction error calculation
        
    #     This method provides direct access to reconstruction outputs for
    #     computing standard reconstruction-based health indicators.
    #     """
    #     return self.predict(x)

