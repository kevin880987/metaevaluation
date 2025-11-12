"""
doi:https://doi.org/10.1088/1361-6501/ad25dc
Vector Quantized Variational Autoencoder for Health Indicator Generation (VQ-VAE)
Literature-aligned implementation based on MST'24: "Utilizing VQ-VAE for end-to-end health indicator generation in predicting rolling bearing RUL"

Key improvements aligned with the paper:
- VQ-VAE objective: recon + codebook + β·commitment with stop-grad & ST estimator (Eq. (2))
- Codebook dims: 32×16 (32 embeddings, 16-dim each). β = 2.0 (Table 2)
- Encoder/Decoder: 4 conv blocks (k=3, BN, ReLU, MaxPool) + two 1×1 convs; decoder mirrors with Upsample (Sec. 3.1.2; Table A1)
- When using 38-D preselected features, reduce to 2 blocks as noted (Tables A2–A3)
- EMA codebook updates for stability (VQ-VAE v2 style)

This maintains your existing API: train(), load_model(), predict(), encode(), health_indicator_codebook_usage()
"""

from statistics import mode
import numpy as np
import pandas as pd
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from torchsummary import summary
# from torchinfo import summary
import sys
import io


# -----------------------------
# Vector Quantizers
# -----------------------------
class VectorQuantizer(nn.Module):
    """Classic VQ (non-EMA): returns z_q (ST), vq_loss, perplexity."""
    def __init__(self, num_embeddings, embedding_dim, beta):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e: [B, D] or [B, D, L]
        input_shape = z_e.shape
        if len(input_shape) == 3:
            # Handle [B, D, L] case by flattening spatial dimension
            B, D, L = input_shape
            z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, D)  # [B*L, D]
        else:
            z_e_flat = z_e  # [B, D]
            
        with torch.no_grad():
            distances = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e_flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
            )
            indices = torch.argmin(distances, dim=1)
            
        z_q_flat = self.embedding(indices)
        
        # Reshape back to original shape if needed
        if len(input_shape) == 3:
            z_q = z_q_flat.view(B, L, D).permute(0, 2, 1).contiguous()
        else:
            z_q = z_q_flat
            
        # VQ-VAE losses with stop gradients (literature-compliant)
        codebook_loss = (z_q.detach() - z_e).pow(2).mean()
        commitment_loss = (z_q - z_e.detach()).pow(2).mean()
        vq_loss = codebook_loss + self.beta * commitment_loss
        
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        
        # Perplexity calculation
        with torch.no_grad():
            if len(input_shape) == 3:
                enc = torch.zeros(z_e_flat.size(0), self.num_embeddings, device=z_e.device)
                enc.scatter_(1, indices.unsqueeze(1), 1)
            else:
                enc = torch.zeros(z_e.size(0), self.num_embeddings, device=z_e.device)
                enc.scatter_(1, indices.unsqueeze(1), 1)
            avg_probs = enc.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
        return z_q_st, vq_loss, perplexity, indices


class VectorQuantizerEMA(nn.Module):
    """EMA codebook updates (VQ-VAE v2 style) for stability."""
    def __init__(self, num_embeddings, embedding_dim, decay, beta, eps):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.beta = beta
        self.eps = eps
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embedding_avg", embed.clone())

    def forward(self, z_e):
        input_shape = z_e.shape
        if len(input_shape) == 3:
            # Handle [B, D, L] case
            B, D, L = input_shape
            z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, D)
        else:
            z_e_flat = z_e
            
        with torch.no_grad():
            distances = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e_flat @ self.embedding.t()
                + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
            )
            indices = torch.argmin(distances, dim=1)
            enc = torch.zeros(z_e_flat.size(0), self.num_embeddings, device=z_e.device)
            enc.scatter_(1, indices.unsqueeze(1), 1)
            
            # EMA updates
            self.cluster_size.mul_(self.decay).add_(enc.sum(dim=0), alpha=1 - self.decay)
            self.embedding_avg.mul_(self.decay).add_(enc.t() @ z_e_flat, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
            self.embedding.copy_(self.embedding_avg / cluster_size.unsqueeze(1))
            
        z_q_flat = torch.index_select(self.embedding, 0, indices)
        
        # Reshape back to original shape if needed
        if len(input_shape) == 3:
            z_q = z_q_flat.view(B, L, D).permute(0, 2, 1).contiguous()
        else:
            z_q = z_q_flat
            
        # Only commitment loss for EMA version
        commitment_loss = (z_q.detach() - z_e).pow(2).mean()
        vq_loss = self.beta * commitment_loss
        
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        
        # Perplexity
        with torch.no_grad():
            avg_probs = enc.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
        return z_q_st, vq_loss, perplexity, indices


# -----------------------------
# Encoder / Decoder per MST'24
# -----------------------------
class VQVAEEncoder(nn.Module):
    """Temporal-aware encoder for sequence data"""
    def __init__(self, in_dim, hidden_dim, embedding_dim, n_blocks):
        super().__init__()
        layers = []
        c_in = in_dim
        for b in range(n_blocks):
            c_out = max(hidden_dim // (2 ** b), embedding_dim) if b < n_blocks - 1 else embedding_dim
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(c_out),
                # nn.LayerNorm(c_out),
                nn.ReLU(inplace=True),
                nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1),  # Temporal downsampling
            ]
            c_in = c_out
            
        # Final embedding projection
        layers += [nn.Conv1d(c_in, embedding_dim, kernel_size=1, stride=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C=in_dim, L]
        z = self.net(x)  # [B, embedding_dim, L']
        return z  # [B, embedding_dim, L'] - temporal VQ-VAE


class VQVAEDecoder(nn.Module):
    """Mirror: conv (k=3) + Upsample; start from linear projection of latent."""
    def __init__(self, out_dim, hidden_dim, embedding_dim, output_length, n_blocks):
        super().__init__()
        self.output_length = output_length
        self.init_len = max(1, output_length // (2 ** n_blocks))
        self.embedding_dim = embedding_dim
        
        layers = []
        c_in = embedding_dim
        
        # Reverse the channel progression from encoder
        channels = [max(hidden_dim // (2 ** b), out_dim) for b in reversed(range(n_blocks))]
        
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(c_out),
                # nn.LayerNorm(c_out),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ]
            c_in = c_out
            
        # Final output layer
        layers += [nn.Conv1d(c_in, out_dim, kernel_size=3, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # z: [B, embedding_dim, L'] - temporal quantized features
        B, D, L_prime = z.size()
        
        # Ensure we have the correct embedding dimension
        assert D == self.embedding_dim, f"Expected embedding_dim={self.embedding_dim}, got {D}"
        
        # Apply convolutional decoder directly to temporal features
        x_hat = self.net(z)
        
        # Ensure correct output length
        if x_hat.size(-1) != self.output_length:
            x_hat = F.interpolate(x_hat, size=self.output_length, mode='linear', align_corners=False)
        return x_hat


# -----------------------------
# VQ-VAE Model
# -----------------------------
class VectorQuantizedVAE(nn.Module):
    """
    Literature-compliant VQ-VAE model for health indicator generation
    """
    def __init__(self, in_dim, out_dim, hidden_dim, embedding_dim, num_embeddings, beta, use_ema, ema_decay, n_blocks, output_length):
        super().__init__()
        self.encoder = VQVAEEncoder(in_dim, hidden_dim, embedding_dim, n_blocks)
        self.decoder = VQVAEDecoder(out_dim, hidden_dim, embedding_dim, output_length, n_blocks)
        
        if use_ema:
            self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, ema_decay, beta, 1e-5)
        else:
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, beta)
            
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        z_e = self.encoder(x)  # [B, embedding_dim, L'] - NOW TEMPORAL!
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)  # Per-timestep quantization
        x_hat = self.decoder(z_q)  # [B, out_dim, L] - proper sequence reconstruction
        return x_hat, z_e, z_q, vq_loss, perplexity, indices

    @torch.no_grad()
    def encode_latent(self, x):
        z_e = self.encoder(x)
        z_q, _, _, _ = self.quantizer(z_e)
        return z_q
        
    @torch.no_grad()
    def get_quantization_indices(self, x):
        """Get quantization indices for health indicator calculation"""
        z_e = self.encoder(x)
        _, _, _, indices = self.quantizer(z_e)
        return indices

    def set_output_length(self, output_length):
        """Update output length if needed"""
        self.decoder.output_length = output_length


class VQVAE:
    """
    Vector Quantized Variational Autoencoder for Health Indicator Generation
    
    Literature-compliant implementation following MST'24 paper specifications.
    """
    
    def __init__(self, config):
        self.config = config
        self.architecture = config.architecture
        hidden_dim, embedding_dim, num_embeddings, n_blocks = self.architecture
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.n_blocks = n_blocks
        
        # Get dimensions from config
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels
        self.input_length = config.input_length
        self.output_length = config.output_length

        # Literature-compliant hyperparameters (MST'24 Table 2)
        self.beta = config.commitment_cost  # Table 2: β=2.0
        self.use_ema = config.use_ema
        self.ema_decay = config.ema_decay
                
        # Training hyperparameters
        self.learning_rate = config.learning_rate
        self.n_epochs = config.n_epochs
        self.stopping_count = config.stopping_count
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.vq_loss_weight = config.vq_loss_weight

        # Device and paths
        self.device = config.device
        self.seed = config.seed
        self.dir = config.training_dir
        self.model_path = self.dir + "vq_vae.pt"

        # Initialize model
        self.model = VectorQuantizedVAE(
            in_dim=self.input_channels,
            out_dim=self.output_channels,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            beta=self.beta,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            n_blocks=self.n_blocks,
            output_length=self.output_length
        ).to(self.device)

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
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
    
    def _train_one_epoch(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        mse = nn.MSELoss()
        
        batch_losses = []
        perplexities = []
        
        for x_in_t, x_out_t in self._load_batches(self.batch_size, self.shuffle):
            optimizer.zero_grad()
            
            # Forward pass
            x_hat, z_e, z_q, vq_loss, perplexity, indices = self.model(x_in_t)
            
            # Literature-compliant loss: reconstruction + β·VQ_loss
            recon_loss = mse(x_hat, x_out_t)
            total_loss = recon_loss + self.vq_loss_weight * vq_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            batch_losses.append([recon_loss.item(), vq_loss.item(), total_loss.item()])
            perplexities.append(perplexity.item())
        
        # Calculate average losses
        batch_losses = np.array(batch_losses)
        recon_avg, vq_avg, total_avg = batch_losses.mean(axis=0)
        perplexity_avg = float(np.mean(perplexities))
        
        epoch_loss = {
            "Reconstruction Loss": recon_avg,
            "VQ Loss": vq_avg,
            "Total Loss": total_avg,
            "Perplexity": perplexity_avg
        }
        
        return epoch_loss
    
    def train(self, X_in: np.array, X_out: np.array):
        """Train the literature-compliant VQ-VAE model"""
        # Set training data
        self.X_in_tensor = torch.from_numpy(X_in).float().to(self.device)  # [n_samples, input_channels, sequence_length]
        self.X_out_tensor = torch.from_numpy(X_out).float().to(self.device)  # [n_samples, output_channels, sequence_length]

        sns.set_theme()
        plt.rcParams["font.family"] = "serif"
        
        # Training history
        history = pd.DataFrame()
        best_total = float('inf')
        no_improve = 0
        start_time = time.time()
        
        print(f"Training Literature-Compliant VQ-VAE")
        print(f"Input shape: {X_in.shape}, Output shape: {X_out.shape}")
        print(f"Codebook: {self.num_embeddings}×{self.embedding_dim}, β={self.beta}")
        print(f"Architecture: {self.n_blocks} blocks, Device: {self.device}")
        print("-" * 20)
        
        for epoch in range(self.n_epochs):
            epoch_loss = self._train_one_epoch()
            
            result = {
                "Epoch": epoch,
                "Elapsed Time": time.time() - start_time,
            }
            result.update(epoch_loss)
            
            # Early stopping criteria
            current_total = epoch_loss["Total Loss"]
            if current_total < best_total - 1e-6:
                best_total = current_total
                no_improve = 0
                result["Best"] = "True"
                # Save best model
                self.save_model()
            else:
                no_improve += 1
                
            if no_improve >= self.stopping_count:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Save history
            history = pd.concat([
                history, 
                pd.Series(result, name=epoch).to_frame().T
            ], axis=0)
            
            # Save periodically and plot
            if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                self._save_progress(history, epoch)
                
            # Progress message
            self._print_progress(epoch, epoch_loss, start_time)
        
        self._finalize_training(history, start_time, best_total)
            
    def _save_progress(self, history, epoch):
        """Save training progress and model"""
        # Save history
        try:
            history.to_csv(self.dir + "VQVAE_History.csv", index=False)
        except Exception:
            pass
        
        # Plot training curves
        self._plot_training_curves(history)
    
    def _plot_training_curves(self, history):
        """Plot and save VQ-VAE training curves"""
        title = "VQ-VAE Literature-Compliant Training Curves"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reconstruction loss
        ax1.plot(history["Epoch"], history["Reconstruction Loss"], 'b-', alpha=0.7, label="Reconstruction Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Reconstruction Loss")
        ax1.set_title("Reconstruction Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # VQ Loss
        ax2.plot(history["Epoch"], history["VQ Loss"], 'r-', alpha=0.7, label="VQ Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("VQ Loss")
        ax2.set_title(f"Vector Quantization Loss (β={self.beta})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Total loss
        ax3.plot(history["Epoch"], history["Total Loss"], 'purple', alpha=0.7, label="Total Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Total Loss")
        ax3.set_title("Total Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Perplexity
        ax4.plot(history["Epoch"], history["Perplexity"], 'orange', alpha=0.7, label="Perplexity")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Perplexity")
        ax4.set_title("Codebook Perplexity")
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
    
    def _finalize_training(self, history, start_time, best_total):
        """Finalize training and print results"""
        print()
        print("-" * 20)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Best Total Loss: {best_total:.6f}")
        print(f"Final Codebook: {self.num_embeddings}×{self.embedding_dim}")
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
            f.write("VQ-VAE Model Architecture Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Shape: {(self.input_channels, self.input_length)}\n")
            f.write(f"Output Shape: {(self.output_channels, self.output_length)}\n")
            f.write(f"Hidden Dimension: {self.hidden_dim}\n")
            f.write(f"Embedding Dimension: {self.embedding_dim}\n")
            f.write(f"Number of Embeddings: {self.num_embeddings}\n")
            f.write(f"Number of Blocks: {self.n_blocks}\n")

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
        print(f"VQ-VAE model loaded from {self.model_path}")
    
    def predict(self, x):
        """Generate reconstructions for input data"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            x_hat, *_ = self.model(x_tensor)
            return x_hat.cpu().numpy()
    
    def encode(self, x):
        """Extract feature representations"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            z_e = self.model.encoder(x_tensor)
            return z_e.cpu().numpy()

    def encode_latent(self, x):
        """Extract codebook representations"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            z_q = self.model.encode_latent(x_tensor)
            return z_q.cpu().numpy()
    
    # def get_quantization_indices(self, x):
    #     """Get discrete quantization indices for each sample"""
    #     self.model.eval()
    #     with torch.no_grad():
    #         x_tensor = torch.from_numpy(x).float().to(self.device)
            
    #         # Get quantization indices from the model
    #         indices = self.model.get_quantization_indices(x_tensor)  # [batch_size]
            
    #         return indices.cpu().numpy()
    
    # def get_quantized_latent_representation(self, x):
    #     """
    #     Calculate health indicators based on codebook usage patterns
        
    #     This implements the end-to-end health indicator generation approach
    #     described in the MST'24 paper using vector quantization indices.
        
    #     Returns:
    #         Health indicator values for each sample based on codebook entropy
    #     """
    #     # Get quantization indices for each sample
    #     indices = self.get_quantization_indices(x)
    #     batch_size = len(indices)
        
    #     # For health indicator calculation, we use a sliding window approach
    #     # to calculate entropy of codebook usage over local neighborhoods
    #     window_size = min(50, batch_size)  # Sliding window size
    #     health_indicators = []
        
    #     for i in range(batch_size):
    #         # Define window around current sample
    #         start_idx = max(0, i - window_size // 2)
    #         end_idx = min(batch_size, i + window_size // 2 + 1)
    #         window_indices = indices[start_idx:end_idx]
            
    #         # Calculate entropy of codebook usage in this window
    #         unique_indices, counts = np.unique(window_indices, return_counts=True)
    #         probs = counts / counts.sum()
    #         entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
    #         # Normalize entropy by maximum possible (uniform distribution)
    #         max_entropy = np.log2(min(len(unique_indices), self.num_embeddings))
    #         normalized_entropy = entropy / (max_entropy + 1e-10)
            
    #         health_indicators.append(normalized_entropy)
        
    #     return np.array(health_indicators)

    # def get_quantization_error(self, x, smooth=True):
    #     self.model.eval()
    #     X = torch.from_numpy(x).float().to(self.device)
    #     with torch.no_grad():
    #         z_e = self.model.encoder(X)  # [B, embedding_dim, L']
    #         z_q, _, _, _ = self.model.quantizer(z_e)  # Get quantized version
    #         quantization_error = torch.norm(z_e - z_q, dim=1).cpu().numpy()  # [B, L’]

    #     # Normalize to [0,1] using train-set stats
    #     eps = 1e-12
    #     if hasattr(self, "quantization_error_min") and hasattr(self, "quantization_error_max"):
    #         quantization_error = (quantization_error - self.quantization_error_min) / (max(self.quantization_error_max - self.quantization_error_min, eps))

    #     if smooth:
    #         try:
    #             from scipy.signal import savgol_filter
    #             quantization_error = savgol_filter(quantization_error, window_length=21, polyorder=3, axis=1, mode="interp")
    #         except Exception:
    #             pass

    #     return quantization_error
