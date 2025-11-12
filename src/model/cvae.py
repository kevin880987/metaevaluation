from xml.parsers.expat import model
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from torchsummary import summary
# from torchinfo import summary
import sys
import io


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, input_length, output_length, encoder_channels, latent_dim, decoder_channels):
        super(ConvolutionalVariationalAutoencoder, self).__init__()
        # Store dimensions for proper reconstruction
        self.input_channels = input_channels  # Number of input features (channels)
        self.output_channels = output_channels  # Number of output features (channels)
        self.input_length = input_length
        self.output_length = output_length
        
        self.encoder_channels = encoder_channels
        self.latent_dim = latent_dim
        self.decoder_channels = decoder_channels

        # Encoder: Progressive downsampling with conv layers
        # Input: [batch, input_channels, input_sequence_length]
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
        
        # Calculate the size after convolutions for FC layers
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Latent space layers (mean and log variance) - now initialized upfront
        self.fc_mu = nn.Linear(self.conv_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, self.latent_dim)
        
        # Decoder FC layer (from latent to conv input)
        self.decoder_fc = nn.Linear(self.latent_dim, self.conv_output_size)
        
        # Decoder: Progressive upsampling with deconv layers
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
            
        # Final layer to output features (no activation for regression)
        decoder_layers.append(
            nn.ConvTranspose1d(in_channels, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def _calculate_conv_output_size(self):
        """Calculate conv output size mathematically to avoid dynamic initialization"""
        # Create a dummy input to calculate output size
        dummy_input = torch.zeros(1, self.input_channels, self.input_length)
        
        # Set eval mode to avoid BatchNorm issues during initialization
        self.encoder_conv.eval()
        with torch.no_grad():
            dummy_output = self.encoder_conv(dummy_input)
        
        # Return to train mode
        self.encoder_conv.train()
        
        return dummy_output.view(1, -1).size(1)
        
    def encode(self, x):
        # Convolutional encoding
        conv_features = self.encoder_conv(x)  # [batch, channels, reduced_length]
        
        # Flatten for FC layers
        conv_features_flat = conv_features.view(conv_features.size(0), -1)
        
        # Get mean and log variance
        mu = self.fc_mu(conv_features_flat)
        logvar = self.fc_logvar(conv_features_flat)
        
        return mu, logvar, conv_features.shape

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, conv_shape):
        """Decode from latent space back to multi-feature time series"""
        # FC layer: latent -> conv input size
        decoded = self.decoder_fc(z)
        
        # Reshape to convolutional format
        decoded = decoded.view(conv_shape)
        
        # Deconvolutional decoding
        reconstruction = self.decoder_conv(decoded)
        
        # Ensure output matches target output length
        if reconstruction.size(-1) != self.output_length:
            reconstruction = torch.nn.functional.interpolate(
                reconstruction, size=self.output_length, mode='linear', align_corners=False
            )
                    
        return reconstruction

    def forward(self, x):
        # Encode
        mu, logvar, conv_shape = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decode(z, conv_shape)
        
        return reconstruction, mu, logvar

"""
conv_output=x.unsqueeze(2)
for i in range(len(self.encoder)):
    conv_output=self.encoder[i](conv_output)
    plt.plot(conv_output.squeeze().detach().numpy())
    plt.show()
z=conv_output.unsqueeze(2)
for i in range(len(self.decoder)):
    z=self.decoder[i](z)
    plt.plot(z.squeeze().detach().numpy())
    plt.show()
"""

# x=x_in
# X_in, X_out, E=X_in.values, X_out.values, E.values
# self=CVAE(config, X_in, X_out, E)
# self=model
class CVAE():
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

        # β-VAE hyperparameters from config
        self.beta_start = config.beta_start
        self.beta_max = config.beta_max
        self.beta_annealing_rate = config.beta_annealing_rate

        # Device and paths
        self.device = config.device
        self.seed = config.seed
        self.dir = config.training_dir
        self.model_path = self.dir + "cvae.pt"
        
        # Initialize current epoch for beta annealing
        self.current_epoch = 0
        
        # Initialize the model with proper dimensions
        self.model = ConvolutionalVariationalAutoencoder(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            input_length=self.input_length,
            output_length=self.output_length,
            encoder_channels=self.encoder_channels,
            latent_dim=self.latent_dim,
            decoder_channels=self.decoder_channels, 
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
        reconstruction_criterion = nn.MSELoss()  # Mean Squared Error loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        batch_losses = []
        for x_in, x_out in self._load_batches(self.batch_size, self.shuffle):
            optimizer.zero_grad()

            # Forward pass - x_in is already in 3D format [batch, n_features, sequence_length]
            reconstruction_outputs, mu, logvar = self.model(x_in)
            
            # Compute losses - both reconstruction_outputs and x_out are 3D
            reconstruction_loss = reconstruction_criterion(reconstruction_outputs, x_out)
            
            # Proper KL divergence calculation (mean over batch, sum over latent dims)
            kl_divergence = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )
            
            # Literature-compliant β-VAE loss with annealing
            # Use configurable β parameters
            beta = min(self.beta_max, self.beta_start * (1 + self.current_epoch * self.beta_annealing_rate))
            total_loss = reconstruction_loss + beta * kl_divergence
            
            batch_losses.append([
                reconstruction_loss.item(), 
                kl_divergence.item(), 
                total_loss.item(), 
                beta
            ])
            
            # Backpropagation and optimization
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # Find average loss over sequences and batches
        epoch_loss = np.array(batch_losses).mean(axis=0)
        epoch_loss = dict(zip([
            "Reconstruction Loss", "KL Divergence", "Total Loss", "Beta"
        ], epoch_loss))

        return epoch_loss

    def train(self, X_in: np.array, X_out: np.array):
        """Train the CVAE model with provided data"""
        # Set training data
        self.X_in_tensor = torch.from_numpy(X_in).float().to(self.device)  # [n_samples, input_channels, sequence_length]
        self.X_out_tensor = torch.from_numpy(X_out).float().to(self.device)  # [n_samples, output_channels, sequence_length]
        
        sns.set_theme()
        plt.rcParams["font.family"] = "serif"

        # Training history
        history = pd.DataFrame()
        best = defaultdict(lambda: np.inf)
        stopping_criteria_ctr = 0
        stop = False
        start_time = time.time()
        
        print(f"Training CVAE with architecture: {self.architecture}")
        print(f"Input shape: {X_in.shape}, Output shape: {X_out.shape}")
        print(f"Device: {self.device}")
        print(f"β-VAE annealing: {self.beta_start} → {self.beta_max} (rate: {self.beta_annealing_rate})")
        print("-" * 20)
        
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch  # Track current epoch for β annealing
            epoch_loss = self._train_one_epoch()
            result = {
                "Epoch": epoch, 
                "Elapsed Time": time.time()-start_time, 
                }
            result.update(epoch_loss)

            # Stopping criteria (only check reconstruction and total loss)
            key_losses = ["Reconstruction Loss", "Total Loss"]
            if np.all([best[k] <= epoch_loss[k] for k in key_losses if k in epoch_loss]):
                stopping_criteria_ctr += 1
            else:
                result["Best"] = "True"
                # Update only the key losses for best tracking
                for k in key_losses:
                    if k in epoch_loss:
                        best[k] = epoch_loss[k]
                stopping_criteria_ctr = 0
                
            if stopping_criteria_ctr >= self.stopping_count:
                stop = True

            # Save history
            history = pd.concat([history, pd.Series(result, name=epoch).to_frame().T], axis=0)

            # Save progress and plot periodically
            if epoch % 50 == 0 or epoch == self.n_epochs - 1 or stop:
                self._save_progress(history, epoch)

            # Progress message
            self._print_progress(epoch, epoch_loss, start_time)

            if stop:
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
        """Plot and save training curves for CVAE"""
        title = "CVAE Training Curves"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reconstruction loss
        ax1.plot(history["Epoch"], history["Reconstruction Loss"], 'b-', alpha=0.7, label="Reconstruction Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Reconstruction Loss")
        ax1.set_title("Reconstruction Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # KL Divergence
        ax2.plot(history["Epoch"], history["KL Divergence"], 'r-', alpha=0.7, label="KL Divergence")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("KL Divergence")
        ax2.set_title("KL Divergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Total loss
        ax3.plot(history["Epoch"], history["Total Loss"], 'purple', alpha=0.7, label="Total Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Total Loss")
        ax3.set_title("Total Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Beta annealing schedule
        ax4.plot(history["Epoch"], history["Beta"], 'orange', alpha=0.7, label="Beta (KL Weight)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Beta Value")
        ax4.set_title(f"β-VAE Annealing: {self.beta_start} → {self.beta_max}")
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
            f.write("CVAE Model Architecture Summary\n")
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
        device = self.device if device is None else device
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"CVAE model loaded from {self.model_path}")

    def predict(self, x):
        # x should already be in 3D format: [batch, n_features, sequence_length]
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

        # Reconstruct
        self.model.to(self.device).eval()
        reconstruction_outputs, mu, logvar = self.model(x)
        reconstruction_outputs = reconstruction_outputs.detach().cpu().numpy()
        return reconstruction_outputs

    def encode(self, x):
        # x should already be in 3D format: [batch, n_features, sequence_length]
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

        # Encode
        self.model.to(self.device).eval()
        mu, logvar, conv_shape = self.model.encode(x)
        latent_space = self.model.reparameterize(mu, logvar)
        latent_space = latent_space.detach().cpu().numpy()
        return latent_space
