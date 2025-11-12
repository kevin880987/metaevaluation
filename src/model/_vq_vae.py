import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict

"""
Literature-aligned VQ-VAE for multi-sensor 1D vibration (MST'24: "Utilizing VQ-VAE for end-to-end HI generation in predicting rolling bearing RUL").
Key points matched to the paper:
- VQ-VAE objective: recon + codebook + β·commitment with stop-grad & ST estimator (Eq. (2)).
- Codebook dims: 32×16 (32 embeddings, 16-dim each). β = 2.0. (Table 2)
- Encoder/Decoder: 4 conv blocks (k=3, BN, ReLU, MaxPool) + two 1×1 convs; decoder mirrors with Upsample. (Sec. 3.1.2; Table A1)
- When using 38-D preselected features, reduce to 2 blocks as noted. (Tables A2–A3)
- ASTCN label/prediction backbone provided (Table 1) for completeness; not used inside VQ-VAE.

Inputs are treated as 1D signals: shape [B, C=1, L].
This wrapper keeps your existing API: train(), load_model(), predict(), encode().
"""

# -----------------------------
# Vector Quantizers
# -----------------------------
class VectorQuantizer(nn.Module):
    """Classic VQ (non-EMA): returns z_q (ST), vq_loss, perplexity."""
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 2.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        # z_e: [B, D]
        with torch.no_grad():
            distances = (
                z_e.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
            )
            indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        codebook_loss = (z_q.detach() - z_e).pow(2).mean()
        commitment_loss = (z_q - z_e.detach()).pow(2).mean()
        vq_loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()
        with torch.no_grad():
            enc = torch.zeros(z_e.size(0), self.num_embeddings, device=z_e.device)
            enc.scatter_(1, indices.unsqueeze(1), 1)
            avg_probs = enc.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return z_q_st, vq_loss, perplexity

class VectorQuantizerEMA(nn.Module):
    """EMA codebook updates (VQ-VAE v2 style) for stability."""
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, beta: float = 2.0, eps: float = 1e-5):
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

    def forward(self, z_e: torch.Tensor):
        with torch.no_grad():
            distances = (
                z_e.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e @ self.embedding.t()
                + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
            )
            indices = torch.argmin(distances, dim=1)
            enc = torch.zeros(z_e.size(0), self.num_embeddings, device=z_e.device)
            enc.scatter_(1, indices.unsqueeze(1), 1)
            # EMA updates
            self.cluster_size.mul_(self.decay).add_(enc.sum(dim=0), alpha=1 - self.decay)
            self.embedding_avg.mul_(self.decay).add_(enc.t() @ z_e, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n
            self.embedding.copy_(self.embedding_avg / cluster_size.unsqueeze(1))
        z_q = torch.index_select(self.embedding, 0, indices)
        commitment_loss = (z_q.detach() - z_e).pow(2).mean()
        vq_loss = self.beta * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()
        with torch.no_grad():
            avg_probs = enc.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return z_q_st, vq_loss, perplexity

# -----------------------------
# Encoder / Decoder per MST'24
# -----------------------------
class Encoder(nn.Module):
    """4 conv blocks (k=3) + two 1×1 convs → GAP over time → latent [B, D]."""
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, n_blocks: int = 4):
        super().__init__()
        layers = []
        c_in = in_dim
        for b in range(n_blocks):
            c_out = max(hidden_dim // (2 ** b), latent_dim) if b < n_blocks - 1 else hidden_dim // (2 ** b)
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        layers += [
            nn.Conv1d(c_in, latent_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1, stride=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C=in_dim, L]
        z = self.net(x)              # [B, D, L']
        z = z.mean(dim=-1)           # global average pool over time → [B, D]
        return z

class Decoder(nn.Module):
    """Mirror: conv (k=3) + Upsample; start from linear projection of latent."""
    def __init__(self, out_dim: int, hidden_dim: int, latent_dim: int, output_length: int, n_blocks: int = 4):
        super().__init__()
        self.output_length = output_length
        self.init_len = max(1, output_length // (2 ** n_blocks))
        self.proj = nn.Linear(latent_dim, latent_dim * self.init_len)
        layers = []
        c_in = latent_dim
        channels = [max(hidden_dim // (2 ** b), out_dim) for b in reversed(range(n_blocks))]
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ]
            c_in = c_out
        layers += [nn.Conv1d(c_in, out_dim, kernel_size=3, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        B = z.size(0)
        h = self.proj(z).view(B, -1, self.init_len)
        x_hat = self.net(h)
        if x_hat.size(-1) != self.output_length:
            x_hat = F.interpolate(x_hat, size=self.output_length, mode='linear', align_corners=False)
        return x_hat

# -----------------------------
# VQ-VAE Model
# -----------------------------
class VQVAEModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, latent_dim: int,
                 num_embeddings: int = 32, beta: float = 2.0, use_ema: bool = True, ema_decay: float = 0.99,
                 n_blocks: int = 4, output_length: int = 256):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, latent_dim, n_blocks=n_blocks)
        self.decoder = Decoder(out_dim, hidden_dim, latent_dim, output_length=output_length, n_blocks=n_blocks)
        if use_ema:
            self.quantizer = VectorQuantizerEMA(num_embeddings, latent_dim, decay=ema_decay, beta=beta)
        else:
            self.quantizer = VectorQuantizer(num_embeddings, latent_dim, beta=beta)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_e, z_q, vq_loss, perplexity

    @torch.no_grad()
    def encode_latent(self, x):
        z_e = self.encoder(x)
        z_q, _, _ = self.quantizer(z_e)
        return z_q

# -----------------------------
# Wrapper
# -----------------------------
class VQVAE:
    def __init__(self, config, X_in: np.ndarray, X_out: np.ndarray, D: np.ndarray):
        self.config = config
        self.X_in = X_in
        self.X_out = X_out
        self.D = D
        # Interpret inputs as 1D signals with C=1, L=features
        input_size = X_in.shape[1]
        output_size = X_out.shape[1]
        hidden_dim = getattr(config, "hidden_dim", 128)
        latent_dim = getattr(config, "latent_dim", 16)  # Table 2: 16
        self.num_embeddings = getattr(config, "num_embeddings", 32)  # Table 2: 32
        self.beta = getattr(config, "commitment_cost", 2.0)          # Table 2: β=2.0
        self.use_ema = getattr(config, "use_ema", True)
        self.ema_decay = getattr(config, "ema_decay", 0.99)
        # Block count: 4 for raw signals; 2 for 38-D features (paper note)
        self.n_blocks = getattr(config, "n_blocks", 4 if input_size > 38 else 2)

        self.learning_rate = getattr(config, "learning_rate", 1e-3)
        self.n_epochs = getattr(config, "n_epochs", 300)
        self.stopping_count = getattr(config, "stopping_count", 30)
        self.batch_size = getattr(config, "batch_size", 256)
        self.shuffle = getattr(config, "shuffle", True)
        self.vq_loss_weight = getattr(config, "vq_loss_weight", 1.0)

        self.DEVICE = getattr(config, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dir = getattr(config, "training_dir", "./")

        torch.manual_seed(getattr(config, "SEED", 42))
        torch.cuda.manual_seed(getattr(config, "SEED", 42))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.model = VQVAEModel(
            in_dim=1,
            out_dim=1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_embeddings=self.num_embeddings,
            beta=self.beta,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            n_blocks=self.n_blocks,
            output_length=output_size,
        ).to(self.DEVICE)

    def _load_batches(self):
        idx = np.arange(self.X_in.shape[0])
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(int(np.ceil(idx.size / self.batch_size))):
            x_in = self.X_in[idx[i * self.batch_size:(i + 1) * self.batch_size]]
            x_out = self.X_out[idx[i * self.batch_size:(i + 1) * self.batch_size]]
            if x_in.shape[0] <= 1:
                break
            yield x_in, x_out

    def _train_one_epoch(self, optimizer):
        self.model.train()
        mse = nn.MSELoss()
        batch_losses = []
        perplexities = []
        for x_in, x_out in self._load_batches():
            x_in_t = torch.from_numpy(x_in).float().to(self.DEVICE).unsqueeze(1)
            x_out_t = torch.from_numpy(x_out).float().to(self.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            x_hat, z_e, z_q, vq_loss, perplexity = self.model(x_in_t)
            recon_loss = mse(x_hat, x_out_t)
            loss = recon_loss + self.vq_loss_weight * vq_loss
            loss.backward()
            optimizer.step()
            batch_losses.append([recon_loss.item(), vq_loss.item(), loss.item()])
            perplexities.append(perplexity.item())
        batch_losses = np.array(batch_losses)
        recon, vq, total = batch_losses.mean(axis=0)
        return recon, vq, total, float(np.mean(perplexities))

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        history = []
        best_total = float('inf')
        no_improve = 0
        start = time.time()
        for epoch in range(self.n_epochs):
            recon, vq, total, ppl = self._train_one_epoch(optimizer)
            history.append({
                "epoch": epoch,
                "elapsed_sec": time.time() - start,
                "recon": recon,
                "vq": vq,
                "total": total,
                "perplexity": ppl,
            })
            if total < best_total - 1e-6:
                best_total = total
                no_improve = 0
                with open(self.dir + "vq_vae.pt", mode="wb") as f:
                    torch.save(self.model.state_dict(), f)
            else:
                no_improve += 1
            if no_improve >= self.stopping_count:
                break
        try:
            pd.DataFrame(history).to_csv(self.dir + "VQVAE_History.csv", index=False)
        except Exception:
            pass

    def load_model(self):
        try:
            state = torch.load(self.dir + "vq_vae.pt", map_location=self.DEVICE)
            self.model.load_state_dict(state)
        except Exception:
            pass

    @torch.no_grad()
    def predict(self, X_in: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_in_t = torch.from_numpy(X_in).float().to(self.DEVICE).unsqueeze(1)
        x_hat, *_ = self.model(X_in_t)
        return x_hat.squeeze(1).cpu().numpy()

    @torch.no_grad().
    def encode(self, X_in: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_in_t = torch.from_numpy(X_in).float().to(self.DEVICE).unsqueeze(1)
        z_q = self.model.encode_latent(X_in_t)
        return z_q.cpu().numpy()

# -----------------------------
# (Optional) ASTCN backbone per MST'24 Table 1 (for label/prediction)
# -----------------------------
class ASBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, leaky=0.2, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False)
        self.lrelu = nn.LeakyReLU(leaky, inplace=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = self.conv(x)
        return self.dropout(self.lrelu(y))

class ASTCN(nn.Module):
    """Minimal ASTCN per MST'24 (for label/prediction), not used inside VQ-VAE."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=12, stride=4),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.3),
        )
        self.b1 = ASBlock(16, 12, kernel_size=3, dilation=1, leaky=0.2, dropout=0.3)
        self.b2 = ASBlock(12, 6, kernel_size=3, dilation=2, leaky=0.2, dropout=0.3)
        self.b3 = ASBlock(6, 4, kernel_size=3, dilation=4, leaky=0.2, dropout=0.3)
        self.head = nn.Conv1d(4, 1, kernel_size=1)
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x)
