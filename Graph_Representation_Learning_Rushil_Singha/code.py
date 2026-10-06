import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import math
import numpy as np
import matplotlib.pyplot as plt
from jetnet.datasets import JetNet
USE_JETNET = True
print("Using JetNet dataset")

import networkx as nx
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import ChebConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeometricDataLoader

from scipy.stats import wasserstein_distance

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Configuration ---
class Config:
    # Data
    max_jets = 3200  # FIXED: Reduced from 50000 to avoid large memory usage
    
    # GNN hyperparameters - FIXED: Made configurable
    gnn_in_dim = 3
    gnn_hidden_dim = 64
    gnn_K = 4
    gnn_out_dim = 32
    
    # Decoder hyperparameters - FIXED: Made configurable
    decoder_hidden_dims = [512, 1024]
    max_particles = 150
    particle_dim = 4
    
    # Diffusion hyperparameters - FIXED: Made configurable
    diffusion_hidden_dim = 512
    diffusion_lr = 1e-4
    diffusion_n_steps = 1000
    diffusion_epochs = 500
    
    # Autoencoder hyperparameters - FIXED: Made configurable
    autoencoder_lr = 0.001
    autoencoder_weight_decay = 1e-5
    autoencoder_epochs = 300
    
    # Training
    batch_size = 8
    
    # Checkpointing - FIXED: Added checkpoint configuration
    save_checkpoints = True
    checkpoint_dir = "checkpoints"
    save_every_n_epochs = 50

config = Config()

# --- 1. Data Loading and Preprocessing ---

def load_jetnet_data():
    """Load JetNet dataset with fallback options"""
    global USE_JETNET

    if USE_JETNET is None:
        raise ImportError("No suitable dataset package available")

    if USE_JETNET:
        # Load JetNet data
        particle_data, jet_data = JetNet.getData(
            jet_type=['g', 'q', 't'],
            data_dir="jetnet_data",
            particle_features=['etarel', 'phirel', 'ptrel', 'mask'],
            jet_features=['pt', 'eta', 'mass', 'num_particles'],
            num_particles=150,
            split='train',
            download=True
        )

        print("Particle shape:", particle_data.shape)
        print("Jet shape:", jet_data.shape)
        print(f"Total number of jets in dataset: {len(particle_data)}")

        return particle_data, jet_data

def get_hits_from_jetnet(jet_particles, jet_info):
    """Extract hits from JetNet particle data"""
    # Get valid particles (mask == 1)
    valid_mask = jet_particles[:, 3] == 1
    valid_particles = jet_particles[valid_mask]

    if len(valid_particles) == 0:
        return None

    # Sort by descending pt_rel
    sorted_idx = np.argsort(-valid_particles[:, 2])
    valid_particles = valid_particles[sorted_idx]

    # Extract features
    eta = valid_particles[:, 0]  # etarel
    phi = valid_particles[:, 1]  # phirel
    pt = valid_particles[:, 2]   # ptrel

    # Create hits array: [eta, phi, pt]
    hits = np.column_stack((eta, phi, pt))

    return hits

def make_graph(hits):
    """Create graph from hits using k-nearest neighbors"""
    if len(hits) < 4:  # Need at least 4 points for k=4
        # Use smaller k for sparse jets
        k = min(len(hits) - 1, 2)
        if k <= 0:
            return None, hits, None
    else:
        k = 4

    coords = hits[:, :2]  # eta, phi coordinates
    adj_matrix = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(adj_matrix)
    return G, hits, adj_matrix

def collect_graph_and_targets(particle_data, jet_data, max_jets=None):
    """Collect graph data and targets for training"""
    if max_jets is None:
        max_jets = config.max_jets
        
    graph_data = []
    target_particles = []
    original_jets = []
    failed_count = 0
    # FIXED: Better error tracking
    error_types = {}

    n_jets = min(len(particle_data), max_jets)
    print(f"Collecting data for {n_jets} jets...")

    for i in range(n_jets):
        if i % 100 == 0:
            print(f"Processing jet {i}/{n_jets}...")

        try:
            jet_particles = particle_data[i]
            jet_info = jet_data[i]

            hits = get_hits_from_jetnet(jet_particles, jet_info)
            if hits is None or len(hits) == 0:
                failed_count += 1
                error_types['no_valid_particles'] = error_types.get('no_valid_particles', 0) + 1
                continue

            G, hits, adj_matrix = make_graph(hits)
            if G is None or adj_matrix is None:
                failed_count += 1
                error_types['graph_construction_failed'] = error_types.get('graph_construction_failed', 0) + 1
                continue

            x = torch.tensor(hits, dtype=torch.float)
            edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

            data = Data(x=x, edge_index=edge_index)
            graph_data.append(data)

            # Sort target particles by descending pt_rel and pad
            valid_mask = jet_particles[:, 3] == 1
            valid_particles = jet_particles[valid_mask]
            if len(valid_particles) > 0:
                sorted_idx = np.argsort(-valid_particles[:, 2])
                sorted_valid = valid_particles[sorted_idx]
            else:
                sorted_valid = np.empty((0, 4))
            padded = np.zeros((150, 4))
            padded[:len(sorted_valid)] = sorted_valid
            padded[:len(sorted_valid), 3] = 1  # set mask
            target_particles.append(torch.tensor(padded, dtype=torch.float))

            original_jets.append((jet_particles, jet_info))

        except Exception as e:
            # FIXED: Better error logging with error type tracking
            error_type = type(e).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            if failed_count < 5:  # Print first few errors for debugging
                print(f"Error processing jet {i}: {error_type}: {e}")
            failed_count += 1
            continue

    print(f"\nCollection complete!")
    print(f"Successfully collected: {len(graph_data)} jets")
    print(f"Failed: {failed_count} jets")
    # FIXED: Print detailed error breakdown
    if error_types:
        print("Error breakdown:")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")

    return graph_data, target_particles, original_jets

# --- 2. Graph Neural Network Model ---

class SimpleChebNet(nn.Module):
    def __init__(self, in_dim=None, hidden_dim=None, K=None, out_dim=None):
        super().__init__()
        # FIXED: Use config defaults if not provided
        in_dim = in_dim if in_dim is not None else config.gnn_in_dim
        hidden_dim = hidden_dim if hidden_dim is not None else config.gnn_hidden_dim
        K = K if K is not None else config.gnn_K
        out_dim = out_dim if out_dim is not None else config.gnn_out_dim
        
        self.conv1 = ChebConv(in_dim, hidden_dim, K)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K)
        self.conv4 = ChebConv(hidden_dim, out_dim, K)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(F.leaky_relu(x))
        x = self.conv2(x, edge_index)
        x = self.bn2(F.leaky_relu(x))
        x = self.conv3(x, edge_index)
        x = self.bn3(F.leaky_relu(x))
        x = self.conv4(x, edge_index)
        x = F.normalize(x, p=2, dim=1)

        if batch is None:
            mean_p = x.mean(dim=0)
            max_p = x.max(dim=0)[0]
            min_p = -(-x).max(dim=0)[0]
            sum_p = x.sum(dim=0)
            pooled = torch.cat([mean_p, max_p, min_p, sum_p], dim=-1)
        else:
            mean_p = global_mean_pool(x, batch)
            max_p = global_max_pool(x, batch)
            min_p = -global_max_pool(-x, batch)
            sum_p = global_add_pool(x, batch)
            pooled = torch.cat([mean_p, max_p, min_p, sum_p], dim=1)

        return pooled

# --- 3. Diffusion Model with Batching ---

class SimpleDiffusionMLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None):
        super().__init__()
        # FIXED: Use config default if not provided
        hidden_dim = hidden_dim if hidden_dim is not None else config.diffusion_hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(emb_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

class DiffusionDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

def beta(t, beta_0=1e-4, beta_1=0.02):
    """Linear beta schedule"""
    return beta_0 + t * (beta_1 - beta_0)

def alpha_bar(t, beta_0=1e-4, beta_1=0.02):
    return torch.exp(-(beta_0 * t + 0.5 * (beta_1 - beta_0) * t * t))

def train_diffusion_model_batched(embeddings, n_steps=None, epochs=None, batch_size=None, checkpoint_path=None):
    """Train diffusion model on embeddings with proper batching"""
    # FIXED: Use config defaults if not provided
    n_steps = n_steps if n_steps is not None else config.diffusion_n_steps
    epochs = epochs if epochs is not None else config.diffusion_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_dim = embeddings.shape[1]
    model = SimpleDiffusionMLP(emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.diffusion_lr)

    # FIXED: Load checkpoint if resuming
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    dataset = DiffusionDataset(embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Training diffusion model with {len(embeddings)} embeddings...")
    print(f"Batch size: {batch_size}, Batches per epoch: {len(dataloader)}")

    model.train()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            batch_size_actual = batch.size(0)

            t = torch.rand((batch_size_actual, 1), device=device)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar(t))
            sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar(t))

            noise = torch.randn_like(batch)
            noisy_data = batch * sqrt_alpha_bar_t + noise * sqrt_1_minus_alpha_bar_t

            pred_noise = model(noisy_data, t)
            loss = ((pred_noise - noise)**2).mean()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches
        if epoch % 10 == 0:
            print(f"Diffusion Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        # FIXED: Save checkpoint periodically
        if config.save_checkpoints and (epoch + 1) % config.save_every_n_epochs == 0:
            import os
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{config.checkpoint_dir}/diffusion_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        torch.cuda.empty_cache()

    return model

def sample_diffusion_model_batched(model, emb_dim, n_samples=10, n_steps=200, batch_size=None):
    """Sample new embeddings from trained diffusion model with batching using deterministic DDIM"""
    batch_size = batch_size if batch_size is not None else config.batch_size
    
    device = next(model.parameters()).device
    model.eval()

    all_samples = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            x = torch.randn((current_batch_size, emb_dim), device=device)

            for step in range(n_steps, 0, -1):
                t_value = step / n_steps
                s_value = (step - 1) / n_steps
                t = torch.full((current_batch_size, 1), t_value, device=device)
                alpha_t = alpha_bar(t)
                alpha_s = alpha_bar(torch.full((current_batch_size, 1), s_value, device=device))
                noise_pred = model(x, t)

                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_alpha_s = torch.sqrt(alpha_s)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                sqrt_one_minus_alpha_s = torch.sqrt(1 - alpha_s)

                pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                x = sqrt_alpha_s * pred_x0 + sqrt_one_minus_alpha_s * noise_pred

            all_samples.append(x.cpu().numpy())

    return np.concatenate(all_samples, axis=0)

# --- 4. Enhanced Decoder Model for Particle Generation ---

class JetGraphDataset(Dataset):
    def __init__(self, graph_data, target_particles):
        self.graph_data = graph_data
        self.target_particles = target_particles

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx], self.target_particles[idx]

def collate_fn(batch):
    graphs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    batched_graph = Batch.from_data_list(graphs)
    batched_targets = torch.stack(targets)
    return batched_graph, batched_targets

class ParticleDecoder(nn.Module):
    def __init__(self, emb_dim, max_particles=None, particle_dim=None, hidden_dims=None):
        super().__init__()
        # FIXED: Use config defaults if not provided
        max_particles = max_particles if max_particles is not None else config.max_particles
        particle_dim = particle_dim if particle_dim is not None else config.particle_dim
        hidden_dims = hidden_dims if hidden_dims is not None else config.decoder_hidden_dims
        
        self.max_particles = max_particles
        self.particle_dim = particle_dim
        self.output_size = max_particles * particle_dim

        layers = []
        current_dim = emb_dim

        layers.extend([
            nn.Linear(current_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        ])
        current_dim = hidden_dims[0]

        layers.extend([
            nn.Linear(current_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        ])
        current_dim = hidden_dims[1]

        layers.extend([
            nn.Linear(current_dim, self.output_size)
        ])

        self.main_net = nn.Sequential(*layers)

        self.residual_projection = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.output_size)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, z):
        main_out = self.main_net(z)
        residual_out = self.residual_projection(z)
        combined = main_out + 0.1 * residual_out
        output = combined.view(-1, self.max_particles, self.particle_dim)

        eta = torch.tanh(output[:, :, 0]) * 2.0
        phi = torch.tanh(output[:, :, 1]) * 2.0
        pt = F.softplus(output[:, :, 2])

        mask_logits = output[:, :, 3]
        mask = torch.sigmoid(mask_logits * 2.0)

        output = torch.stack([eta, phi, pt, mask], dim=2)
        return output

def train_autoencoder(gnn, decoder, graph_data, target_particles, epochs=None, batch_size=None, checkpoint_path=None):
    """Train GNN and decoder jointly as autoencoder"""
    # FIXED: Use config defaults if not provided
    epochs = epochs if epochs is not None else config.autoencoder_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training autoencoder on device: {device}")

    dataset = JetGraphDataset(graph_data, target_particles)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    optimizer = optim.AdamW(list(gnn.parameters()) + list(decoder.parameters()), lr=config.autoencoder_lr, weight_decay=config.autoencoder_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs//4, eta_min=1e-6)

    # FIXED: Load checkpoint if resuming
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        gnn.load_state_dict(checkpoint['gnn_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()

    print(f"Training autoencoder with {len(dataset)} samples, batch size {batch_size}")
    print(f"Total parameters: {sum(p.numel() for p in gnn.parameters()) + sum(p.numel() for p in decoder.parameters()):,}")

    gnn.train()
    decoder.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (batch_graph, batch_target) in enumerate(loader):
            batch_graph = batch_graph.to(device)
            batch_target = batch_target.to(device)

            pooled = gnn(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
            pred = decoder(pooled)

            eta_loss = mse_criterion(pred[:, :, 0], batch_target[:, :, 0])
            phi_loss = mse_criterion(pred[:, :, 1], batch_target[:, :, 1])
            pt_loss = mse_criterion(pred[:, :, 2], batch_target[:, :, 2])
            mask_loss = bce_criterion(pred[:, :, 3], batch_target[:, :, 3])
            target_num = torch.sum(batch_target[:, :, 3], dim=1)
            pred_num = torch.sum(pred[:, :, 3], dim=1)
            count_loss = nn.MSELoss()(pred_num, target_num)
            loss = eta_loss*4 + phi_loss*4 + pt_loss + mask_loss*2 + 0.1 * count_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(decoder.parameters()), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_batches

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"  Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Autoencoder Epoch {epoch}/{epochs}")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")
            print(f"  Best Loss: {best_loss:.6f}, Patience: {patience_counter}")

        # FIXED: Save checkpoint periodically
        if config.save_checkpoints and (epoch + 1) % config.save_every_n_epochs == 0:
            import os
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{config.checkpoint_dir}/autoencoder_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': gnn.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

        torch.cuda.empty_cache()

    return gnn, decoder

# --- 5. Visualization Functions ---

def visualize_particle_jets(particle_data_list, titles, n_cols=2):
    """Visualize jets as scatter plots using particle data"""
    n_jets = len(particle_data_list)
    n_rows = (n_jets + n_cols - 1) // n_cols

    plt.style.use('default')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
    fig.patch.set_facecolor('white')

    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (particles, title) in enumerate(zip(particle_data_list, titles)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        valid_mask = particles[:, 3] > 0.5
        if valid_mask.sum() == 0:
            ax.text(0.5, 0.5, 'No valid particles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue

        valid_particles = particles[valid_mask]
        eta = valid_particles[:, 0]
        phi = valid_particles[:, 1]
        pt = valid_particles[:, 2]

        scatter = ax.scatter(eta, phi, c=pt, s=50, cmap='viridis', alpha=0.7)

        ax.set_title(f"{title}\nParticles: {len(valid_particles)}", fontsize=12, fontweight='bold')
        ax.set_xlabel('η_rel', fontsize=10)
        ax.set_ylabel('φ_rel', fontsize=10)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('p_T', fontsize=10)

        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)

    for i in range(n_jets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

def calculate_kl_divergence(real_data, generated_data, n_bins=100):
    """KL divergence following the paper's methodology exactly"""
    real_data = real_data[np.isfinite(real_data)]
    generated_data = generated_data[np.isfinite(generated_data)]

    if len(real_data) == 0 or len(generated_data) == 0:
        return float('inf')

    min_val, max_val = real_data.min(), real_data.max()
    generated_data_clipped = np.clip(generated_data, min_val, max_val)

    bin_edges = np.quantile(real_data, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10

    real_hist, _ = np.histogram(real_data, bins=bin_edges)
    gen_hist, _ = np.histogram(generated_data_clipped, bins=bin_edges)

    real_prob = real_hist / real_hist.sum()
    gen_prob = gen_hist / gen_hist.sum()

    epsilon = 1e-10
    real_prob = np.maximum(real_prob, epsilon)
    gen_prob = np.maximum(gen_prob, epsilon)

    kl_div = np.sum(real_prob * np.log(real_prob / gen_prob))
    return kl_div

def compare_particle_reconstruction(original_particles, reconstructed_particles, idx):
    """Compare original and reconstructed particle data"""
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    orig_valid = original_particles[:, 3] == 1
    if orig_valid.sum() > 0:
        orig_valid_particles = original_particles[orig_valid]
        scatter1 = ax1.scatter(orig_valid_particles[:, 0], orig_valid_particles[:, 1],
                             c=orig_valid_particles[:, 2], s=50, cmap='viridis', alpha=0.7)
        ax1.set_title(f'Original Jet {idx+1} ({orig_valid.sum()} particles)')
        ax1.set_xlabel('η_rel')
        ax1.set_ylabel('φ_rel')
        ax1.set_xlim(-0.8, 0.8)
        ax1.set_ylim(-0.8, 0.8)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='p_T')

    recon_valid = reconstructed_particles[:, 3] > 0.5
    if recon_valid.sum() > 0:
        recon_valid_particles = reconstructed_particles[recon_valid]
        scatter2 = ax2.scatter(recon_valid_particles[:, 0], recon_valid_particles[:, 1],
                             c=recon_valid_particles[:, 2], s=50, cmap='viridis', alpha=0.7)
        ax2.set_title(f'Reconstructed Jet {idx+1} ({recon_valid.sum()} particles)')
        ax2.set_xlabel('η_rel')
        ax2.set_ylabel('φ_rel')
        ax2.set_xlim(-0.8, 0.8)
        ax2.set_ylim(-0.8, 0.8)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='p_T')

    if orig_valid.sum() > 0 and recon_valid.sum() > 0:
        orig_valid_particles = original_particles[orig_valid]
        recon_valid_particles = reconstructed_particles[recon_valid]
        
        ax3.hist(orig_valid_particles[:, 0], bins=100, alpha=0.5, label='Original', density=True)
        ax3.hist(recon_valid_particles[:, 0], bins=100, alpha=0.5, label='Reconstructed', density=True)
        ax3.set_xlabel('η_rel')
        ax3.set_ylabel('Density')
        ax3.set_title('η Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.hist(orig_valid_particles[:, 2], bins=100, alpha=0.5, label='Original', density=True)
        ax4.hist(recon_valid_particles[:, 2], bins=100, alpha=0.5, label='Reconstructed', density=True)
        ax4.set_xlabel('p_T')
        ax4.set_ylabel('Density')
        ax4.set_title('p_T Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ws_eta = wasserstein_distance(orig_valid_particles[:, 0], recon_valid_particles[:, 0])
        ws_phi = wasserstein_distance(orig_valid_particles[:, 1], recon_valid_particles[:, 1])
        ws_pt = wasserstein_distance(orig_valid_particles[:, 2], recon_valid_particles[:, 2])
        kl_eta = calculate_kl_divergence(orig_valid_particles[:, 0], recon_valid_particles[:, 0])
        kl_phi = calculate_kl_divergence(orig_valid_particles[:, 1], recon_valid_particles[:, 1])
        kl_pt = calculate_kl_divergence(orig_valid_particles[:, 2], recon_valid_particles[:, 2])

        print(f"Jet {idx+1} Metrics:")
        print(f"Wasserstein η: {ws_eta:.4f}, φ: {ws_phi:.4f}, p_T: {ws_pt:.4f}")
        print(f"KL Divergence η: {kl_eta:.4f}, φ: {kl_phi:.4f}, p_T: {kl_pt:.4f}")

    plt.tight_layout()
    plt.show()

    print(f"Jet {idx+1} Statistics:")
    print(f"Original particles: {orig_valid.sum()}")
    print(f"Reconstructed particles: {recon_valid.sum()}")
    if orig_valid.sum() > 0:
        print(f"Original η range: [{orig_valid_particles[:, 0].min():.3f}, {orig_valid_particles[:, 0].max():.3f}]")
        print(f"Original p_T range: [{orig_valid_particles[:, 2].min():.3f}, {orig_valid_particles[:, 2].max():.3f}]")
    if recon_valid.sum() > 0:
        print(f"Reconstructed η range: [{recon_valid_particles[:, 0].min():.3f}, {recon_valid_particles[:, 0].max():.3f}]")
        print(f"Reconstructed p_T range: [{recon_valid_particles[:, 2].min():.3f}, {recon_valid_particles[:, 2].max():.3f}]")
    print()

# --- 6. Main Pipeline with JetNet Data ---

def main_pipeline_jetnet(batch_size=None):
    """Main pipeline using JetNet dataset"""
    batch_size = batch_size if batch_size is not None else config.batch_size

    print("=== Loading JetNet data ===")
    particle_data, jet_data = load_jetnet_data()

    print("\n=== Collecting graph data ===")
    # FIXED: Uses config.max_jets (3200) instead of 50000
    graph_data, target_particles, original_jets = collect_graph_and_targets(particle_data, jet_data)

    if len(graph_data) == 0:
        print("No jets were successfully collected!")
        return None

    print("\n=== Training autoencoder ===")
    gnn = SimpleChebNet().to(device)
    decoder = ParticleDecoder(emb_dim=128).to(device)
    gnn, decoder = train_autoencoder(gnn, decoder, graph_data, target_particles, batch_size=batch_size)

    print("\n=== Extracting pooled embeddings ===")
    gnn.eval()
    all_pooled_embeddings = []
    for data in graph_data:
        data = data.to(device)
        with torch.no_grad():
            pooled = gnn(data.x, data.edge_index).cpu().numpy()
        all_pooled_embeddings.append(pooled)
    all_pooled_embeddings = np.stack(all_pooled_embeddings)
    
    # FIXED: Save embeddings for potential resumption
    if config.save_checkpoints:
        import os
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        np.save(f"{config.checkpoint_dir}/embeddings.npy", all_pooled_embeddings)
        print(f"Saved embeddings to {config.checkpoint_dir}/embeddings.npy")

    print(f"\n=== Training diffusion model (batch_size={batch_size}) ===")
    diff_model = train_diffusion_model_batched(all_pooled_embeddings, batch_size=batch_size)

    print("\n=== Generating new samples ===")
    n_samples = 8
    new_embeddings = sample_diffusion_model_batched(
        diff_model, emb_dim=all_pooled_embeddings.shape[1],
        n_samples=n_samples, batch_size=batch_size
    )

    decoder.eval()
    with torch.no_grad():
        generated_particles = []
        for i in range(0, len(new_embeddings), batch_size):
            batch_embeddings = new_embeddings[i:i+batch_size]
            batch_embeddings_torch = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
            batch_particles = decoder(batch_embeddings_torch).cpu().numpy()
            generated_particles.append(batch_particles)

        generated_particles = np.concatenate(generated_particles, axis=0)

    print(f"\n=== Visualizing {n_samples} generated jets ===")
    titles = [f'Generated Jet {i+1}' for i in range(n_samples)]
    visualize_particle_jets(generated_particles, titles)

    print("\n=== Particle decoder reconstruction comparison ===")
    n_compare = min(3, len(all_pooled_embeddings))
    comparison_embeddings = all_pooled_embeddings[:n_compare]

    with torch.no_grad():
        comparison_embeddings_torch = torch.tensor(comparison_embeddings, dtype=torch.float32).to(device)
        decoder_reconstructions = decoder(comparison_embeddings_torch).cpu().numpy()

    for i in range(n_compare):
        original_particles, _ = original_jets[i]
        decoder_reconstruction = decoder_reconstructions[i]
        compare_particle_reconstruction(original_particles, decoder_reconstruction, i)

    ws_eta_list, ws_phi_list, ws_pt_list = [], [], []
    kl_eta_list, kl_phi_list, kl_pt_list = [], [], []

    for i in range(n_compare):
        original_particles, _ = original_jets[i]
        decoder_reconstruction = decoder_reconstructions[i]
        orig_valid = original_particles[:, 3] == 1
        recon_valid = decoder_reconstruction[:, 3] > 0.5

        if orig_valid.sum() > 0 and recon_valid.sum() > 0:
            orig_valid_particles = original_particles[orig_valid]
            recon_valid_particles = decoder_reconstruction[recon_valid]

            ws_eta = wasserstein_distance(orig_valid_particles[:, 0], recon_valid_particles[:, 0])
            ws_phi = wasserstein_distance(orig_valid_particles[:, 1], recon_valid_particles[:, 1])
            ws_pt = wasserstein_distance(orig_valid_particles[:, 2], recon_valid_particles[:, 2])

            kl_eta = calculate_kl_divergence(orig_valid_particles[:, 0], recon_valid_particles[:, 0])
            kl_phi = calculate_kl_divergence(orig_valid_particles[:, 1], recon_valid_particles[:, 1])
            kl_pt = calculate_kl_divergence(orig_valid_particles[:, 2], recon_valid_particles[:, 2])

            ws_eta_list.append(ws_eta)
            ws_phi_list.append(ws_phi)
            ws_pt_list.append(ws_pt)
            kl_eta_list.append(kl_eta)
            kl_phi_list.append(kl_phi)
            kl_pt_list.append(kl_pt)

    if ws_eta_list:
        avg_ws_eta = sum(ws_eta_list) / len(ws_eta_list)
        avg_ws_phi = sum(ws_phi_list) / len(ws_phi_list)
        avg_ws_pt = sum(ws_pt_list) / len(ws_pt_list)

        avg_kl_eta = sum(kl_eta_list) / len(kl_eta_list)
        avg_kl_phi = sum(kl_phi_list) / len(kl_phi_list)
        avg_kl_pt = sum(kl_pt_list) / len(kl_pt_list)

        print("\nComponent-wise Average Metrics (over comparisons):")
        print(f"Wasserstein - η: {avg_ws_eta:.4f}, φ: {avg_ws_phi:.4f}, p_T: {avg_ws_pt:.4f}")
        print(f"KL Divergence - η: {avg_kl_eta:.4f}, φ: {avg_kl_phi:.4f}, p_T: {avg_kl_pt:.4f}")

    print("\n=== Random real jets for comparison ===")
    n_compare = 3
    random_indices = np.random.choice(len(original_jets), n_compare, replace=False)
    random_jets = [original_jets[idx][0] for idx in random_indices]
    random_titles = [f'Random Real Jet (Index {idx})' for idx in random_indices]
    visualize_particle_jets(random_jets, random_titles)

    print(f"\n=== Summary ===")
    print(f"Total jets in dataset: {len(particle_data)}")
    print(f"Successfully processed: {len(all_pooled_embeddings)}")
    print(f"Processing success rate: {len(all_pooled_embeddings)/len(particle_data)*100:.1f}%")
    print(f"Embedding dimension: {all_pooled_embeddings.shape[1]}")
    print(f"Generated samples: {n_samples}")
    print(f"Max particles per jet: 150")
    print(f"Batch size used for training: {batch_size}")

    return {
        'embeddings': all_pooled_embeddings,
        'original_jets': original_jets,
        'diffusion_model': diff_model,
        'decoder': decoder,
        'generated_particles': generated_particles,
        'particle_data': particle_data,
        'jet_data': jet_data
    }

# --- 7. Additional Utility Functions ---

def plot_jetnet_sample_jets(particle_data, jet_data, jet_nums=[0, 1, 2]):
    """Plot sample jets from JetNet dataset for initial visualization"""
    for idx in jet_nums:
        jet_particles = particle_data[idx]
        jet_info = jet_data[idx]

        valid = jet_particles[jet_particles[:, 3] == 1]

        if len(valid) == 0:
            print(f"Jet {idx} has no valid particles, skipping...")
            continue

        eta = valid[:, 0]
        phi = valid[:, 1]
        pt = valid[:, 2]
        energy = pt * jet_info[0]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(eta, phi, c=energy, s=50, cmap='viridis', alpha=0.7)
        plt.title(f"Jet {idx}\npT: {jet_info[0]:.2f} GeV, eta: {jet_info[1]:.2f}")
        plt.xlabel("eta_rel")
        plt.ylabel("phi_rel")
        plt.colorbar(scatter, label="Energy")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(pt, bins=20, alpha=0.7, label='p_T distribution')
        plt.xlabel('p_T')
        plt.ylabel('Count')
        plt.title(f'p_T Distribution (Jet {idx})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"jetnet_jet_{idx}.png")
        plt.show()

        print(f"Jet {idx} Stats:")
        print("Particles:", len(valid))
        print("Jet info:", jet_info)
        print("eta mean:", np.mean(eta))
        print("phi mean:", np.mean(phi))
        print("pt mean:", np.mean(pt))
        print()

def analyze_jetnet_dataset(particle_data, jet_data):
    """Analyze the JetNet dataset structure and statistics"""
    print("=== JetNet Dataset Analysis ===")
    print(f"Particle data shape: {particle_data.shape}")
    print(f"Jet data shape: {jet_data.shape}")

    valid_particles_per_jet = []
    for i in range(len(particle_data)):
        valid_count = (particle_data[i][:, 3] == 1).sum()
        valid_particles_per_jet.append(valid_count)

    valid_particles_per_jet = np.array(valid_particles_per_jet)

    print(f"\nParticle Statistics:")
    print(f"Average particles per jet: {valid_particles_per_jet.mean():.2f}")
    print(f"Std particles per jet: {valid_particles_per_jet.std():.2f}")
    print(f"Min particles per jet: {valid_particles_per_jet.min()}")
    print(f"Max particles per jet: {valid_particles_per_jet.max()}")

    print(f"\nJet Statistics:")
    print(f"Jet pT range: [{jet_data[:, 0].min():.2f}, {jet_data[:, 0].max():.2f}] GeV")
    print(f"Jet eta range: [{jet_data[:, 1].min():.2f}, {jet_data[:, 1].max():.2f}]")
    print(f"Jet mass range: [{jet_data[:, 2].min():.2f}, {jet_data[:, 2].max():.2f}] GeV")
    print(f"Reported num_particles range: [{jet_data[:, 3].min():.0f}, {jet_data[:, 3].max():.0f}]")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.hist(valid_particles_per_jet, bins=50, alpha=0.7)
    plt.xlabel('Number of particles')
    plt.ylabel('Count')
    plt.title('Particles per jet distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.hist(jet_data[:, 0], bins=50, alpha=0.7)
    plt.xlabel('Jet pT (GeV)')
    plt.ylabel('Count')
    plt.title('Jet pT distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.hist(jet_data[:, 1], bins=50, alpha=0.7)
    plt.xlabel('Jet eta')
    plt.ylabel('Count')
    plt.title('Jet eta distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.hist(jet_data[:, 2], bins=50, alpha=0.7)
    plt.xlabel('Jet mass (GeV)')
    plt.ylabel('Count')
    plt.title('Jet mass distribution')
    plt.grid(True, alpha=0.3)

    all_valid_particles = []
    for i in range(min(1000, len(particle_data))):
        valid_mask = particle_data[i][:, 3] == 1
        if valid_mask.sum() > 0:
            all_valid_particles.append(particle_data[i][valid_mask])

    if all_valid_particles:
        all_valid_particles = np.concatenate(all_valid_particles, axis=0)

        plt.subplot(2, 3, 5)
        plt.hist(all_valid_particles[:, 0], bins=50, alpha=0.7)
        plt.xlabel('eta_rel')
        plt.ylabel('Count')
        plt.title('Particle eta_rel distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 6)
        plt.hist(all_valid_particles[:, 2], bins=50, alpha=0.7)
        plt.xlabel('pt_rel')
        plt.ylabel('Count')
        plt.title('Particle pt_rel distribution')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- 8. Run the pipeline ---

if __name__ == "__main__":
    import os
    
    BATCH_SIZE = config.batch_size

    print("Loading JetNet data for analysis...")
    particle_data, jet_data = load_jetnet_data()

    print("\n=== Sample JetNet jets ===")
    plot_jetnet_sample_jets(particle_data, jet_data, [0, 1, 2])

    print("\n=== Running complete JetNet pipeline ===")
    results = main_pipeline_jetnet(batch_size=BATCH_SIZE)

    if results is not None:
        # FIXED: Save final models
        if config.save_checkpoints:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(results['diffusion_model'].state_dict(), f'{config.checkpoint_dir}/diffusion_model_final.pth')
            torch.save(results['decoder'].state_dict(), f'{config.checkpoint_dir}/decoder_model_final.pth')
            np.save(f'{config.checkpoint_dir}/generated_particles.npy', results['generated_particles'])
            print(f"\nSaved final models to {config.checkpoint_dir}/")

        print("\nJetNet pipeline completed successfully!")

        print("\n=== Generated Jet Analysis ===")
        generated_particles = results['generated_particles']

        generated_valid_counts = []
        for i in range(len(generated_particles)):
            valid_count = (generated_particles[i][:, 3] > 0.5).sum()
            generated_valid_counts.append(valid_count)

        generated_valid_counts = np.array(generated_valid_counts)

        print(f"Generated jets - Average particles: {generated_valid_counts.mean():.2f}")
        print(f"Generated jets - Std particles: {generated_valid_counts.std():.2f}")
        print(f"Generated jets - Range: [{generated_valid_counts.min()}, {generated_valid_counts.max()}]")

        original_valid_counts = []
        for jet_particles, _ in results['original_jets'][:len(generated_particles)]:
            valid_count = (jet_particles[:, 3] == 1).sum()
            original_valid_counts.append(valid_count)

        original_valid_counts = np.array(original_valid_counts)

        print(f"Original jets - Average particles: {original_valid_counts.mean():.2f}")
        print(f"Original jets - Std particles: {original_valid_counts.std():.2f}")
        print(f"Original jets - Range: [{original_valid_counts.min()}, {original_valid_counts.max()}]")

    else:
        print("\nPipeline failed - no jets were successfully processed.")