# Custom Training Examples

This page provides advanced examples for customizing Free Transformer training.

## Custom Loss Functions

### Adding Custom Loss Components

```python
import torch
import torch.nn.functional as F
from free_transformer.losses import free_transformer_loss

def custom_free_transformer_loss(logits, z_logits, targets, config, **kwargs):
    """Custom loss function with additional components."""
    
    # Standard Free Transformer loss
    base_loss = free_transformer_loss(
        logits=logits,
        z_logits=z_logits,
        targets=targets,
        latent_dim=config.latent_dim,
        kl_weight=config.kl_weight,
        free_bits=config.free_bits
    )
    
    # Custom loss components
    custom_losses = {}
    
    # 1. Diversity loss - encourage different plans for different sequences
    if 'diversity_weight' in kwargs:
        diversity_loss = compute_diversity_loss(z_logits)
        custom_losses['diversity_loss'] = diversity_loss
        base_loss['total_loss'] += kwargs['diversity_weight'] * diversity_loss
    
    # 2. Consistency loss - similar inputs should have similar plans
    if 'consistency_weight' in kwargs and 'similarity_matrix' in kwargs:
        consistency_loss = compute_consistency_loss(z_logits, kwargs['similarity_matrix'])
        custom_losses['consistency_loss'] = consistency_loss
        base_loss['total_loss'] += kwargs['consistency_weight'] * consistency_loss
    
    # 3. Sparsity loss - encourage sparse plan usage
    if 'sparsity_weight' in kwargs:
        sparsity_loss = compute_sparsity_loss(z_logits)
        custom_losses['sparsity_loss'] = sparsity_loss
        base_loss['total_loss'] += kwargs['sparsity_weight'] * sparsity_loss
    
    # Combine all losses
    base_loss.update(custom_losses)
    return base_loss

def compute_diversity_loss(z_logits):
    """Encourage diversity in latent plans."""
    # Convert to probabilities
    probs = torch.sigmoid(z_logits)
    
    # Compute pairwise distances
    batch_size = probs.size(0)
    distances = torch.cdist(probs, probs, p=2)
    
    # Encourage larger distances (more diversity)
    diversity_loss = -distances.mean()
    return diversity_loss

def compute_consistency_loss(z_logits, similarity_matrix):
    """Encourage consistent plans for similar inputs."""
    probs = torch.sigmoid(z_logits)
    
    # Compute plan distances
    plan_distances = torch.cdist(probs, probs, p=2)
    
    # Consistency loss: similar inputs should have similar plans
    consistency_loss = (similarity_matrix * plan_distances).mean()
    return consistency_loss

def compute_sparsity_loss(z_logits):
    """Encourage sparse plan usage."""
    probs = torch.sigmoid(z_logits)
    
    # L1 penalty on probabilities (encourages 0 or 1)
    sparsity_loss = torch.abs(probs - 0.5).mean()
    return -sparsity_loss  # Negative because we want to maximize sparsity
```

### Using Custom Loss in Training

```python
def train_with_custom_loss(model, dataloader, config):
    """Training loop with custom loss function."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            tokens = batch['input_ids']
            
            # Forward pass
            logits, z_logits = model(tokens, mode='training')
            
            # Compute similarity matrix (example: based on first few tokens)
            similarity_matrix = compute_input_similarity(tokens)
            
            # Custom loss with additional components
            loss_dict = custom_free_transformer_loss(
                logits=logits,
                z_logits=z_logits,
                targets=tokens,
                config=config,
                diversity_weight=0.1,
                consistency_weight=0.05,
                sparsity_weight=0.02,
                similarity_matrix=similarity_matrix
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Recon Loss: {loss_dict['recon_loss'].item():.4f}")
                print(f"  KL Loss: {loss_dict['kl_loss'].item():.4f}")
                if 'diversity_loss' in loss_dict:
                    print(f"  Diversity Loss: {loss_dict['diversity_loss'].item():.4f}")

def compute_input_similarity(tokens):
    """Compute similarity matrix based on input tokens."""
    batch_size = tokens.size(0)
    similarity_matrix = torch.zeros(batch_size, batch_size)
    
    for i in range(batch_size):
        for j in range(batch_size):
            # Simple similarity: overlap in first 10 tokens
            overlap = (tokens[i, :10] == tokens[j, :10]).float().mean()
            similarity_matrix[i, j] = overlap
    
    return similarity_matrix.to(tokens.device)
```

## Custom Training Schedules

### KL Annealing Schedule

```python
class KLAnnealingScheduler:
    """Custom KL weight annealing scheduler."""
    
    def __init__(self, initial_weight=0.0, final_weight=1.0, annealing_steps=10000, schedule_type='linear'):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.annealing_steps = annealing_steps
        self.schedule_type = schedule_type
        self.step_count = 0
    
    def get_kl_weight(self):
        """Get current KL weight."""
        if self.step_count >= self.annealing_steps:
            return self.final_weight
        
        progress = self.step_count / self.annealing_steps
        
        if self.schedule_type == 'linear':
            weight = self.initial_weight + progress * (self.final_weight - self.initial_weight)
        elif self.schedule_type == 'cosine':
            weight = self.initial_weight + 0.5 * (self.final_weight - self.initial_weight) * (1 - torch.cos(torch.tensor(progress * 3.14159)))
        elif self.schedule_type == 'exponential':
            weight = self.initial_weight * (self.final_weight / self.initial_weight) ** progress
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return float(weight)
    
    def step(self):
        """Update step count."""
        self.step_count += 1

# Usage in training loop
kl_scheduler = KLAnnealingScheduler(
    initial_weight=0.0,
    final_weight=0.1,
    annealing_steps=5000,
    schedule_type='cosine'
)

for batch in dataloader:
    # Get current KL weight
    current_kl_weight = kl_scheduler.get_kl_weight()
    
    # Use in loss computation
    loss_dict = free_transformer_loss(
        logits=logits,
        z_logits=z_logits,
        targets=targets,
        latent_dim=config.latent_dim,
        kl_weight=current_kl_weight,
        free_bits=config.free_bits
    )
    
    # Update scheduler
    kl_scheduler.step()
```

### Free Bits Scheduling

```python
class FreeBitsScheduler:
    """Dynamic free bits scheduling."""
    
    def __init__(self, initial_bits=2.0, final_bits=0.5, schedule_steps=8000):
        self.initial_bits = initial_bits
        self.final_bits = final_bits
        self.schedule_steps = schedule_steps
        self.step_count = 0
        self.kl_history = []
    
    def get_free_bits(self, current_kl=None):
        """Get current free bits threshold."""
        if current_kl is not None:
            self.kl_history.append(current_kl)
        
        # Adaptive scheduling based on KL history
        if len(self.kl_history) > 100:
            recent_kl = torch.tensor(self.kl_history[-100:]).mean()
            
            # If KL is too low, increase free bits
            if recent_kl < 0.1:
                return min(self.initial_bits, self.get_scheduled_bits() + 0.5)
            # If KL is too high, decrease free bits
            elif recent_kl > 2.0:
                return max(self.final_bits, self.get_scheduled_bits() - 0.5)
        
        return self.get_scheduled_bits()
    
    def get_scheduled_bits(self):
        """Get scheduled free bits value."""
        if self.step_count >= self.schedule_steps:
            return self.final_bits
        
        progress = self.step_count / self.schedule_steps
        return self.initial_bits + progress * (self.final_bits - self.initial_bits)
    
    def step(self):
        """Update step count."""
        self.step_count += 1
```

## Custom Data Loading

### Advanced Data Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AdvancedSyntheticDataset(Dataset):
    """Advanced synthetic dataset with custom features."""
    
    def __init__(self, sequences, config):
        self.sequences = sequences
        self.config = config
        self.max_length = config.max_seq_len
        
        # Precompute sequence features
        self.sequence_features = self._compute_features()
    
    def _compute_features(self):
        """Compute features for each sequence."""
        features = []
        for seq in self.sequences:
            feature = {
                'length': len(seq),
                'unique_tokens': len(set(seq)),
                'repetition_ratio': self._compute_repetition_ratio(seq),
                'entropy': self._compute_entropy(seq),
            }
            features.append(feature)
        return features
    
    def _compute_repetition_ratio(self, sequence):
        """Compute repetition ratio in sequence."""
        if len(sequence) < 2:
            return 0.0
        
        repeats = sum(1 for i in range(1, len(sequence)) if sequence[i] == sequence[i-1])
        return repeats / (len(sequence) - 1)
    
    def _compute_entropy(self, sequence):
        """Compute entropy of sequence."""
        if not sequence:
            return 0.0
        
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        
        entropy = 0
        for count in counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        features = self.sequence_features[idx]
        
        # Truncate or pad sequence
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            # Pad with special token (assuming 0 is padding)
            sequence = sequence + [0] * (self.max_length - len(sequence))
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(self.sequences[idx]) + [0] * max(0, self.max_length - len(self.sequences[idx])), dtype=torch.long),
            'features': features,
            'original_length': len(self.sequences[idx])
        }

class CurriculumDataLoader:
    """Data loader with curriculum learning."""
    
    def __init__(self, dataset, batch_size, curriculum_stages):
        self.dataset = dataset
        self.batch_size = batch_size
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.stage_step = 0
    
    def get_current_dataloader(self):
        """Get data loader for current curriculum stage."""
        stage_config = self.curriculum_stages[self.current_stage]
        
        # Filter dataset based on current stage criteria
        filtered_indices = self._filter_indices(stage_config)
        
        # Create subset
        subset = torch.utils.data.Subset(self.dataset, filtered_indices)
        
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
    
    def _filter_indices(self, stage_config):
        """Filter dataset indices based on stage criteria."""
        indices = []
        
        for idx, features in enumerate(self.dataset.sequence_features):
            # Check if sequence meets stage criteria
            if (features['length'] >= stage_config.get('min_length', 0) and
                features['length'] <= stage_config.get('max_length', float('inf')) and
                features['entropy'] >= stage_config.get('min_entropy', 0) and
                features['entropy'] <= stage_config.get('max_entropy', float('inf'))):
                indices.append(idx)
        
        return indices
    
    def _collate_fn(self, batch):
        """Custom collate function."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'features': [item['features'] for item in batch],
            'original_lengths': [item['original_length'] for item in batch]
        }
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.stage_step = 0
            print(f"Advanced to curriculum stage {self.current_stage}")
    
    def step(self):
        """Update step count and check for stage advancement."""
        self.stage_step += 1
        
        stage_config = self.curriculum_stages[self.current_stage]
        if self.stage_step >= stage_config.get('steps', float('inf')):
            self.advance_stage()

# Usage example
curriculum_stages = [
    {'min_length': 0, 'max_length': 128, 'min_entropy': 0, 'max_entropy': 2, 'steps': 2000},
    {'min_length': 64, 'max_length': 256, 'min_entropy': 1, 'max_entropy': 4, 'steps': 3000},
    {'min_length': 128, 'max_length': 512, 'min_entropy': 2, 'max_entropy': 8, 'steps': 5000},
]

dataset = AdvancedSyntheticDataset(sequences, config)
curriculum_loader = CurriculumDataLoader(dataset, batch_size=32, curriculum_stages=curriculum_stages)
```

## Custom Model Components

### Custom Injection Mechanisms

```python
import torch.nn as nn

class CrossAttentionInjection(nn.Module):
    """Inject plan using cross-attention mechanism."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.num_heads = config.num_heads
        
        # Project plan to hidden dimension
        self.plan_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
    
    def forward(self, decoder_hidden, binary_plan):
        """Apply cross-attention injection."""
        batch_size, seq_len, hidden_dim = decoder_hidden.shape
        
        # Project plan to hidden dimension
        plan_repr = self.plan_projection(binary_plan)  # [batch_size, hidden_dim]
        plan_repr = plan_repr.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention: decoder hidden as query, plan as key/value
        attended, _ = self.cross_attention(
            query=decoder_hidden,
            key=plan_repr,
            value=plan_repr
        )
        
        # Residual connection and normalization
        output = self.norm(decoder_hidden + attended)
        
        # Feedforward
        ff_output = self.feedforward(output)
        output = self.norm(output + ff_output)
        
        return output

class MultiLevelInjection(nn.Module):
    """Inject plan at multiple levels with different transformations."""
    
    def __init__(self, config):
        super().__init__()
        self.num_levels = config.num_injection_levels
        
        # Different projections for each level
        self.level_projections = nn.ModuleList([
            nn.Linear(config.latent_dim, config.hidden_dim)
            for _ in range(self.num_levels)
        ])
        
        # Level-specific gates
        self.level_gates = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(self.num_levels)
        ])
        
        # Attention weights for combining levels
        self.level_attention = nn.Linear(config.hidden_dim, self.num_levels)
    
    def forward(self, decoder_hidden, binary_plan, level_weights=None):
        """Apply multi-level injection."""
        batch_size, seq_len, hidden_dim = decoder_hidden.shape
        
        # Generate level-specific representations
        level_reprs = []
        for i, (proj, gate) in enumerate(zip(self.level_projections, self.level_gates)):
            # Project plan for this level
            level_repr = proj(binary_plan)  # [batch_size, hidden_dim]
            level_repr = level_repr.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Apply level-specific gating
            gate_values = torch.sigmoid(gate(decoder_hidden))
            gated_repr = gate_values * level_repr
            
            level_reprs.append(gated_repr)
        
        # Compute attention weights for combining levels
        if level_weights is None:
            attention_logits = self.level_attention(decoder_hidden)  # [batch_size, seq_len, num_levels]
            level_weights = torch.softmax(attention_logits, dim=-1)
        
        # Combine level representations
        combined_repr = torch.zeros_like(decoder_hidden)
        for i, level_repr in enumerate(level_reprs):
            weight = level_weights[..., i:i+1]  # [batch_size, seq_len, 1]
            combined_repr += weight * level_repr
        
        return decoder_hidden + combined_repr
```

### Custom Encoder Architectures

```python
class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for multi-scale planning."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_levels = config.encoder_levels
        
        # Encoders for different scales
        self.local_encoder = self._build_encoder(config, scale='local')
        self.global_encoder = self._build_encoder(config, scale='global')
        
        # Fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.latent_dim)
    
    def _build_encoder(self, config, scale):
        """Build encoder for specific scale."""
        if scale == 'local':
            # Local encoder: smaller receptive field
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads // 2,
                    dim_feedforward=config.hidden_dim * 2,
                    batch_first=True
                ),
                num_layers=2
            )
        else:
            # Global encoder: larger receptive field
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    batch_first=True
                ),
                num_layers=3
            )
    
    def forward(self, hidden_states):
        """Forward pass with hierarchical encoding."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Local encoding: process overlapping windows
        local_features = self._encode_local(hidden_states)
        
        # Global encoding: process entire sequence
        global_features = self._encode_global(hidden_states)
        
        # Fuse local and global features
        fused_features = self.fusion(torch.cat([local_features, global_features], dim=-1))
        
        # Project to latent dimension
        latent_repr = self.output_projection(fused_features)
        
        return latent_repr
    
    def _encode_local(self, hidden_states):
        """Encode local features using sliding windows."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        window_size = min(64, seq_len // 4)
        stride = window_size // 2
        
        local_features = []
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window = hidden_states[:, start:end, :]
            
            # Encode window
            encoded_window = self.local_encoder(window)
            
            # Pool to single representation
            pooled = encoded_window.mean(dim=1)  # [batch_size, hidden_dim]
            local_features.append(pooled)
        
        # Combine local features
        if local_features:
            local_repr = torch.stack(local_features, dim=1).mean(dim=1)
        else:
            local_repr = hidden_states.mean(dim=1)
        
        return local_repr
    
    def _encode_global(self, hidden_states):
        """Encode global features using full sequence."""
        # Use learned query for global aggregation
        global_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        batch_size = hidden_states.size(0)
        
        query = global_query.expand(batch_size, -1, -1)
        
        # Concatenate query with hidden states
        input_with_query = torch.cat([query, hidden_states], dim=1)
        
        # Encode
        encoded = self.global_encoder(input_with_query)
        
        # Extract global representation (first token)
        global_repr = encoded[:, 0, :]
        
        return global_repr
```

## Advanced Training Techniques

### Adversarial Training

```python
class AdversarialTrainer:
    """Adversarial training for Free Transformer."""
    
    def __init__(self, model, discriminator, config):
        self.model = model
        self.discriminator = discriminator
        self.config = config
        
        # Optimizers
        self.model_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config.disc_learning_rate)
    
    def train_step(self, batch):
        """Single adversarial training step."""
        tokens = batch['input_ids']
        
        # Train discriminator
        disc_loss = self._train_discriminator(tokens)
        
        # Train generator (model)
        gen_loss = self._train_generator(tokens)
        
        return {
            'discriminator_loss': disc_loss,
            'generator_loss': gen_loss
        }
    
    def _train_discriminator(self, real_tokens):
        """Train discriminator to distinguish real from generated sequences."""
        self.disc_optimizer.zero_grad()
        
        # Real sequences
        real_logits = self.discriminator(real_tokens)
        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, 
            torch.ones_like(real_logits)
        )
        
        # Generated sequences
        with torch.no_grad():
            generated_tokens = self.model.generate(
                real_tokens[:, :10], 
                max_new_tokens=real_tokens.size(1) - 10
            )
        
        fake_logits = self.discriminator(generated_tokens)
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.zeros_like(fake_logits)
        )
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return disc_loss.item()
    
    def _train_generator(self, tokens):
        """Train generator to fool discriminator."""
        self.model_optimizer.zero_grad()
        
        # Standard Free Transformer loss
        logits, z_logits = self.model(tokens, mode='training')
        standard_loss = free_transformer_loss(
            logits=logits,
            z_logits=z_logits,
            targets=tokens,
            latent_dim=self.config.latent_dim,
            kl_weight=self.config.kl_weight,
            free_bits=self.config.free_bits
        )
        
        # Adversarial loss
        generated_tokens = self.model.generate(
            tokens[:, :10],
            max_new_tokens=tokens.size(1) - 10
        )
        
        fake_logits = self.discriminator(generated_tokens)
        adversarial_loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits)  # Want discriminator to think it's real
        )
        
        # Combined loss
        total_loss = standard_loss['total_loss'] + self.config.adversarial_weight * adversarial_loss
        total_loss.backward()
        self.model_optimizer.step()
        
        return total_loss.item()

class SimpleDiscriminator(nn.Module):
    """Simple discriminator for adversarial training."""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, tokens):
        """Forward pass."""
        embedded = self.embedding(tokens)
        encoded = self.encoder(embedded)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
```

## Next Steps

- **[Evaluation Examples](evaluation.md)**: Advanced evaluation techniques
- **[API Reference](../api/model.md)**: Complete API documentation
- **[Training Guide](../training/guide.md)**: Standard training practices