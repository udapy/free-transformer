# Latent Planning Mechanism

This page provides an in-depth explanation of the latent planning mechanism that makes the Free Transformer unique.

## Core Concept

Traditional autoregressive models generate tokens sequentially, making decisions based only on the tokens seen so far. The Free Transformer introduces **explicit latent planning** - the model first creates an abstract "plan" for the entire sequence, then generates tokens to fulfill that plan.

## Planning vs Reactive Generation

### Reactive Generation (Standard Transformers)
```
Token 1 → Token 2 → Token 3 → ... → Token N
   ↑        ↑        ↑              ↑
   |        |        |              |
Context  Context  Context       Context
```

Each token depends only on previous tokens, leading to:
- **Local coherence** but potential global inconsistency
- **Difficulty with long-range planning**
- **Limited controllability**

### Plan-Based Generation (Free Transformer)
```
Full Context → Abstract Plan Z → Token 1, Token 2, ..., Token N
                     ↓              ↑       ↑            ↑
                     └──────────────┴───────┴────────────┘
```

The model first creates a plan, then generates all tokens conditioned on that plan:
- **Global coherence** through explicit planning
- **Better long-range dependencies**
- **Controllable generation** via plan manipulation

## Mathematical Formulation

### Standard Autoregressive Model
$$P(x_1, ..., x_T) = \prod_{t=1}^T P(x_t | x_{<t})$$

### Free Transformer (Conditional VAE)
$$P(x_1, ..., x_T) = \int P(x_1, ..., x_T | z) P(z) dz$$

Where:
- $z$ is the latent plan variable
- $P(z)$ is the prior distribution (uniform for binary plans)
- $P(x_1, ..., x_T | z)$ is the conditional generation model

## Plan Representation

### Binary Plans
The Free Transformer uses binary latent variables:
$$z \in \{0, 1\}^d$$

Where $d$ is the latent dimension (typically 16-64).

### Why Binary?
1. **Interpretability**: Each bit can represent a discrete choice
2. **Efficiency**: Compact representation
3. **Controllability**: Easy to manipulate specific aspects
4. **Stability**: Avoids posterior collapse issues common with continuous latents

### Plan Semantics
Each bit in the plan can potentially encode:
- **Style**: Formal vs informal, technical vs casual
- **Structure**: Narrative vs expository, linear vs non-linear  
- **Content**: Topic focus, emotional tone
- **Length**: Short vs long form content

## Architecture Components

### 1. Non-Causal Encoder

The encoder creates the latent plan from the full sequence:

```python
class NonCausalEncoder(nn.Module):
    def __init__(self, config):
        self.attention_layers = nn.ModuleList([
            NonCausalAttention(config) for _ in range(config.encoder_layers)
        ])
        self.learned_query = nn.Parameter(torch.randn(config.hidden_dim))
        
    def forward(self, hidden_states):
        # Use learned query to aggregate sequence information
        query = self.learned_query.expand(hidden_states.size(0), 1, -1)
        
        # Non-causal attention over entire sequence
        for layer in self.attention_layers:
            query = layer(query, hidden_states, hidden_states)
            
        return query.squeeze(1)
```

**Key Features:**
- **Non-causal attention**: Can see the entire sequence
- **Learned query**: Single vector that aggregates information
- **Separate parameters**: Independent from decoder

### 2. Binary Mapping

Converts continuous encoder output to discrete binary plan:

```python
class BinaryMapper(nn.Module):
    def __init__(self, config):
        self.projection = nn.Linear(config.hidden_dim, config.latent_dim)
        self.temperature = config.gumbel_temperature
        
    def forward(self, encoder_output, training=True):
        logits = self.projection(encoder_output)
        
        if training:
            # Gumbel-Softmax for differentiable sampling
            binary_soft = F.gumbel_softmax(
                torch.stack([logits, -logits], dim=-1),
                tau=self.temperature,
                hard=True
            )
            return binary_soft[..., 0], logits
        else:
            # Hard binary sampling
            return (logits > 0).float(), logits
```

**Gumbel-Softmax Trick:**
- Enables gradient flow through discrete sampling
- Temperature controls discreteness vs continuity
- Hard sampling during forward, soft during backward

### 3. Plan Injection

Integrates the binary plan into decoder representations:

```python
class PlanInjection(nn.Module):
    def __init__(self, config):
        self.plan_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, decoder_hidden, binary_plan):
        # Project plan to hidden dimension
        plan_repr = self.plan_projection(binary_plan)
        
        # Broadcast to sequence length
        plan_repr = plan_repr.unsqueeze(1).expand(-1, decoder_hidden.size(1), -1)
        
        # Gated injection
        gate_values = torch.sigmoid(self.gate(decoder_hidden))
        return decoder_hidden + gate_values * plan_repr
```

**Injection Strategies:**
1. **Additive**: `hidden + plan_projection(z)`
2. **Gated**: `hidden + gate * plan_projection(z)` (used)
3. **Concatenation**: `concat(hidden, plan_projection(z))`
4. **Cross-attention**: Plan as keys/values

## Training Dynamics

### Variational Objective

The model is trained to maximize the Evidence Lower Bound (ELBO):

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot KL(q(z|x) || p(z))$$

Where:
- **Reconstruction term**: $\mathbb{E}_{q(z|x)}[\log p(x|z)]$ - how well the model generates given the plan
- **Regularization term**: $KL(q(z|x) || p(z))$ - keeps posterior close to prior
- **β-VAE weight**: Controls trade-off between reconstruction and regularization

### Free Bits Regularization

To prevent posterior collapse, we use free bits:

$$KL_{regularized} = \max(KL(q(z|x) || p(z)), \text{free\_bits})$$

This ensures the model uses at least `free_bits` nats of information in the latent variable.

### Training vs Inference

**Training Mode:**
1. Encode full sequence → latent plan
2. Inject plan into decoder
3. Optimize reconstruction + KL loss

**Inference Mode:**
1. Sample plan from prior: $z \sim p(z) = \text{Uniform}(\{0,1\}^d)$
2. Inject sampled plan into decoder
3. Generate autoregressively

## Plan Analysis and Interpretation

### Plan Utilization

Monitor whether the model actually uses the latent variable:

```python
def analyze_plan_usage(model, dataloader):
    """Analyze how much the model uses the latent plan."""
    kl_divergences = []
    
    for batch in dataloader:
        with torch.no_grad():
            _, z_logits = model(batch['input_ids'], mode='training')
            
            # Compute KL divergence for each sample
            posterior = torch.sigmoid(z_logits)
            prior = torch.full_like(posterior, 0.5)
            
            kl = F.kl_div(torch.log(posterior + 1e-8), prior, reduction='none')
            kl_divergences.append(kl.sum(dim=-1))
    
    kl_divergences = torch.cat(kl_divergences)
    
    print(f"Mean KL divergence: {kl_divergences.mean():.4f}")
    print(f"Std KL divergence: {kl_divergences.std():.4f}")
    print(f"Min KL divergence: {kl_divergences.min():.4f}")
    print(f"Max KL divergence: {kl_divergences.max():.4f}")
    
    return kl_divergences
```

### Plan Interpolation

Explore the latent space by interpolating between plans:

```python
def interpolate_plans(model, prompt, plan1, plan2, steps=5):
    """Generate text with interpolated plans."""
    generations = []
    
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated_plan = (1 - alpha) * plan1 + alpha * plan2
        interpolated_plan = (interpolated_plan > 0.5).float()
        
        # Generate with interpolated plan
        with torch.no_grad():
            generated = model.generate_with_plan(
                prompt, 
                interpolated_plan,
                max_new_tokens=50
            )
        generations.append(generated)
    
    return generations
```

### Plan Manipulation

Control generation by modifying specific plan bits:

```python
def manipulate_plan(model, prompt, bit_index, value):
    """Generate text with specific plan bit set to value."""
    # Sample random plan
    plan = torch.bernoulli(torch.full((1, model.config.latent_dim), 0.5))
    
    # Set specific bit
    plan[0, bit_index] = value
    
    # Generate with modified plan
    with torch.no_grad():
        generated = model.generate_with_plan(prompt, plan, max_new_tokens=100)
    
    return generated

# Example: Compare generations with bit 5 set to 0 vs 1
gen_0 = manipulate_plan(model, prompt, bit_index=5, value=0)
gen_1 = manipulate_plan(model, prompt, bit_index=5, value=1)
```

## Advanced Planning Techniques

### Hierarchical Planning

Use multiple latent variables at different levels:

```python
class HierarchicalPlanner(nn.Module):
    def __init__(self, config):
        self.global_encoder = NonCausalEncoder(config)
        self.local_encoders = nn.ModuleList([
            NonCausalEncoder(config) for _ in range(config.num_local_levels)
        ])
        
    def forward(self, hidden_states):
        # Global plan for entire sequence
        global_plan = self.global_encoder(hidden_states)
        
        # Local plans for subsequences
        local_plans = []
        chunk_size = hidden_states.size(1) // len(self.local_encoders)
        
        for i, encoder in enumerate(self.local_encoders):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = hidden_states[:, start:end]
            local_plan = encoder(chunk)
            local_plans.append(local_plan)
        
        return global_plan, local_plans
```

### Conditional Planning

Condition plans on external information:

```python
class ConditionalPlanner(nn.Module):
    def __init__(self, config):
        self.encoder = NonCausalEncoder(config)
        self.condition_projection = nn.Linear(config.condition_dim, config.hidden_dim)
        
    def forward(self, hidden_states, condition):
        # Project condition to hidden space
        condition_repr = self.condition_projection(condition)
        
        # Add condition to hidden states
        conditioned_hidden = hidden_states + condition_repr.unsqueeze(1)
        
        # Encode with condition
        plan = self.encoder(conditioned_hidden)
        return plan
```

## Troubleshooting Planning Issues

### Posterior Collapse
**Symptoms**: KL loss drops to zero, model ignores latent variable
**Solutions**:
- Increase free bits threshold
- Reduce KL weight (β)
- Use KL annealing
- Check encoder capacity

### Plan Underutilization
**Symptoms**: Low KL divergence, similar generations
**Solutions**:
- Increase latent dimension
- Improve encoder architecture
- Use stronger regularization
- Check injection mechanism

### Training Instability
**Symptoms**: Loss spikes, gradient explosions
**Solutions**:
- Gradient clipping
- Lower learning rate
- Reduce Gumbel temperature
- Use warmup schedule

## Evaluation Metrics

### Plan Quality Metrics
1. **KL Divergence**: Measures plan utilization
2. **Mutual Information**: I(X; Z) between input and plan
3. **Plan Consistency**: Similarity of plans for similar inputs
4. **Generation Diversity**: Variety in outputs for different plans

### Implementation
```python
def evaluate_planning(model, dataloader):
    """Comprehensive evaluation of planning mechanism."""
    metrics = {
        'kl_divergence': [],
        'plan_entropy': [],
        'generation_diversity': []
    }
    
    for batch in dataloader:
        with torch.no_grad():
            logits, z_logits = model(batch['input_ids'], mode='training')
            
            # KL divergence
            posterior = torch.sigmoid(z_logits)
            kl = compute_kl_divergence(posterior)
            metrics['kl_divergence'].append(kl)
            
            # Plan entropy
            entropy = -posterior * torch.log(posterior + 1e-8) - (1 - posterior) * torch.log(1 - posterior + 1e-8)
            metrics['plan_entropy'].append(entropy.sum(dim=-1))
    
    return {k: torch.cat(v).mean().item() for k, v in metrics.items()}
```

## Next Steps

- **[Free Transformer Architecture](free-transformer.md)**: Complete architecture overview
- **[Training Guide](../training/guide.md)**: How to train with latent planning
- **[Examples](../examples/basic.md)**: Practical usage examples