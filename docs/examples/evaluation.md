# Evaluation Examples

This page provides comprehensive examples for evaluating Free Transformer models.

## Basic Evaluation

### Model Comparison

```python
import torch
import torch.nn.functional as F
from free_transformer import FreeTransformer, TransformerBaseline, ModelConfig
from free_transformer.losses import free_transformer_loss

def compare_models(free_model, baseline_model, test_dataloader, device='cuda'):
    """Compare Free Transformer with baseline model."""
    
    results = {
        'free_transformer': {'perplexity': 0, 'loss': 0, 'samples': 0},
        'baseline': {'perplexity': 0, 'loss': 0, 'samples': 0}
    }
    
    # Evaluate Free Transformer
    free_model.eval()
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        
        for batch in test_dataloader:
            tokens = batch['input_ids'].to(device)
            
            # Forward pass
            logits, z_logits = free_model(tokens, mode='training')
            
            # Compute loss (only reconstruction part for fair comparison)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += tokens.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        results['free_transformer']['loss'] = avg_loss
        results['free_transformer']['perplexity'] = perplexity.item()
        results['free_transformer']['samples'] = total_tokens
    
    # Evaluate Baseline
    baseline_model.eval()
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        
        for batch in test_dataloader:
            tokens = batch['input_ids'].to(device)
            
            # Forward pass
            logits = baseline_model(tokens)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += tokens.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        results['baseline']['loss'] = avg_loss
        results['baseline']['perplexity'] = perplexity.item()
        results['baseline']['samples'] = total_tokens
    
    return results

# Example usage
config = ModelConfig(vocab_size=10000, hidden_dim=512, num_layers=12)
free_model = FreeTransformer(config).to('cuda')
baseline_model = TransformerBaseline(config).to('cuda')

# Load trained models
free_model.load_state_dict(torch.load('checkpoints/free/model.pt'))
baseline_model.load_state_dict(torch.load('checkpoints/baseline/model.pt'))

results = compare_models(free_model, baseline_model, test_dataloader)
print(f"Free Transformer Perplexity: {results['free_transformer']['perplexity']:.2f}")
print(f"Baseline Perplexity: {results['baseline']['perplexity']:.2f}")
```

### Generation Quality Evaluation

```python
def evaluate_generation_quality(model, prompts, max_length=100, num_samples=5):
    """Evaluate generation quality with multiple metrics."""
    
    model.eval()
    results = {
        'diversity': [],
        'coherence': [],
        'fluency': [],
        'generations': []
    }
    
    for prompt in prompts:
        prompt_results = {
            'prompt': prompt,
            'generations': [],
            'diversity_score': 0,
            'coherence_score': 0,
            'fluency_score': 0
        }
        
        # Generate multiple samples for diversity measurement
        generations = []
        for _ in range(num_samples):
            with torch.no_grad():
                generated = model.generate(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=0.8,
                    top_k=40,
                    do_sample=True
                )
            generations.append(generated)
        
        prompt_results['generations'] = generations
        
        # Compute diversity (average pairwise distance)
        diversity_score = compute_diversity(generations)
        prompt_results['diversity_score'] = diversity_score
        results['diversity'].append(diversity_score)
        
        # Compute coherence (consistency within each generation)
        coherence_scores = [compute_coherence(gen) for gen in generations]
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        prompt_results['coherence_score'] = avg_coherence
        results['coherence'].append(avg_coherence)
        
        # Compute fluency (language model score)
        fluency_scores = [compute_fluency(gen) for gen in generations]
        avg_fluency = sum(fluency_scores) / len(fluency_scores)
        prompt_results['fluency_score'] = avg_fluency
        results['fluency'].append(avg_fluency)
        
        results['generations'].append(prompt_results)
    
    # Aggregate results
    results['avg_diversity'] = sum(results['diversity']) / len(results['diversity'])
    results['avg_coherence'] = sum(results['coherence']) / len(results['coherence'])
    results['avg_fluency'] = sum(results['fluency']) / len(results['fluency'])
    
    return results

def compute_diversity(generations):
    """Compute diversity score for a set of generations."""
    if len(generations) < 2:
        return 0.0
    
    # Convert to text (assuming we have a tokenizer)
    # For now, compute token-level diversity
    total_distance = 0
    num_pairs = 0
    
    for i in range(len(generations)):
        for j in range(i + 1, len(generations)):
            # Compute edit distance or Jaccard similarity
            distance = compute_sequence_distance(generations[i], generations[j])
            total_distance += distance
            num_pairs += 1
    
    return total_distance / num_pairs if num_pairs > 0 else 0.0

def compute_sequence_distance(seq1, seq2):
    """Compute distance between two sequences."""
    # Simple Jaccard distance
    set1 = set(seq1.flatten().tolist())
    set2 = set(seq2.flatten().tolist())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return 1 - (intersection / union) if union > 0 else 1.0

def compute_coherence(generation):
    """Compute coherence score for a single generation."""
    # Simple coherence: repetition ratio
    tokens = generation.flatten().tolist()
    if len(tokens) < 2:
        return 0.0
    
    # Count adjacent repetitions
    repetitions = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
    return 1 - (repetitions / (len(tokens) - 1))

def compute_fluency(generation):
    """Compute fluency score using a reference language model."""
    # Placeholder - in practice, use a trained language model
    # For now, return entropy-based score
    tokens = generation.flatten().tolist()
    if not tokens:
        return 0.0
    
    from collections import Counter
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    
    entropy = 0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * torch.log(torch.tensor(prob))
    
    # Normalize entropy (higher entropy = more fluent, up to a point)
    max_entropy = torch.log(torch.tensor(len(token_counts)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy.item()
```

## Advanced Evaluation

### Latent Space Analysis

```python
def analyze_latent_space(model, dataloader, device='cuda'):
    """Analyze the learned latent space."""
    
    model.eval()
    latent_codes = []
    input_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['input_ids'].to(device)
            
            # Get latent codes
            _, z_logits = model(tokens, mode='training')
            z_probs = torch.sigmoid(z_logits)
            
            latent_codes.append(z_probs.cpu())
            
            # Extract input features for analysis
            features = extract_input_features(tokens)
            input_features.extend(features)
    
    latent_codes = torch.cat(latent_codes, dim=0)
    
    analysis_results = {
        'latent_utilization': analyze_latent_utilization(latent_codes),
        'latent_clustering': analyze_latent_clustering(latent_codes, input_features),
        'latent_interpolation': analyze_latent_interpolation(model, latent_codes),
        'bit_importance': analyze_bit_importance(latent_codes, input_features)
    }
    
    return analysis_results

def analyze_latent_utilization(latent_codes):
    """Analyze how much each latent dimension is used."""
    # Compute statistics for each dimension
    utilization = {}
    
    for dim in range(latent_codes.size(1)):
        dim_values = latent_codes[:, dim]
        
        utilization[f'dim_{dim}'] = {
            'mean': dim_values.mean().item(),
            'std': dim_values.std().item(),
            'entropy': compute_entropy_continuous(dim_values),
            'active_ratio': ((dim_values > 0.1) & (dim_values < 0.9)).float().mean().item()
        }
    
    # Overall utilization metrics
    utilization['overall'] = {
        'avg_entropy': sum(d['entropy'] for d in utilization.values() if isinstance(d, dict)) / latent_codes.size(1),
        'avg_active_ratio': sum(d['active_ratio'] for d in utilization.values() if isinstance(d, dict)) / latent_codes.size(1)
    }
    
    return utilization

def compute_entropy_continuous(values):
    """Compute entropy for continuous values using binning."""
    # Bin values and compute entropy
    hist, _ = torch.histogram(values, bins=20, range=(0, 1))
    probs = hist.float() / hist.sum()
    probs = probs[probs > 0]  # Remove zero probabilities
    
    entropy = -(probs * torch.log2(probs)).sum()
    return entropy.item()

def analyze_latent_clustering(latent_codes, input_features):
    """Analyze clustering in latent space."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Convert to numpy for sklearn
    latent_np = latent_codes.numpy()
    
    clustering_results = {}
    
    # Try different numbers of clusters
    for n_clusters in [2, 4, 8, 16]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_np)
        
        # Compute silhouette score
        silhouette = silhouette_score(latent_np, cluster_labels)
        
        clustering_results[f'k_{n_clusters}'] = {
            'silhouette_score': silhouette,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_labels': cluster_labels.tolist()
        }
    
    return clustering_results

def extract_input_features(tokens):
    """Extract features from input tokens for analysis."""
    features = []
    
    for seq in tokens:
        seq_features = {
            'length': len(seq),
            'unique_tokens': len(torch.unique(seq)),
            'repetition_ratio': compute_repetition_ratio(seq),
            'first_token': seq[0].item() if len(seq) > 0 else -1,
            'last_token': seq[-1].item() if len(seq) > 0 else -1
        }
        features.append(seq_features)
    
    return features

def compute_repetition_ratio(sequence):
    """Compute repetition ratio in sequence."""
    if len(sequence) < 2:
        return 0.0
    
    repeats = sum(1 for i in range(1, len(sequence)) if sequence[i] == sequence[i-1])
    return repeats / (len(sequence) - 1)
```

### Plan Manipulation Experiments

```python
def experiment_plan_manipulation(model, test_prompts, device='cuda'):
    """Experiment with plan manipulation for controllable generation."""
    
    model.eval()
    experiments = {}
    
    # Experiment 1: Bit flipping
    experiments['bit_flipping'] = bit_flipping_experiment(model, test_prompts, device)
    
    # Experiment 2: Plan interpolation
    experiments['plan_interpolation'] = plan_interpolation_experiment(model, test_prompts, device)
    
    # Experiment 3: Plan conditioning
    experiments['plan_conditioning'] = plan_conditioning_experiment(model, test_prompts, device)
    
    return experiments

def bit_flipping_experiment(model, prompts, device):
    """Test effect of flipping individual bits in the plan."""
    results = []
    
    for prompt in prompts:
        prompt_tensor = prompt.to(device)
        
        # Generate with random plan
        with torch.no_grad():
            baseline_gen = model.generate(prompt_tensor, max_new_tokens=50)
        
        # Test flipping each bit
        bit_effects = []
        for bit_idx in range(model.config.latent_dim):
            # Sample a plan
            plan = torch.bernoulli(torch.full((1, model.config.latent_dim), 0.5)).to(device)
            
            # Generate with original plan
            original_gen = model.generate_with_plan(prompt_tensor, plan, max_new_tokens=50)
            
            # Flip the bit and generate again
            flipped_plan = plan.clone()
            flipped_plan[0, bit_idx] = 1 - flipped_plan[0, bit_idx]
            flipped_gen = model.generate_with_plan(prompt_tensor, flipped_plan, max_new_tokens=50)
            
            # Measure difference
            difference = compute_sequence_distance(original_gen, flipped_gen)
            
            bit_effects.append({
                'bit_index': bit_idx,
                'difference': difference,
                'original_generation': original_gen,
                'flipped_generation': flipped_gen
            })
        
        results.append({
            'prompt': prompt,
            'baseline_generation': baseline_gen,
            'bit_effects': bit_effects
        })
    
    return results

def plan_interpolation_experiment(model, prompts, device):
    """Test plan interpolation for smooth generation transitions."""
    results = []
    
    for prompt in prompts:
        prompt_tensor = prompt.to(device)
        
        # Sample two random plans
        plan1 = torch.bernoulli(torch.full((1, model.config.latent_dim), 0.5)).to(device)
        plan2 = torch.bernoulli(torch.full((1, model.config.latent_dim), 0.5)).to(device)
        
        # Generate interpolated plans
        interpolation_steps = 5
        interpolated_generations = []
        
        for i in range(interpolation_steps):
            alpha = i / (interpolation_steps - 1)
            
            # Linear interpolation in probability space
            plan1_probs = plan1.float()
            plan2_probs = plan2.float()
            interpolated_probs = (1 - alpha) * plan1_probs + alpha * plan2_probs
            
            # Sample from interpolated probabilities
            interpolated_plan = torch.bernoulli(interpolated_probs)
            
            # Generate with interpolated plan
            with torch.no_grad():
                generation = model.generate_with_plan(
                    prompt_tensor, 
                    interpolated_plan, 
                    max_new_tokens=50
                )
            
            interpolated_generations.append({
                'alpha': alpha,
                'plan': interpolated_plan,
                'generation': generation
            })
        
        results.append({
            'prompt': prompt,
            'plan1': plan1,
            'plan2': plan2,
            'interpolations': interpolated_generations
        })
    
    return results

def plan_conditioning_experiment(model, prompts, device):
    """Test conditioning generation on specific plan patterns."""
    results = []
    
    # Define specific plan patterns to test
    patterns = {
        'all_zeros': torch.zeros(1, model.config.latent_dim),
        'all_ones': torch.ones(1, model.config.latent_dim),
        'alternating': torch.tensor([[i % 2 for i in range(model.config.latent_dim)]]).float(),
        'first_half_ones': torch.cat([
            torch.ones(1, model.config.latent_dim // 2),
            torch.zeros(1, model.config.latent_dim - model.config.latent_dim // 2)
        ], dim=1)
    }
    
    for prompt in prompts:
        prompt_tensor = prompt.to(device)
        pattern_results = {}
        
        for pattern_name, pattern_plan in patterns.items():
            pattern_plan = pattern_plan.to(device)
            
            # Generate multiple samples with this pattern
            generations = []
            for _ in range(3):
                with torch.no_grad():
                    generation = model.generate_with_plan(
                        prompt_tensor,
                        pattern_plan,
                        max_new_tokens=50
                    )
                generations.append(generation)
            
            pattern_results[pattern_name] = {
                'plan': pattern_plan,
                'generations': generations,
                'diversity': compute_diversity(generations)
            }
        
        results.append({
            'prompt': prompt,
            'patterns': pattern_results
        })
    
    return results
```

## Evaluation Metrics

### Comprehensive Metrics Suite

```python
class EvaluationSuite:
    """Comprehensive evaluation suite for Free Transformer."""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = {}
    
    def evaluate_all(self, test_dataloader, generation_prompts=None):
        """Run all evaluation metrics."""
        
        # Language modeling metrics
        self.metrics['language_modeling'] = self.evaluate_language_modeling(test_dataloader)
        
        # Generation metrics
        if generation_prompts:
            self.metrics['generation'] = self.evaluate_generation(generation_prompts)
        
        # Latent space metrics
        self.metrics['latent_space'] = self.evaluate_latent_space(test_dataloader)
        
        # Plan utilization metrics
        self.metrics['plan_utilization'] = self.evaluate_plan_utilization(test_dataloader)
        
        return self.metrics
    
    def evaluate_language_modeling(self, dataloader):
        """Evaluate language modeling performance."""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_kl = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch['input_ids']
                
                logits, z_logits = self.model(tokens, mode='training')
                
                # Reconstruction loss
                recon_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tokens.view(-1),
                    reduction='sum'
                )
                
                # KL loss
                posterior = torch.sigmoid(z_logits)
                prior = torch.full_like(posterior, 0.5)
                kl_loss = F.kl_div(
                    torch.log(posterior + 1e-8),
                    prior,
                    reduction='sum'
                )
                
                total_loss += recon_loss.item()
                total_tokens += tokens.numel()
                total_kl += kl_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        avg_kl = total_kl / num_batches
        
        return {
            'perplexity': perplexity.item(),
            'avg_loss': avg_loss,
            'avg_kl_divergence': avg_kl,
            'total_tokens': total_tokens
        }
    
    def evaluate_generation(self, prompts):
        """Evaluate generation quality."""
        self.model.eval()
        
        all_generations = []
        diversity_scores = []
        coherence_scores = []
        
        for prompt in prompts:
            # Generate multiple samples
            generations = []
            for _ in range(5):
                with torch.no_grad():
                    generated = self.model.generate(
                        prompt,
                        max_new_tokens=100,
                        temperature=0.8,
                        top_k=40,
                        do_sample=True
                    )
                generations.append(generated)
            
            all_generations.extend(generations)
            
            # Compute diversity for this prompt
            diversity = compute_diversity(generations)
            diversity_scores.append(diversity)
            
            # Compute coherence for each generation
            coherence = [compute_coherence(gen) for gen in generations]
            coherence_scores.extend(coherence)
        
        return {
            'avg_diversity': sum(diversity_scores) / len(diversity_scores),
            'avg_coherence': sum(coherence_scores) / len(coherence_scores),
            'total_generations': len(all_generations)
        }
    
    def evaluate_latent_space(self, dataloader):
        """Evaluate latent space properties."""
        self.model.eval()
        
        all_z_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch['input_ids']
                _, z_logits = self.model(tokens, mode='training')
                all_z_logits.append(z_logits)
        
        all_z_logits = torch.cat(all_z_logits, dim=0)
        z_probs = torch.sigmoid(all_z_logits)
        
        # Compute metrics
        metrics = {}
        
        # Utilization per dimension
        dim_utilization = []
        for dim in range(z_probs.size(1)):
            dim_values = z_probs[:, dim]
            utilization = ((dim_values > 0.1) & (dim_values < 0.9)).float().mean()
            dim_utilization.append(utilization.item())
        
        metrics['avg_dimension_utilization'] = sum(dim_utilization) / len(dim_utilization)
        metrics['dimension_utilization'] = dim_utilization
        
        # Overall entropy
        total_entropy = 0
        for dim in range(z_probs.size(1)):
            dim_entropy = compute_entropy_continuous(z_probs[:, dim])
            total_entropy += dim_entropy
        
        metrics['avg_entropy'] = total_entropy / z_probs.size(1)
        
        return metrics
    
    def evaluate_plan_utilization(self, dataloader):
        """Evaluate how well the model uses the latent plan."""
        self.model.eval()
        
        # Test plan manipulation effects
        manipulation_effects = []
        
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch['input_ids']
                if tokens.size(0) == 0:
                    continue
                
                # Take first sequence as test case
                test_tokens = tokens[0:1]
                
                # Generate with original plan
                _, z_logits = self.model(test_tokens, mode='training')
                original_plan = torch.sigmoid(z_logits)
                
                # Generate with modified plans
                for bit_idx in range(min(8, self.model.config.latent_dim)):  # Test first 8 bits
                    modified_plan = original_plan.clone()
                    modified_plan[0, bit_idx] = 1 - modified_plan[0, bit_idx]
                    
                    # This would require a generate_with_plan method
                    # For now, just measure the difference in representations
                    # In practice, you'd compare generated sequences
                    
                    plan_diff = torch.abs(original_plan - modified_plan).sum().item()
                    manipulation_effects.append(plan_diff)
                
                # Only test a few batches for efficiency
                if len(manipulation_effects) > 50:
                    break
        
        return {
            'avg_manipulation_effect': sum(manipulation_effects) / len(manipulation_effects) if manipulation_effects else 0,
            'manipulation_effects': manipulation_effects
        }

# Usage example
def run_comprehensive_evaluation():
    """Run comprehensive evaluation on trained models."""
    
    # Load models
    config = ModelConfig(vocab_size=10000, hidden_dim=512, num_layers=12)
    model = FreeTransformer(config)
    model.load_state_dict(torch.load('checkpoints/free/model.pt'))
    model.eval()
    
    # Create evaluation suite
    evaluator = EvaluationSuite(model)
    
    # Run evaluation
    results = evaluator.evaluate_all(test_dataloader, generation_prompts)
    
    # Print results
    print("=== Free Transformer Evaluation Results ===")
    print(f"Perplexity: {results['language_modeling']['perplexity']:.2f}")
    print(f"Average KL Divergence: {results['language_modeling']['avg_kl_divergence']:.4f}")
    print(f"Generation Diversity: {results['generation']['avg_diversity']:.4f}")
    print(f"Generation Coherence: {results['generation']['avg_coherence']:.4f}")
    print(f"Latent Dimension Utilization: {results['latent_space']['avg_dimension_utilization']:.4f}")
    print(f"Latent Space Entropy: {results['latent_space']['avg_entropy']:.4f}")
    
    return results
```

## Next Steps

- **[Custom Training](custom-training.md)**: Advanced training techniques
- **[API Reference](../api/model.md)**: Complete API documentation
- **[Training Guide](../training/guide.md)**: Standard training practices