# Synthetic Data Generation

This guide covers generating synthetic training data for Free Transformer experiments and prototyping.

## Overview

Synthetic data generation enables:
- **Fast prototyping**: Test models without large datasets
- **Controlled experiments**: Known data properties for analysis
- **Ablation studies**: Isolate specific model behaviors
- **Development**: Quick iteration during model development

## Quick Start

### Using Makefile

```bash
# Generate large synthetic dataset
make generate-data

# Generate small dataset for testing
make generate-data-small
```

### Using Python Script

```bash
# Basic usage
python examples/generate_data.py --output-dir data/synthetic

# Custom parameters
python examples/generate_data.py \
  --output-dir data/custom \
  --vocab-size 5000 \
  --seq-length 256 \
  --num-train 10000 \
  --num-val 1000 \
  --seed 42
```

## Data Generation Types

### 1. Random Token Sequences

Simple random sequences for basic testing:

```python
from free_transformer.synthetic_data import generate_random_sequences

data = generate_random_sequences(
    vocab_size=1000,
    seq_length=128,
    num_samples=5000,
    seed=42
)
```

**Use cases:**
- Model architecture testing
- Memory usage profiling
- Basic training pipeline validation

### 2. Pattern-Based Sequences

Sequences with embedded patterns for coherence testing:

```python
from free_transformer.synthetic_data import generate_pattern_sequences

data = generate_pattern_sequences(
    vocab_size=1000,
    seq_length=256,
    num_samples=10000,
    pattern_types=['repetition', 'alternation', 'progression'],
    pattern_probability=0.3,
    seed=42
)
```

**Pattern types:**
- **Repetition**: `[A, B, C, A, B, C, ...]`
- **Alternation**: `[A, B, A, B, A, B, ...]`
- **Progression**: `[1, 2, 3, 4, 5, ...]`
- **Reversal**: `[A, B, C, C, B, A, ...]`

### 3. Hierarchical Sequences

Multi-level structured sequences:

```python
from free_transformer.synthetic_data import generate_hierarchical_sequences

data = generate_hierarchical_sequences(
    vocab_size=1000,
    seq_length=512,
    num_samples=20000,
    hierarchy_levels=3,
    structure_probability=0.5,
    seed=42
)
```

**Structure levels:**
- **Global**: Overall sequence theme
- **Local**: Subsequence patterns
- **Token**: Individual token relationships

### 4. Conditional Sequences

Sequences conditioned on control tokens:

```python
from free_transformer.synthetic_data import generate_conditional_sequences

data = generate_conditional_sequences(
    vocab_size=1000,
    seq_length=256,
    num_samples=15000,
    num_conditions=10,
    condition_strength=0.8,
    seed=42
)
```

**Conditioning:**
- First token determines sequence properties
- Style, length, or pattern controlled by condition
- Tests model's ability to follow instructions

## Configuration Options

### Basic Parameters

```python
config = {
    'vocab_size': 10000,        # Vocabulary size
    'seq_length': 512,          # Sequence length
    'num_train': 50000,         # Training samples
    'num_val': 5000,            # Validation samples
    'seed': 42,                 # Random seed
}
```

### Advanced Parameters

```python
config = {
    # Pattern control
    'pattern_probability': 0.3,  # Probability of pattern occurrence
    'pattern_length': 10,        # Average pattern length
    'pattern_types': ['rep', 'alt', 'prog'],  # Pattern types to use
    
    # Hierarchy control
    'hierarchy_levels': 3,       # Number of hierarchy levels
    'structure_probability': 0.5, # Probability of structured content
    'global_coherence': 0.7,     # Global coherence strength
    
    # Conditioning
    'num_conditions': 20,        # Number of condition types
    'condition_strength': 0.8,   # How strongly condition affects sequence
    'condition_tokens': [0, 1, 2], # Special condition tokens
    
    # Noise and variation
    'noise_probability': 0.1,    # Random token injection probability
    'length_variation': 0.2,     # Sequence length variation
    'vocab_distribution': 'uniform', # Token distribution (uniform, zipf)
}
```

## Data Analysis

### Sequence Statistics

```python
from free_transformer.synthetic_data import analyze_sequences

def analyze_dataset(sequences):
    """Analyze synthetic dataset properties."""
    stats = {
        'num_sequences': len(sequences),
        'avg_length': np.mean([len(seq) for seq in sequences]),
        'vocab_usage': len(set(token for seq in sequences for token in seq)),
        'token_distribution': Counter(token for seq in sequences for token in seq),
        'pattern_detection': detect_patterns(sequences),
    }
    return stats

# Example usage
train_data = load_synthetic_data('data/synthetic/train.jsonl')
stats = analyze_dataset(train_data)
print(f"Dataset contains {stats['num_sequences']} sequences")
print(f"Average length: {stats['avg_length']:.1f} tokens")
print(f"Vocabulary usage: {stats['vocab_usage']}/{config['vocab_size']}")
```

### Pattern Detection

```python
def detect_patterns(sequences, min_length=3):
    """Detect common patterns in sequences."""
    patterns = Counter()
    
    for seq in sequences:
        for i in range(len(seq) - min_length + 1):
            pattern = tuple(seq[i:i + min_length])
            patterns[pattern] += 1
    
    # Find most common patterns
    common_patterns = patterns.most_common(10)
    return common_patterns

# Example usage
patterns = detect_patterns(train_data)
print("Most common patterns:")
for pattern, count in patterns:
    print(f"  {pattern}: {count} occurrences")
```

### Coherence Metrics

```python
def measure_coherence(sequences):
    """Measure sequence coherence metrics."""
    metrics = {}
    
    # Local coherence (adjacent token similarity)
    local_coherence = []
    for seq in sequences:
        similarities = []
        for i in range(len(seq) - 1):
            # Simple similarity: same token = 1, different = 0
            sim = 1 if seq[i] == seq[i+1] else 0
            similarities.append(sim)
        local_coherence.append(np.mean(similarities))
    
    metrics['local_coherence'] = np.mean(local_coherence)
    
    # Global coherence (sequence-level consistency)
    global_coherence = []
    for seq in sequences:
        # Measure repetition of tokens
        token_counts = Counter(seq)
        max_count = max(token_counts.values())
        coherence = max_count / len(seq)
        global_coherence.append(coherence)
    
    metrics['global_coherence'] = np.mean(global_coherence)
    
    return metrics
```

## Custom Data Generators

### Creating Custom Generators

```python
from free_transformer.synthetic_data import BaseDataGenerator

class CustomDataGenerator(BaseDataGenerator):
    def __init__(self, vocab_size, seq_length, **kwargs):
        super().__init__(vocab_size, seq_length, **kwargs)
        self.custom_param = kwargs.get('custom_param', 0.5)
    
    def generate_sequence(self):
        """Generate a single sequence with custom logic."""
        sequence = []
        
        # Custom generation logic here
        for i in range(self.seq_length):
            if np.random.random() < self.custom_param:
                # Custom behavior
                token = self.generate_special_token(i)
            else:
                # Random token
                token = np.random.randint(0, self.vocab_size)
            sequence.append(token)
        
        return sequence
    
    def generate_special_token(self, position):
        """Generate special token based on position."""
        # Example: position-dependent token
        return position % 100

# Usage
generator = CustomDataGenerator(
    vocab_size=1000,
    seq_length=256,
    custom_param=0.3
)

sequences = generator.generate_dataset(num_samples=10000)
```

### Template-Based Generation

```python
class TemplateDataGenerator(BaseDataGenerator):
    def __init__(self, vocab_size, seq_length, templates, **kwargs):
        super().__init__(vocab_size, seq_length, **kwargs)
        self.templates = templates
    
    def generate_sequence(self):
        """Generate sequence from template."""
        template = np.random.choice(self.templates)
        sequence = []
        
        for element in template:
            if element == 'RANDOM':
                token = np.random.randint(0, self.vocab_size)
            elif element == 'PATTERN':
                token = self.generate_pattern_token(len(sequence))
            else:
                token = element
            sequence.append(token)
        
        # Pad or truncate to desired length
        sequence = self.adjust_length(sequence)
        return sequence

# Define templates
templates = [
    [1, 'RANDOM', 'RANDOM', 2, 'PATTERN', 'PATTERN'],
    [3, 'PATTERN', 'RANDOM', 'PATTERN', 4],
    ['RANDOM'] * 10 + [5] + ['PATTERN'] * 5,
]

generator = TemplateDataGenerator(
    vocab_size=1000,
    seq_length=256,
    templates=templates
)
```

## Data Quality Control

### Validation Checks

```python
def validate_synthetic_data(sequences, config):
    """Validate generated synthetic data."""
    issues = []
    
    # Check sequence lengths
    lengths = [len(seq) for seq in sequences]
    if min(lengths) < config['min_length']:
        issues.append(f"Sequences too short: min={min(lengths)}")
    if max(lengths) > config['max_length']:
        issues.append(f"Sequences too long: max={max(lengths)}")
    
    # Check vocabulary usage
    all_tokens = set(token for seq in sequences for token in seq)
    if max(all_tokens) >= config['vocab_size']:
        issues.append(f"Invalid tokens: max={max(all_tokens)}")
    if min(all_tokens) < 0:
        issues.append(f"Negative tokens: min={min(all_tokens)}")
    
    # Check for empty sequences
    empty_count = sum(1 for seq in sequences if len(seq) == 0)
    if empty_count > 0:
        issues.append(f"Empty sequences: {empty_count}")
    
    return issues

# Usage
issues = validate_synthetic_data(train_data, config)
if issues:
    print("Data validation issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Data validation passed!")
```

### Quality Metrics

```python
def compute_quality_metrics(sequences):
    """Compute data quality metrics."""
    metrics = {}
    
    # Diversity metrics
    unique_sequences = len(set(tuple(seq) for seq in sequences))
    metrics['sequence_diversity'] = unique_sequences / len(sequences)
    
    all_tokens = [token for seq in sequences for token in seq]
    unique_tokens = len(set(all_tokens))
    metrics['token_diversity'] = unique_tokens
    
    # Complexity metrics
    avg_entropy = np.mean([compute_entropy(seq) for seq in sequences])
    metrics['average_entropy'] = avg_entropy
    
    # Pattern metrics
    pattern_count = sum(1 for seq in sequences if has_pattern(seq))
    metrics['pattern_ratio'] = pattern_count / len(sequences)
    
    return metrics

def compute_entropy(sequence):
    """Compute entropy of a sequence."""
    token_counts = Counter(sequence)
    total_tokens = len(sequence)
    
    entropy = 0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * np.log2(prob)
    
    return entropy
```

## Integration with Training

### Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, sequences, max_length=None):
        self.sequences = sequences
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        if self.max_length:
            sequence = sequence[:self.max_length]
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'length': len(sequence)
        }

# Usage
train_sequences = load_synthetic_data('data/synthetic/train.jsonl')
train_dataset = SyntheticDataset(train_sequences, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Curriculum Learning

```python
def create_curriculum_data(base_sequences, stages):
    """Create curriculum learning datasets."""
    curriculum_data = {}
    
    for stage_name, stage_config in stages.items():
        # Filter sequences by complexity
        filtered_sequences = filter_by_complexity(
            base_sequences, 
            stage_config['min_complexity'],
            stage_config['max_complexity']
        )
        
        # Adjust sequence lengths
        adjusted_sequences = adjust_sequence_lengths(
            filtered_sequences,
            stage_config['max_length']
        )
        
        curriculum_data[stage_name] = adjusted_sequences
    
    return curriculum_data

# Define curriculum stages
stages = {
    'stage1': {'min_complexity': 0.0, 'max_complexity': 0.3, 'max_length': 128},
    'stage2': {'min_complexity': 0.2, 'max_complexity': 0.6, 'max_length': 256},
    'stage3': {'min_complexity': 0.5, 'max_complexity': 1.0, 'max_length': 512},
}

curriculum_data = create_curriculum_data(train_sequences, stages)
```

## Best Practices

1. **Start simple**: Begin with basic random sequences
2. **Add complexity gradually**: Introduce patterns incrementally
3. **Validate data**: Always check generated data quality
4. **Monitor diversity**: Ensure sufficient sequence variation
5. **Use appropriate size**: Match real data characteristics
6. **Document generation**: Keep track of generation parameters
7. **Version control**: Save generation configs and seeds

## Troubleshooting

### Common Issues

**Low diversity**
- Increase vocabulary size
- Reduce pattern probability
- Add noise to generation

**Poor patterns**
- Adjust pattern parameters
- Check pattern detection logic
- Validate pattern implementation

**Memory issues**
- Generate data in batches
- Use streaming data loading
- Reduce sequence length

**Training instability**
- Check data distribution
- Validate sequence lengths
- Ensure proper tokenization

## Next Steps

- **[Training Guide](guide.md)**: Use synthetic data for training
- **[Configuration](configuration.md)**: Configure data parameters
- **[Examples](../examples/basic.md)**: See synthetic data in action