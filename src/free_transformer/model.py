"""Core Free Transformer implementation."""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderBlock
from .injection import InjectionMechanism
from .latent import LatentPlan


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        cos_cached = getattr(self, "cos_cached", None)
        sin_cached = getattr(self, "sin_cached", None)

        if cos_cached is None or sin_cached is None:
            raise RuntimeError("RoPE embeddings not properly initialized")

        cos = cos_cached[:, :seq_len, :, :]
        sin = sin_cached[:, :seq_len, :, :]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        silu_out = F.silu(w1_out)
        result = self.w2(silu_out * w3_out)
        assert isinstance(result, torch.Tensor)
        return result


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with optional causal masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.use_rope = use_rope

        # Pre-normalization
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        # Grouped-Query Attention projections
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # RoPE
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Feed-forward
        self.ffn = SwiGLU(dim, ffn_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_input: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query input [batch, seq_q, dim]
            kv_input: Key/Value input [batch, seq_kv, dim]. If None, uses x.
            mask: Attention mask [batch, 1, seq_q, seq_kv]
        """
        batch_size, seq_len, _ = x.shape

        # Attention with pre-normalization
        residual = x
        x_norm = self.attn_norm(x)

        # Use separate kv_input if provided (for injection)
        kv_input = kv_input if kv_input is not None else x_norm

        # Project Q, K, V
        q = self.wq(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(kv_input).view(batch_size, -1, self.num_kv_heads, self.head_dim)
        v = self.wv(kv_input).view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle GQA: repeat k, v if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if needed
        if self.causal and mask is None:
            seq_q, seq_k = scores.shape[-2], scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        attn_output = self.wo(attn_output)
        x = residual + self.dropout(attn_output)

        # Feed-forward with pre-normalization
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x


class FreeTransformer(nn.Module):
    """
    Free Transformer: Conditional VAE-based language model with latent planning.

    Implements the architecture from the Free Transformer paper with:
    - Split decoder stack (first half for context, second half for generation)
    - Non-causal encoder for latent plan inference
    - Binary mapper for differentiable discrete sampling
    - Injection mechanism for plan integration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # First half of decoder blocks (context processing)
        self.first_half_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    ffn_dim=config.ffn_hidden_dim,
                    dropout=config.dropout,
                    causal=True,
                    use_rope=config.use_rope,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.split_layer)
            ]
        )

        # Encoder module (non-causal, for plan inference)
        self.encoder = EncoderBlock(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_hidden_dim,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )

        # Binary mapper and latent plan handler
        self.latent_plan = LatentPlan(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
        )

        # Injection mechanism
        self.injection = InjectionMechanism(config.hidden_dim)

        # Second half of decoder blocks (generation with plan)
        second_half_layers = config.num_layers - config.split_layer
        if second_half_layers <= 0:
            raise ValueError(
                f"split_layer ({config.split_layer}) must be less than num_layers ({config.num_layers})"
            )

        self.second_half_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    ffn_dim=config.ffn_hidden_dim,
                    dropout=config.dropout,
                    causal=True,
                    use_rope=config.use_rope,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(second_half_layers)
            ]
        )

        # Final output
        self.norm = RMSNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.output.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        mode: Literal["training", "inference"] = "training",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with mode switching.

        Args:
            tokens: Input token IDs [batch, seq_len]
            mode: 'training' uses encoder path, 'inference' samples random Z

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            z_logits: Encoder logits for Z (only in training mode) [batch, seq_len, latent_dim]
        """
        batch_size, seq_len = tokens.shape

        # 1. Embed tokens
        x = self.token_embedding(tokens)

        # 2. Process through first half of decoder
        for block in self.first_half_blocks:
            x = block(x)

        # 3. Generate or sample latent plan Z
        if mode == "training":
            # Encoder path: infer Z from context
            z_logits = self.encoder(x)  # [batch, seq_len, latent_dim]
            z_onehot = self.latent_plan.sample_from_logits(
                z_logits
            )  # [batch, seq_len, 2^latent_dim]
        else:  # inference
            # Sample Z from uniform prior
            z_onehot = self.latent_plan.sample_from_prior(batch_size, seq_len, device=tokens.device)
            z_logits = None

        # 4. Project Z to hidden dimension and inject into decoder
        z_projected = self.latent_plan.project_to_hidden(z_onehot)
        x_with_z = self.injection(x, z_projected)

        # 5. Process through second half of decoder
        # First block uses injected kv, rest use standard self-attention
        x = self.second_half_blocks[0](x, kv_input=x_with_z)
        for block in self.second_half_blocks[1:]:
            x = block(x)

        # 6. Final output projection
        x = self.norm(x)
        logits = self.output(x)

        assert isinstance(logits, torch.Tensor)
        return logits, z_logits

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with random latent plans.

        Args:
            prompt_tokens: Initial tokens [batch, prompt_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated tokens [batch, prompt_len + max_new_tokens]
        """
        tokens = prompt_tokens

        for _ in range(max_new_tokens):
            # Forward pass in inference mode
            logits, _ = self.forward(tokens, mode="inference")

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Truncate if exceeds max length
            if tokens.shape[1] > self.config.max_seq_len:
                tokens = tokens[:, -self.config.max_seq_len :]

        return tokens
