"""
Transformer Model Architecture - Decoder-only for ESP32
Target: ~2M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import yaml


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - more parameter efficient"""

    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings if provided
        if rope is not None:
            cos, sin = rope
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache for inference
        if use_cache:
            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output, new_kv_cache


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm architecture (more stable)
        # Self-attention
        normed = self.norm1(x)
        attn_out, new_kv_cache = self.attention(normed, mask, rope, kv_cache, use_cache)
        x = x + self.dropout(attn_out)

        # Feed-forward
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)

        return x, new_kv_cache


class GPTModel(nn.Module):
    """
    Decoder-only transformer model for text generation
    Target: ~2M parameters
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config['model']

        self.vocab_size = model_config['vocab_size']
        self.d_model = model_config['d_model']
        self.n_layers = model_config['n_layers']
        self.n_heads = model_config['n_heads']
        self.d_ff = model_config['d_ff']
        self.max_seq_len = model_config['max_seq_len']
        self.dropout = model_config['dropout']

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Rotary positional embeddings
        self.rope = RotaryPositionalEmbedding(self.d_model // self.n_heads, self.max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])

        # Final layer norm
        self.norm_f = nn.LayerNorm(self.d_model)

        # Output projection
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Tie weights between embedding and output projection (reduces parameters)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            kv_cache: List of (k, v) tuples for each layer
            use_cache: Whether to return KV cache

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            new_kv_cache: List of (k, v) tuples if use_cache=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Get rotary embeddings
        cos, sin = self.rope(seq_len, device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        rope = (cos, sin)

        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)

        # Create causal attention mask
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Combine with padding mask if provided
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        mask = causal_mask * attention_mask

        # Pass through transformer blocks
        new_kv_cache = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv_cache = block(x, mask, rope, layer_kv_cache, use_cache)

            if use_cache:
                new_kv_cache.append(layer_new_kv_cache)

        # Final layer norm
        x = self.norm_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits, new_kv_cache

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params_by_component(self) -> dict:
        """Get parameter count breakdown by component"""
        return {
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'blocks': sum(p.numel() for p in self.blocks.parameters()),
            'norm_f': sum(p.numel() for p in self.norm_f.parameters()),
            'total': self.count_parameters()
        }


def create_model(config_path: str = "config.yaml") -> GPTModel:
    """Create model and print parameter count"""
    model = GPTModel(config_path)

    total_params = model.count_parameters()
    param_breakdown = model.get_num_params_by_component()

    print(f"Model created with {total_params:,} parameters")
    print(f"  Token embedding: {param_breakdown['token_embedding']:,}")
    print(f"  Transformer blocks: {param_breakdown['blocks']:,}")
    print(f"  Final norm: {param_breakdown['norm_f']:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits, _ = model(input_ids)
        print(f"  Output shape: {logits.shape}")

    print("\n✓ Model architecture test passed")
