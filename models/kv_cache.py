"""
KV Cache for efficient inference
"""

import torch
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KVCache:
    """
    Key-Value cache for efficient token-by-token generation
    Stores past key and value tensors to avoid recomputation
    """

    def __init__(self, n_layers: int, batch_size: int, max_seq_len: int,
                 n_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Initialize KV cache

        Args:
            n_layers: Number of transformer layers
            batch_size: Batch size
            max_seq_len: Maximum sequence length
            n_heads: Number of attention heads
            head_dim: Dimension of each head
            device: Device to store cache on
            dtype: Data type for cache
        """
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Initialize cache for each layer
        # Shape: (batch_size, n_heads, max_seq_len, head_dim)
        self.key_cache = [
            torch.zeros(
                (batch_size, n_heads, max_seq_len, head_dim),
                device=device,
                dtype=dtype
            )
            for _ in range(n_layers)
        ]

        self.value_cache = [
            torch.zeros(
                (batch_size, n_heads, max_seq_len, head_dim),
                device=device,
                dtype=dtype
            )
            for _ in range(n_layers)
        ]

        # Track current position
        self.current_length = 0

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """
        Update cache for a specific layer

        Args:
            layer_idx: Index of the layer
            key: New key tensor (batch_size, n_heads, seq_len, head_dim)
            value: New value tensor (batch_size, n_heads, seq_len, head_dim)
        """
        seq_len = key.shape[2]

        # Check if we have space
        if self.current_length + seq_len > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: current_length={self.current_length}, "
                f"new_seq_len={seq_len}, max_seq_len={self.max_seq_len}"
            )

        # Update cache
        self.key_cache[layer_idx][:, :, self.current_length:self.current_length + seq_len, :] = key
        self.value_cache[layer_idx][:, :, self.current_length:self.current_length + seq_len, :] = value

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key and value for a specific layer

        Returns:
            key: Cached key tensor up to current length
            value: Cached value tensor up to current length
        """
        if self.current_length == 0:
            return None, None

        key = self.key_cache[layer_idx][:, :, :self.current_length, :]
        value = self.value_cache[layer_idx][:, :, :self.current_length, :]

        return key, value

    def increment_length(self, increment: int = 1):
        """Increment the current length counter"""
        self.current_length += increment

        if self.current_length > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: current_length={self.current_length}, "
                f"max_seq_len={self.max_seq_len}"
            )

    def reset(self):
        """Reset cache to empty state"""
        self.current_length = 0

        # Optionally zero out cache (not strictly necessary)
        for i in range(self.n_layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def get_memory_usage(self) -> int:
        """Get memory usage in bytes"""
        # Each cache entry: batch_size * n_heads * max_seq_len * head_dim * dtype_size
        dtype_size = 4 if self.dtype == torch.float32 else 2  # float32=4, float16=2

        per_cache_size = (
            self.batch_size * self.n_heads * self.max_seq_len * self.head_dim * dtype_size
        )

        # Key + Value cache for all layers
        total_size = 2 * self.n_layers * per_cache_size

        return total_size


class SimplifiedKVCache:
    """
    Simplified KV cache that stores as list of tuples
    Compatible with model's forward pass
    """

    def __init__(self):
        self.cache = None

    def get(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get cache"""
        return self.cache

    def update(self, new_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update cache with new key-value pairs"""
        self.cache = new_cache

    def reset(self):
        """Reset cache"""
        self.cache = None

    def is_empty(self) -> bool:
        """Check if cache is empty"""
        return self.cache is None


def test_kv_cache():
    """Test KV cache functionality"""
    print("Testing KV Cache...")

    # Configuration
    n_layers = 8
    batch_size = 1
    max_seq_len = 256
    n_heads = 8
    head_dim = 32
    device = torch.device('cpu')

    # Create cache
    cache = KVCache(
        n_layers=n_layers,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device=device
    )

    print(f"Cache created:")
    print(f"  Memory usage: {cache.get_memory_usage():,} bytes")

    # Simulate adding tokens
    seq_len = 10

    for layer_idx in range(n_layers):
        key = torch.randn(batch_size, n_heads, seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, seq_len, head_dim)

        cache.update(layer_idx, key, value)

    cache.increment_length(seq_len)

    print(f"  Current length: {cache.current_length}")

    # Retrieve cache
    for layer_idx in range(n_layers):
        key, value = cache.get(layer_idx)
        assert key.shape == (batch_size, n_heads, seq_len, head_dim)
        assert value.shape == (batch_size, n_heads, seq_len, head_dim)

    print("  ✓ Cache update and retrieval works")

    # Add more tokens
    new_seq_len = 5

    for layer_idx in range(n_layers):
        key = torch.randn(batch_size, n_heads, new_seq_len, head_dim)
        value = torch.randn(batch_size, n_heads, new_seq_len, head_dim)

        cache.update(layer_idx, key, value)

    cache.increment_length(new_seq_len)

    print(f"  Current length: {cache.current_length}")

    # Retrieve updated cache
    for layer_idx in range(n_layers):
        key, value = cache.get(layer_idx)
        expected_len = seq_len + new_seq_len
        assert key.shape == (batch_size, n_heads, expected_len, head_dim)
        assert value.shape == (batch_size, n_heads, expected_len, head_dim)

    print("  ✓ Cache incremental update works")

    # Test reset
    cache.reset()
    assert cache.current_length == 0
    print("  ✓ Cache reset works")

    print("\n✓ All KV cache tests passed")


if __name__ == "__main__":
    test_kv_cache()
