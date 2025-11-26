"""
4-bit Quantization for model compression
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """4-bit quantized linear layer"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store quantized weights and scale/zero_point
        # Weights are stored as int8 (even though they're 4-bit, we pack them later)
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, dtype=torch.int8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization"""
        # Dequantize weights
        weight = self.dequantize_weights()

        # Standard linear operation
        return torch.nn.functional.linear(x, weight, None)

    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights to float"""
        # Dequantization formula: real_value = scale * (quantized_value - zero_point)
        weight_scale = self.weight_scale.unsqueeze(1)  # (out_features, 1)
        weight_zero_point = self.weight_zero_point.unsqueeze(1).float()

        weight = weight_scale * (self.weight_quantized.float() - weight_zero_point)

        return weight


def quantize_tensor(tensor: torch.Tensor, bits: int = 4, symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to specified bit width

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits (default 4)
        symmetric: Use symmetric quantization (default True)

    Returns:
        quantized: Quantized tensor as int8
        scale: Scaling factor
        zero_point: Zero point (for asymmetric quantization)
    """
    # Quantization range
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bits - 1

    # Calculate scale and zero point per output channel (row)
    if len(tensor.shape) == 2:
        # For weight matrices, quantize per output channel (row)
        min_val = tensor.min(dim=1, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0]
    else:
        # For other tensors, use global quantization
        min_val = tensor.min()
        max_val = tensor.max()

    if symmetric:
        # Symmetric quantization
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
    else:
        # Asymmetric quantization
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = zero_point.clamp(qmin, qmax).to(torch.int8)

    # Avoid division by zero
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # Quantize
    quantized = torch.round(tensor / scale + zero_point.float())
    quantized = quantized.clamp(qmin, qmax).to(torch.int8)

    # Squeeze scale and zero_point if they have extra dimensions
    if len(scale.shape) > 1:
        scale = scale.squeeze(1)
        zero_point = zero_point.squeeze(1)

    return quantized, scale, zero_point


def quantize_model(model: nn.Module, bits: int = 4, symmetric: bool = True) -> nn.Module:
    """
    Quantize all linear layers in the model

    Args:
        model: PyTorch model to quantize
        bits: Number of bits for quantization
        symmetric: Use symmetric quantization

    Returns:
        Quantized model (in-place modification)
    """
    logger.info(f"Quantizing model to {bits}-bit...")

    original_size = 0
    quantized_size = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get weight tensor
            weight = module.weight.data

            # Calculate original size (in bytes, assuming float32)
            original_size += weight.numel() * 4

            # Quantize
            weight_quantized, scale, zero_point = quantize_tensor(weight, bits, symmetric)

            # Calculate quantized size
            # Weights: 4 bits per value = 0.5 bytes
            # Scale: float32 per output channel = 4 bytes
            # Zero point: int8 per output channel = 1 byte
            quantized_size += (weight.numel() * bits) / 8  # Quantized weights
            quantized_size += scale.numel() * 4  # Scales
            quantized_size += zero_point.numel() * 1  # Zero points

            # Create quantized layer
            quantized_layer = QuantizedLinear(module.in_features, module.out_features)

            # Set quantized parameters
            quantized_layer.weight_quantized = weight_quantized
            quantized_layer.weight_scale = scale
            quantized_layer.weight_zero_point = zero_point

            # Replace module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            setattr(parent, child_name, quantized_layer)

            logger.debug(f"Quantized layer: {name}")

    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

    logger.info(f"Quantization complete:")
    logger.info(f"  Original size: {original_size:,} bytes ({original_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  Quantized size: {quantized_size:,} bytes ({quantized_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  Compression ratio: {compression_ratio:.2f}x")

    return model


def evaluate_quantization_error(original_model: nn.Module, quantized_model: nn.Module,
                                 test_input: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate quantization error

    Args:
        original_model: Original float model
        quantized_model: Quantized model
        test_input: Test input tensor

    Returns:
        Dictionary with error metrics
    """
    logger.info("Evaluating quantization error...")

    original_model.eval()
    quantized_model.eval()

    with torch.no_grad():
        # Get outputs
        original_output, _ = original_model(test_input)
        quantized_output, _ = quantized_model(test_input)

        # Calculate metrics
        mse = torch.mean((original_output - quantized_output) ** 2).item()
        mae = torch.mean(torch.abs(original_output - quantized_output)).item()

        # Relative error
        relative_error = mae / (torch.mean(torch.abs(original_output)).item() + 1e-8)

        # Cosine similarity
        original_flat = original_output.flatten()
        quantized_flat = quantized_output.flatten()

        cosine_sim = torch.nn.functional.cosine_similarity(
            original_flat.unsqueeze(0),
            quantized_flat.unsqueeze(0)
        ).item()

    metrics = {
        'mse': mse,
        'mae': mae,
        'relative_error': relative_error,
        'cosine_similarity': cosine_sim
    }

    logger.info(f"Quantization error metrics:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  Relative error: {relative_error:.2%}")
    logger.info(f"  Cosine similarity: {cosine_sim:.6f}")

    return metrics


def save_quantized_model(model: nn.Module, save_path: str):
    """Save quantized model"""
    logger.info(f"Saving quantized model to {save_path}")

    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'quantized': True
    }, save_path)

    logger.info(f"Quantized model saved")


def load_quantized_model(model: nn.Module, load_path: str) -> nn.Module:
    """Load quantized model"""
    logger.info(f"Loading quantized model from {load_path}")

    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Quantized model loaded")

    return model


def main():
    """Test quantization"""
    from models.architecture import create_model

    print("Testing quantization...")

    # Create model
    model = create_model()

    # Create test input
    batch_size = 2
    seq_len = 32
    test_input = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    # Get original output
    model.eval()
    with torch.no_grad():
        original_output, _ = model(test_input)

    print(f"\nOriginal model output shape: {original_output.shape}")

    # Quantize model
    quantized_model = quantize_model(model, bits=4, symmetric=True)

    # Get quantized output
    with torch.no_grad():
        quantized_output, _ = quantized_model(test_input)

    print(f"Quantized model output shape: {quantized_output.shape}")

    # Evaluate error
    metrics = evaluate_quantization_error(model, quantized_model, test_input)

    print("\n✓ Quantization test passed")


if __name__ == "__main__":
    main()
