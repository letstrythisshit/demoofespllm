"""
Inference Engine - Text generation with model
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
import logging
import yaml
from pathlib import Path

from models.architecture import GPTModel
from models.tokenizer import BPETokenizer
from models.kv_cache import SimplifiedKVCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Text generation engine"""

    def __init__(self, model_path: str, tokenizer_path: str, config_path: str = "config.yaml"):
        """
        Initialize inference engine

        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer directory
            config_path: Path to config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.inference_config = self.config['inference']

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = GPTModel(config_path).to(self.device)
        self._load_model(model_path)
        self.model.eval()

        logger.info("Inference engine initialized")

    def _load_model(self, model_path: str):
        """Load model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling
            stream: Whether to stream tokens (for streaming, use generate_stream)

        Returns:
            Generated text
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.inference_config['max_new_tokens']
        temperature = temperature or self.inference_config['temperature']
        top_k = top_k or self.inference_config['top_k']
        top_p = top_p or self.inference_config['top_p']
        do_sample = do_sample if do_sample is not None else self.inference_config['do_sample']

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self._generate_tokens(
                input_ids,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                do_sample
            )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        return generated_text

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = None
    ):
        """
        Generate text with streaming (yields tokens as they're generated)

        Yields:
            Generated tokens as strings
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.inference_config['max_new_tokens']
        temperature = temperature or self.inference_config['temperature']
        top_k = top_k or self.inference_config['top_k']
        top_p = top_p or self.inference_config['top_p']
        do_sample = do_sample if do_sample is not None else self.inference_config['do_sample']

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate with streaming
        kv_cache = SimplifiedKVCache()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                if kv_cache.is_empty():
                    # First pass - use full sequence
                    logits, new_cache = self.model(input_ids, use_cache=True)
                    kv_cache.update(new_cache)
                else:
                    # Subsequent passes - use only last token
                    last_token = input_ids[:, -1:]
                    logits, new_cache = self.model(
                        last_token,
                        kv_cache=kv_cache.get(),
                        use_cache=True
                    )
                    kv_cache.update(new_cache)

                # Get logits for last token
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Sample next token
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[:, indices_to_remove] = float('-inf')

                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Decode token
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)

                yield token_text

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool
    ) -> torch.Tensor:
        """Generate tokens (non-streaming)"""
        kv_cache = SimplifiedKVCache()

        for _ in range(max_new_tokens):
            # Forward pass
            if kv_cache.is_empty():
                logits, new_cache = self.model(input_ids, use_cache=True)
                kv_cache.update(new_cache)
            else:
                last_token = input_ids[:, -1:]
                logits, new_cache = self.model(
                    last_token,
                    kv_cache=kv_cache.get(),
                    use_cache=True
                )
                kv_cache.update(new_cache)

            # Get next token logits
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Sample
            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def main():
    """Test inference engine"""
    import argparse

    parser = argparse.ArgumentParser(description='Test inference engine')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer',
                       help='Path to tokenizer')
    parser.add_argument('--prompt', type=str,
                       help='Test prompt')
    parser.add_argument('--stream', action='store_true',
                       help='Use streaming generation')

    args = parser.parse_args()

    engine = InferenceEngine(args.model, args.tokenizer)

    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerated text:")
        print("-" * 80)

        if args.stream:
            for token in engine.generate_stream(args.prompt):
                print(token, end='', flush=True)
            print()
        else:
            output = engine.generate(args.prompt)
            print(output)

    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit.")

        while True:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            if not prompt:
                continue

            print("\nGenerated:")
            print("-" * 80)

            for token in engine.generate_stream(prompt):
                print(token, end='', flush=True)

            print("\n")


if __name__ == "__main__":
    main()
