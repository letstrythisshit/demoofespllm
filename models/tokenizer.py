"""
BPE Tokenizer for domain-specific vocabulary
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import re
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding Tokenizer"""

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        # Vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = []

        # Token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def train(self, texts: List[str], save_path: str = None):
        """Train BPE tokenizer on corpus"""
        logger.info(f"Training tokenizer on {len(texts)} texts")

        # Initialize vocabulary with characters
        vocab = self._initialize_vocab(texts)

        # Learn merges
        logger.info("Learning BPE merges...")
        self.merges = self._learn_merges(texts, vocab)

        # Build final vocabulary
        logger.info("Building final vocabulary...")
        self._build_vocab()

        # Save if path provided
        if save_path:
            self.save(save_path)

        logger.info(f"Tokenizer trained with {len(self.vocab)} tokens")

    def _initialize_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Initialize vocabulary with characters"""
        vocab = {}

        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            vocab[token] = i

        # Add all characters
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)

        # Add characters sorted by frequency
        for char, _ in char_freq.most_common():
            if char not in vocab and len(vocab) < self.vocab_size:
                vocab[char] = len(vocab)

        return vocab

    def _learn_merges(self, texts: List[str], vocab: Dict[str, int], num_merges: int = None) -> List[Tuple[str, str]]:
        """Learn BPE merges"""
        if num_merges is None:
            num_merges = self.vocab_size - len(vocab)

        # Pre-tokenize texts into words
        words = []
        for text in texts:
            words.extend(self._pre_tokenize(text))

        # Convert words to character sequences
        word_freqs = Counter(words)
        splits = {word: list(word) for word in word_freqs.keys()}

        merges = []

        for i in range(num_merges):
            # Count pairs
            pair_freqs = defaultdict(int)

            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) < 2:
                    continue

                for j in range(len(split) - 1):
                    pair = (split[j], split[j + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Get most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            merges.append(best_pair)

            # Merge the pair in all words
            splits = self._merge_pair(best_pair, splits)

            if (i + 1) % 100 == 0:
                logger.info(f"  Learned {i + 1} merges")

        logger.info(f"Learned {len(merges)} merges")
        return merges

    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words"""
        # Split on whitespace and punctuation
        pattern = r'\w+|[^\w\s]'
        words = re.findall(pattern, text.lower())
        return words

    def _merge_pair(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge a pair in all word splits"""
        new_splits = {}

        for word, split in splits.items():
            if len(split) < 2:
                new_splits[word] = split
                continue

            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1

            new_splits[word] = new_split

        return new_splits

    def _build_vocab(self):
        """Build final vocabulary from merges"""
        # Start with special tokens
        self.vocab = {}
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i

        # Add single characters (that appear in merges)
        chars = set()
        for pair in self.merges:
            chars.add(pair[0][0]) if len(pair[0]) == 1 else None
            chars.add(pair[1][0]) if len(pair[1]) == 1 else None

        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Add merged tokens
        for pair in self.merges:
            merged = pair[0] + pair[1]
            if merged not in self.vocab and len(self.vocab) < self.vocab_size:
                self.vocab[merged] = len(self.vocab)

        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")

        # Pre-tokenize
        words = self._pre_tokenize(text)

        # Tokenize each word
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        for word in words:
            word_tokens = self._tokenize_word(word)
            token_ids.extend(word_tokens)

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        return token_ids

    def _tokenize_word(self, word: str) -> List[int]:
        """Tokenize a single word using BPE"""
        if not word:
            return []

        # Start with characters
        tokens = list(word)

        # Apply merges
        for pair in self.merges:
            if len(tokens) < 2:
                break

            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # Convert to IDs
        token_ids = []
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []

        for token_id in token_ids:
            if skip_special_tokens and token_id in [
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id
            ]:
                continue

            token = self.reverse_vocab.get(token_id, self.unk_token)
            tokens.append(token)

        # Join tokens
        text = ''.join(tokens)

        # Add spaces around punctuation for readability
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def save(self, path: str):
        """Save tokenizer to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = path / "vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)

        # Save merges
        merges_path = path / "merges.txt"
        with open(merges_path, 'w') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id
        }

        config_path = path / "tokenizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from disk"""
        path = Path(path)

        # Load config
        config_path = path / "tokenizer_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        tokenizer = cls(config['vocab_size'])

        # Load vocabulary
        vocab_path = path / "vocab.json"
        with open(vocab_path, 'r') as f:
            tokenizer.vocab = json.load(f)

        # Convert string keys to proper types
        tokenizer.vocab = {k: int(v) for k, v in tokenizer.vocab.items()}

        # Load merges
        merges_path = path / "merges.txt"
        merges = []
        with open(merges_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        tokenizer.merges = merges

        # Create reverse vocab
        tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

        # Set special token IDs
        tokenizer.pad_token_id = config['pad_token_id']
        tokenizer.unk_token_id = config['unk_token_id']
        tokenizer.bos_token_id = config['bos_token_id']
        tokenizer.eos_token_id = config['eos_token_id']

        logger.info(f"Tokenizer loaded from {path}")

        return tokenizer


def train_tokenizer_from_articles(
    articles_dir: str,
    output_dir: str,
    vocab_size: int = 8192
):
    """Train tokenizer on article corpus"""
    articles_dir = Path(articles_dir)

    # Load all articles
    logger.info(f"Loading articles from {articles_dir}")

    texts = []
    article_files = list(articles_dir.glob("**/*.json"))
    article_files = [f for f in article_files if f.name != "master_index.json"]

    for article_path in article_files:
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                article = json.load(f)
                texts.append(article['title'])
                texts.append(article['content'])
        except Exception as e:
            logger.error(f"Error loading {article_path}: {e}")

    logger.info(f"Loaded {len(texts)} texts from {len(article_files)} articles")

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size)
    tokenizer.train(texts, output_dir)

    return tokenizer


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--articles-dir', type=str,
                       help='Path to articles directory')
    parser.add_argument('--output-dir', type=str, default='models/tokenizer',
                       help='Output directory')
    parser.add_argument('--vocab-size', type=int, default=8192,
                       help='Vocabulary size')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    articles_dir = args.articles_dir or config['knowledge_base']['articles_dir']

    tokenizer = train_tokenizer_from_articles(
        articles_dir,
        args.output_dir,
        args.vocab_size
    )

    # Test tokenizer
    test_text = "How do I start a fire in wet conditions?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nTest encoding/decoding:")
    print(f"  Original: {test_text}")
    print(f"  Encoded: {encoded[:20]}...")
    print(f"  Decoded: {decoded}")

    print(f"\n✓ Tokenizer trained successfully")
    print(f"✓ Vocabulary size: {len(tokenizer.vocab)}")


if __name__ == "__main__":
    main()
