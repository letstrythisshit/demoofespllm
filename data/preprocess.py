"""
Data Preprocessing for Training
"""

import jsonlines
import logging
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurvivalQADataset(Dataset):
    """Dataset for survival Q&A training"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        """
        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.examples = self._load_data()

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file"""
        examples = []

        with jsonlines.open(self.data_path) as reader:
            for example in reader:
                examples.append(example)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example"""
        example = self.examples[idx]

        # Format input: [BOS] Query: {query} Facts: {facts} Response: {response} [EOS]
        query = example['query']
        facts = example['facts']
        response = example['response']

        # Create input text
        input_text = f"Query: {query}\nFacts: {facts}\nResponse: {response}"

        # Tokenize
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Pad if needed
        padding_length = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # For language modeling, labels are same as input_ids (shifted in the model)
        labels = input_ids.clone()

        # Mask padding tokens in labels (-100 is ignored by loss function)
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def analyze_dataset(data_path: str):
    """Analyze dataset statistics"""
    logger.info(f"Analyzing dataset: {data_path}")

    examples = []
    with jsonlines.open(data_path) as reader:
        for example in reader:
            examples.append(example)

    # Statistics
    total = len(examples)

    query_lengths = [len(e['query'].split()) for e in examples]
    response_lengths = [len(e['response'].split()) for e in examples]
    facts_lengths = [len(e['facts'].split()) for e in examples]

    # Category distribution
    from collections import Counter
    categories = Counter(e['category'] for e in examples)
    query_types = Counter(e.get('query_type', 'unknown') for e in examples)

    print(f"\nDataset Statistics:")
    print(f"  Total examples: {total}")
    print(f"\nLength Statistics (in words):")
    print(f"  Query - Mean: {sum(query_lengths)/len(query_lengths):.1f}, "
          f"Min: {min(query_lengths)}, Max: {max(query_lengths)}")
    print(f"  Facts - Mean: {sum(facts_lengths)/len(facts_lengths):.1f}, "
          f"Min: {min(facts_lengths)}, Max: {max(facts_lengths)}")
    print(f"  Response - Mean: {sum(response_lengths)/len(response_lengths):.1f}, "
          f"Min: {min(response_lengths)}, Max: {max(response_lengths)}")

    print(f"\nCategory Distribution:")
    for category, count in categories.most_common():
        print(f"  {category}: {count} ({count/total*100:.1f}%)")

    print(f"\nQuery Type Distribution:")
    for qtype, count in query_types.most_common():
        print(f"  {qtype}: {count} ({count/total*100:.1f}%)")

    # Show sample
    print(f"\nSample Example:")
    sample = examples[0]
    print(f"  Query: {sample['query']}")
    print(f"  Facts: {sample['facts'][:100]}...")
    print(f"  Response: {sample['response'][:100]}...")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to JSONL data file')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset statistics')

    args = parser.parse_args()

    if args.analyze:
        analyze_dataset(args.data_path)
    else:
        print("Use --analyze to see dataset statistics")


if __name__ == "__main__":
    main()
