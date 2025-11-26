"""
Dataset Generation using Ollama
Generate training examples from article corpus
"""

import json
import jsonlines
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import yaml
import random
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate training dataset using Ollama"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.articles_dir = Path(self.config['knowledge_base']['articles_dir'])
        self.output_dir = Path(self.config['dataset']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ollama configuration
        self.ollama_config = self.config['dataset']['ollama']
        self.model = self.ollama_config['model']
        self.temperature = self.ollama_config['temperature']
        self.max_retries = self.ollama_config['max_retries']

        # Query distribution
        self.query_dist = self.config['dataset']['query_distribution']

        # Response length
        self.response_config = self.config['dataset']['response_length']

        # Initialize Ollama client
        self.client = ollama.Client(host=self.ollama_config['endpoint'])

        # Check if model is available
        self._check_model()

    def _check_model(self):
        """Check if Ollama model is available"""
        try:
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if self.model not in model_names and not any(self.model.startswith(m.split(':')[0]) for m in model_names):
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                logger.info("Attempting to pull model...")
                try:
                    self.client.pull(self.model)
                    logger.info(f"Successfully pulled {self.model}")
                except Exception as e:
                    logger.error(f"Failed to pull model: {e}")
                    logger.error("Please install Ollama and pull the model manually:")
                    logger.error(f"  ollama pull {self.model}")
            else:
                logger.info(f"Using Ollama model: {self.model}")

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Please ensure Ollama is running:")
            logger.error("  ollama serve")

    def generate_dataset(self, target_examples: int, resume: bool = True):
        """
        Generate complete training dataset

        Args:
            target_examples: Number of examples to generate
            resume: Resume from existing checkpoint
        """
        logger.info(f"Generating dataset with {target_examples} examples")

        # Load articles
        articles = self._load_articles()

        if len(articles) == 0:
            logger.error("No articles found! Please run corpus_builder.py first")
            return

        logger.info(f"Loaded {len(articles)} articles")

        # Check for existing data
        output_file = self.output_dir / "training_data.jsonl"
        existing_count = 0

        if resume and output_file.exists():
            existing_count = sum(1 for _ in open(output_file))
            logger.info(f"Found {existing_count} existing examples, resuming...")

        if existing_count >= target_examples:
            logger.info("Target already reached!")
            return

        # Calculate examples needed
        examples_needed = target_examples - existing_count

        # Open output file in append mode
        mode = 'a' if resume else 'w'

        with jsonlines.open(output_file, mode=mode) as writer:
            examples_generated = 0
            failed_count = 0

            # Progress bar
            pbar = tqdm(total=examples_needed, initial=0, desc="Generating examples")

            while examples_generated < examples_needed:
                # Sample random article
                article = random.choice(articles)

                try:
                    # Generate examples for this article
                    examples = self.generate_examples_for_article(article)

                    for example in examples:
                        if examples_generated >= examples_needed:
                            break

                        # Write example
                        writer.write(example)
                        examples_generated += 1
                        pbar.update(1)

                        # Avoid rate limiting
                        time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error generating examples for article {article['id']}: {e}")
                    failed_count += 1

                    if failed_count > 10:
                        logger.warning("Too many failures, waiting 10 seconds...")
                        time.sleep(10)
                        failed_count = 0

            pbar.close()

        logger.info(f"Dataset generation complete!")
        logger.info(f"  Generated: {examples_generated} examples")
        logger.info(f"  Total examples: {existing_count + examples_generated}")
        logger.info(f"  Output: {output_file}")

        # Generate validation split
        self._create_validation_split(output_file)

    def _load_articles(self) -> List[Dict]:
        """Load all articles"""
        article_files = list(self.articles_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        articles = []
        for article_path in article_files:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    # Only use cleaned articles
                    if article.get('cleaned', False):
                        articles.append(article)
            except Exception as e:
                logger.error(f"Error loading {article_path}: {e}")

        return articles

    def generate_examples_for_article(self, article: Dict, num_examples: int = 5) -> List[Dict]:
        """Generate multiple training examples for a single article"""
        examples = []

        # Extract key facts from article
        facts = self._extract_facts(article)

        if not facts:
            return examples

        # Generate diverse queries
        query_types = self._sample_query_types(num_examples)

        for query_type in query_types:
            try:
                query = self._generate_query(article, query_type)
                response = self._generate_response(article, query, facts)

                if query and response:
                    example = {
                        'query': query,
                        'facts': facts,
                        'response': response,
                        'article_id': article['id'],
                        'category': article['category'],
                        'query_type': query_type
                    }
                    examples.append(example)

            except Exception as e:
                logger.debug(f"Failed to generate {query_type} example: {e}")

        return examples

    def _extract_facts(self, article: Dict) -> str:
        """Extract key facts from article"""
        content = article['content']
        title = article['title']

        # Limit content length for context
        max_content_len = 2000
        if len(content) > max_content_len:
            # Take first part of content
            content = content[:max_content_len] + "..."

        prompt = f"""Extract the 3-5 most important facts from this article.
Focus on practical, actionable information.

Title: {title}

Content:
{content}

Provide the facts as a concise paragraph (2-4 sentences)."""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for fact extraction
                    'num_predict': 150
                }
            )

            facts = response['response'].strip()
            return facts

        except Exception as e:
            logger.debug(f"Failed to extract facts: {e}")
            # Fallback: use first few sentences
            sentences = content.split('.')[:3]
            return '. '.join(sentences) + '.'

    def _sample_query_types(self, num: int) -> List[str]:
        """Sample query types based on distribution"""
        query_types = []
        dist = self.query_dist

        types = list(dist.keys())
        weights = list(dist.values())

        for _ in range(num):
            query_type = random.choices(types, weights=weights)[0]
            query_types.append(query_type)

        return query_types

    def _generate_query(self, article: Dict, query_type: str) -> str:
        """Generate a query for the article"""
        title = article['title']
        category = article['category']

        # Query generation prompts
        prompts = {
            'how_to': f"Generate a 'how to' question about: {title}\nCategory: {category}\nQuestion:",
            'what_is': f"Generate a 'what is' question about: {title}\nCategory: {category}\nQuestion:",
            'why': f"Generate a 'why' question about: {title}\nCategory: {category}\nQuestion:",
            'troubleshooting': f"Generate a troubleshooting question about: {title}\nCategory: {category}\nQuestion:",
            'follow_up': f"Generate a follow-up question about: {title}\nCategory: {category}\nQuestion:"
        }

        prompt = prompts.get(query_type, prompts['how_to'])

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': 50
                }
            )

            query = response['response'].strip()

            # Clean up query
            query = query.replace('\n', ' ').strip()
            if not query.endswith('?'):
                query += '?'

            return query

        except Exception as e:
            logger.debug(f"Failed to generate query: {e}")
            # Fallback
            return f"How do I {title.lower()}?"

    def _generate_response(self, article: Dict, query: str, facts: str) -> str:
        """Generate response using facts"""
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided facts.

Facts:
{facts}

Question: {query}

Provide a helpful, natural response (2-5 sentences). Use the facts but rephrase naturally.

Answer:"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.response_config['max_tokens']
                }
            )

            answer = response['response'].strip()

            # Clean up response
            answer = answer.replace('\n\n', ' ')

            return answer

        except Exception as e:
            logger.debug(f"Failed to generate response: {e}")
            # Fallback: use facts directly
            return facts

    def _create_validation_split(self, train_file: Path):
        """Create validation split"""
        val_split = self.config['dataset']['validation_split']

        logger.info(f"Creating validation split ({val_split:.1%})...")

        # Read all examples
        examples = []
        with jsonlines.open(train_file) as reader:
            for example in reader:
                examples.append(example)

        # Shuffle
        random.shuffle(examples)

        # Split
        val_size = int(len(examples) * val_split)
        train_size = len(examples) - val_size

        train_examples = examples[:train_size]
        val_examples = examples[val_size:]

        # Save
        train_output = self.output_dir / "train.jsonl"
        val_output = self.output_dir / "val.jsonl"

        with jsonlines.open(train_output, 'w') as writer:
            writer.write_all(train_examples)

        with jsonlines.open(val_output, 'w') as writer:
            writer.write_all(val_examples)

        logger.info(f"  Train: {len(train_examples)} examples -> {train_output}")
        logger.info(f"  Val: {len(val_examples)} examples -> {val_output}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate training dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--target', type=int,
                       help='Target number of examples (overrides config)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from scratch (ignore existing data)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    target = args.target or config['dataset']['target_examples']

    generator = DatasetGenerator(args.config)
    generator.generate_dataset(target, resume=not args.no_resume)

    print(f"\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
