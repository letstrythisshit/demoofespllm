"""
Article Compressor - LZ4 compression for articles
"""

import json
import lz4.frame
import logging
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleCompressor:
    """Compress articles using LZ4"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.articles_dir = Path(self.config['knowledge_base']['articles_dir'])
        self.compressed_dir = Path(self.config['knowledge_base']['compressed_dir'])
        self.compressed_dir.mkdir(parents=True, exist_ok=True)

        # LZ4 compression level (0-16, higher = better compression)
        self.compression_level = self.config['knowledge_base']['compression'].get('level', 9)

    def compress_all_articles(self) -> Tuple[int, Dict]:
        """Compress all articles in corpus"""
        article_files = list(self.articles_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        logger.info(f"Compressing {len(article_files)} articles")

        total_original_size = 0
        total_compressed_size = 0
        compressed_count = 0
        failed_count = 0

        compression_stats = []

        for article_path in tqdm(article_files, desc="Compressing articles"):
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)

                # Compress article
                original_size, compressed_size = self.compress_article(article)

                total_original_size += original_size
                total_compressed_size += compressed_size
                compressed_count += 1

                ratio = compressed_size / original_size if original_size > 0 else 0
                compression_stats.append({
                    'id': article['id'],
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'ratio': ratio
                })

            except Exception as e:
                logger.error(f"Error compressing {article_path}: {e}")
                failed_count += 1

        # Calculate overall statistics
        overall_ratio = total_compressed_size / total_original_size if total_original_size > 0 else 0

        stats = {
            'compressed_articles': compressed_count,
            'failed_articles': failed_count,
            'total_original_size': total_original_size,
            'total_compressed_size': total_compressed_size,
            'compression_ratio': overall_ratio,
            'space_saved': total_original_size - total_compressed_size,
            'space_saved_percent': (1 - overall_ratio) * 100,
            'article_stats': compression_stats
        }

        # Save compression statistics
        stats_path = self.compressed_dir / "compression_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Compression complete: {compressed_count} articles")
        logger.info(f"Original size: {total_original_size:,} bytes")
        logger.info(f"Compressed size: {total_compressed_size:,} bytes")
        logger.info(f"Compression ratio: {overall_ratio:.2%}")
        logger.info(f"Space saved: {stats['space_saved_percent']:.1f}%")

        return compressed_count, stats

    def compress_article(self, article: Dict) -> Tuple[int, int]:
        """Compress a single article"""
        # Prepare data for compression
        # We'll compress just the content, keep metadata separate for fast access
        content = article['content']
        content_bytes = content.encode('utf-8')

        original_size = len(content_bytes)

        # Compress using LZ4
        compressed_data = lz4.frame.compress(
            content_bytes,
            compression_level=self.compression_level
        )

        compressed_size = len(compressed_data)

        # Prepare compressed article package
        compressed_article = {
            'id': article['id'],
            'title': article['title'],
            'category': article['category'],
            'keywords': article['keywords'],
            'source': article['source'],
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / original_size
        }

        # Save metadata as JSON
        article_dir = self.compressed_dir / article['category']
        article_dir.mkdir(exist_ok=True)

        metadata_path = article_dir / f"article_{article['id']:05d}_meta.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(compressed_article, f, indent=2)

        # Save compressed content as binary
        content_path = article_dir / f"article_{article['id']:05d}_content.lz4"
        with open(content_path, 'wb') as f:
            f.write(compressed_data)

        return original_size, compressed_size

    def decompress_article(self, article_id: int, category: str = None) -> str:
        """Decompress and return article content"""
        # If category not provided, search for it
        if category is None:
            category = self._find_article_category(article_id)
            if category is None:
                raise ValueError(f"Article {article_id} not found")

        content_path = self.compressed_dir / category / f"article_{article_id:05d}_content.lz4"

        if not content_path.exists():
            raise FileNotFoundError(f"Compressed article not found: {content_path}")

        # Read compressed data
        with open(content_path, 'rb') as f:
            compressed_data = f.read()

        # Decompress
        decompressed_data = lz4.frame.decompress(compressed_data)
        content = decompressed_data.decode('utf-8')

        return content

    def load_article_metadata(self, article_id: int, category: str = None) -> Dict:
        """Load article metadata without decompressing content"""
        if category is None:
            category = self._find_article_category(article_id)
            if category is None:
                raise ValueError(f"Article {article_id} not found")

        metadata_path = self.compressed_dir / category / f"article_{article_id:05d}_meta.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Article metadata not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def _find_article_category(self, article_id: int) -> str:
        """Find which category an article belongs to"""
        for category_dir in self.compressed_dir.iterdir():
            if not category_dir.is_dir():
                continue

            metadata_path = category_dir / f"article_{article_id:05d}_meta.json"
            if metadata_path.exists():
                return category_dir.name

        return None

    def test_compression(self, num_samples: int = 10):
        """Test compression/decompression on sample articles"""
        logger.info(f"Testing compression on {num_samples} sample articles")

        article_files = list(self.articles_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        if len(article_files) == 0:
            logger.error("No articles found to test")
            return

        import random
        sample_files = random.sample(article_files, min(num_samples, len(article_files)))

        success_count = 0
        for article_path in sample_files:
            try:
                # Load original
                with open(article_path, 'r', encoding='utf-8') as f:
                    original_article = json.load(f)

                original_content = original_article['content']
                article_id = original_article['id']
                category = original_article['category']

                # Decompress
                decompressed_content = self.decompress_article(article_id, category)

                # Compare
                if original_content == decompressed_content:
                    logger.info(f"✓ Article {article_id} compression test passed")
                    success_count += 1
                else:
                    logger.error(f"✗ Article {article_id} content mismatch!")

            except Exception as e:
                logger.error(f"Error testing article {article_path}: {e}")

        logger.info(f"Compression test results: {success_count}/{len(sample_files)} passed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Compress article corpus')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--test', action='store_true',
                       help='Test compression/decompression')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to test')

    args = parser.parse_args()

    compressor = ArticleCompressor(args.config)

    if args.test:
        compressor.test_compression(args.samples)
    else:
        count, stats = compressor.compress_all_articles()
        print(f"\n✓ Successfully compressed {count} articles")
        print(f"✓ Original size: {stats['total_original_size']:,} bytes")
        print(f"✓ Compressed size: {stats['total_compressed_size']:,} bytes")
        print(f"✓ Space saved: {stats['space_saved_percent']:.1f}%")


if __name__ == "__main__":
    main()
