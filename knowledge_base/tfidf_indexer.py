"""
TF-IDF Indexer - Build search index using BM25
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
import yaml
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFIndexer:
    """Build TF-IDF search index with BM25 ranking"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.articles_dir = Path(self.config['knowledge_base']['articles_dir'])
        self.index_dir = Path(self.config['knowledge_base']['index_dir'])
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # BM25 parameters
        self.k1 = self.config['knowledge_base']['retrieval']['bm25_k1']
        self.b = self.config['knowledge_base']['retrieval']['bm25_b']

        # Stopwords
        self.stopwords = self._load_stopwords()

        # Index structures
        self.vocabulary = {}  # term -> term_id
        self.inverted_index = defaultdict(list)  # term_id -> [(doc_id, term_freq)]
        self.doc_lengths = {}  # doc_id -> length
        self.doc_info = {}  # doc_id -> metadata
        self.document_count = 0
        self.avg_doc_length = 0
        self.idf = {}  # term_id -> idf score

    def _load_stopwords(self) -> set:
        """Load common stopwords"""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this',
            'but', 'they', 'have', 'had', 'what', 'when', 'where', 'who',
            'which', 'why', 'how', 'or', 'can', 'could', 'would', 'should',
            'may', 'might', 'must', 'shall'
        }
        return stopwords

    def build_index(self):
        """Build complete TF-IDF index"""
        logger.info("Building TF-IDF index...")

        # Load all articles
        articles = self._load_articles()

        if len(articles) == 0:
            logger.error("No articles found!")
            return

        self.document_count = len(articles)

        # Build vocabulary and calculate term frequencies
        logger.info("Building vocabulary and term frequencies...")
        for doc_id, article in enumerate(tqdm(articles, desc="Processing articles")):
            self._process_document(doc_id, article)

        # Calculate average document length
        self.avg_doc_length = np.mean(list(self.doc_lengths.values()))

        # Calculate IDF scores
        logger.info("Calculating IDF scores...")
        self._calculate_idf()

        # Save index
        logger.info("Saving index...")
        self._save_index()

        logger.info(f"Index built successfully!")
        logger.info(f"  Documents: {self.document_count}")
        logger.info(f"  Vocabulary size: {len(self.vocabulary)}")
        logger.info(f"  Average document length: {self.avg_doc_length:.1f}")

    def _load_articles(self) -> List[Dict]:
        """Load all articles from disk"""
        article_files = list(self.articles_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        articles = []
        for article_path in article_files:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    articles.append(article)
            except Exception as e:
                logger.error(f"Error loading {article_path}: {e}")

        # Sort by ID for consistency
        articles.sort(key=lambda x: x['id'])

        return articles

    def _process_document(self, doc_id: int, article: Dict):
        """Process a single document"""
        # Extract text
        text = article['content']
        title = article['title']

        # Tokenize (title gets higher weight)
        title_tokens = self._tokenize(title)
        content_tokens = self._tokenize(text)

        # Combine with title boost (title tokens counted 3x)
        all_tokens = title_tokens * 3 + content_tokens

        # Calculate term frequencies
        term_freq = Counter(all_tokens)

        # Store document length
        self.doc_lengths[doc_id] = len(all_tokens)

        # Store document info
        self.doc_info[doc_id] = {
            'id': article['id'],
            'title': article['title'],
            'category': article['category'],
            'length': len(text)
        }

        # Update vocabulary and inverted index
        for term, freq in term_freq.items():
            if term not in self.vocabulary:
                self.vocabulary[term] = len(self.vocabulary)

            term_id = self.vocabulary[term]
            self.inverted_index[term_id].append((doc_id, freq))

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        # Lowercase
        text = text.lower()

        # Extract words
        words = re.findall(r'\b[a-z][a-z]+\b', text)

        # Remove stopwords and short words
        words = [w for w in words if w not in self.stopwords and len(w) > 2]

        # Simple stemming (remove common suffixes)
        words = [self._stem(w) for w in words]

        return words

    def _stem(self, word: str) -> str:
        """Simple stemming"""
        # Remove common suffixes
        suffixes = ['ing', 'ed', 'ly', 'er', 'est', 's']

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]

        return word

    def _calculate_idf(self):
        """Calculate IDF scores for all terms"""
        N = self.document_count

        for term_id, postings in self.inverted_index.items():
            df = len(postings)  # document frequency

            # BM25 IDF formula
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term_id] = idf

    def _save_index(self):
        """Save index to disk"""
        index_data = {
            'vocabulary': self.vocabulary,
            'inverted_index': dict(self.inverted_index),
            'doc_lengths': self.doc_lengths,
            'doc_info': self.doc_info,
            'document_count': self.document_count,
            'avg_doc_length': self.avg_doc_length,
            'idf': self.idf,
            'config': {
                'k1': self.k1,
                'b': self.b
            }
        }

        index_path = self.index_dir / "tfidf_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)

        # Also save vocabulary as JSON for inspection
        vocab_path = self.index_dir / "vocabulary.json"
        vocab_list = sorted(self.vocabulary.items(), key=lambda x: x[1])
        with open(vocab_path, 'w') as f:
            json.dump(dict(vocab_list), f, indent=2)

        # Save statistics
        stats = {
            'document_count': self.document_count,
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': self.avg_doc_length,
            'index_size_bytes': index_path.stat().st_size
        }

        stats_path = self.index_dir / "index_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Index saved to {index_path}")
        logger.info(f"Index size: {stats['index_size_bytes']:,} bytes")

    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        stats_path = self.index_dir / "index_stats.json"

        if not stats_path.exists():
            return {}

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Build TF-IDF search index')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--stats', action='store_true',
                       help='Show index statistics')

    args = parser.parse_args()

    indexer = TFIDFIndexer(args.config)

    if args.stats:
        stats = indexer.get_index_stats()
        if stats:
            print("\nIndex Statistics:")
            print(f"  Documents: {stats['document_count']}")
            print(f"  Vocabulary size: {stats['vocabulary_size']}")
            print(f"  Average document length: {stats['avg_doc_length']:.1f}")
            print(f"  Index size: {stats['index_size_bytes']:,} bytes")
        else:
            print("No index found. Run without --stats to build index.")
    else:
        indexer.build_index()
        print("\n✓ Index built successfully")


if __name__ == "__main__":
    main()
