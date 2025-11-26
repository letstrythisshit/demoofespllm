"""
Retrieval Engine - Search and retrieve articles using BM25
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import re
import time
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Search engine using BM25 ranking"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.index_dir = Path(self.config['knowledge_base']['index_dir'])
        self.top_k = self.config['knowledge_base']['retrieval']['top_k']

        # Load index
        self._load_index()

        # Cache for article content
        self.cache_size = 100
        self.cache_hits = 0
        self.cache_misses = 0

        # Stopwords (same as indexer)
        self.stopwords = self._load_stopwords()

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

    def _load_index(self):
        """Load TF-IDF index from disk"""
        index_path = self.index_dir / "tfidf_index.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                "Please run tfidf_indexer.py first."
            )

        logger.info(f"Loading index from {index_path}")

        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)

        self.vocabulary = index_data['vocabulary']
        self.inverted_index = index_data['inverted_index']
        self.doc_lengths = index_data['doc_lengths']
        self.doc_info = index_data['doc_info']
        self.document_count = index_data['document_count']
        self.avg_doc_length = index_data['avg_doc_length']
        self.idf = index_data['idf']

        # BM25 parameters
        config = index_data.get('config', {})
        self.k1 = config.get('k1', 1.5)
        self.b = config.get('b', 0.75)

        # Create reverse vocabulary for lookup
        self.reverse_vocab = {v: k for k, v in self.vocabulary.items()}

        logger.info(f"Index loaded: {self.document_count} documents, "
                   f"{len(self.vocabulary)} terms")

    def search(self, query: str, top_k: int = None) -> List[Tuple[int, float, Dict]]:
        """
        Search for relevant documents

        Args:
            query: Search query string
            top_k: Number of results to return (default from config)

        Returns:
            List of (doc_id, score, metadata) tuples, sorted by relevance
        """
        start_time = time.time()

        if top_k is None:
            top_k = self.top_k

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("Query produced no valid tokens")
            return []

        # Calculate BM25 scores
        scores = self._calculate_bm25_scores(query_tokens)

        # Get top-k results
        top_doc_ids = np.argsort(scores)[::-1][:top_k]

        results = []
        for doc_id in top_doc_ids:
            score = scores[doc_id]
            if score > 0:  # Only include documents with positive scores
                metadata = self.doc_info[doc_id]
                results.append((doc_id, float(score), metadata))

        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(f"Search completed in {elapsed:.2f}ms, found {len(results)} results")

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (same as indexer)"""
        # Lowercase
        text = text.lower()

        # Extract words
        words = re.findall(r'\b[a-z][a-z]+\b', text)

        # Remove stopwords and short words
        words = [w for w in words if w not in self.stopwords and len(w) > 2]

        # Simple stemming
        words = [self._stem(w) for w in words]

        return words

    def _stem(self, word: str) -> str:
        """Simple stemming (same as indexer)"""
        suffixes = ['ing', 'ed', 'ly', 'er', 'est', 's']

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]

        return word

    def _calculate_bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        """Calculate BM25 scores for all documents"""
        scores = np.zeros(self.document_count)

        for term in query_tokens:
            if term not in self.vocabulary:
                continue

            term_id = self.vocabulary[term]

            # Get IDF score
            idf_score = self.idf.get(term_id, 0)

            # Get postings list
            postings = self.inverted_index.get(term_id, [])

            for doc_id, term_freq in postings:
                # BM25 formula
                doc_length = self.doc_lengths[doc_id]

                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                scores[doc_id] += idf_score * (numerator / denominator)

        return scores

    @lru_cache(maxsize=100)
    def get_article_content(self, doc_id: int) -> str:
        """
        Get article content (with caching)

        Note: This loads from the original articles, not compressed.
        For production, use article_compressor to load compressed articles.
        """
        metadata = self.doc_info[doc_id]
        article_id = metadata['id']
        category = metadata['category']

        # Try to load from compressed first
        compressed_dir = Path(self.config['knowledge_base']['compressed_dir'])
        content_path = compressed_dir / category / f"article_{article_id:05d}_content.lz4"

        if content_path.exists():
            # Load from compressed
            from knowledge_base.article_compressor import ArticleCompressor
            compressor = ArticleCompressor()
            try:
                content = compressor.decompress_article(article_id, category)
                self.cache_hits += 1
                return content
            except Exception as e:
                logger.warning(f"Failed to decompress article {article_id}: {e}")

        # Fallback to original
        articles_dir = Path(self.config['knowledge_base']['articles_dir'])
        article_path = articles_dir / category / f"article_{article_id:05d}.json"

        if not article_path.exists():
            raise FileNotFoundError(f"Article not found: {article_path}")

        import json
        with open(article_path, 'r', encoding='utf-8') as f:
            article = json.load(f)

        self.cache_misses += 1
        return article['content']

    def search_and_retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search and retrieve full article content

        Returns:
            List of dictionaries with metadata and content
        """
        results = self.search(query, top_k)

        retrieved = []
        for doc_id, score, metadata in results:
            try:
                content = self.get_article_content(doc_id)

                retrieved.append({
                    'doc_id': doc_id,
                    'score': score,
                    'title': metadata['title'],
                    'category': metadata['category'],
                    'content': content
                })
            except Exception as e:
                logger.error(f"Error retrieving article {doc_id}: {e}")

        return retrieved

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }


def main():
    """Main entry point - Interactive search"""
    import argparse

    parser = argparse.ArgumentParser(description='Search articles')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--query', type=str,
                       help='Search query')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of results')

    args = parser.parse_args()

    engine = RetrievalEngine(args.config)

    if args.query:
        # Single query
        results = engine.search_and_retrieve(args.query, args.top_k)

        print(f"\nSearch results for: '{args.query}'")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} (Score: {result['score']:.3f})")
            print(f"   Category: {result['category']}")
            print(f"   Preview: {result['content'][:200]}...")

    else:
        # Interactive mode
        print("Interactive Search Mode")
        print("=" * 80)
        print("Enter your query (or 'quit' to exit)")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            results = engine.search_and_retrieve(query, args.top_k)

            print(f"\nFound {len(results)} results:")
            print("-" * 80)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (Score: {result['score']:.3f})")
                print(f"   Category: {result['category']}")
                print(f"   Preview: {result['content'][:200]}...")

        # Show cache stats
        stats = engine.get_cache_stats()
        print(f"\nCache statistics:")
        print(f"  Hits: {stats['cache_hits']}")
        print(f"  Misses: {stats['cache_misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
