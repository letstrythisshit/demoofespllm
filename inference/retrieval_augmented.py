"""
Retrieval-Augmented Generation (RAG) Pipeline
Combines retrieval with generation for factual responses
"""

import logging
import time
from typing import List, Dict
import yaml

from inference.engine import InferenceEngine
from knowledge_base.retrieval_engine import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config_path: str = "config.yaml"
    ):
        """
        Initialize RAG pipeline

        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
            config_path: Path to config
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info("Initializing RAG pipeline...")

        # Initialize retrieval engine
        logger.info("Loading retrieval engine...")
        self.retrieval_engine = RetrievalEngine(config_path)

        # Initialize inference engine
        logger.info("Loading inference engine...")
        self.inference_engine = InferenceEngine(model_path, tokenizer_path, config_path)

        # Configuration
        self.top_k = self.config['knowledge_base']['retrieval']['top_k']

        logger.info("RAG pipeline ready!")

    def query(self, question: str, verbose: bool = True) -> Dict:
        """
        Process a query through the RAG pipeline

        Args:
            question: User query
            verbose: Print intermediate steps

        Returns:
            Dictionary with query results
        """
        start_time = time.time()

        if verbose:
            print(f"\nQuery: {question}")
            print("=" * 80)

        # Step 1: Retrieve relevant articles
        retrieval_start = time.time()
        retrieved_articles = self.retrieval_engine.search_and_retrieve(question, self.top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        if verbose:
            print(f"\n[1] Retrieved {len(retrieved_articles)} articles ({retrieval_time:.1f}ms)")
            for i, article in enumerate(retrieved_articles, 1):
                print(f"    {i}. {article['title']} (score: {article['score']:.3f})")

        # Step 2: Extract facts from retrieved articles
        facts = self._extract_facts_from_articles(retrieved_articles)

        if verbose:
            print(f"\n[2] Extracted facts:")
            print(f"    {facts[:200]}...")

        # Step 3: Generate response
        generation_start = time.time()
        prompt = self._format_prompt(question, facts)

        if verbose:
            print(f"\n[3] Generating response...")

        response = self.inference_engine.generate(prompt)

        generation_time = (time.time() - generation_start) * 1000

        # Total time
        total_time = (time.time() - start_time) * 1000

        if verbose:
            print(f"\n[4] Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            print(f"\nTimings:")
            print(f"  Retrieval: {retrieval_time:.1f}ms")
            print(f"  Generation: {generation_time:.1f}ms")
            print(f"  Total: {total_time:.1f}ms")

        return {
            'question': question,
            'response': response,
            'retrieved_articles': retrieved_articles,
            'facts': facts,
            'timing': {
                'retrieval_ms': retrieval_time,
                'generation_ms': generation_time,
                'total_ms': total_time
            }
        }

    def query_stream(self, question: str):
        """
        Process query with streaming response

        Yields:
            Response tokens as they're generated
        """
        # Retrieve articles
        retrieved_articles = self.retrieval_engine.search_and_retrieve(question, self.top_k)

        # Extract facts
        facts = self._extract_facts_from_articles(retrieved_articles)

        # Format prompt
        prompt = self._format_prompt(question, facts)

        # Generate with streaming
        for token in self.inference_engine.generate_stream(prompt):
            yield token

    def _extract_facts_from_articles(self, articles: List[Dict]) -> str:
        """Extract and combine facts from multiple articles"""
        facts_parts = []

        for article in articles:
            content = article['content']

            # Take first few sentences from each article
            sentences = content.split('.')[:5]
            excerpt = '. '.join(sentences) + '.'

            facts_parts.append(f"From {article['title']}: {excerpt}")

        # Combine all facts
        facts = ' '.join(facts_parts)

        # Limit total length
        max_facts_length = 500
        if len(facts) > max_facts_length:
            facts = facts[:max_facts_length] + "..."

        return facts

    def _format_prompt(self, question: str, facts: str) -> str:
        """Format prompt for generation"""
        prompt = f"""Query: {question}
Facts: {facts}
Response:"""

        return prompt

    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process multiple queries"""
        results = []

        for question in questions:
            result = self.query(question, verbose=False)
            results.append(result)

        return results


def main():
    """Interactive RAG demo"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG Pipeline Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer',
                       help='Path to tokenizer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config')
    parser.add_argument('--query', type=str,
                       help='Single query to process')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RAGPipeline(args.model, args.tokenizer, args.config)

    if args.query:
        # Single query
        pipeline.query(args.query, verbose=True)
    else:
        # Interactive mode
        print("\n" + "=" * 80)
        print("RAG Pipeline - Interactive Mode")
        print("=" * 80)
        print("Ask questions about survival topics.")
        print("Type 'quit' to exit.")
        print("=" * 80)

        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            try:
                pipeline.query(question, verbose=True)
            except Exception as e:
                print(f"Error processing query: {e}")
                logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
