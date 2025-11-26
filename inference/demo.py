"""
Interactive Demo - Beautiful CLI interface for RAG system
"""

import sys
from pathlib import Path
from colorama import init, Fore, Style
import time

# Initialize colorama for cross-platform colored output
init(autoreset=True)

from inference.retrieval_augmented import RAGPipeline


class InteractiveDemo:
    """Interactive demo with nice formatting"""

    def __init__(self, model_path: str, tokenizer_path: str, config_path: str = "config.yaml"):
        print(Fore.CYAN + "\n╔═══════════════════════════════════════════════════════════════════════╗")
        print(Fore.CYAN + "║          ESP32 Offline AI Assistant - Interactive Demo              ║")
        print(Fore.CYAN + "╚═══════════════════════════════════════════════════════════════════════╝")

        print(Fore.YELLOW + "\nInitializing system...")

        # Initialize RAG pipeline
        self.pipeline = RAGPipeline(model_path, tokenizer_path, config_path)

        print(Fore.GREEN + "✓ System ready!\n")

        # Statistics
        self.queries_processed = 0
        self.total_retrieval_time = 0
        self.total_generation_time = 0

    def print_header(self):
        """Print header"""
        print("\n" + Fore.CYAN + "═" * 80)
        print(Fore.CYAN + "  ESP32 OFFLINE AI ASSISTANT - SURVIVAL KNOWLEDGE SYSTEM")
        print(Fore.CYAN + "═" * 80)
        print(Fore.WHITE + """
This system can answer questions about:
  • Fire starting techniques
  • Water purification methods
  • Shelter building
  • First aid procedures
  • Navigation skills
  • Food foraging
  • Emergency signaling
  • Weather prediction
  • Knots and tools
  • Emergency procedures

Commands:
  - Type your question and press Enter
  - Type 'help' for this message
  - Type 'stats' for system statistics
  - Type 'quit' or 'exit' to leave
""")
        print(Fore.CYAN + "═" * 80 + "\n")

    def process_query(self, question: str):
        """Process and display a query"""
        print(Fore.YELLOW + f"\n📝 Question: {question}")
        print(Fore.CYAN + "─" * 80)

        # Retrieve articles
        print(Fore.BLUE + "\n🔍 Searching knowledge base...")
        start_time = time.time()

        retrieved_articles = self.pipeline.retrieval_engine.search_and_retrieve(question, self.pipeline.top_k)
        retrieval_time = (time.time() - start_time) * 1000

        print(Fore.GREEN + f"✓ Found {len(retrieved_articles)} relevant articles ({retrieval_time:.1f}ms)")

        # Display articles
        for i, article in enumerate(retrieved_articles, 1):
            score_bar = "█" * int(article['score'] * 10)
            print(Fore.WHITE + f"  {i}. " + Fore.CYAN + f"{article['title']}")
            print(Fore.WHITE + f"     Score: {score_bar} {article['score']:.3f} | Category: {article['category']}")

        # Extract facts
        facts = self.pipeline._extract_facts_from_articles(retrieved_articles)

        print(Fore.BLUE + "\n💡 Key facts extracted:")
        print(Fore.WHITE + f"   {facts[:150]}...")

        # Generate response
        print(Fore.BLUE + "\n🤖 Generating response...")

        prompt = self.pipeline._format_prompt(question, facts)

        gen_start = time.time()
        print(Fore.GREEN + "\n" + "─" * 80)
        print(Fore.GREEN + "RESPONSE:")
        print(Fore.GREEN + "─" * 80)
        print(Fore.WHITE, end='')

        # Stream response
        response_parts = []
        for token in self.pipeline.inference_engine.generate_stream(prompt):
            print(token, end='', flush=True)
            response_parts.append(token)

        generation_time = (time.time() - gen_start) * 1000

        print("\n" + Fore.GREEN + "─" * 80)

        # Timing info
        total_time = retrieval_time + generation_time
        print(Fore.MAGENTA + f"\n⏱️  Retrieval: {retrieval_time:.1f}ms | Generation: {generation_time:.1f}ms | Total: {total_time:.1f}ms")

        # Update statistics
        self.queries_processed += 1
        self.total_retrieval_time += retrieval_time
        self.total_generation_time += generation_time

    def show_stats(self):
        """Show system statistics"""
        print(Fore.CYAN + "\n╔═══════════════════════════════════════════════════════════════════════╗")
        print(Fore.CYAN + "║                         SYSTEM STATISTICS                            ║")
        print(Fore.CYAN + "╚═══════════════════════════════════════════════════════════════════════╝")

        print(Fore.WHITE + f"\nSession Statistics:")
        print(Fore.GREEN + f"  Queries processed: {self.queries_processed}")

        if self.queries_processed > 0:
            avg_retrieval = self.total_retrieval_time / self.queries_processed
            avg_generation = self.total_generation_time / self.queries_processed

            print(Fore.YELLOW + f"  Average retrieval time: {avg_retrieval:.1f}ms")
            print(Fore.YELLOW + f"  Average generation time: {avg_generation:.1f}ms")
            print(Fore.YELLOW + f"  Average total time: {avg_retrieval + avg_generation:.1f}ms")

        # Cache statistics
        cache_stats = self.pipeline.retrieval_engine.get_cache_stats()
        print(Fore.WHITE + f"\nRetrieval Cache:")
        print(Fore.GREEN + f"  Cache hits: {cache_stats['cache_hits']}")
        print(Fore.GREEN + f"  Cache misses: {cache_stats['cache_misses']}")
        if cache_stats['cache_hits'] + cache_stats['cache_misses'] > 0:
            print(Fore.GREEN + f"  Hit rate: {cache_stats['hit_rate']:.1%}")

        # Model info
        print(Fore.WHITE + f"\nModel Information:")
        print(Fore.CYAN + f"  Parameters: {self.pipeline.inference_engine.model.count_parameters():,}")
        print(Fore.CYAN + f"  Vocabulary size: {self.pipeline.inference_engine.tokenizer.vocab_size}")

    def run(self):
        """Run interactive demo"""
        self.print_header()

        # Example queries
        examples = [
            "How do I start a fire without matches?",
            "What are the best methods to purify water?",
            "How can I build an emergency shelter?",
        ]

        print(Fore.YELLOW + "💡 Example questions you can ask:")
        for i, example in enumerate(examples, 1):
            print(Fore.WHITE + f"   {i}. {example}")

        print()

        while True:
            try:
                # Get user input
                user_input = input(Fore.CYAN + "Your question: " + Fore.WHITE).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(Fore.YELLOW + "\n👋 Thank you for using the ESP32 AI Assistant!")
                    if self.queries_processed > 0:
                        print(Fore.GREEN + f"   Processed {self.queries_processed} queries this session.")
                    print()
                    break

                elif user_input.lower() == 'help':
                    self.print_header()
                    continue

                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue

                elif user_input.lower() == 'clear':
                    # Clear screen (works on most terminals)
                    print("\033[2J\033[H")
                    self.print_header()
                    continue

                # Process query
                self.process_query(user_input)

            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\n👋 Interrupted. Type 'quit' to exit.")
                continue

            except Exception as e:
                print(Fore.RED + f"\n❌ Error: {e}")
                print(Fore.YELLOW + "Please try again or type 'quit' to exit.")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Interactive RAG Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer',
                       help='Path to tokenizer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.model).exists():
        print(Fore.RED + f"Error: Model not found at {args.model}")
        print(Fore.YELLOW + "Please train the model first: python training/train.py")
        sys.exit(1)

    if not Path(args.tokenizer).exists():
        print(Fore.RED + f"Error: Tokenizer not found at {args.tokenizer}")
        print(Fore.YELLOW + "Please train the tokenizer first: python models/tokenizer.py")
        sys.exit(1)

    # Run demo
    demo = InteractiveDemo(args.model, args.tokenizer, args.config)
    demo.run()


if __name__ == "__main__":
    main()
