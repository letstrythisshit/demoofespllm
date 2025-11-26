"""
End-to-end system test
"""

import sys
from pathlib import Path
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_corpus_building():
    """Test corpus building"""
    print("\n[TEST] Corpus Building")
    print("-" * 80)

    from knowledge_base.corpus_builder import CorpusBuilder

    try:
        builder = CorpusBuilder()

        # Check if articles exist
        article_dir = Path("knowledge_base/articles")

        if not article_dir.exists():
            print("✗ Articles directory not found")
            return False

        article_files = list(article_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        if len(article_files) == 0:
            print("✗ No articles found")
            print("  Run: python knowledge_base/corpus_builder.py --target 100")
            return False

        print(f"✓ Found {len(article_files)} articles")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_article_cleaning():
    """Test article cleaning"""
    print("\n[TEST] Article Cleaning")
    print("-" * 80)

    from knowledge_base.article_cleaner import ArticleCleaner

    try:
        cleaner = ArticleCleaner()
        stats = cleaner.get_cleaning_stats()

        if stats['cleaned_articles'] == 0:
            print("✗ No cleaned articles found")
            print("  Run: python knowledge_base/article_cleaner.py")
            return False

        print(f"✓ {stats['cleaned_articles']} articles cleaned")
        print(f"  Average length: {stats['avg_length']:.0f} chars")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_index_building():
    """Test TF-IDF index"""
    print("\n[TEST] TF-IDF Index")
    print("-" * 80)

    from knowledge_base.tfidf_indexer import TFIDFIndexer

    try:
        indexer = TFIDFIndexer()
        stats = indexer.get_index_stats()

        if not stats:
            print("✗ Index not found")
            print("  Run: python knowledge_base/tfidf_indexer.py")
            return False

        print(f"✓ Index built with {stats['document_count']} documents")
        print(f"  Vocabulary size: {stats['vocabulary_size']}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_retrieval():
    """Test retrieval engine"""
    print("\n[TEST] Retrieval Engine")
    print("-" * 80)

    from knowledge_base.retrieval_engine import RetrievalEngine

    try:
        engine = RetrievalEngine()

        # Test search
        test_query = "How do I start a fire?"
        start_time = time.time()
        results = engine.search(test_query, top_k=3)
        elapsed = (time.time() - start_time) * 1000

        if len(results) == 0:
            print("✗ No results returned")
            return False

        print(f"✓ Retrieval works (found {len(results)} results in {elapsed:.2f}ms)")

        for i, (doc_id, score, metadata) in enumerate(results, 1):
            print(f"  {i}. {metadata['title']} (score: {score:.3f})")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_tokenizer():
    """Test tokenizer"""
    print("\n[TEST] Tokenizer")
    print("-" * 80)

    from models.tokenizer import BPETokenizer

    try:
        tokenizer_path = Path("models/tokenizer")

        if not tokenizer_path.exists():
            print("✗ Tokenizer not found")
            print("  Run: python models/tokenizer.py --articles-dir knowledge_base/articles")
            return False

        tokenizer = BPETokenizer.load(str(tokenizer_path))

        # Test encoding/decoding
        test_text = "How do I start a fire without matches?"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        print(f"✓ Tokenizer works (vocab size: {len(tokenizer.vocab)})")
        print(f"  Test text: {test_text}")
        print(f"  Encoded length: {len(encoded)} tokens")
        print(f"  Decoded: {decoded}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_architecture():
    """Test model architecture"""
    print("\n[TEST] Model Architecture")
    print("-" * 80)

    from models.architecture import GPTModel
    import torch

    try:
        model = GPTModel()
        param_count = model.count_parameters()

        print(f"✓ Model created with {param_count:,} parameters")

        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = model(input_ids)

        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {logits.shape}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_dataset():
    """Test dataset generation"""
    print("\n[TEST] Dataset Generation")
    print("-" * 80)

    dataset_path = Path("data/generated/train.jsonl")

    if not dataset_path.exists():
        print("✗ Training dataset not found")
        print("  Run: python data/generate_dataset.py --target 100")
        return False

    # Count examples
    count = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            count += 1

    print(f"✓ Found {count} training examples")

    return True


def test_training():
    """Test if model is trained"""
    print("\n[TEST] Model Training")
    print("-" * 80)

    checkpoint_dir = Path("models/checkpoints")

    if not checkpoint_dir.exists():
        print("✗ No checkpoints found")
        print("  Run: python training/train.py")
        return False

    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if len(checkpoints) == 0:
        print("✗ No model checkpoints found")
        print("  Run: python training/train.py")
        return False

    print(f"✓ Found {len(checkpoints)} checkpoint(s)")

    # Check for best model
    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        print(f"  Best model: {best_model}")
    else:
        print(f"  Latest checkpoint: {checkpoints[-1]}")

    return True


def test_inference():
    """Test inference"""
    print("\n[TEST] Inference Engine")
    print("-" * 80)

    from inference.engine import InferenceEngine
    from pathlib import Path

    try:
        # Find model checkpoint
        checkpoint_dir = Path("models/checkpoints")
        best_model = checkpoint_dir / "best_model.pt"

        if not best_model.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if len(checkpoints) == 0:
                print("✗ No model checkpoint found")
                return False
            best_model = checkpoints[0]

        tokenizer_path = "models/tokenizer"

        engine = InferenceEngine(str(best_model), tokenizer_path)

        # Test generation
        test_prompt = "Query: How do I start a fire? Facts: To start a fire, you need tinder, kindling, and fuel. Response:"

        output = engine.generate(test_prompt, max_new_tokens=20)

        print(f"✓ Inference works")
        print(f"  Prompt: {test_prompt[:50]}...")
        print(f"  Output: {output[:100]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline():
    """Test full RAG pipeline"""
    print("\n[TEST] RAG Pipeline (End-to-End)")
    print("-" * 80)

    from inference.retrieval_augmented import RAGPipeline
    from pathlib import Path

    try:
        # Find model
        checkpoint_dir = Path("models/checkpoints")
        best_model = checkpoint_dir / "best_model.pt"

        if not best_model.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if len(checkpoints) == 0:
                print("✗ No model checkpoint found")
                return False
            best_model = checkpoints[0]

        tokenizer_path = "models/tokenizer"

        pipeline = RAGPipeline(str(best_model), tokenizer_path)

        # Test query
        test_query = "How do I purify water?"

        result = pipeline.query(test_query, verbose=False)

        print(f"✓ RAG pipeline works")
        print(f"  Query: {test_query}")
        print(f"  Retrieved {len(result['retrieved_articles'])} articles")
        print(f"  Response length: {len(result['response'])} chars")
        print(f"  Total time: {result['timing']['total_ms']:.1f}ms")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "=" * 80)
    print("END-TO-END SYSTEM TEST")
    print("=" * 80)

    tests = [
        ("Corpus Building", test_corpus_building),
        ("Article Cleaning", test_article_cleaning),
        ("TF-IDF Index", test_index_building),
        ("Retrieval Engine", test_retrieval),
        ("Tokenizer", test_tokenizer),
        ("Model Architecture", test_model_architecture),
        ("Dataset Generation", test_dataset),
        ("Model Training", test_training),
        ("Inference Engine", test_inference),
        ("RAG Pipeline", test_rag_pipeline),
    ]

    results = []

    for test_name, test_func in tests:
        passed = test_func()
        results.append((test_name, passed))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 80)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    if passed_count == total_count:
        print("\n✓ ALL TESTS PASSED - System is ready!")
    else:
        print("\n✗ Some tests failed - Please fix issues above")

    return passed_count == total_count


if __name__ == "__main__":
    run_all_tests()
