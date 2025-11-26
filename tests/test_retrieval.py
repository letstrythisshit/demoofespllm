"""
Test retrieval system quality
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.retrieval_engine import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalTester:
    """Test retrieval quality"""

    def __init__(self, config_path: str = "config.yaml"):
        self.engine = RetrievalEngine(config_path)

        # Test queries with expected categories
        self.test_queries = [
            # Fire starting
            ("How do I start a fire without matches?", "fire_starting"),
            ("What is the best way to make a friction fire?", "fire_starting"),
            ("How do I prepare tinder for fire starting?", "fire_starting"),

            # Water purification
            ("How can I purify water in the wilderness?", "water_purification"),
            ("What are methods to make water safe to drink?", "water_purification"),
            ("How do I boil water for purification?", "water_purification"),

            # Shelter building
            ("How do I build an emergency shelter?", "shelter_building"),
            ("What is a lean-to shelter?", "shelter_building"),
            ("How can I stay warm in my shelter?", "shelter_building"),

            # First aid
            ("How do I treat a cut in the wilderness?", "first_aid"),
            ("What should I do for hypothermia?", "first_aid"),
            ("How do I perform CPR?", "first_aid"),

            # Navigation
            ("How can I navigate without a compass?", "navigation"),
            ("How do I use stars for navigation?", "navigation"),
            ("How can I read a topographic map?", "navigation"),

            # Food foraging
            ("What wild plants are safe to eat?", "food_foraging"),
            ("How do I identify edible mushrooms?", "food_foraging"),
            ("What insects are safe to eat?", "food_foraging"),

            # Signaling
            ("How do I signal for rescue?", "signaling"),
            ("What is an SOS signal?", "signaling"),
            ("How do I use a signal mirror?", "signaling"),

            # Weather
            ("How can I predict weather in nature?", "weather"),
            ("What do different clouds mean for weather?", "weather"),
            ("How do I stay safe in a lightning storm?", "weather"),

            # Knots and tools
            ("What are essential survival knots?", "knots_tools"),
            ("How do I tie a bowline knot?", "knots_tools"),
            ("How do I safely use an axe?", "knots_tools"),

            # Emergency procedures
            ("What should I do first in an emergency?", "emergency_procedures"),
            ("What is the rule of threes in survival?", "emergency_procedures"),
            ("How do I stay calm in an emergency?", "emergency_procedures"),
        ]

    def test_retrieval_accuracy(self, top_k: int = 3):
        """
        Test if retrieved articles match expected category

        Args:
            top_k: Number of results to retrieve
        """
        logger.info(f"Testing retrieval accuracy with {len(self.test_queries)} queries")

        correct = 0
        total = len(self.test_queries)

        results = []

        for query, expected_category in self.test_queries:
            # Search
            search_results = self.engine.search(query, top_k=top_k)

            # Check if any result matches expected category
            found_match = False
            for doc_id, score, metadata in search_results:
                if metadata['category'] == expected_category:
                    found_match = True
                    break

            if found_match:
                correct += 1

            results.append({
                'query': query,
                'expected_category': expected_category,
                'found_match': found_match,
                'results': search_results
            })

        accuracy = correct / total

        logger.info(f"Retrieval accuracy: {accuracy:.2%} ({correct}/{total})")

        return accuracy, results

    def test_retrieval_speed(self, num_queries: int = 100):
        """Test retrieval speed"""
        logger.info(f"Testing retrieval speed with {num_queries} queries")

        import time

        times = []

        # Sample queries
        queries = [q for q, _ in self.test_queries] * (num_queries // len(self.test_queries) + 1)
        queries = queries[:num_queries]

        for query in queries:
            start = time.time()
            self.engine.search(query, top_k=3)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = np.mean(times)
        median_time = np.median(times)
        p95_time = np.percentile(times, 95)

        logger.info(f"Retrieval speed:")
        logger.info(f"  Average: {avg_time:.2f}ms")
        logger.info(f"  Median: {median_time:.2f}ms")
        logger.info(f"  P95: {p95_time:.2f}ms")

        return {
            'avg_ms': avg_time,
            'median_ms': median_time,
            'p95_ms': p95_time
        }

    def test_precision_at_k(self, k_values: list = [1, 3, 5]):
        """Calculate Precision@K"""
        logger.info("Calculating Precision@K")

        precision_scores = {k: [] for k in k_values}

        for query, expected_category in self.test_queries:
            # Get results for maximum k
            max_k = max(k_values)
            results = self.engine.search(query, top_k=max_k)

            # Calculate precision at each k
            for k in k_values:
                top_k_results = results[:k]

                # Count relevant results (matching category)
                relevant = sum(
                    1 for _, _, metadata in top_k_results
                    if metadata['category'] == expected_category
                )

                precision = relevant / k if k > 0 else 0
                precision_scores[k].append(precision)

        # Calculate average precision for each k
        avg_precision = {
            k: np.mean(scores)
            for k, scores in precision_scores.items()
        }

        for k, precision in avg_precision.items():
            logger.info(f"  Precision@{k}: {precision:.3f}")

        return avg_precision

    def run_all_tests(self):
        """Run all retrieval tests"""
        print("\n" + "=" * 80)
        print("RETRIEVAL SYSTEM TESTS")
        print("=" * 80)

        # Test accuracy
        print("\n[1] Testing Retrieval Accuracy...")
        print("-" * 80)
        accuracy, _ = self.test_retrieval_accuracy(top_k=3)

        # Test speed
        print("\n[2] Testing Retrieval Speed...")
        print("-" * 80)
        speed_stats = self.test_retrieval_speed(num_queries=100)

        # Test precision
        print("\n[3] Testing Precision@K...")
        print("-" * 80)
        precision_scores = self.test_precision_at_k([1, 3, 5])

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Retrieval Accuracy: {accuracy:.2%}")
        print(f"Average Speed: {speed_stats['avg_ms']:.2f}ms")
        print(f"Median Speed: {speed_stats['median_ms']:.2f}ms")
        print(f"P95 Speed: {speed_stats['p95_ms']:.2f}ms")

        for k, precision in sorted(precision_scores.items()):
            print(f"Precision@{k}: {precision:.3f}")

        # Pass/fail
        print("\n" + "=" * 80)

        passed = True

        if accuracy < 0.5:
            print("✗ Accuracy test FAILED (< 50%)")
            passed = False
        else:
            print("✓ Accuracy test PASSED")

        if speed_stats['p95_ms'] > 100:
            print("✗ Speed test FAILED (P95 > 100ms)")
            passed = False
        else:
            print("✓ Speed test PASSED")

        if precision_scores[3] < 0.3:
            print("✗ Precision test FAILED (P@3 < 0.3)")
            passed = False
        else:
            print("✓ Precision test PASSED")

        if passed:
            print("\n✓ ALL TESTS PASSED")
        else:
            print("\n✗ SOME TESTS FAILED")

        print("=" * 80)

        return passed


def main():
    """Main entry point"""
    tester = RetrievalTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
