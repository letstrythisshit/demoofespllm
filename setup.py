#!/usr/bin/env python3
"""
Quick Setup Script for ESP32 Offline AI Assistant
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def check_requirements():
    """Check if requirements are met"""
    print_header("Checking Requirements")

    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check if pip is available
    try:
        subprocess.run(["pip", "--version"], check=True, capture_output=True)
        print("✓ pip is available")
    except:
        print("✗ pip is not available")
        return False

    # Check if Ollama is available
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Ollama is available")

        # Check if model is pulled
        if "llama3.2" in result.stdout or "mistral" in result.stdout:
            print("✓ Ollama model is available")
        else:
            print("⚠ Ollama model not found. Will need to pull llama3.2:3b")

    except:
        print("⚠ Ollama not found. Dataset generation will not work.")
        print("  Install from: https://ollama.ai")

    return True


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")

    return run_command(
        "pip install -r requirements.txt",
        "Installing Python packages"
    )


def setup_knowledge_base():
    """Set up knowledge base with sample articles"""
    print_header("Setting Up Knowledge Base")

    # Check if articles exist
    article_dir = Path("knowledge_base/articles")
    article_files = list(article_dir.glob("**/*.json"))
    article_files = [f for f in article_files if f.name != "master_index.json"]

    if len(article_files) >= 3:
        print(f"✓ Found {len(article_files)} articles")
    else:
        print("⚠ Only sample articles available")
        print("  Run: python knowledge_base/corpus_builder.py --target 100")

    # Clean articles
    run_command(
        "python knowledge_base/article_cleaner.py",
        "Cleaning articles",
        check=False
    )

    # Compress articles
    run_command(
        "python knowledge_base/article_compressor.py",
        "Compressing articles",
        check=False
    )

    # Build index
    return run_command(
        "python knowledge_base/tfidf_indexer.py",
        "Building TF-IDF index"
    )


def train_tokenizer():
    """Train tokenizer"""
    print_header("Training Tokenizer")

    # Check if tokenizer exists
    tokenizer_path = Path("models/tokenizer")

    if tokenizer_path.exists():
        print("✓ Tokenizer already exists")
        return True

    return run_command(
        "python models/tokenizer.py --articles-dir knowledge_base/articles",
        "Training tokenizer"
    )


def test_system():
    """Run basic system tests"""
    print_header("Testing System")

    return run_command(
        "python tests/test_end_to_end.py",
        "Running system tests",
        check=False
    )


def main():
    """Main setup flow"""
    print("\n" + "🚀" * 40)
    print("ESP32 Offline AI Assistant - Quick Setup")
    print("🚀" * 40)

    # Step 1: Check requirements
    if not check_requirements():
        print("\n✗ Requirements check failed. Please install missing dependencies.")
        sys.exit(1)

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n✗ Dependency installation failed.")
        sys.exit(1)

    # Step 3: Setup knowledge base
    if not setup_knowledge_base():
        print("\n✗ Knowledge base setup failed.")
        sys.exit(1)

    # Step 4: Train tokenizer
    if not train_tokenizer():
        print("\n✗ Tokenizer training failed.")
        sys.exit(1)

    # Step 5: Test system
    test_system()

    # Summary
    print_header("Setup Summary")

    print("\n✓ Basic setup complete!")

    print("\nNext Steps:")
    print("  1. Generate more articles:")
    print("     python knowledge_base/corpus_builder.py --target 100")
    print()
    print("  2. Generate training dataset:")
    print("     python data/generate_dataset.py --target 10000")
    print()
    print("  3. Train the model:")
    print("     python training/train.py")
    print()
    print("  4. Test the system:")
    print("     python tests/test_end_to_end.py")
    print()
    print("  5. Run interactive demo:")
    print("     python inference/demo.py --model models/checkpoints/best_model.pt")

    print("\nFor full documentation, see README.md")
    print("\n" + "✨" * 40)


if __name__ == "__main__":
    main()
