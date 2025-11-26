# Quick Start Guide

Get the ESP32 Offline AI Assistant running in 5 minutes!

## Prerequisites

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **Ollama** (for dataset generation)
   - Download from https://ollama.ai
   - Pull model: `ollama pull llama3.2:3b`

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
# Run setup script
python setup.py
```

This will:
- Install all dependencies
- Set up knowledge base with sample articles
- Train tokenizer
- Run basic tests

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process sample articles
python knowledge_base/article_cleaner.py
python knowledge_base/article_compressor.py
python knowledge_base/tfidf_indexer.py

# 3. Train tokenizer
python models/tokenizer.py --articles-dir knowledge_base/articles

# 4. Verify
python tests/test_end_to_end.py
```

## Getting More Articles

The project includes 3 sample articles. For a better experience:

```bash
# Download 100 articles from Wikipedia/WikiHow
python knowledge_base/corpus_builder.py --target 100

# This takes ~15 minutes and requires internet
```

## Training Pipeline

### 1. Generate Training Data

```bash
# Generate 10,000 examples (takes ~2-3 hours)
python data/generate_dataset.py --target 10000

# Or start smaller for testing
python data/generate_dataset.py --target 100
```

### 2. Train Model

```bash
# Train model (4-8 hours on GPU, 2-3 days on CPU)
python training/train.py

# Monitor with TensorBoard
tensorboard --logdir models/checkpoints/logs
```

### 3. Test Inference

```bash
# Interactive RAG demo
python inference/demo.py --model models/checkpoints/best_model.pt
```

## Testing Without Training

If you don't want to train a model yet, you can still test components:

```bash
# Test retrieval system
python knowledge_base/retrieval_engine.py --query "How do I start a fire?"

# Test model architecture
python models/architecture.py

# Test tokenizer
python models/tokenizer.py

# Run all tests
python tests/test_end_to_end.py
```

## What Each Component Does

### Knowledge Base
- **corpus_builder.py**: Downloads articles from web sources
- **article_cleaner.py**: Removes HTML and normalizes text
- **article_compressor.py**: Compresses articles with LZ4
- **tfidf_indexer.py**: Builds search index
- **retrieval_engine.py**: Searches for relevant articles

### Model
- **architecture.py**: 2M parameter transformer
- **tokenizer.py**: BPE tokenizer with 8192 vocab
- **kv_cache.py**: Key-value cache for fast generation
- **quantization.py**: 4-bit quantization

### Data
- **generate_dataset.py**: Creates training data using Ollama
- **preprocess.py**: Prepares data for training

### Training
- **train.py**: Complete training pipeline with all features

### Inference
- **engine.py**: Text generation
- **retrieval_augmented.py**: RAG pipeline
- **demo.py**: Interactive CLI demo

### Export
- **to_tflite.py**: Export to TensorFlow Lite
- **to_c_array.py**: Generate C headers for ESP32

## Common Issues

### "Ollama connection failed"
```bash
# Start Ollama server in a separate terminal
ollama serve

# In another terminal
ollama pull llama3.2:3b
```

### "No articles found"
```bash
# The project includes 3 sample articles
# Download more with:
python knowledge_base/corpus_builder.py --target 100
```

### "Tokenizer not found"
```bash
# Train tokenizer
python models/tokenizer.py --articles-dir knowledge_base/articles
```

### "CUDA out of memory"
Edit `config.yaml`:
```yaml
training:
  batch_size: 4  # Reduce from 16
  mixed_precision: true
```

## Expected Timeline

| Task | Time | Hardware |
|------|------|----------|
| Setup | 5 min | Any |
| Download articles | 15 min | Internet required |
| Generate dataset (10K) | 2-3 hours | CPU with Ollama |
| Train model | 4-8 hours | GPU (RTX 3080) |
| Train model | 2-3 days | CPU (16 cores) |

## Minimal Working Example

To see the system work without full training:

```bash
# 1. Setup (5 minutes)
python setup.py

# 2. Test retrieval (works immediately)
python knowledge_base/retrieval_engine.py --query "How do I purify water?"

# 3. Generate small dataset (30 minutes)
python data/generate_dataset.py --target 100

# 4. Quick training test (1 hour on GPU)
# Edit config.yaml: num_epochs: 1
python training/train.py

# 5. Test inference
python inference/demo.py --model models/checkpoints/checkpoint_epoch_1.pt
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [config.yaml](config.yaml) to customize settings
3. Run tests to verify everything works
4. Train your model on a larger dataset
5. Export for ESP32 deployment

## Getting Help

- Run tests: `python tests/test_end_to_end.py`
- Check logs in `logs/` directory
- See README.md for troubleshooting
- Open an issue on GitHub

## Performance Expectations

With sample articles (3):
- ✓ Retrieval works
- ✗ Limited knowledge coverage
- ✗ Can't generate good training data

With 100 articles:
- ✓ Retrieval works well
- ✓ Decent knowledge coverage
- ✓ Can generate 10K+ examples
- ✓ Model trains well

With 1000+ articles:
- ✓ Excellent retrieval
- ✓ Comprehensive knowledge
- ✓ Can generate 50K+ examples
- ✓ Production quality

## Quick Commands Reference

```bash
# Setup
python setup.py

# Knowledge Base
python knowledge_base/corpus_builder.py --target 100
python knowledge_base/article_cleaner.py
python knowledge_base/article_compressor.py
python knowledge_base/tfidf_indexer.py

# Tokenizer
python models/tokenizer.py --articles-dir knowledge_base/articles

# Dataset
python data/generate_dataset.py --target 10000
python data/preprocess.py --data-path data/generated/train.jsonl --analyze

# Training
python training/train.py
tensorboard --logdir models/checkpoints/logs

# Testing
python tests/test_end_to_end.py
python tests/test_retrieval.py

# Inference
python inference/demo.py --model models/checkpoints/best_model.pt

# Export
python export/to_c_array.py --model models/checkpoints/best_model.pt --quantize
```

---

**Ready to start? Run `python setup.py`**
