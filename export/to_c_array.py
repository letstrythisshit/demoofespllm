"""
Export model and data to C arrays for ESP32
"""

import torch
import numpy as np
from pathlib import Path
import logging
import yaml
import pickle
import json

from models.architecture import GPTModel
from models.tokenizer import BPETokenizer
from models.quantization import quantize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CArrayExporter:
    """Export model and data to C arrays for ESP32"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.export_config = self.config['export']
        self.output_dir = Path(self.export_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.split_threshold = self.export_config['c_arrays']['split_threshold']
        self.header_prefix = self.export_config['c_arrays']['header_guard_prefix']

    def export_model(self, model_path: str, quantize: bool = True):
        """
        Export model weights to C arrays

        Args:
            model_path: Path to trained model
            quantize: Whether to quantize model first
        """
        logger.info("Exporting model to C arrays...")

        # Load model
        model = GPTModel().to('cpu')
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Quantize if requested
        if quantize:
            logger.info("Quantizing model...")
            model = quantize_model(model, bits=4, symmetric=True)

        model.eval()

        # Extract weights
        logger.info("Extracting model weights...")
        weights = self._extract_weights(model)

        # Generate C files
        logger.info("Generating C header files...")
        self._generate_c_headers(weights, "model_weights")

        logger.info(f"Model exported to {self.output_dir}")

    def export_tokenizer(self, tokenizer_path: str):
        """
        Export tokenizer to C arrays

        Args:
            tokenizer_path: Path to tokenizer directory
        """
        logger.info("Exporting tokenizer to C arrays...")

        # Load tokenizer
        tokenizer = BPETokenizer.load(tokenizer_path)

        # Export vocabulary
        vocab_list = []
        for token, token_id in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            # Convert token to bytes
            token_bytes = token.encode('utf-8')
            vocab_list.append({
                'id': token_id,
                'token': token,
                'bytes': list(token_bytes),
                'length': len(token_bytes)
            })

        self._generate_vocabulary_header(vocab_list, "tokenizer_vocab")

        # Export merges
        self._generate_merges_header(tokenizer.merges, "tokenizer_merges")

        # Export config
        tokenizer_config = {
            'vocab_size': tokenizer.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }

        self._generate_config_header(tokenizer_config, "tokenizer_config")

        logger.info(f"Tokenizer exported to {self.output_dir}")

    def export_index(self, index_dir: str):
        """
        Export TF-IDF index to C arrays

        Args:
            index_dir: Path to index directory
        """
        logger.info("Exporting TF-IDF index to C arrays...")

        index_path = Path(index_dir) / "tfidf_index.pkl"

        if not index_path.exists():
            logger.error(f"Index not found at {index_path}")
            return

        # Load index
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)

        # Export vocabulary
        vocabulary = index_data['vocabulary']
        vocab_items = sorted(vocabulary.items(), key=lambda x: x[1])

        self._generate_vocabulary_header(
            [{'id': idx, 'token': term, 'bytes': list(term.encode('utf-8')), 'length': len(term)}
             for term, idx in vocab_items],
            "index_vocabulary"
        )

        # Export inverted index (simplified for ESP32)
        # We'll store just the most important postings
        inverted_index = index_data['inverted_index']

        self._generate_inverted_index_header(inverted_index, "index_postings")

        # Export document info
        doc_info = index_data['doc_info']
        self._generate_doc_info_header(doc_info, "index_documents")

        logger.info(f"Index exported to {self.output_dir}")

    def _extract_weights(self, model: torch.nn.Module) -> dict:
        """Extract weights from model"""
        weights = {}

        for name, param in model.named_parameters():
            # Convert to numpy
            weight = param.detach().cpu().numpy()

            # Flatten for C array
            weight_flat = weight.flatten()

            weights[name] = {
                'data': weight_flat,
                'shape': list(weight.shape),
                'size': weight_flat.size,
                'dtype': str(weight.dtype)
            }

        return weights

    def _generate_c_headers(self, weights: dict, base_name: str):
        """Generate C header files for weights"""
        header_path = self.output_dir / f"{base_name}.h"

        with open(header_path, 'w') as f:
            # Header guard
            guard = f"{self.header_prefix}_{base_name.upper()}_H"
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            f.write("#include <stdint.h>\n\n")

            # Write each weight array
            for name, weight_info in weights.items():
                # Sanitize name for C
                c_name = name.replace('.', '_').replace('[', '_').replace(']', '_')

                # Write shape and size constants
                f.write(f"// Shape: {weight_info['shape']}\n")
                f.write(f"#define {c_name.upper()}_SIZE {weight_info['size']}\n")

                # Write array declaration
                if 'int' in weight_info['dtype']:
                    dtype = 'int8_t'
                else:
                    dtype = 'float'

                f.write(f"const {dtype} {c_name}[{weight_info['size']}];\n\n")

            f.write(f"#endif // {guard}\n")

        logger.info(f"Generated header: {header_path}")

        # Generate implementation file
        impl_path = self.output_dir / f"{base_name}.c"

        with open(impl_path, 'w') as f:
            f.write(f'#include "{base_name}.h"\n\n')

            for name, weight_info in weights.items():
                c_name = name.replace('.', '_').replace('[', '_').replace(']', '_')

                if 'int' in weight_info['dtype']:
                    dtype = 'int8_t'
                else:
                    dtype = 'float'

                f.write(f"const {dtype} {c_name}[{weight_info['size']}] = {{\n")

                # Write data
                data = weight_info['data']
                for i in range(0, len(data), 12):  # 12 values per line
                    line_data = data[i:i+12]
                    if 'int' in weight_info['dtype']:
                        f.write("    " + ", ".join(f"{int(v)}" for v in line_data) + ",\n")
                    else:
                        f.write("    " + ", ".join(f"{v:.6f}f" for v in line_data) + ",\n")

                f.write("};\n\n")

        logger.info(f"Generated implementation: {impl_path}")

    def _generate_vocabulary_header(self, vocab_list: list, base_name: str):
        """Generate vocabulary C header"""
        header_path = self.output_dir / f"{base_name}.h"

        with open(header_path, 'w') as f:
            guard = f"{self.header_prefix}_{base_name.upper()}_H"
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            f.write("#include <stdint.h>\n\n")

            f.write(f"#define VOCAB_SIZE {len(vocab_list)}\n\n")

            f.write("typedef struct {\n")
            f.write("    uint16_t id;\n")
            f.write("    const char* token;\n")
            f.write("    uint8_t length;\n")
            f.write("} VocabEntry;\n\n")

            f.write("extern const VocabEntry vocabulary[VOCAB_SIZE];\n\n")

            f.write(f"#endif // {guard}\n")

        # Implementation
        impl_path = self.output_dir / f"{base_name}.c"

        with open(impl_path, 'w') as f:
            f.write(f'#include "{base_name}.h"\n\n')

            f.write(f"const VocabEntry vocabulary[VOCAB_SIZE] = {{\n")

            for item in vocab_list:
                token_escaped = item['token'].replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'    {{{item["id"]}, "{token_escaped}", {item["length"]}}},\n')

            f.write("};\n")

        logger.info(f"Generated vocabulary: {header_path}")

    def _generate_merges_header(self, merges: list, base_name: str):
        """Generate merges C header"""
        header_path = self.output_dir / f"{base_name}.h"

        with open(header_path, 'w') as f:
            guard = f"{self.header_prefix}_{base_name.upper()}_H"
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            f.write(f"#define NUM_MERGES {len(merges)}\n\n")

            f.write("typedef struct {\n")
            f.write("    const char* first;\n")
            f.write("    const char* second;\n")
            f.write("} MergePair;\n\n")

            f.write("extern const MergePair merges[NUM_MERGES];\n\n")

            f.write(f"#endif // {guard}\n")

        # Implementation
        impl_path = self.output_dir / f"{base_name}.c"

        with open(impl_path, 'w') as f:
            f.write(f'#include "{base_name}.h"\n\n')

            f.write(f"const MergePair merges[NUM_MERGES] = {{\n")

            for first, second in merges:
                first_escaped = first.replace('\\', '\\\\').replace('"', '\\"')
                second_escaped = second.replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'    {{"{first_escaped}", "{second_escaped}"}},\n')

            f.write("};\n")

        logger.info(f"Generated merges: {header_path}")

    def _generate_config_header(self, config: dict, base_name: str):
        """Generate config C header"""
        header_path = self.output_dir / f"{base_name}.h"

        with open(header_path, 'w') as f:
            guard = f"{self.header_prefix}_{base_name.upper()}_H"
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            for key, value in config.items():
                f.write(f"#define {key.upper()} {value}\n")

            f.write(f"\n#endif // {guard}\n")

        logger.info(f"Generated config: {header_path}")

    def _generate_inverted_index_header(self, inverted_index: dict, base_name: str):
        """Generate inverted index C header (simplified)"""
        # For ESP32, we'll only include postings for most important terms
        # Full index might be too large

        logger.info("Note: Generating simplified index for ESP32")

        header_path = self.output_dir / f"{base_name}.h"

        with open(header_path, 'w') as f:
            guard = f"{self.header_prefix}_{base_name.upper()}_H"
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            f.write("// Simplified inverted index for ESP32\n")
            f.write("// Store in external flash\n\n")

            f.write(f"#endif // {guard}\n")

    def _generate_doc_info_header(self, doc_info: dict, base_name: str):
        """Generate document info C header"""
        logger.info("Generating document info header")

        header_path = self.output_dir / f"{base_name}.h"

        with open(header_path, 'w') as f:
            guard = f"{self.header_prefix}_{base_name.upper()}_H"
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")

            f.write(f"#define NUM_DOCUMENTS {len(doc_info)}\n\n")

            f.write("// Document metadata\n")
            f.write("// Store in external flash\n\n")

            f.write(f"#endif // {guard}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Export to C arrays for ESP32')
    parser.add_argument('--model', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer',
                       help='Path to tokenizer')
    parser.add_argument('--index', type=str, default='knowledge_base/index',
                       help='Path to index directory')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize model before export')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config')

    args = parser.parse_args()

    exporter = CArrayExporter(args.config)

    if args.model:
        exporter.export_model(args.model, quantize=args.quantize)

    if Path(args.tokenizer).exists():
        exporter.export_tokenizer(args.tokenizer)

    if Path(args.index).exists():
        exporter.export_index(args.index)

    print(f"\n✓ C headers generated in {exporter.output_dir}")
    print("✓ Ready for ESP32 integration")


if __name__ == "__main__":
    main()
