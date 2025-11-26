"""
Export model to TensorFlow Lite format
"""

import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import yaml

from models.architecture import GPTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFLiteExporter:
    """Export PyTorch model to TensorFlow Lite"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.export_config = self.config['export']
        self.model_config = self.config['model']

        self.output_dir = Path(self.export_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, model_path: str, output_name: str = "model.tflite"):
        """
        Export model to TFLite format

        Args:
            model_path: Path to PyTorch model checkpoint
            output_name: Output filename for TFLite model
        """
        logger.info(f"Exporting model from {model_path} to TFLite...")

        # Load PyTorch model
        logger.info("Loading PyTorch model...")
        model = GPTModel().to('cpu')
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Convert to ONNX first (intermediate step)
        logger.info("Converting to ONNX...")
        onnx_path = self.output_dir / "model.onnx"
        self._export_to_onnx(model, onnx_path)

        # Convert ONNX to TensorFlow
        logger.info("Converting ONNX to TensorFlow...")
        try:
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)

            tf_model_dir = self.output_dir / "tf_model"
            tf_rep.export_graph(str(tf_model_dir))

            logger.info(f"TensorFlow model saved to {tf_model_dir}")

        except ImportError:
            logger.error("onnx-tf not installed. Cannot convert to TensorFlow.")
            logger.info("Install with: pip install onnx onnx-tf")
            return None

        # Convert to TFLite
        logger.info("Converting to TFLite...")
        tflite_path = self.output_dir / output_name
        self._convert_to_tflite(tf_model_dir, tflite_path)

        logger.info(f"TFLite model exported to {tflite_path}")

        # Get file size
        size_bytes = tflite_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        logger.info(f"Model size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

        return tflite_path

    def _export_to_onnx(self, model: torch.nn.Module, output_path: Path):
        """Export PyTorch model to ONNX"""
        # Create dummy input
        batch_size = 1
        seq_len = self.model_config['max_seq_len']
        dummy_input = torch.randint(
            0,
            self.model_config['vocab_size'],
            (batch_size, seq_len),
            dtype=torch.long
        )

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )

        logger.info(f"ONNX model saved to {output_path}")

    def _convert_to_tflite(self, tf_model_dir: Path, output_path: Path):
        """Convert TensorFlow model to TFLite"""
        # Load TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_dir))

        # Optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for quantization (optional)
        def representative_dataset():
            for _ in range(self.export_config['tflite']['representative_dataset_size']):
                # Random input
                data = np.random.randint(
                    0,
                    self.model_config['vocab_size'],
                    size=(1, self.model_config['max_seq_len'])
                ).astype(np.int32)
                yield [data]

        converter.representative_dataset = representative_dataset

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Export model to TFLite')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--output', type=str, default='model.tflite',
                       help='Output filename')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    exporter = TFLiteExporter(args.config)

    try:
        output_path = exporter.export(args.model, args.output)

        if output_path:
            print(f"\n✓ Successfully exported to {output_path}")
        else:
            print("\n✗ Export failed. See logs for details.")

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        print(f"\n✗ Export failed: {e}")


if __name__ == "__main__":
    main()
