"""
Training Pipeline - Complete training with all features
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import time
import math

from models.architecture import GPTModel
from models.tokenizer import BPETokenizer
from data.preprocess import SurvivalQADataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Complete training pipeline"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_config = self.config['training']
        self.model_config = self.config['model']
        self.data_config = self.config['dataset']

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(self.train_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        logger.info("Initializing model...")
        self.model = GPTModel(config_path).to(self.device)
        logger.info(f"Model has {self.model.count_parameters():,} parameters")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer_path = "models/tokenizer"
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Please run: python models/tokenizer.py first"
            )
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        # Load datasets
        logger.info("Loading datasets...")
        train_path = Path(self.data_config['output_dir']) / "train.jsonl"
        val_path = Path(self.data_config['output_dir']) / "val.jsonl"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Please run: python data/generate_dataset.py first"
            )

        self.train_dataset = SurvivalQADataset(
            train_path,
            self.tokenizer,
            self.model_config['max_seq_len']
        )

        self.val_dataset = SurvivalQADataset(
            val_path,
            self.tokenizer,
            self.model_config['max_seq_len']
        ) if val_path.exists() else None

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for simpler debugging
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        ) if self.val_dataset else None

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )

        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * self.train_config['num_epochs']
        self.scheduler = self._create_scheduler(total_steps)

        # Mixed precision training
        self.use_amp = self.train_config.get('mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler with warmup and cosine annealing"""
        warmup_steps = self.train_config['warmup_steps']

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.train_config['num_epochs']}")
        logger.info(f"  Batch size: {self.train_config['batch_size']}")
        logger.info(f"  Training samples: {len(self.train_dataset)}")
        logger.info(f"  Validation samples: {len(self.val_dataset) if self.val_dataset else 0}")
        logger.info(f"  Steps per epoch: {len(self.train_loader)}")

        for epoch in range(self.train_config['num_epochs']):
            self.epoch = epoch

            logger.info(f"\nEpoch {epoch + 1}/{self.train_config['num_epochs']}")

            # Train epoch
            train_loss = self.train_epoch()

            logger.info(f"Train loss: {train_loss:.4f}")
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)

            # Validate
            if self.val_loader:
                val_loss = self.validate()
                logger.info(f"Validation loss: {val_loss:.4f}")
                self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    # Save best model
                    self.save_checkpoint('best_model.pt', is_best=True)
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement. Patience: {self.patience_counter}/"
                               f"{self.train_config['early_stopping_patience']}")

                # Early stopping
                if self.patience_counter >= self.train_config['early_stopping_patience']:
                    logger.info("Early stopping triggered!")
                    break

            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        # Save final model
        self.save_checkpoint('final_model.pt')

        logger.info("\nTraining complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        self.writer.close()

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training")

        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits, _ = self.model(input_ids, attention_mask)

                    # Reshape for loss calculation
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)

                    loss = self.criterion(logits, labels)

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config['gradient_clip']
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard forward pass
                logits, _ = self.model(input_ids, attention_mask)

                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                loss = self.criterion(logits, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config['gradient_clip']
                )

                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

            # Logging
            if self.global_step % self.train_config['logging_steps'] == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], self.global_step)

            # Save checkpoint
            if self.global_step % self.train_config['save_steps'] == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate model"""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits, _ = self.model(input_ids, attention_mask)

                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)

        if is_best:
            logger.info(f"Saved best model to {save_path}")
        else:
            logger.debug(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        load_path = self.output_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        logger.info(f"Loading checkpoint from {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    trainer = Trainer(args.config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    print("\n✓ Training complete!")
    print(f"✓ Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"✓ Models saved to {trainer.output_dir}")


if __name__ == "__main__":
    main()
