"""
MTUP Fixed Training Script
Based on successful Baseline approach with instruction masking
"""

import os
import sys
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "config"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import config_mtup_fixed as config
from prompt_templates_fixed import format_mtup_training_example

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / f"train_mtup_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MTUPPreprocessor:
    """Preprocessor for MTUP format"""

    def __init__(self):
        self.stats = {'processed': 0, 'errors': 0}

    def remove_variables(self, amr_string: str) -> str:
        """
        Remove variables from AMR: (var / concept) → (concept)
        Creates Task 1 output - AMR without variables
        """
        # Remove variable declarations
        cleaned = re.sub(r'\([^\s/:()]+\s*/', r'(', amr_string)
        return cleaned.strip()

    def linearize(self, amr_string: str) -> str:
        """Convert multi-line AMR to single line"""
        # Join lines
        linear = ' '.join(amr_string.split())
        # Normalize spaces
        linear = re.sub(r'\s+', ' ', linear)
        # Clean parentheses spacing
        linear = re.sub(r'\s*\(\s*', '(', linear)
        linear = re.sub(r'\s*\)\s*', ')', linear)
        # Keep spaces after colons for readability
        linear = re.sub(r'\s*:\s*', ': ', linear)
        return linear.strip()

    def preprocess(self, sentence: str, amr_with_vars: str) -> tuple:
        """
        Preprocess sentence-AMR pair for MTUP

        Returns:
            (amr_no_vars, amr_with_vars) - both ready for template
        """
        try:
            # Generate AMR without variables (Task 1)
            amr_no_vars = self.remove_variables(amr_with_vars)
            amr_no_vars = self.linearize(amr_no_vars)

            # Keep AMR with variables as-is (Task 2) - already in Penman format
            amr_with_vars_clean = amr_with_vars.strip()

            self.stats['processed'] += 1
            return amr_no_vars, amr_with_vars_clean

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Preprocessing error: {e}")
            # Fallback
            return amr_with_vars.strip(), amr_with_vars.strip()


class MTUPDataset:
    """Dataset handler for MTUP with instruction masking"""

    def __init__(self, tokenizer, config_dict):
        self.tokenizer = tokenizer
        self.config = config_dict
        self.preprocessor = MTUPPreprocessor()
        self.max_length = config.MAX_SEQ_LENGTH

    def load_data(self, file_paths: List[str]) -> List[Dict]:
        """Load and preprocess AMR data"""
        examples = []

        for file_path in file_paths:
            full_path = config.DATA_DIR / file_path
            logger.info(f"Loading: {full_path}")

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by double newline
            pairs = content.strip().split('\n\n')

            for pair in pairs:
                lines = pair.strip().split('\n')
                if len(lines) < 2:
                    continue

                # Parse sentence and AMR
                sentence_line = lines[0]
                amr_lines = lines[1:]

                # Extract sentence
                if sentence_line.startswith('#::snt'):
                    sentence = sentence_line.replace('#::snt', '').strip()
                else:
                    sentence = sentence_line.strip()

                # Extract AMR
                amr = '\n'.join(amr_lines).strip()

                if sentence and amr:
                    examples.append({
                        'sentence': sentence,
                        'amr_with_vars': amr
                    })

        logger.info(f"Loaded {len(examples)} examples")
        return examples

    def create_training_example_with_masking(self, sentence: str, amr_no_vars: str, amr_with_vars: str):
        """
        Create training example with proper instruction masking
        Following baseline's successful approach
        """
        # Build full prompt
        full_prompt = format_mtup_training_example(
            sentence=sentence,
            amr_no_vars=amr_no_vars,
            amr_with_vars=amr_with_vars,
            template_type=self.config.get('template_type', 'ultra_minimal')
        )

        # CRITICAL: Encode parts separately (like baseline)
        # Find the split point - everything before last occurrence of amr_with_vars is prompt
        # The model should only learn to generate amr_with_vars

        # Split into instruction and target
        # Target is the final AMR with variables
        split_marker = "AMR chuẩn PENMAN:\n"
        if split_marker in full_prompt:
            parts = full_prompt.split(split_marker)
            instruction = parts[0] + split_marker
            target = amr_with_vars  # Just the final AMR
        else:
            # Fallback: treat everything before last AMR as instruction
            logger.warning("Could not find split marker, using fallback")
            instruction = full_prompt
            target = ""

        # Encode separately WITHOUT special tokens (avoid tokenization mismatch)
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        eos_ids = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)

        # Concatenate
        input_ids = instruction_ids + target_ids + eos_ids

        # Create labels: train only on target + EOS
        labels = input_ids.copy()
        for i in range(len(instruction_ids)):
            labels[i] = -100  # Mask instruction

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            labels += [-100] * padding_length

        attention_mask = [1] * len(input_ids) if padding_length == 0 else \
                        [1] * (self.max_length - padding_length) + [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

    def prepare_dataset(self, examples: List[Dict]) -> Dataset:
        """Prepare HuggingFace dataset with instruction masking"""
        processed_examples = []

        for example in examples:
            sentence = example['sentence']
            amr_with_vars = example['amr_with_vars']

            # Preprocess to get both versions
            amr_no_vars, amr_with_vars_clean = self.preprocessor.preprocess(
                sentence, amr_with_vars
            )

            # Create training example with masking
            if self.config.get('use_instruction_masking', True):
                train_example = self.create_training_example_with_masking(
                    sentence, amr_no_vars, amr_with_vars_clean
                )
            else:
                # Old way (not recommended)
                full_text = format_mtup_training_example(
                    sentence, amr_no_vars, amr_with_vars_clean,
                    template_type=self.config.get('template_type')
                )
                encodings = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                train_example = {
                    'input_ids': encodings['input_ids'][0],
                    'attention_mask': encodings['attention_mask'][0],
                    'labels': encodings['input_ids'][0].clone()
                }

            processed_examples.append(train_example)

        return Dataset.from_dict({
            'input_ids': [ex['input_ids'] for ex in processed_examples],
            'attention_mask': [ex['attention_mask'] for ex in processed_examples],
            'labels': [ex['labels'] for ex in processed_examples],
        })


def main():
    logger.info("=" * 80)
    logger.info("MTUP FIXED TRAINING - Based on Baseline Success")
    logger.info("=" * 80)

    # Print config
    config.print_config()

    # Create output directory
    output_dir = config.OUTPUT_DIR / f"{config.OUTPUT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Important: Set padding side to LEFT for decoder-only models
    tokenizer.padding_side = 'left'

    # Load model
    logger.info(f"Loading model: {config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(**config.LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset_handler = MTUPDataset(tokenizer, config.MTUP_CONFIG)

    # Load training data
    train_examples = dataset_handler.load_data(config.DATA_CONFIG['train_files'])

    # Split train/val
    val_split = config.DATA_CONFIG['validation_split']
    split_idx = int(len(train_examples) * (1 - val_split))
    train_data = train_examples[:split_idx]
    val_data = train_examples[split_idx:]

    logger.info(f"Train examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")

    # Prepare datasets
    train_dataset = dataset_handler.prepare_dataset(train_data)
    val_dataset = dataset_handler.prepare_dataset(val_data)

    # Show sample
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE TRAINING EXAMPLE:")
    logger.info("=" * 80)
    sample = train_data[0]
    amr_no_vars, amr_with_vars = dataset_handler.preprocessor.preprocess(
        sample['sentence'], sample['amr_with_vars']
    )
    sample_text = format_mtup_training_example(
        sample['sentence'], amr_no_vars, amr_with_vars,
        template_type=config.MTUP_CONFIG.get('template_type')
    )
    logger.info(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    logger.info("=" * 80 + "\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **config.TRAINING_CONFIG,
        logging_dir=str(config.LOG_DIR),
        report_to=[],  # Disable wandb
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    logger.info(f"Training complete! Model saved to: {final_model_path}")
    logger.info(f"Preprocessor stats: {dataset_handler.preprocessor.stats}")

    # Print next steps
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Test checkpoints:")
    logger.info(f"   bash TEST_MTUP_CHECKPOINTS.sh {output_dir}")
    logger.info("2. Evaluate best checkpoint:")
    logger.info(f"   python predict_mtup_fixed.py --model {output_dir}/checkpoint-XXX")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
