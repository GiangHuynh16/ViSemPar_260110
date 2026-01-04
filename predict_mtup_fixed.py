"""
MTUP Fixed Prediction Script
Two-stage inference with proper AMR extraction
"""

import os
import sys
import argparse
import logging
import torch
import re
from pathlib import Path
from typing import List, Tuple

# Add to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "config"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config_mtup_fixed as config
from prompt_templates_fixed import MTUP_INFERENCE_TEMPLATE, MTUP_INFERENCE_STEP2_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MTUPPredictor:
    """Two-stage MTUP predictor"""

    def __init__(self, model_path: str):
        logger.info(f"Loading model from: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load LoRA weights
        logger.info("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()

        logger.info("Model loaded successfully!")

    def extract_amr_step1(self, generated_text: str) -> str:
        """
        Extract AMR from Step 1 (AMR without variables)
        Look for content after "AMR không biến:"
        """
        # Split by marker
        if "AMR không biến:" in generated_text:
            parts = generated_text.split("AMR không biến:")
            if len(parts) > 1:
                amr_section = parts[1]
            else:
                amr_section = generated_text
        else:
            # Fallback: split by "Bước 1"
            if "Bước 1" in generated_text:
                amr_section = generated_text.split("Bước 1")[1]
            else:
                amr_section = generated_text

        # Extract until balanced parentheses or next section
        lines = amr_section.split('\n')
        amr_lines = []

        for line in lines:
            # Stop at Step 2 marker
            if "AMR chuẩn PENMAN" in line or "Bước 2" in line:
                break

            amr_lines.append(line)

            # Check if we have balanced AMR
            accumulated = '\n'.join(amr_lines).strip()
            if accumulated.count('(') == accumulated.count(')') > 0:
                break

        result = '\n'.join(amr_lines).strip()

        # Clean up
        result = result.split('\n')[0] if result else ""  # Take first line only
        return result.strip()

    def extract_amr_step2(self, generated_text: str) -> str:
        """
        Extract final AMR (with variables) from Step 2
        Look for content after "AMR chuẩn PENMAN:"
        """
        # Remove EOS token
        if self.tokenizer.eos_token in generated_text:
            generated_text = generated_text.split(self.tokenizer.eos_token)[0]

        # Find Step 2 section
        if "AMR chuẩn PENMAN:" in generated_text:
            parts = generated_text.split("AMR chuẩn PENMAN:")
            amr_section = parts[-1]  # Take last occurrence
        elif "Bước 2" in generated_text:
            amr_section = generated_text.split("Bước 2")[-1]
        else:
            # Fallback: take everything after last marker
            amr_section = generated_text

        # Extract until balanced parentheses
        lines = amr_section.split('\n')
        amr_lines = []

        for line in lines:
            # Skip empty lines at the start
            if not amr_lines and not line.strip():
                continue

            amr_lines.append(line)

            # Check balance on accumulated text
            accumulated = '\n'.join(amr_lines)
            open_count = accumulated.count('(')
            close_count = accumulated.count(')')

            if open_count == close_count > 0:
                break  # Found complete AMR

        return '\n'.join(amr_lines).strip()

    def generate_step1(self, sentence: str) -> str:
        """
        Step 1: Generate AMR without variables
        """
        # Build prompt
        prompt = MTUP_INFERENCE_TEMPLATE.format(sentence=sentence)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
            truncation=True
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.INFERENCE_CONFIG['max_new_tokens'],
                temperature=config.INFERENCE_CONFIG['temperature'],
                top_p=config.INFERENCE_CONFIG['top_p'],
                repetition_penalty=config.INFERENCE_CONFIG['repetition_penalty'],
                do_sample=config.INFERENCE_CONFIG['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract AMR
        amr_no_vars = self.extract_amr_step1(generated_text)

        return amr_no_vars

    def generate_step2(self, amr_no_vars: str) -> str:
        """
        Step 2: Add variables to AMR
        """
        # Build prompt for Step 2
        prompt = MTUP_INFERENCE_STEP2_TEMPLATE.format(amr_no_vars=amr_no_vars)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
            truncation=True
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.INFERENCE_CONFIG['max_new_tokens'],
                temperature=config.INFERENCE_CONFIG['temperature'],
                top_p=config.INFERENCE_CONFIG['top_p'],
                repetition_penalty=config.INFERENCE_CONFIG['repetition_penalty'],
                do_sample=config.INFERENCE_CONFIG['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract final AMR
        amr_with_vars = self.extract_amr_step2(generated_text)

        return amr_with_vars

    def predict(self, sentence: str, verbose: bool = False) -> str:
        """
        Full two-stage prediction

        Args:
            sentence: Vietnamese sentence
            verbose: Print intermediate results

        Returns:
            Final AMR with variables (Penman format)
        """
        if verbose:
            logger.info(f"\nInput: {sentence}")

        # Step 1: Generate AMR without variables
        amr_no_vars = self.generate_step1(sentence)

        if verbose:
            logger.info(f"Step 1 (no vars): {amr_no_vars}")

        # Step 2: Add variables
        amr_with_vars = self.generate_step2(amr_no_vars)

        if verbose:
            logger.info(f"Step 2 (with vars):\n{amr_with_vars}")

        return amr_with_vars

    def predict_batch(self, sentences: List[str], verbose: bool = False) -> List[str]:
        """Predict for multiple sentences"""
        results = []
        for i, sentence in enumerate(sentences, 1):
            if verbose:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing {i}/{len(sentences)}")
                logger.info(f"{'=' * 80}")

            result = self.predict(sentence, verbose=verbose)
            results.append(result)

        return results


def validate_amr(amr: str) -> Tuple[bool, List[str]]:
    """Validate AMR structure"""
    errors = []

    # Check balanced parentheses
    if amr.count('(') != amr.count(')'):
        errors.append(f"Unbalanced: {amr.count('(')} open, {amr.count(')')} close")

    # Check duplicate nodes
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr)
    duplicates = [n for n in nodes if nodes.count(n) > 1]
    if duplicates:
        errors.append(f"Duplicate nodes: {duplicates}")

    # Check non-empty
    if not amr.strip() or '(' not in amr:
        errors.append("Empty or invalid")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="MTUP Fixed Prediction")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-file", type=str, default="data/public_test.txt", help="Test file")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--sample", type=int, default=None, help="Test on first N samples only")

    args = parser.parse_args()

    # Load predictor
    predictor = MTUPPredictor(args.model)

    # Load test data
    logger.info(f"Loading test data: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if args.sample:
        sentences = sentences[:args.sample]
        logger.info(f"Testing on first {args.sample} samples")

    logger.info(f"Total sentences: {len(sentences)}")

    # Predict
    logger.info("Starting prediction...")
    predictions = predictor.predict_batch(sentences, verbose=args.verbose)

    # Validate
    valid_count = 0
    invalid_count = 0
    for i, pred in enumerate(predictions, 1):
        is_valid, errors = validate_amr(pred)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            if args.verbose:
                logger.warning(f"Invalid AMR #{i}: {errors}")

    logger.info(f"\nValidation: {valid_count}/{len(predictions)} valid ({valid_count/len(predictions)*100:.1f}%)")

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(predictions))

        logger.info(f"Predictions saved to: {output_path}")
    else:
        # Print to console
        print("\n" + "=" * 80)
        print("PREDICTIONS")
        print("=" * 80)
        for i, pred in enumerate(predictions, 1):
            print(f"\n# Sentence {i}")
            print(pred)

    logger.info("Done!")


if __name__ == "__main__":
    main()
