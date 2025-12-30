#!/usr/bin/env python3
"""
Evaluate Baseline Model on Test Data
Generates predictions and computes SMATCH scores
"""

import sys
import torch
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, 'src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import AMRDataLoader


def post_process_amr_conservative(amr_string: str) -> str:
    """
    Minimal post-processing - extract AMR only, NO heavy processing
    Philosophy: Let LLM output speak for itself, trust the model
    """
    if not amr_string or len(amr_string) < 3:
        return "(amr-empty)"

    amr = amr_string.strip()

    # Simply find the first '(' and take everything from there
    # This removes prompt text but keeps AMR as-is
    if '(' in amr:
        amr = amr[amr.index('('):]

    # Basic whitespace normalization only (no structural changes!)
    amr = re.sub(r'\s+', ' ', amr).strip()

    return amr


def load_model(checkpoint_path: str, base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load baseline model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    print(f"Base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("✓ Model loaded successfully")
    return model, tokenizer


def generate_baseline_prediction(model, tokenizer, sentence: str, max_length=512):
    """Generate AMR using baseline (single-task) approach"""

    # Use same prompt template as training (UPDATED - Vietnamese instructions)
    prompt = f"""Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation).

Quy tắc quan trọng:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới (ví dụ: chủ_tịch, môi_trường)
- Gán biến cho mỗi khái niệm (ví dụ: c / chủ_tịch)
- Sử dụng quan hệ chuẩn AMR (:ARG0, :ARG1, :time, :location, etc.)
- Giữ nguyên cấu trúc cây với dấu ngoặc đơn cân bằng
- Đảm bảo tất cả biến được định nghĩa trước khi sử dụng

Câu tiếng Việt: {sentence}

AMR:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Greedy decoding
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AMR from response (UPDATED for Vietnamese markers)
    if "AMR:" in result:
        # Split by "AMR:" and take the last part
        amr = result.split("AMR:")[-1].strip()
    elif "### Response:" in result:
        # Fallback for old English template
        amr = result.split("### Response:")[-1].strip()
    else:
        # Fallback: try to find first '('
        if '(' in result:
            amr = result[result.index('('):]
        else:
            amr = result

    # Post-process
    amr = post_process_amr_conservative(amr)

    return amr


def evaluate_on_test_data(model, tokenizer, test_file: Path, max_samples=None):
    """Evaluate model on test set"""
    print(f"\nLoading test data from: {test_file}")

    # Load test data
    loader = AMRDataLoader()
    examples = loader.load_from_file(str(test_file))

    if max_samples:
        examples = examples[:max_samples]

    print(f"✓ Loaded {len(examples)} test examples")

    # Generate predictions
    predictions = []
    ground_truth = []
    errors = 0

    print("\nGenerating predictions...")
    for i, example in enumerate(tqdm(examples, desc="Generating")):
        sentence = example['sentence']
        gold_amr = example['amr']

        try:
            pred_amr = generate_baseline_prediction(model, tokenizer, sentence)
            predictions.append(pred_amr)
            ground_truth.append(gold_amr)
        except Exception as e:
            print(f"\nError on example {i+1}: {e}")
            predictions.append("(error)")
            ground_truth.append(gold_amr)
            errors += 1

    print(f"\n✓ Generated {len(predictions)} predictions")
    if errors > 0:
        print(f"⚠️  {errors} errors during generation")

    # Compute SMATCH scores
    print("\nComputing SMATCH scores...")

    try:
        import smatch
    except ImportError:
        print("❌ smatch package not found. Install with: pip install smatch")
        return None

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    valid_count = 0

    # Import linearize function
    from amr_utils import linearize_amr

    for i, (pred, gold) in enumerate(tqdm(list(zip(predictions, ground_truth)), desc="Evaluating")):
        try:
            # Linearize AMRs for SMATCH
            pred_linear = linearize_amr(pred)
            gold_linear = linearize_amr(gold)

            # Compute SMATCH
            best, test, gold_t = smatch.get_amr_match(
                pred_linear, gold_linear,
                justinstance=False, justattribute=False, justrelation=False
            )

            if test > 0:
                precision = best / test
            else:
                precision = 0.0

            if gold_t > 0:
                recall = best / gold_t
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            valid_count += 1

        except Exception as e:
            if i < 3:  # Show first 3 errors
                print(f"\nError computing SMATCH for example {i+1}: {e}")
            continue

    # Calculate averages
    if valid_count > 0:
        avg_precision = precision_sum / valid_count
        avg_recall = recall_sum / valid_count
        avg_f1 = f1_sum / valid_count
    else:
        avg_precision = avg_recall = avg_f1 = 0.0

    results = {
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4),
        "f1": round(avg_f1, 4),
        "valid": valid_count,
        "total": len(examples),
        "errors": len(examples) - valid_count
    }

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Processed: {results['valid']}/{results['total']} examples")
    print(f"Errors:    {results['errors']}")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test data file')
    parser.add_argument('--output', type=str, default='results/baseline_evaluation.json',
                        help='Output file for results')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of test samples')
    parser.add_argument('--base-model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Base model name')

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.base_model)

    # Evaluate
    test_file = Path(args.test_file)
    results = evaluate_on_test_data(model, tokenizer, test_file, args.max_samples)

    # Save results
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
