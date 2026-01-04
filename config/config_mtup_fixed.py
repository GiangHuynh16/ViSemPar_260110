"""
MTUP Fixed Configuration - Based on Baseline Success
Minimal prompt + Instruction masking + Early stopping
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512  # Same as baseline (sufficient for Vietnamese AMR)

# ==============================================================================
# LORA CONFIGURATION - SAME AS BASELINE
# ==============================================================================

LORA_CONFIG = {
    "r": 64,                    # Same as baseline
    "lora_alpha": 128,          # 2x rank
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
}

# ==============================================================================
# TRAINING CONFIGURATION - BASED ON BASELINE SUCCESS
# ==============================================================================

TRAINING_CONFIG = {
    "learning_rate": 2e-4,              # Same as baseline
    "num_train_epochs": 2,              # FIXED: 2 epochs (baseline proven optimal)
    "per_device_train_batch_size": 1,   # Same as baseline
    "gradient_accumulation_steps": 16,  # Effective batch = 16
    "warmup_steps": 50,                 # Same as baseline
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 100,                  # FIXED: Save every 100 steps (find sweet spot)
    "save_total_limit": 12,             # Keep all checkpoints for 2 epochs
    "fp16": False,                      # Use bfloat16 instead
    "bf16": True,                       # FIXED: bfloat16 like baseline
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "dataloader_num_workers": 4,
    "remove_unused_columns": False,     # Important for custom data
}

# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================

INFERENCE_CONFIG = {
    "temperature": 0.3,              # Same as baseline
    "top_p": 0.95,                   # Same as baseline
    "repetition_penalty": 1.2,       # Same as baseline
    "max_new_tokens": 512,           # Same as baseline
    "do_sample": False,              # Greedy decoding
}

# ==============================================================================
# MTUP SPECIFIC CONFIGURATION
# ==============================================================================

MTUP_CONFIG = {
    # Template type: 'minimal', 'ultra_minimal', or 'recommended'
    "template_type": "ultra_minimal",    # Minimal prompt with Penman example

    # Instruction masking (CRITICAL FIX)
    "use_instruction_masking": True,     # NEW: Mask prompt like baseline

    # Format
    "use_graph_format": True,            # Keep Penman multi-line format

    # Two-stage markers for extraction
    "step1_marker": "AMR không biến:",
    "step2_marker": "AMR chuẩn PENMAN:",
}

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

DATA_CONFIG = {
    "train_files": ["train_amr_1.txt", "train_amr_2.txt"],
    "public_test_file": "public_test.txt",
    "public_test_ground_truth": "public_test_ground_truth.txt",
    "validation_split": 0.05,         # 5% like baseline
    "max_samples": None,              # Use all data
}

# ==============================================================================
# COMPARISON: MTUP FIXED vs BASELINE
# ==============================================================================

COMPARISON_TABLE = """
┌─────────────────────┬──────────────┬──────────────┐
│ Configuration       │ Baseline     │ MTUP Fixed   │
├─────────────────────┼──────────────┼──────────────┤
│ Prompt Lines        │ 3            │ 10 (w/example)│
│ Instruction Mask    │ ✅ Yes       │ ✅ Yes       │
│ Epochs              │ 2            │ 2            │
│ Save Steps          │ 100          │ 100          │
│ Learning Rate       │ 2e-4         │ 2e-4         │
│ Precision           │ bfloat16     │ bfloat16     │
│ Max Length          │ 512          │ 512          │
│ LoRA Rank           │ 64           │ 64           │
│ Batch Size (eff)    │ 16           │ 16           │
└─────────────────────┴──────────────┴──────────────┘

Key Improvements over Old MTUP:
✅ Minimal prompt (was 20+ lines with verbose instructions)
✅ Instruction masking (was missing, caused copying behavior)
✅ 2 epochs (was 15, caused severe overfitting)
✅ Penman examples in prompt (was missing)
✅ Save every 100 steps (was 200, missed sweet spot)
✅ bfloat16 precision (was fp16)

Expected Improvement:
- Old MTUP: Generated explanations, not valid Penman
- MTUP Fixed: Valid Penman format, ~90%+ validity
- Hypothesis: MTUP may outperform Baseline by 3-5% F1
"""

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

OUTPUT_PREFIX = "mtup_fixed"
MODEL_SAVE_NAME = f"{OUTPUT_PREFIX}_qwen7b"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_config():
    """Print configuration summary"""
    print("=" * 80)
    print("MTUP FIXED CONFIGURATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Template: {MTUP_CONFIG['template_type']}")
    print(f"Instruction Masking: {MTUP_CONFIG['use_instruction_masking']}")
    print(f"Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"Effective Batch Size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"LoRA Rank: {LORA_CONFIG['r']}")
    print(f"Max Length: {MAX_SEQ_LENGTH}")
    print("=" * 80)
    print(COMPARISON_TABLE)


if __name__ == "__main__":
    print_config()
