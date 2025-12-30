# Baseline 7B Model Training Guide

**Date**: 2025-12-30
**Status**: Ready to Train
**Purpose**: Fair comparison with MTUP 7B

---

## Overview

This guide describes how to train the **Baseline (Single-Task)** model with Qwen 2.5 7B for Vietnamese AMR parsing. The baseline uses a standard direct mapping approach (Sentence → AMR) without multi-task decomposition, providing a fair comparison point for the MTUP approach.

---

## 1. Comparison: Baseline vs MTUP

| Aspect | Baseline (Single-Task) | MTUP (Multi-Task) |
|--------|------------------------|-------------------|
| **Approach** | Direct Sentence → AMR | Two-Stage: (1) Structure (2) Variables |
| **Prompt** | Single instruction | Multi-task unified prompt |
| **Supervision** | One output signal | Two supervision signals |
| **Model Size** | Qwen 2.5 7B | Qwen 2.5 7B |
| **LoRA Rank** | 128 | 128 |
| **Training Time** | ~12-15 hours | ~12-15 hours |
| **Expected F1** | To be determined | 0.51-0.52 (target) |

**Key Difference**: Baseline learns to generate complete AMR in one step, while MTUP decomposes the task into structure generation + variable binding.

---

## 2. Configuration

### 2.1 Model Configuration

**File**: `config/config.py`

```python
# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Same as MTUP
MAX_SEQ_LENGTH = 2048

# Quantization
USE_4BIT_QUANTIZATION = False  # Disabled - same as MTUP

# LoRA
LORA_CONFIG = {
    "r": 128,                    # Same as MTUP
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
}
```

### 2.2 Training Configuration

```python
TRAINING_CONFIG = {
    "learning_rate": 2e-4,              # Same as MTUP
    "num_train_epochs": 15,             # Same as MTUP
    "per_device_train_batch_size": 2,   # Same as MTUP
    "gradient_accumulation_steps": 8,   # Same as MTUP
    "warmup_steps": 100,
    "optim": "adamw_torch",             # Standard AdamW
    "lr_scheduler_type": "cosine",
    "save_steps": 200,
    "save_total_limit": 5,
}
```

**Note**: All hyperparameters are matched to MTUP 7B for fair comparison.

### 2.3 Prompt Template

```python
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the following Vietnamese sentence to Abstract Meaning Representation (AMR) format. Ensure proper concept alignment and preserve co-references.

### Input:
{sentence}

### Response:
"""
```

**Contrast with MTUP**: MTUP uses Vietnamese prompts with explicit two-stage instructions.

---

## 3. Training

### 3.1 Quick Start

```bash
# Start tmux session
tmux new -s baseline_7b

# Run training
bash START_BASELINE_7B_TRAINING.sh
```

### 3.2 Manual Training

```bash
# With all parameters explicit
python train_baseline.py \
  --epochs 15 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --val-split 0.1 \
  --show-sample
```

### 3.3 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 15 | Number of training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 8 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--val-split` | 0.1 | Validation split ratio |
| `--no-quantize` | False | Disable quantization (auto-disabled) |
| `--max-samples` | None | Limit training samples (for testing) |
| `--show-sample` | False | Display sample training example |

### 3.4 Expected Training Details

```
Model: Qwen/Qwen2.5-7B-Instruct
Total params: ~7B
Trainable params: ~239M (LoRA)
Trainable %: ~3.4%

Training:
- Epochs: 15
- Batch size: 2
- Gradient accumulation: 8
- Effective batch size: 16
- Learning rate: 2e-4
- Total steps: ~1545
- Optimizer: AdamW (torch)
- FP16: Enabled
- Gradient checkpointing: Enabled
```

### 3.5 Monitor Training

```bash
# Attach to tmux
tmux attach -t baseline_7b

# Check logs
tail -f logs/training.log

# GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir outputs/logs
```

---

## 4. Evaluation

### 4.1 Evaluation Command

```bash
python evaluate_baseline_model.py \
  --checkpoint outputs/checkpoints/baseline_7b_final \
  --test-file data/public_test_ground_truth.txt \
  --output results/baseline_7b_evaluation.json \
  --base-model Qwen/Qwen2.5-7B-Instruct
```

### 4.2 Expected Output Format

```json
{
  "precision": 0.XX,
  "recall": 0.XX,
  "f1": 0.XX,
  "valid": XXX,
  "total": 150,
  "errors": X
}
```

### 4.3 Comparison with MTUP

```bash
# After evaluation, compare results
echo "=== Baseline vs MTUP Comparison ==="
echo ""
echo "Baseline 7B:"
cat results/baseline_7b_evaluation.json
echo ""
echo "MTUP 7B:"
cat results/mtup_7b_evaluation.json
```

---

## 5. Push to HuggingFace

### 5.1 When to Push

Push to HuggingFace if:
- Training completed successfully
- F1 score is reasonable (e.g., ≥ 0.40)
- Want to share for comparison/research

### 5.2 Upload Command

```bash
# First, check your HuggingFace username
huggingface-cli whoami

# Then upload (replace YOUR-USERNAME)
hf upload YOUR-USERNAME/vietnamese-amr-baseline-7b \
  outputs/checkpoints/baseline_7b_final \
  --commit-message "Baseline 7B model - F1=X.XX, 15 epochs, LoRA rank 128"
```

### 5.3 Model Card Example

Create `README.md` in checkpoint directory:

```markdown
# Vietnamese AMR Parser - Baseline 7B

Single-task baseline for Vietnamese AMR parsing using Qwen 2.5 7B.

## Performance

- **F1 Score**: 0.XX (public test set, 150 examples)
- **Precision**: 0.XX
- **Recall**: 0.XX
- **Success Rate**: XX% (XXX/150)

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Training**: LoRA (rank 128, alpha 256)
- **Trainable Params**: 239M (~3.4%)
- **Training Data**: VLSP 2025 AMR Corpus (1,842 examples)
- **Epochs**: 15
- **Approach**: Direct single-task mapping (Sentence → AMR)

## Comparison with MTUP

This model serves as a baseline for comparison with the MTUP (Multi-Task Unified Prompt) approach, which decomposes AMR parsing into two explicit subtasks.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "YOUR-USERNAME/vietnamese-amr-baseline-7b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```
```

---

## 6. Files Created

| File | Purpose |
|------|---------|
| `config/config.py` | Baseline configuration (updated to 7B) |
| `train_baseline.py` | Baseline training script |
| `START_BASELINE_7B_TRAINING.sh` | Training launcher script |
| `evaluate_baseline_model.py` | Evaluation script |
| `BASELINE_7B_TRAINING_GUIDE.md` | This documentation |

---

## 7. Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: The configuration is already optimized for 24GB VRAM. If OOM occurs:
- Reduce batch size: `--batch-size 1`
- Increase gradient accumulation: `--grad-accum 16`
- Check other processes using GPU: `nvidia-smi`

### Issue: Slow Training

**Expected**: ~12-15 hours for 15 epochs on Quadro RTX 6000
- Monitor with: `watch -n 1 nvidia-smi`
- Check GPU utilization (should be ~90%+)

### Issue: Model Not Loading in Evaluation

**Check**:
1. Checkpoint path exists: `ls outputs/checkpoints/baseline_7b_final`
2. Required files present: `adapter_model.safetensors`, `adapter_config.json`
3. Base model name matches: `Qwen/Qwen2.5-7B-Instruct`

---

## 8. Next Steps After Training

### 8.1 Immediate

1. ✅ Run evaluation on public test set
2. ✅ Compare F1 with MTUP 7B
3. ✅ Analyze error patterns

### 8.2 Research Analysis

1. **Performance Comparison**:
   - Is MTUP better than baseline?
   - By how much? (absolute and relative improvement)
   - Statistical significance?

2. **Error Analysis**:
   - What types of errors does baseline make?
   - Does MTUP fix specific error categories?
   - Variable binding accuracy comparison

3. **Efficiency Analysis**:
   - Training time comparison
   - Inference speed comparison
   - Model complexity vs performance trade-off

### 8.3 Thesis Integration

Update thesis with:
- Baseline 7B results
- MTUP 7B vs Baseline 7B comparison
- Analysis of multi-task decomposition benefits
- Updated conclusion based on final results

---

## 9. Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Configuration setup | 10 min | ✅ Done |
| Training (15 epochs) | 12-15 hours | ⏳ Pending |
| Evaluation | 10-15 min | ⏳ Pending |
| Analysis & comparison | 30-60 min | ⏳ Pending |
| HuggingFace upload | 5-10 min | ⏳ Pending |
| Thesis update | 1-2 hours | ⏳ Pending |

**Total**: ~1 day (mostly training time)

---

## 10. References

- **MTUP Training Guide**: `UPGRADE_TO_7B_MODEL.md`
- **Config**: `config/config.py`
- **Training Script**: `train_baseline.py`
- **Evaluation**: `evaluate_baseline_model.py`
- **Thesis Chapter**: `THESIS_CHAPTER_MTUP.md`

---

**Ready to start**: Run `bash START_BASELINE_7B_TRAINING.sh` in tmux session!
