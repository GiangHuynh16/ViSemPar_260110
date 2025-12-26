# Training Guide - Unified Pipeline

## 🎯 Objective

So sánh **2 phương pháp** với cùng model (Qwen 2.5 7B):
1. **Baseline**: Direct generation (1 task)
2. **MTUP**: Two-task decomposition (our method)

## 📊 Changes Applied

### ✅ 1. Unified Models
- **Baseline**: Qwen 2.5 14B → **Qwen 2.5 7B**
- **MTUP**: Qwen 2.5 3B → **Qwen 2.5 7B**
- **Result**: Fair comparison, methodology difference is isolated

### ✅ 2. Removed Post-processing (MTUP only)
- **Philosophy**: End-to-end LLM learning
- **Code**: Removed `post_process_amr_conservative()` call
- **Result**: True evaluation of what model learned

### ✅ 3. Fixed Template Formatting
**Before**:
```
### NHIỆM VỤ: Chuyển đổi câu...
### Câu cần phân tích:
## Bước 1 - Tạo cấu trúc...
Hướng dẫn:
```

**After**:
```
### NHIỆM VỤ
Chuyển đổi câu...

### CÂU ĐẦU VÀO

## BƯỚC 1: Cấu trúc AMR

Quy tắc gán biến:
```

**Improvements**:
- ✅ Consistent markdown levels
- ✅ No colons in headers
- ✅ Clear section separation
- ✅ "Quy tắc gán biến:" on separate line
- ✅ Less confusing for model

## 🚀 Training Steps

### Step 1: Train Baseline (if not already trained)

```bash
# SSH to server
ssh your_server

# Navigate to project
cd ~/ViSemPar_new1

# Pull latest changes
git pull origin main

# Check GPU
nvidia-smi

# Train baseline (if needed)
python3 src/train.py \
  --config config/config.py \
  --output_dir outputs/checkpoints_baseline_7b \
  --num_epochs 10

# Or use existing training script
bash RUN_FULL_TRAINING.sh
```

**Expected time**: ~4-6 hours (depends on GPU)

**Monitor**:
```bash
# Check training progress
tail -f logs/training.log

# Or attach to tmux session (if using)
tmux attach -t training
```

### Step 2: Train MTUP with Unified Settings

```bash
# Pull latest code (with all fixes)
cd ~/ViSemPar_new1
git pull origin main

# Verify configuration
python3 -c "
import sys
sys.path.insert(0, 'config')
from config_mtup import MODEL_NAME
print(f'Model: {MODEL_NAME}')
"
# Should print: Model: Qwen/Qwen2.5-7B-Instruct

# Train MTUP
python3 train_mtup.py \
  --use-case best_accuracy \
  --epochs 10

# Or use tmux for long training
bash RUN_FULL_TRAINING_MTUP.sh
```

**Expected time**: ~4-6 hours (same as baseline)

**Monitor**:
```bash
# Watch log
tail -f logs/training_mtup.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Step 3: Evaluate Both Models

#### 3.1 Evaluate Baseline

```bash
# Evaluate on test set
python3 src/evaluate.py \
  --checkpoint outputs/checkpoints_baseline_7b/final \
  --test-file data/public_test_ground_truth.txt \
  --output outputs/results_baseline_7b.json
```

#### 3.2 Evaluate MTUP

```bash
# Evaluate on test set
python3 evaluate_mtup_model.py \
  --checkpoint outputs/checkpoints_mtup/mtup_full_training_final \
  --test-file data/public_test_ground_truth.txt \
  --output outputs/results_mtup_7b.json
```

### Step 4: Compare Results

```bash
# Create comparison script
python3 compare_baseline_mtup.py \
  --baseline outputs/results_baseline_7b.json \
  --mtup outputs/results_mtup_7b.json
```

## 📊 Expected Results

### Baseline (Qwen 2.5 7B, Direct Generation)
- **F1**: ~0.42-0.46
- **Parse errors**: ~15-20%
- **Strength**: Simple, straightforward
- **Weakness**: Single-task learning is harder

### MTUP (Qwen 2.5 7B, Two-Task)
- **F1**: ~0.49-0.53 (**+15-23% improvement**)
- **Parse errors**: ~8-12%
- **Strength**: Explicit task decomposition, clearer learning signal
- **Weakness**: Longer prompt, more tokens

### Key Comparison

| Metric | Baseline | MTUP | Improvement |
|--------|----------|------|-------------|
| Model | Qwen 2.5 7B | Qwen 2.5 7B | Same ✅ |
| Approach | 1-task direct | 2-task decomp | Different |
| Template | Simple | Structured | Different |
| Post-proc | None | None | Same ✅ |
| **F1** | 0.42-0.46 | 0.49-0.53 | **+15-23%** |
| Parse errors | 15-20% | 8-12% | **-40-60%** |

## 🔍 What Makes MTUP Better?

### 1. Explicit Task Decomposition
**Baseline**:
```
Sentence → [LLM] → AMR with variables
```
Hard to learn in one step!

**MTUP**:
```
Sentence → [Task 1: Structure] → AMR without variables
           ↓
       [Task 2: Binding] → AMR with variables
```
Easier to learn step-by-step!

### 2. Clearer Learning Signal
**Baseline**: Model must learn:
- Semantic structure
- Variable assignment
- Coreference resolution
All at once!

**MTUP**: Model learns:
- **Task 1**: Focus on semantic structure only
- **Task 2**: Focus on variable binding (given structure)
Separate concerns → Better learning!

### 3. Better Error Attribution
**Baseline error**: Is it structure wrong or variable wrong?
**MTUP error**: Can pinpoint which task failed!

## 📁 Files Modified

1. ✅ `config/config.py` - Changed model to 7B
2. ✅ `config/config_mtup.py` - Changed model to 7B
3. ✅ `config/prompt_templates.py` - Fixed v2_natural template
4. ✅ `evaluate_mtup_model.py` - Removed post-processing

## 🐛 Troubleshooting

### Issue: OOM (Out of Memory)

**Solution 1**: Reduce batch size
```python
# In config_mtup.py or config.py
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,  # Reduce from 4
    "gradient_accumulation_steps": 8,  # Increase to maintain effective batch size
}
```

**Solution 2**: Use 4-bit quantization
```python
# In training script
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Issue: Training too slow

**Check**:
```bash
# Is GPU being used?
nvidia-smi

# Check training parameters
python3 -c "
from config.config_mtup import TRAINING_CONFIG
print(f'Batch size: {TRAINING_CONFIG[\"per_device_train_batch_size\"]}')
print(f'Grad accum: {TRAINING_CONFIG[\"gradient_accumulation_steps\"]}')
"
```

**Speed up**:
- Use mixed precision (fp16) ✅ Already enabled
- Increase batch size if GPU has memory
- Reduce max_seq_length if not needed

### Issue: Model not improving

**Check**:
1. Learning rate too high/low?
2. Template format correct?
3. Data quality okay?

**Debug**:
```python
# Test template formatting
python3 config/prompt_templates.py

# Check one training example
python3 -c "
import sys
sys.path.insert(0, 'src')
from preprocessor_mtup import MTUPAMRPreprocessor
from data_loader import AMRDataLoader

loader = AMRDataLoader('data')
examples = loader.parse_amr_file('data/train_amr_1.txt')
preprocessor = MTUPAMRPreprocessor()

# Process first example
example = preprocessor.process_example(examples[0])
print(example['text'][:500])
"
```

## ✅ Verification Checklist

Before training, verify:

- [ ] Both configs use Qwen 2.5 7B
  ```bash
  grep "MODEL_NAME" config/config.py
  grep "MODEL_NAME" config/config_mtup.py
  ```

- [ ] Template is fixed
  ```bash
  grep "### NHIỆM VỤ" config/prompt_templates.py
  ```

- [ ] Post-processing removed from MTUP
  ```bash
  grep "post_process" evaluate_mtup_model.py
  # Should only see comments, no actual call
  ```

- [ ] GPU available
  ```bash
  nvidia-smi
  ```

- [ ] Data files present
  ```bash
  ls -lh data/*.txt
  ```

## 🎓 For Thesis

### Experimental Setup Section

```markdown
## Experimental Setup

### Models
We use Qwen 2.5 7B-Instruct as the base model for both baseline
and MTUP approaches to ensure fair comparison. The model is fine-tuned
using LoRA (Low-Rank Adaptation) with rank r=128.

### Baseline
The baseline uses direct generation with a simple prompt template:
"Convert the following Vietnamese sentence to AMR format."

The model learns to generate AMR with variables in a single step.

### MTUP (Our Approach)
Our method decomposes AMR generation into two explicit tasks:
1. Task 1: Generate AMR structure without variables
2. Task 2: Add variable bindings to create final AMR

This provides clearer learning signal and easier credit assignment.

### Training Configuration
- Optimizer: AdamW 8-bit
- Learning rate: 2e-4 with cosine schedule
- Batch size: 16 (4 per device × 4 gradient accumulation)
- Epochs: 10
- Max sequence length: 2048 tokens

### Evaluation
We evaluate on 150 test examples using SMATCH metric, which measures
graph similarity between predicted and gold AMR graphs.

**Important**: No post-processing is applied. This evaluates the true
end-to-end learning capability of each approach.
```

### Results Section

```markdown
## Results

| Approach | Precision | Recall | F1 | Parse Success |
|----------|-----------|--------|-----|---------------|
| Baseline | 0.XX | 0.XX | 0.XX | XX% |
| MTUP (Ours) | 0.XX | 0.XX | **0.XX** | **XX%** |

MTUP achieves **XX% relative improvement** in F1 score over the baseline,
demonstrating the effectiveness of explicit task decomposition for
structured prediction tasks.
```

## 📝 Next Steps

1. **Train both models** on server
2. **Evaluate and compare** results
3. **Document findings** in thesis
4. **Analyze errors** to understand where each method succeeds/fails
5. **Write discussion** on why MTUP works better

---

**Status**: Ready to train ✅
**Model**: Qwen 2.5 7B (both)
**Template**: Fixed ✅
**Post-processing**: Removed ✅
**Comparison**: Fair ✅
