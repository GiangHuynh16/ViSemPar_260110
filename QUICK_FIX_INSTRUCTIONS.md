# 🚨 QUICK FIX INSTRUCTIONS - Training Output Issue

## ❌ Current Problem

Model output is completely WRONG:
```
(bi_kich / bi_kich)
(la / la)
(oc / oc)
```

Expected:
```
Task 1: (bi_kịch :domain(chỗ :mod(đó)))
Task 2: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
```

## 🔍 Root Cause

**quick_training_test.py has WRONG data splitting logic (line 30-42)**

Current (WRONG):
```python
for line in content.split('\n'):
    if line.startswith('<|im_start|>system') and current:
        conversations.append('\n'.join(current))
```

This splits by LINES, breaking the conversation blocks!

## ✅ Solution

Use the CORRECT splitting logic from `train_mtup_no_quant.py` (line 86):

```python
blocks = re.split(r'<\|im_end\|>\n\n(?=<\|im_start\|>system)', content.strip())
blocks = [b + '<|im_end|>' if not b.endswith('<|im_end|>') else b for b in blocks]
```

## 🚀 Steps to Fix

### Option 1: Use Existing Working Script (RECOMMENDED)

**Don't use quick_training_test.py!** Use the verified script instead:

```bash
# Copy training script and modify for quick test
python mtup_v2/scripts/train_mtup_no_quant.py \
  --data_path data/train_mtup_unified.txt \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir outputs/quick_test \
  --epochs 1
```

Then modify the script to load only 10 samples:
- Change `load_and_validate_dataset()` to return first 10 samples
- Or create a small data file with 10 samples

### Option 2: Fix quick_training_test.py

Replace the `create_small_test_dataset()` function with:

```python
def create_small_test_dataset(data_path, num_samples=10):
    """Load only first N samples for quick test"""
    import re
    print(f"📦 Loading {num_samples} samples for quick test...")

    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # CORRECT splitting logic
    blocks = re.split(r'<\|im_end\|>\n\n(?=<\|im_start\|>system)', content.strip())
    blocks = [b + '<|im_end|>' if not b.endswith('<|im_end|>') else b for b in blocks]
    blocks = [b.strip() for b in blocks if b.strip()]

    # Take first N samples
    small_dataset = blocks[:num_samples]

    dataset = Dataset.from_dict({"text": small_dataset})
    print(f"✅ Created test dataset with {len(dataset)} samples\n")
    return dataset
```

## 🧪 Verify Before Training

Run debug script to check data is loaded correctly:

```bash
python mtup_v2/scripts/debug_tokenization.py \
  --data_path data/train_mtup_unified.txt
```

Check output has:
- ✅ Full conversation blocks (system + user + assistant)
- ✅ "Task 1:" and "Task 2:" in assistant part
- ✅ Proper AMR format with parentheses

## ⚡ Quick Test After Fix

```bash
# Run with 10 samples, 1 epoch (~5-10 min)
python mtup_v2/scripts/quick_training_test_FIXED.py \
  --data_path data/train_mtup_unified.txt \
  --output_dir outputs/quick_test
```

Expected output after fix:
```
Task 1: (bi_kịch :domain(chỗ :mod(đó)))
Task 2: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
```

## 📊 Full Training (after quick test passes)

```bash
python mtup_v2/scripts/train_mtup_higher_capacity.py \
  --data_path data/train_mtup_unified.txt \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir outputs/mtup_v2_rank64 \
  --epochs 20
```

---

**TL;DR**: quick_training_test.py has wrong data splitting. Fix by using the regex split from train_mtup_no_quant.py line 86, or just use the working script directly with --epochs 1.
