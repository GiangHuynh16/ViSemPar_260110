# 🔍 Investigation: Is max_length=512 Causing Invalid AMRs?

## 📋 Current Situation

**Results from checkpoint-1500:**
- ✅ **137/150 valid AMRs (91.3%)** - EXCELLENT!
- ❌ **13/150 invalid AMRs (8.7%)**

**Concern:** The 13 invalid AMRs might be caused by `max_length=512` truncating long sentences.

---

## 🎯 Investigation Steps

### Step 1: Identify Invalid AMR Indices (2 minutes)

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

python3 identify_invalid_amrs.py
```

**Expected output:**
- List of 13 sentence indices with invalid AMRs
- Error types for each (unbalanced parentheses, duplicate nodes, etc.)
- Saves to `invalid_amr_indices.txt`

---

### Step 2: Analyze Sentence Lengths (3 minutes)

```bash
bash CHECK_MAX_LENGTH.sh
```

**This will show:**
1. **Total sentence length statistics**
   - Min, max, average prompt lengths
   - How many sentences exceed 400, 450, 500, 512 tokens

2. **Top 10 longest sentences**
   - With their token counts
   - Helps identify problematic cases

3. **Correlation with invalid AMRs**
   - Checks if invalid AMRs correspond to long sentences
   - Estimates total tokens (prompt + output)

4. **Clear recommendation**
   - Whether to retrain with larger max_length
   - Or accept current 91.3% as excellent

---

## 📊 How to Interpret Results

### Scenario 1: Truncation IS the Problem ❌

**Signs:**
- Many sentences have prompts >450 tokens
- Invalid AMRs mostly occur on long sentences
- Total tokens (prompt + AMR) > 512 for invalid cases

**Example output:**
```
⚠️ 15 sentences have prompts exceeding 450 tokens
❌ 8 invalid AMRs correlate with long sentences
⚠️ Sentence #47: 487 tokens (prompt) + 156 (AMR) = 643 > 512
   LIKELY TRUNCATED
```

**Recommendation:** Retrain with `max_length=768` or `1024`

---

### Scenario 2: Truncation is NOT the Problem ✅

**Signs:**
- All prompts <400 tokens
- Invalid AMRs occur randomly (not just on long sentences)
- Most sentences have room for AMR output

**Example output:**
```
✅ Max prompt length: 387 tokens
✅ All prompts fit within 512 with room for output
✅ Invalid AMRs spread across short and long sentences
```

**Recommendation:**
- **Accept 91.3% valid as EXCELLENT result** (exceeds 80-90% target)
- The 13 invalid AMRs are model errors, not truncation
- Document limitation in thesis: "91.3% structural validity"
- Calculate SMATCH on 137 valid AMRs only

---

## 🔧 If Retraining is Needed

### Option A: Increase max_length to 768

**Edit:** `config/config_fixed.py`
```python
MAX_SEQ_LENGTH = 768  # Changed from 512
```

**Pros:**
- Handles longer sentences
- Potentially fixes all 13 invalid AMRs

**Cons:**
- Training time: +30-40% (3.5-4 hours instead of 2.5)
- Memory usage: +40% (may need to reduce batch size)
- Slower inference

**Command:**
```bash
bash TRAIN_BASELINE_FIXED.sh
```

---

### Option B: Increase max_length to 1024

**Edit:** `config/config_fixed.py`
```python
MAX_SEQ_LENGTH = 1024  # Changed from 512
```

**Pros:**
- Handles ALL sentences comfortably

**Cons:**
- Training time: +60-80% (4-5 hours)
- Memory usage: +80% (may OOM, need gradient_accumulation=32)
- Much slower inference

**Only use if:** Analysis shows many sentences >600 tokens

---

## 🎯 If NOT Retraining (91.3% is Great!)

### How to Calculate SMATCH on Valid AMRs Only

**Option 1: Filter predictions file**

```bash
python3 filter_valid_amrs.py \
    --predictions evaluation_results/checkpoint_comparison/checkpoint-1500.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred evaluation_results/checkpoint-1500_valid_only.txt \
    --output-gold evaluation_results/gold_valid_only.txt
```

Then run SMATCH:
```bash
python -m smatch -f \
    evaluation_results/checkpoint-1500_valid_only.txt \
    evaluation_results/gold_valid_only.txt \
    --significant 4
```

---

**Option 2: Manual calculation**

Calculate SMATCH and document:
- SMATCH F1 on 137/150 valid AMRs: X.XX
- 13 AMRs (8.7%) excluded due to structural errors
- Structural validity: 91.3%

---

## 📈 Decision Framework

```
┌─────────────────────────────────────┐
│ Run: bash CHECK_MAX_LENGTH.sh      │
└─────────────────┬───────────────────┘
                  │
         ┌────────▼──────────┐
         │ Check Results     │
         └────────┬──────────┘
                  │
        ┌─────────▼────────────┐
        │ Sentences >450 tokens?│
        └─────────┬────────────┘
                  │
        ┌─────────▼─────────────┐
        │ YES (>5 sentences)    │
        │ → Retrain max=768     │
        └───────────────────────┘
                  │
        ┌─────────▼─────────────┐
        │ NO (<5 sentences)     │
        │ → Accept 91.3% valid  │
        │ → Calculate SMATCH    │
        │ → Document limitation │
        └───────────────────────┘
```

---

## ✅ Success Criteria Reminder

**Original targets (from START_HERE.md):**
- Valid AMRs: 80-90% ✅ **Achieved: 91.3%!**
- All 150 samples generated: ✅ **Yes**
- SMATCH F1: ~0.55 (to calculate)

**Current status:** EXCEEDED expectations on structural validity!

---

## 🚀 Quick Commands

```bash
# 1. Identify invalid AMRs
python3 identify_invalid_amrs.py

# 2. Check sentence lengths
bash CHECK_MAX_LENGTH.sh

# 3. Based on results, either:
#    - Accept 91.3% and calculate SMATCH on valid AMRs
#    - OR retrain with larger max_length
```

---

## 📝 For Thesis Documentation

**If accepting 91.3%:**

Document in Section 4.7.2:

> Our baseline model achieved **91.3% structural validity** (137/150 AMRs),
> exceeding our target of 80-90%. The remaining 8.7% (13 AMRs) contained
> structural errors (unbalanced parentheses or duplicate node variables),
> primarily due to model generation errors rather than input truncation.
> SMATCH F1 was calculated on the 137 structurally valid AMRs, yielding
> F1 = X.XX.

**If retraining with larger max_length:**

Document the change:

> Initial training with max_length=512 achieved 91.3% validity. Analysis
> revealed that N sentences required longer context windows. We retrained
> with max_length=768, achieving XX% validity (YYY/150 AMRs).

---

## 💡 My Recommendation

Based on the excellent 91.3% result:

1. **Run the investigation** to confirm it's not truncation
2. **If truncation is not the issue:**
   - Accept 91.3% as excellent (exceeds target!)
   - Calculate SMATCH on 137 valid AMRs
   - Document the 8.7% limitation
   - Move on to MTUP comparison

3. **Only retrain if:**
   - Analysis clearly shows truncation is the cause
   - AND you have time (~4 hours for retraining)
   - AND memory allows (may need batch size adjustments)

**Time saved by not retraining:** 4 hours
**Quality difference:** Likely minimal (maybe 92-95% vs 91.3%)

---

**Next:** Run `bash CHECK_MAX_LENGTH.sh` and report results!
