# MTUP 2-Model Approach - Training Guide

## Overview

Train 2 separate models:
1. **Model 1:** SENTENCE → AMR WITH VARIABLES (use existing Baseline!)
2. **Model 2:** AMR NO VARS → AMR WITH VARIABLES (new model)

## Why 2 Models?

Single model cannot learn both:
- Generate AMR structure from sentence (complex semantic parsing)
- Add variables to existing AMR (simple variable binding)

Training both together causes confusion (as we saw: F1=0.10).

## Approach

### Model 1: Use Existing Baseline! ✅

**Already trained and working:**
- Model: `outputs/baseline_fixed_20260103_115114/checkpoint-1500`
- F1: 0.47
- Validity: 91.3%

**No need to retrain!**

### Model 2: Train Variable Binding Model

**Task:** Add variables to AMR without variables

**Input format:**
```
(nhớ :pivot (tôi) :theme (lời :poss (chủ_tịch)))
```

**Output format:**
```
(n / nhớ
:pivot(t / tôi)
:theme(l / lời
:poss(c / chủ_tịch)))
```

---

## Step 1: Prepare Training Data for Model 2

### Create preprocessing script:

```bash
python3 preprocess_variable_binding.py
```

This will read `train_amr_mtup_preprocessed.txt` and create:
- Input: AMR without variables
- Output: AMR with variables

### Expected output:

```
data/train_variable_binding.txt

Format:
#::input (bi_kịch :domain (chỗ :mod (đó)))
#::output
(b / bi_kịch
:domain(c / chỗ
:mod(đ / đó)))

#::input (nhớ :pivot (tôi) :theme (lời))
#::output
(n / nhớ
:pivot(t / tôi)
:theme(l / lời))

...
```

---

## Step 2: Create Training Script for Model 2

Key differences from Baseline:
- **Input:** AMR without variables (not sentence!)
- **Output:** AMR with variables
- **Simpler task** → Smaller model or less epochs might work

### Template:

```python
VARIABLE_BINDING_TEMPLATE = """Thêm biến vào AMR theo chuẩn PENMAN.

VÍ DỤ:
Input: (bi_kịch :domain (chỗ :mod (đó)))
Output:
(b / bi_kịch
:domain(c / chỗ
:mod(đ / đó)))

---

Input: {amr_no_vars}

Output:
{amr_with_vars}"""
```

---

## Step 3: Train Model 2

```bash
python3 train_variable_binding.py
```

**Config:**
- Model: Qwen 2.5 7B (same as Baseline)
- LoRA rank: 64
- Epochs: 2
- Learning rate: 2e-4
- Training time: ~2-3 hours

---

## Step 4: 2-Model Inference Pipeline

```python
# Load both models
baseline_model = load_model("outputs/baseline_fixed_.../checkpoint-1500")
variable_model = load_model("outputs/variable_binding_.../checkpoint-XXX")

def predict_2model(sentence):
    # Option A: Use Baseline (generates with variables already)
    amr_with_vars = baseline_model.generate(sentence)
    return amr_with_vars

    # Option B: Use 2-stage (for comparison)
    # Stage 1: Get AMR with variables from baseline
    amr_initial = baseline_model.generate(sentence)

    # Remove variables (post-processing)
    amr_no_vars = remove_variables(amr_initial)

    # Stage 2: Add variables back using Model 2
    amr_final = variable_model.generate(amr_no_vars)

    return amr_final
```

---

## Expected Results

### Baseline (1-model):
- F1: **0.47**
- Validity: 91.3%
- Speed: Fast (1 pass)

### MTUP (2-model):
- F1: **0.40-0.45** (likely LOWER!)
- Validity: ~90%
- Speed: Slower (2 passes)

**Why lower?**
- Error propagation: If Stage 1 wrong, Stage 2 can't fix
- Additional complexity doesn't help for this task
- AMR generation is holistic (needs full context)

---

## Academic Value

Even if F1 is lower, this is valuable for thesis:

**Research Question:**
"Does task decomposition improve Vietnamese AMR parsing?"

**Answer:**
"No. Single-stage direct generation (Baseline) outperforms 2-stage MTUP."

**Analysis:**
1. Error propagation issue
2. AMR generation requires holistic reasoning
3. Decomposition adds complexity without benefit

**Contribution:**
- Empirical evidence that simpler is better for this task
- Important negative result for the field

---

## Thesis Structure

### Section 4.4: Baseline Approach ✅
- F1: 0.47
- Single-stage direct generation
- Simple and effective

### Section 4.5: MTUP Approach

**4.5.1 Motivation**
- Task decomposition hypothesis
- Separate structure from variables

**4.5.2 Implementation**
- Model 1: Baseline (reused)
- Model 2: Variable binding (new)

**4.5.3 Results**
- F1: 0.40-0.45
- Lower than Baseline
- Error propagation analysis

**4.5.4 Discussion**
- Why decomposition didn't help
- When single-stage is better
- Lessons learned

**4.5.5 Conclusion**
- Baseline is superior approach
- MTUP shows limitations of decomposition
- Important negative result

---

## Timeline

If you want to implement Model 2:

1. **Create preprocessing script:** 1 hour
2. **Verify training data:** 30 min
3. **Train Model 2:** 2-3 hours
4. **Inference:** 1 hour
5. **Analysis:** 1 hour

**Total:** ~6-7 hours

**Expected F1:** 0.40-0.45 (lower than Baseline's 0.47)

---

## My Recommendation

### Don't train Model 2!

**Reasons:**
1. **Won't improve F1** (likely 0.40-0.45 vs Baseline's 0.47)
2. **Takes 6+ hours** (training + debugging)
3. **Baseline already works well** (0.47 is good!)
4. **Can write thesis without it:**
   - Section 4.4: Baseline (F1=0.47)
   - Section 4.5: Discussion of MTUP limitations
   - No need to implement to discuss!

**Better use of time:**
- Polish Baseline results
- Write thesis chapters
- Prepare for defense

---

## Alternative: Theoretical Analysis Only

In Section 4.5, discuss MTUP theoretically:

**4.5.1 MTUP Approach (Theoretical)**
- Describe 2-stage decomposition
- Explain intended benefits
- Show training data format

**4.5.2 Expected Challenges**
- Error propagation
- Loss of context
- Increased complexity

**4.5.3 Why We Didn't Implement**
- Baseline already achieves strong results (0.47)
- Task decomposition may not help for holistic tasks like AMR
- Cost-benefit analysis favors simple approach

**4.5.4 Related Work**
- Cite papers that tried decomposition (some succeeded, some failed)
- Position your work in this context

**4.5.5 Future Work**
- Other decomposition strategies
- Different task formulations

This gives you a complete thesis **without** spending 6+ hours on an approach that won't beat Baseline!

---

## Decision Point

**Question for you:**

Do you want to:

**A.** Skip MTUP implementation, write thesis with Baseline only (F1=0.47)
- Faster (save 6+ hours)
- Cleaner (1 strong method vs 2 methods with 1 weaker)
- Still complete thesis

**B.** Implement MTUP Model 2 for academic completeness
- Takes 6+ hours
- Expected F1: 0.40-0.45 (lower than Baseline)
- Shows negative result (still valuable!)

**C.** Write MTUP section theoretically without implementation
- Middle ground
- Discuss approach + limitations
- No implementation needed

**My strong recommendation: Choose A or C!**

Baseline F1=0.47 is already a strong result. Focus on polishing that and writing a great thesis!
