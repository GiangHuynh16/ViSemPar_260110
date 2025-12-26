# Pipeline Thống Nhất - Vietnamese AMR Parser

## 🎯 Mục Tiêu So Sánh

So sánh **2 phương pháp** với cùng model để đánh giá hiệu quả của MTUP:

| Aspect | Baseline | MTUP (Ours) |
|--------|----------|-------------|
| **Model** | Qwen 2.5 7B | Qwen 2.5 7B |
| **Approach** | Direct generation (1 task) | Two-task decomposition |
| **Template** | Simple prompt | Structured 2-step prompt |
| **Post-processing** | ❌ None (end-to-end LLM) | ❌ None (end-to-end LLM) |
| **Philosophy** | LLM learns directly | LLM learns via task decomposition |

## 📊 Current Issues to Fix

### 1. Model Inconsistency ❌
- **Baseline**: Qwen 2.5 14B (`config/config.py` line 20)
- **MTUP**: Qwen 2.5 3B (`config/config_mtup.py` line 42)
- **Fix**: Both use **Qwen 2.5 7B**

### 2. Post-processing ❌
- **Current**: Conservative post-processing in `evaluate_mtup_model.py`
- **Philosophy**: MTUP should be **end-to-end LLM**, not relying on post-processing
- **Fix**: Remove all post-processing

### 3. Template Formatting Issues ❌
**Current template** (`v2_natural` in `prompt_templates.py` line 34-51):

```python
MTUP_TEMPLATE_V2_NATURAL = """### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
{amr_no_vars}

## Bước 2 - Gán biến cho các khái niệm:
Hướng dẫn:
• Mỗi khái niệm được gán một biến riêng (ví dụ: n, n2, p, c...)
• Khái niệm xuất hiện nhiều lần → dùng chung một biến (đồng tham chiếu)
• Format: (biến / khái_niệm :quan_hệ...)

AMR hoàn chỉnh:
{amr_with_vars}"""
```

**Problems**:
- ❌ Mixed markdown levels (`###` vs `##`)
- ❌ Inconsistent spacing after colons
- ❌ "Hướng dẫn:" and "AMR hoàn chỉnh:" on same line (should be separated)
- ❌ Free text makes parsing harder
- ❌ Not structured like JSON (harder for model to learn boundaries)

**Impact on errors**:
- Unmatched parens likely caused by unclear boundaries
- Model confused about where output starts/ends
- Free text "Hướng dẫn..." may leak into output

## 🔧 Fixes to Implement

### Fix 1: Unify Models to Qwen 2.5 7B

**File**: `config/config.py` (Baseline)
```python
# Line 20: Change from
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# To:
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
```

**File**: `config/config_mtup.py` (MTUP)
```python
# Line 42: Change from
MODEL_NAME = MODELS['qwen2.5-3b']

# To:
MODEL_NAME = MODELS['qwen2.5-7b']
```

### Fix 2: Remove Post-processing

**File**: `evaluate_mtup_model.py`
```python
# Line 162-163: Remove these lines
# POST-PROCESSING: Apply conservative repair pipeline (minimal changes)
final_amr = post_process_amr_conservative(final_amr)

# Keep only:
return final_amr
```

**Also remove**:
- `post_process_amr()` function (line 29-107)
- `post_process_amr_conservative()` function (line 62-107)

### Fix 3: Clean Template Format

**File**: `config/prompt_templates.py`

Replace `MTUP_TEMPLATE_V2_NATURAL` with cleaner version:

```python
MTUP_TEMPLATE_V2_NATURAL = """### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{sentence}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR (chưa có biến)
{amr_no_vars}

## BƯỚC 2: Gán biến

Quy tắc:
- Mỗi khái niệm → một biến (ví dụ: n, p, c)
- Khái niệm lặp lại → dùng chung biến (đồng tham chiếu)
- Format: (biến / khái_niệm :quan_hệ ...)

AMR hoàn chỉnh:
{amr_with_vars}"""
```

**Changes**:
- ✅ Consistent markdown levels (`###` for main, `##` for steps)
- ✅ No space after colon in headers ("### NHIỆM VỤ" not "### NHIỆM VỤ:")
- ✅ Separated "Quy tắc:" section with newline
- ✅ Clearer boundaries
- ✅ Less free text, more structured
- ✅ "AMR hoàn chỉnh:" on separate line from content

### Fix 4: Clean Input Data

**File**: `src/data_loader.py` (if exists) or preprocessing script

Add trimming for trailing `...`:
```python
def clean_sentence(sentence: str) -> str:
    """Clean input sentence"""
    # Remove trailing ...
    sentence = re.sub(r'\.\.\.+\s*$', '', sentence)
    # Trim whitespace
    sentence = sentence.strip()
    return sentence
```

## 📁 Updated Pipeline Structure

```
Input Sentence
      ↓
[PREPROCESSING]
  - Clean sentence (remove ..., trim)
  - Parse AMR from dataset
  - Remove variables → Task 1 target
  - Keep variables → Task 2 target
      ↓
[TRAINING]
  Model: Qwen 2.5 7B
  Template: v2_natural (cleaned version)

  Baseline:
    Prompt: Simple "Convert to AMR"
    Output: AMR with variables

  MTUP:
    Prompt: Structured 2-step
    Output: Step 1 + Step 2 AMRs
      ↓
[INFERENCE]
  - Generate AMR from trained model
  - NO post-processing ✅
  - Extract final AMR from Step 2
      ↓
[EVALUATION]
  - SMATCH scoring
  - Compare Baseline vs MTUP
```

## 🎓 Why These Changes?

### 1. Same Model = Fair Comparison
- Baseline 14B vs MTUP 3B → Not fair (size difference dominates)
- Both 7B → Isolates the effect of **MTUP methodology**

### 2. No Post-processing = True End-to-End
- **Philosophy**: MTUP teaches LLM to generate correct AMR directly
- Post-processing = admission that LLM failed
- If errors occur → improve training, not add band-aids
- **Result**: Cleaner evaluation of what LLM actually learned

### 3. Clean Template = Better Learning
- Clearer boundaries → Model knows where to output
- Consistent format → Easier to learn
- Less free text → Less confusion
- Structured sections → Better separation of tasks

### 4. Clean Input = Less Noise
- Trailing `...` confuses model
- Clean data → Clean learning

## 📊 Expected Impact

### Before Fixes
| Metric | Baseline | MTUP | Issue |
|--------|----------|------|-------|
| Model | 14B | 3B | Unfair comparison |
| F1 | ??? | 0.48 | Can't compare |
| Parse errors | ??? | 30% | Post-processing hides real errors |

### After Fixes
| Metric | Baseline | MTUP | Comparison |
|--------|----------|------|------------|
| Model | 7B | 7B | ✅ Fair |
| F1 (expected) | ~0.42-0.45 | ~0.48-0.52 | +13-23% improvement from MTUP |
| Parse errors | ~15-20% | ~10-15% | MTUP should have fewer errors |

**Hypothesis**: MTUP with 7B should **outperform** Baseline with 7B because:
- Task decomposition is easier to learn
- Explicit structure guidance
- Two-stage supervision

## 🚀 Next Steps

1. ✅ **Unify models** → Both use Qwen 2.5 7B
2. ✅ **Remove post-processing** → End-to-end LLM only
3. ✅ **Fix template** → Cleaner format
4. ✅ **Update code** → Apply all changes
5. ✅ **Train Baseline** → Get baseline F1 score
6. ✅ **Re-train MTUP** → With same model, clean template
7. ✅ **Compare** → Evaluate improvement

## 📝 Files to Modify

1. `config/config.py` - Change model to 7B
2. `config/config_mtup.py` - Change model to 7B
3. `config/prompt_templates.py` - Fix v2_natural template
4. `evaluate_mtup_model.py` - Remove post-processing
5. `src/preprocessor_mtup.py` - Add input cleaning (if needed)

---

**Philosophy**: End-to-end LLM learning with clean, structured prompts → Better generalization than post-processing hacks
