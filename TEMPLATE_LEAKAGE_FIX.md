# Template Leakage Fix - MTUP Training

## Problem Discovery

**Date**: 2025-12-29
**Training Duration**: ~9 hours
**Result**: Complete failure - all 150 evaluation examples failed

### Symptom
Model was outputting template placeholder text literally:
```
(biến / khái_niệm :quan_hệ ...) AMR cuối cùng: (n / nhớ :pivot(t / tôi) ...)
```

This caused parsing errors on 100% of test cases.

### Root Cause

The MTUP template contained example placeholder text that the model learned to output:

**Before (WRONG)**:
```python
# config/prompt_templates.py line 50
## BƯỚC 2: AMR hoàn chỉnh với biến

Quy tắc gán biến:
- Mỗi khái niệm → một biến duy nhất
- Khái niệm lặp lại → dùng chung biến
- Format: (biến / khái_niệm :quan_hệ ...)  # ❌ MODEL LEARNED THIS!

AMR cuối cùng:  # ❌ MODEL LEARNED THIS TOO!
{amr_with_vars}
```

The model treated `"- Format: (biến / khái_niệm :quan_hệ ...)"` and `"AMR cuối cùng:"` as part of the expected output pattern, not as instructions.

## Fix Applied

### Template V2_NATURAL (Current training template)

**File**: `config/prompt_templates.py` lines 45-51

**After (CORRECT)**:
```python
## BƯỚC 2: AMR hoàn chỉnh với biến

Quy tắc gán biến:
- Mỗi khái niệm được gán một biến duy nhất
- Khái niệm lặp lại sử dụng chung biến đã gán

{amr_with_vars}
```

**Changes**:
1. ✅ Removed `- Format: (biến / khái_niệm :quan_hệ ...)` placeholder
2. ✅ Removed `AMR cuối cùng:` header
3. ✅ Made instructions more natural and descriptive
4. ✅ Direct transition from rules to actual AMR content

### Template V5_COT (Future-proofing)

**File**: `config/prompt_templates.py` lines 117-125

Also removed the same placeholder from V5_COT template:
```python
- Format cuối: (biến / khái_niệm :quan_hệ ...)  # ❌ REMOVED
```

## Verification

Confirmed no more placeholder text remains:
```bash
grep -r "Format.*biến.*khái_niệm" config/  # No matches ✅
grep -r "AMR cuối cùng:" config/           # No matches ✅
```

## Impact

### Failed Model
- **Training time**: ~9 hours wasted
- **Disk space**: ~2-3 GB (needs cleanup)
- **Usability**: 0% - completely unusable
- **F1 score**: Cannot be measured (parsing fails)

### Required Action
1. ✅ Fix template (DONE)
2. ⏳ Clean up failed model using `bash CLEANUP_FAILED_MODEL.sh`
3. ⏳ Re-train model (~9 hours)
4. ⏳ Verify evaluation works correctly

## Cleanup Command

```bash
bash CLEANUP_FAILED_MODEL.sh
```

This will remove:
- `models/mtup_reentrancy_final/` (~1-2 GB)
- `models/checkpoints/mtup_reentrancy/` (~1-2 GB)
- `results/evaluation/mtup_reentrancy_eval.json`

## Prevention

**Key Lesson**: Templates should contain only:
1. Task instructions (what to do)
2. Input placeholders (`{sentence}`, `{amr_no_vars}`, `{amr_with_vars}`)
3. Formatting rules (descriptive text)

**Never include**:
- Example placeholder AMR structures like `(biến / khái_niệm :quan_hệ ...)`
- Section headers that could be learned as output patterns like `AMR cuối cùng:`
- Symbolic representations that look like target output

The model should learn AMR structure from the actual training data (VLSP corpus), not from placeholder examples in the prompt.

## Files Modified

1. `config/prompt_templates.py`
   - Line 45-51: Fixed MTUP_TEMPLATE_V2_NATURAL
   - Line 117-125: Fixed MTUP_TEMPLATE_V5_COT

2. `CLEANUP_FAILED_MODEL.sh` (NEW)
   - Script to remove failed training outputs

## Next Steps

1. User runs: `bash CLEANUP_FAILED_MODEL.sh`
2. User runs: `bash scripts/run_training_mtup.sh` (in tmux session)
3. Training completes in ~9 hours
4. Evaluate and verify no template leakage
5. If successful, continue with baseline training
